/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package sessionstate provides a plugin that tracks agentic session
// history and publishes it before flow control for session-aware plugins.
package sessionstate

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/duration"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

const (
	defaultEvictionTTL           = time.Hour
	defaultEvictionSweepInterval = 5 * time.Minute
	// defaultMaxSessions holds about 65 MiB of session state when every
	// identity is defaultMaxIdentityBytes long.
	defaultMaxSessions = 100000
	// defaultMaxIdentityBytes is several times the length of the session IDs
	// the built-in agent headers carry.
	defaultMaxIdentityBytes = 256
)

// Parameters configures idle session eviction and the bounds on tracked state.
type Parameters struct {
	// EvictionTTLSeconds is the maximum idle time before session state is removed.
	// Zero disables eviction.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds,omitempty"`
	// EvictionSweepSeconds is how often idle session state is scanned.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds,omitempty"`
	// MaxSessions bounds the number of tracked sessions; the least recently
	// active idle session is evicted when the bound is reached. Zero disables
	// the bound.
	MaxSessions int `json:"maxSessions,omitempty"`
	// MaxIdentityBytes is the longest agent identity tracked; requests with a
	// longer identity get no session state. Zero disables the limit.
	MaxIdentityBytes int `json:"maxIdentityBytes,omitempty"`
}

func (p Parameters) validate() error {
	if p.EvictionTTLSeconds < 0 {
		return fmt.Errorf("evictionTtlSeconds must be >= 0, got %v", p.EvictionTTLSeconds)
	}
	if p.EvictionTTLSeconds > 0 {
		if err := duration.ValidateSeconds("evictionTtlSeconds", p.EvictionTTLSeconds); err != nil {
			return err
		}
	}
	if err := duration.ValidateSeconds("evictionSweepSeconds", p.EvictionSweepSeconds); err != nil {
		return err
	}
	if p.MaxSessions < 0 {
		return fmt.Errorf("maxSessions must be >= 0, got %d", p.MaxSessions)
	}
	if p.MaxIdentityBytes < 0 {
		return fmt.Errorf("maxIdentityBytes must be >= 0, got %d", p.MaxIdentityBytes)
	}
	return nil
}

var (
	_ fwkrh.RequestHeaderProcessor = &Producer{}
	_ fwkrh.DataProducer           = &Producer{}
	_ fwkrh.PreRequest             = &Producer{}
	_ fwkrh.ResponseBodyProcessor  = &Producer{}
	_ fwkrh.RequestAbandoned       = &Producer{}
	_ fwkplugin.ConsumerPlugin     = &Producer{}
)

// Producer tracks session history within one EPP instance.
type Producer struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey

	registry SessionStateRegistry

	evictionTTL           time.Duration
	evictionSweepInterval time.Duration
	maxIdentityBytes      int
}

// Factory builds a session-state producer.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		EvictionTTLSeconds:   defaultEvictionTTL.Seconds(),
		EvictionSweepSeconds: defaultEvictionSweepInterval.Seconds(),
		MaxSessions:          defaultMaxSessions,
		MaxIdentityBytes:     defaultMaxIdentityBytes,
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", SessionStateProducerType, name, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", SessionStateProducerType, name, err)
	}

	p := &Producer{
		typedName:             fwkplugin.TypedName{Type: SessionStateProducerType, Name: name},
		dk:                    SessionStateDataKey.WithNonEmptyProducerName(name),
		evictionTTL:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		evictionSweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
		maxIdentityBytes:      params.MaxIdentityBytes,
	}
	p.registry.maxSessions = params.MaxSessions
	if handle != nil && p.evictionTTL > 0 {
		go p.runEviction(handle.Context())
	}
	return p, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the SessionState request attribute written by this producer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: SessionState{}}
}

// Consumes declares agent identity as a required input. The agent-identity
// plugin is intentionally not a default producer, so operators must enable an
// identity provider explicitly rather than having one auto-created.
func (p *Producer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// RequestHeader marks the session as seen and publishes its state before flow
// control. The attribute resolves on every read, so a request waiting in a
// queue observes turns and responses recorded for the session in the meantime.
// The session is not evicted while the request is pending.
func (p *Producer) RequestHeader(ctx context.Context, request *fwksched.InferenceRequest) error {
	identity, tracked := p.readAgentIdentity(request)
	if !tracked {
		if identity != "" {
			log.FromContext(ctx).V(logging.DEBUG).Info("Agent identity exceeds maxIdentityBytes, session state is not tracked",
				"identityBytes", len(identity), "maxIdentityBytes", p.maxIdentityBytes)
		}
		return nil
	}
	obs := p.registry.observe(identity, request.RequestID)
	request.PutAttribute(p.dk, &fwksched.DynamicRequestAttribute{Get: func() any { return obs.state() }})
	return nil
}

// Produce is a no-op: RequestHeader already published the session state.
func (p *Producer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	return nil
}

// RequestAbandoned ends the pending state of a request that leaves before
// PreRequest, for example when flow control rejects it.
func (p *Producer) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	p.registry.ReleasePending(request.RequestID)
}

// PreRequest records one dispatched turn. A request counts once even when
// several profiles run.
func (p *Producer) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	identity, ok := p.readAgentIdentity(request)
	if !ok {
		return nil
	}

	p.registry.RecordDispatch(identity, request.RequestID)
	return nil
}

// ResponseBody updates session counters after the response lifecycle ends.
// Only naturally completed responses contribute completion and token totals.
func (p *Producer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrh.Response,
	_ *fwkdl.EndpointMetadata,
) {
	if response == nil || !response.EndOfStream {
		return
	}
	identity, ok := p.readAgentIdentity(request)
	if !ok {
		return
	}

	var cachedTokens int64
	if details := response.Usage.PromptTokenDetails; details != nil {
		cachedTokens = int64(details.CachedTokens)
	}
	p.registry.RecordResponse(
		identity,
		response.TerminationCause == fwkrh.TerminationCauseNatural,
		TurnUsage{
			InputTokens:  int64(response.Usage.PromptTokens),
			OutputTokens: int64(response.Usage.CompletionTokens),
			CachedTokens: cachedTokens,
		},
	)
}

// readAgentIdentity returns the agent identity of the request and whether its
// session is tracked. An identity longer than maxIdentityBytes is not.
func (p *Producer) readAgentIdentity(request *fwksched.InferenceRequest) (identity string, tracked bool) {
	if request == nil {
		return "", false
	}
	identity, _ = fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return identity, identity != "" && (p.maxIdentityBytes == 0 || len(identity) <= p.maxIdentityBytes)
}

func (p *Producer) runEviction(ctx context.Context) {
	ticker := time.NewTicker(p.evictionSweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.evictIdle(time.Now())
		}
	}
}

func (p *Producer) evictIdle(now time.Time) {
	p.registry.EvictIdle(now, p.evictionTTL)
}
