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
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

const (
	defaultEvictionTTL           = time.Hour
	defaultEvictionSweepInterval = 5 * time.Minute
)

// Parameters configures idle session eviction.
type Parameters struct {
	// EvictionTTLSeconds is the maximum idle time before session state is removed.
	// Zero disables eviction.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds,omitempty"`
	// EvictionSweepSeconds is how often idle session state is scanned.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds,omitempty"`
}

func (p Parameters) validate() error {
	if p.EvictionTTLSeconds < 0 {
		return fmt.Errorf("evictionTtlSeconds must be >= 0, got %v", p.EvictionTTLSeconds)
	}
	if p.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", p.EvictionSweepSeconds)
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
	_ fwkdl.Registrant             = &Producer{}
	_ fwkdl.EndpointExtractor      = (*Producer)(nil)
)

// Producer tracks session history within one EPP instance.
type Producer struct {
	typedName          fwkplugin.TypedName
	dk                 fwkplugin.DataKey
	endpointSessionsDK fwkplugin.DataKey

	registry            SessionStateRegistry
	endpointMu          sync.Mutex
	registeredEndpoints map[string]fwkdl.Endpoint // key: NamespacedName, guarded by endpointMu

	evictionTTL           time.Duration
	evictionSweepInterval time.Duration
}

// Factory builds a session-state producer.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		EvictionTTLSeconds:   defaultEvictionTTL.Seconds(),
		EvictionSweepSeconds: defaultEvictionSweepInterval.Seconds(),
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
		endpointSessionsDK:    EndpointSessionsDataKey.WithNonEmptyProducerName(name),
		registeredEndpoints:   make(map[string]fwkdl.Endpoint),
		evictionTTL:           time.Duration(params.EvictionTTLSeconds * float64(time.Second)),
		evictionSweepInterval: time.Duration(params.EvictionSweepSeconds * float64(time.Second)),
	}
	if handle != nil && p.evictionTTL > 0 {
		go p.runEviction(handle.Context())
	}
	return p, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the request and endpoint attributes written by this producer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.dk:                 SessionState{},
		p.endpointSessionsDK: EndpointSessions{},
	}
}

// Consumes declares agent identity as a required input. The agent-identity
// plugin is intentionally not a default producer, so operators must enable an
// identity provider explicitly rather than having one auto-created.
func (p *Producer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// RegisterDependencies subscribes the producer to endpoint lifecycle events so
// it can attach dynamic session snapshots to new endpoints and unbind sessions
// when an endpoint leaves the pool.
func (p *Producer) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract attaches dynamic EndpointSessions attributes on endpoint creation or
// update and clears session bindings when an endpoint is removed.
func (p *Producer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}

	id := event.Endpoint.GetMetadata().ID.String()

	p.endpointMu.Lock()
	defer p.endpointMu.Unlock()

	switch event.Type {
	case fwkdl.EventDelete:
		// Guard against an out-of-order delete for a replaced endpoint object.
		if registered, ok := p.registeredEndpoints[id]; ok && registered != event.Endpoint {
			log.FromContext(ctx).V(logging.DEFAULT).Info("Ignoring stale delete for replaced endpoint", "endpoint", id)
			break
		}
		delete(p.registeredEndpoints, id)
		p.registry.RemoveEndpoint(id)
		log.FromContext(ctx).V(logging.DEFAULT).Info("Cleared session bindings for deleted endpoint", "endpoint", id)
	case fwkdl.EventAddOrUpdate:
		if p.registeredEndpoints == nil {
			p.registeredEndpoints = make(map[string]fwkdl.Endpoint)
		}
		if registered, ok := p.registeredEndpoints[id]; ok && registered != event.Endpoint {
			// A new Endpoint object under the same ID replaces an older pod whose
			// EventDelete has not arrived yet; unbind sessions from the old pod.
			p.registry.RemoveEndpoint(id)
		}
		p.registeredEndpoints[id] = event.Endpoint
		if attrs := event.Endpoint.GetAttributes(); attrs != nil {
			attrs.Put(p.endpointSessionsDK, &fwkdl.DynamicAttribute{
				Get: func() fwkdl.Cloneable {
					return p.registry.EndpointSessions(id)
				},
			})
		}
		log.FromContext(ctx).V(logging.DEBUG).Info("Attached dynamic session attribute to endpoint", "key", p.endpointSessionsDK.String(), "endpoint", id)
	}
	return nil
}

// RequestHeader marks the session as seen and publishes its state before flow
// control. The attribute resolves on every read, so a request waiting in a
// queue observes turns and responses recorded for the session in the meantime.
// The session is not evicted while the request is pending.
func (p *Producer) RequestHeader(_ context.Context, request *fwksched.InferenceRequest) error {
	identity, ok := readAgentIdentity(request)
	if !ok {
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

// PreRequest records one dispatched turn, binds the primary profile's picked
// endpoint when present, and charges the prompt token estimate into
// InFlightTokens. A request counts once even when several profiles run.
func (p *Producer) PreRequest(_ context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	identity, ok := readAgentIdentity(request)
	if !ok {
		return nil
	}

	p.registry.RecordDispatch(identity, request.RequestID, primaryEndpointID(result), EstimatePromptTokens(request.RequestSizeBytes))
	return nil
}

func primaryEndpointID(result *fwksched.SchedulingResult) string {
	endpoint := result.PrimaryEndpoint()
	if endpoint == nil {
		return ""
	}
	md := endpoint.GetMetadata()
	if md == nil || (md.ID.Namespace == "" && md.ID.Name == "") {
		return ""
	}
	return md.ID.String()
}

// ResponseBody settles the turn after the response lifecycle ends.
func (p *Producer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrh.Response,
	_ *fwkdl.EndpointMetadata,
) {
	if response == nil || !response.EndOfStream {
		return
	}
	identity, ok := readAgentIdentity(request)
	if !ok {
		return
	}

	var cachedTokens int64
	if details := response.Usage.PromptTokenDetails; details != nil {
		cachedTokens = int64(details.CachedTokens)
	}
	p.registry.RecordResponse(
		identity,
		request.RequestID,
		response.TerminationCause == fwkrh.TerminationCauseNatural,
		TurnUsage{
			InputTokens:  int64(response.Usage.PromptTokens),
			OutputTokens: int64(response.Usage.CompletionTokens),
			TotalTokens:  int64(response.Usage.TotalTokens),
			CachedTokens: cachedTokens,
		},
	)
}

func readAgentIdentity(request *fwksched.InferenceRequest) (string, bool) {
	if request == nil {
		return "", false
	}
	identity, ok := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return identity, ok && identity != ""
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
