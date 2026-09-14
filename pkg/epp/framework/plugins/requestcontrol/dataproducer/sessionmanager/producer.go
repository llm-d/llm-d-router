/*
Copyright 2026 The llm-d Authors.

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

// Package sessionmanager provides scoped session identity and request/event
// correlation without inferring continuations or publishing cache prefixes.
package sessionmanager

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const PluginType = "session-manager"

var (
	_ requestcontrol.SessionCacheManager = (*Producer)(nil)
	_ fwkplugin.ConsumerPlugin           = (*Producer)(nil)
	_ requestcontrol.PreRequest          = (*Producer)(nil)
	_ fwkplugin.StateDumper              = (*Producer)(nil)
)

// Producer publishes scoped identities and optional precise-cache requests.
type Producer struct {
	typedName       fwkplugin.TypedName
	identityKey     fwkplugin.DataKey
	cacheRequestKey fwkplugin.DataKey
	tokenKey        fwkplugin.DataKey

	deploymentID string
	hmacKey      []byte
	scopeVersion string
	correlation  bool
	stamps       *stampGenerator
	bindings     *bindingStore
	metrics      *managerMetrics
}

// Factory constructs a configured session-manager.
func Factory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var config Config
	if decoder != nil {
		if err := decoder.Decode(&config); err != nil {
			return nil, fmt.Errorf("decode %s parameters: %w", PluginType, err)
		}
	}
	resolved, err := config.resolve()
	if err != nil {
		return nil, fmt.Errorf("configure %s: %w", PluginType, err)
	}
	if name == "" {
		name = PluginType
	}
	var stamps *stampGenerator
	if resolved.eventCorrelationEnabled {
		stamps, err = newStampGenerator()
		if err != nil {
			return nil, fmt.Errorf("initialize request stamps: %w", err)
		}
	}
	bindings := newBindingStore(resolved.bindingTTL, resolved.maxBindings)
	var registerer fwkplugin.MetricsRecorder
	if handle != nil {
		registerer = handle.Metrics()
	}
	metrics, err := newManagerMetrics(name, bindings, registerer)
	if err != nil {
		return nil, err
	}
	_, scopeVersion := deriveIdentity(resolved.hmacKey, resolved.deploymentID, "")
	return &Producer{
		typedName:       fwkplugin.TypedName{Type: PluginType, Name: name},
		identityKey:     requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(name),
		cacheRequestKey: requestcontrol.SessionCacheRequestDataKey.WithNonEmptyProducerName(name),
		tokenKey:        tokenproducer.TokenizedPromptDataKey.WithNonEmptyProducerName(resolved.tokenProducer),
		deploymentID:    resolved.deploymentID,
		hmacKey:         resolved.hmacKey,
		scopeVersion:    scopeVersion,
		correlation:     resolved.eventCorrelationEnabled,
		stamps:          stamps,
		bindings:        bindings,
		metrics:         metrics,
	}, nil
}

func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.identityKey:     requestcontrol.SessionIdentity{},
		p.cacheRequestKey: requestcontrol.SessionCacheRequest{},
	}
}

func (p *Producer) Consumes() fwkplugin.DataDependencies {
	required := map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""}
	if p.correlation {
		required[p.tokenKey] = fwkrh.TokenizedRequest{}
	}
	return fwkplugin.DataDependencies{Required: required}
}

// Produce always publishes valid identity when an alias is present. Event
// correlation remains opt-in and fails open for unsupported request shapes.
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	alias, ok := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	if !ok || alias == "" {
		p.metrics.identityOutcomes.WithLabelValues("missing").Inc()
		return nil
	}
	sessionTag, _ := deriveIdentity(p.hmacKey, p.deploymentID, alias)
	identity := requestcontrol.SessionIdentity{
		SessionTag:     sessionTag,
		IdentitySource: requestcontrol.IdentitySourceAgentIdentityAttribute,
		ScopeVersion:   p.scopeVersion,
	}
	request.PutAttribute(p.identityKey, identity)
	p.metrics.identityOutcomes.WithLabelValues("published").Inc()

	if !p.correlation {
		p.metrics.requestOutcomes.WithLabelValues("disabled").Inc()
		return nil
	}
	if request.Body == nil {
		p.metrics.requestOutcomes.WithLabelValues("unsupported_body").Inc()
		return nil
	}
	if _, ok := request.Body.Payload.(fwkrh.PayloadMap); !ok {
		p.metrics.requestOutcomes.WithLabelValues("unsupported_envelope").Inc()
		return nil
	}
	if request.TargetModel == "" {
		p.metrics.requestOutcomes.WithLabelValues("unsupported_model").Inc()
		return nil
	}
	tokens := request.Body.TokenizedRequest
	if tokens == nil || len(tokens.Prompts) != 1 || len(tokens.Prompts[0].TokenIDs) == 0 {
		p.metrics.requestOutcomes.WithLabelValues("unsupported_tokens").Inc()
		return nil
	}
	stamp, err := p.stamps.next()
	if err != nil {
		return err
	}
	p.bindings.put(stamp, sessionTag, request.TargetModel)
	request.PutAttribute(p.cacheRequestKey, requestcontrol.SessionCacheRequest{
		Stamp:       stamp,
		FullReport:  false,
		TotalTokens: len(tokens.Prompts[0].TokenIDs),
		Prefixes:    nil,
	})
	p.metrics.requestOutcomes.WithLabelValues("published").Inc()
	return nil
}

// PreRequest binds a request stamp to the endpoint selected for dispatch.
func (p *Producer) PreRequest(
	_ context.Context,
	request *fwksched.InferenceRequest,
	result *fwksched.SchedulingResult,
) error {
	if !p.correlation || request == nil || result == nil {
		return nil
	}
	cacheRequest, ok := fwksched.ReadRequestAttribute[requestcontrol.SessionCacheRequest](
		request,
		p.cacheRequestKey,
	)
	if !ok || cacheRequest.Stamp == "" {
		return nil
	}
	profile := result.ProfileResults[result.PrimaryProfileName]
	if profile == nil || len(profile.TargetEndpoints) != 1 || profile.TargetEndpoints[0] == nil {
		return nil
	}
	metadata := profile.TargetEndpoints[0].GetMetadata()
	if metadata == nil || metadata.Address == "" || metadata.Port == "" {
		return nil
	}
	p.bindings.bindEndpoint(cacheRequest.Stamp, fmt.Sprintf("%s:%s", metadata.Address, metadata.Port))
	return nil
}

// CacheNamespace is empty in v1 because the manager publishes no engine-block
// prefixes. Compatibility is therefore neither asserted nor operator-configured.
func (p *Producer) CacheNamespace(kvevents.EventSource, *int) string { return "" }

// ProcessEvents validates known request stamps for observability only.
func (p *Producer) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, event := range batch.Events {
		stored, ok := event.(*kvevents.BlockStoredEvent)
		if !ok || !kvevents.IsIndexableLocalGPUStore(stored) {
			continue
		}
		if stored.SessionID == nil || *stored.SessionID == "" {
			p.metrics.eventOutcomes.WithLabelValues("unstamped").Inc()
			continue
		}
		known, duplicate, mismatch, stale := p.bindings.observe(
			*stored.SessionID,
			source.ModelName,
			source.Endpoint,
		)
		switch {
		case !known:
			p.metrics.eventOutcomes.WithLabelValues("unknown").Inc()
		case mismatch:
			p.metrics.eventOutcomes.WithLabelValues("request_mismatch").Inc()
		case stale:
			p.metrics.eventOutcomes.WithLabelValues("reset_stale").Inc()
		case duplicate:
			p.metrics.eventOutcomes.WithLabelValues("duplicate").Inc()
		default:
			p.metrics.eventOutcomes.WithLabelValues("known").Inc()
		}
	}
	return nil
}

// Reset invalidates bindings dispatched before the source reset. The manager
// retains no physical residency.
func (p *Producer) Reset(ctx context.Context, endpoint string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.correlation {
		p.bindings.resetEndpoint(endpoint)
	}
	p.metrics.eventOutcomes.WithLabelValues("reset").Inc()
	return nil
}

func (p *Producer) DumpState() (json.RawMessage, error) {
	if p == nil || p.bindings == nil {
		return nil, errors.New("session-manager is not initialized")
	}
	return json.Marshal(struct {
		ActiveBindings int `json:"activeBindings"`
		MaxBindings    int `json:"maxBindings"`
	}{
		ActiveBindings: p.bindings.len(),
		MaxBindings:    p.bindings.capacity,
	})
}
