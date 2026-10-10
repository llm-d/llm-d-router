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

// Package sessionmanager provides scoped session identity and per-request
// stamps without inferring continuations or publishing cache prefixes.
package sessionmanager

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

const PluginType = "session-manager"

// SessionTagDataKey identifies the plain string SessionTag published for
// generic attribute consumers such as session-affinity.
var SessionTagDataKey = fwkplugin.NewDataKey("session-tag", "")

var (
	_ fwkplugin.ConsumerPlugin              = (*Producer)(nil)
	_ requestcontrol.RequestHeaderProcessor = (*Producer)(nil)
	_ requestcontrol.DataProducer           = (*Producer)(nil)
)

// Producer publishes scoped identities, plain string tags, and cache requests.
type Producer struct {
	typedName       fwkplugin.TypedName
	identityKey     fwkplugin.DataKey
	sessionTagKey   fwkplugin.DataKey
	cacheRequestKey fwkplugin.DataKey

	deploymentID string
	hmacKey      []byte
	scopeVersion string
	stamps       *stampGenerator
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
	stamps, err := newStampGenerator()
	if err != nil {
		return nil, fmt.Errorf("initialize request stamps: %w", err)
	}
	var registerer fwkplugin.MetricsRecorder
	if handle != nil {
		registerer = handle.Metrics()
	}
	metrics, err := newManagerMetrics(name, registerer)
	if err != nil {
		return nil, err
	}
	_, scopeVersion := deriveIdentity(resolved.hmacKey, resolved.deploymentID, "")
	return &Producer{
		typedName:       fwkplugin.TypedName{Type: PluginType, Name: name},
		identityKey:     requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(name),
		sessionTagKey:   SessionTagDataKey.WithNonEmptyProducerName(name),
		cacheRequestKey: attrsession.SessionCacheRequestDataKey.WithNonEmptyProducerName(name),
		deploymentID:    resolved.deploymentID,
		hmacKey:         resolved.hmacKey,
		scopeVersion:    scopeVersion,
		stamps:          stamps,
		metrics:         metrics,
	}, nil
}

func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.identityKey:     requestcontrol.SessionIdentity{},
		p.sessionTagKey:   "",
		p.cacheRequestKey: attrsession.SessionCacheRequest{},
	}
}

func (p *Producer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// RequestHeader publishes identity before admission control. Missing identity
// fails open, and the hook always returns nil because its errors fail requests.
func (p *Producer) RequestHeader(_ context.Context, request *fwksched.InferenceRequest) error {
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
	request.PutAttribute(p.sessionTagKey, sessionTag)
	p.metrics.identityOutcomes.WithLabelValues("published").Inc()
	return nil
}

// Produce publishes only the request stamp consumed by
// session-prefix-cache-producer. Token data is intentionally unavailable here
// because tokenization runs after admission control.
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	if _, ok := requestcontrol.ReadSessionIdentity(request, p.typedName.Name); !ok {
		p.metrics.requestOutcomes.WithLabelValues("missing_identity").Inc()
		return nil
	}
	stamp, err := p.stamps.next()
	if err != nil {
		return err
	}
	request.PutAttribute(p.cacheRequestKey, attrsession.SessionCacheRequest{
		SessionID:   stamp,
		FullReport:  false,
		TotalTokens: 0,
		Prefixes:    nil,
	})
	p.metrics.requestOutcomes.WithLabelValues("published").Inc()
	return nil
}
