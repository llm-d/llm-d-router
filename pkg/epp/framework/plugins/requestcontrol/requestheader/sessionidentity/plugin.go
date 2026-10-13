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

// Package sessionidentity derives a scoped session identity before flow-control
// admission from the agent-identity request attribute.
package sessionidentity

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

const PluginType = "session-identity"

// SessionTagDataKey identifies the plain string SessionTag published for
// generic attribute consumers such as fairness and session affinity.
var SessionTagDataKey = fwkplugin.NewDataKey("session-tag", "")

var (
	_ fwkplugin.ConsumerPlugin              = (*Plugin)(nil)
	_ fwkplugin.ProducerPlugin              = (*Plugin)(nil)
	_ requestcontrol.RequestHeaderProcessor = (*Plugin)(nil)
)

// Plugin publishes scoped typed identity and its plain string tag.
type Plugin struct {
	typedName     fwkplugin.TypedName
	identityKey   fwkplugin.DataKey
	sessionTagKey fwkplugin.DataKey

	deploymentID string
	hmacKey      []byte
	scopeVersion string
	metrics      *identityMetrics
}

// Factory constructs a configured session-identity plugin.
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
	var registerer fwkplugin.MetricsRecorder
	if handle != nil {
		registerer = handle.Metrics()
	}
	metrics, err := newIdentityMetrics(name, registerer)
	if err != nil {
		return nil, err
	}
	_, scopeVersion := deriveIdentity(resolved.hmacKey, resolved.deploymentID, "")
	return &Plugin{
		typedName:     fwkplugin.TypedName{Type: PluginType, Name: name},
		identityKey:   requestcontrol.SessionIdentityDataKey.WithNonEmptyProducerName(name),
		sessionTagKey: SessionTagDataKey.WithNonEmptyProducerName(name),
		deploymentID:  resolved.deploymentID,
		hmacKey:       resolved.hmacKey,
		scopeVersion:  scopeVersion,
		metrics:       metrics,
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.identityKey:   requestcontrol.SessionIdentity{},
		p.sessionTagKey: "",
	}
}

func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// RequestHeader publishes identity before admission control. Missing identity
// fails open, and the hook always returns nil because its errors fail requests.
func (p *Plugin) RequestHeader(_ context.Context, request *fwksched.InferenceRequest) error {
	if request == nil {
		return nil
	}
	alias, ok := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	if !ok || alias == "" {
		p.metrics.outcomes.WithLabelValues("missing").Inc()
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
	p.metrics.outcomes.WithLabelValues("published").Inc()
	return nil
}
