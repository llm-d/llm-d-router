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

// Package thunderagent provides session level admission control for agentic
// workloads, a minimal implementation of ThunderAgent (arXiv 2602.13692).
//
// A session (an agent trajectory identified by the session id the
// agent-identity plugin publishes) is bound to one pod and its KV token footprint is tracked from usage reports
// plus in-flight estimates. The engine's own KV utilization cannot serve this
// purpose: a session waiting on a tool call still owns its context in the
// prefix cache, but those blocks sit on the free list and are reported as
// unused, so on an agentic workload the reported utilization stays low while
// the cache is in fact full. The ledger tracked here counts idle sessions,
// which is the quantity that decides whether one more session fits.
//
// This file wires the plugin; the ledger lives in manager.go and the request
// hooks in accounting.go.
package thunderagent

import (
	"encoding/json"
	"fmt"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// ThunderAgentPluginType is the plugin type registered with the framework.
const ThunderAgentPluginType = "thunder-agent"

// inflightEstimateKey is the request attribute under which PreRequest stashes
// the token estimate it charged to the session, so ResponseBody removes
// exactly what was added.
var inflightEstimateKey = fwkplugin.NewDataKey("inflight-estimate", ThunderAgentPluginType)

var (
	_ fwkrc.PreRequest            = &ThunderAgent{}
	_ fwkrc.ResponseBodyProcessor = &ThunderAgent{}
	_ fwkplugin.ConsumerPlugin    = &ThunderAgent{}
)

// ThunderAgent is a single named instance shared by every hookup, so all of
// them read and write the same session ledger.
type ThunderAgent struct {
	typedName fwkplugin.TypedName

	capacityTokens float64

	mgr     *sessionManager
	metrics *thunderMetrics
}

// Factory builds a ThunderAgent from raw plugin parameters and registers its
// metrics.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' plugin: %w", ThunderAgentPluginType, err)
	}

	a := newThunderAgent(name, cfg)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := a.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("failed to register metrics of the '%s' plugin: %w", ThunderAgentPluginType, err)
			}
		}
	}
	return a, nil
}

func newThunderAgent(name string, cfg Config) *ThunderAgent {
	mgr := newSessionManager(cfg)
	return &ThunderAgent{
		typedName:      fwkplugin.TypedName{Type: ThunderAgentPluginType, Name: name},
		capacityTokens: float64(cfg.CapacityTokens),
		mgr:            mgr,
		metrics:        newThunderMetrics(mgr),
	}
}

func (a *ThunderAgent) TypedName() fwkplugin.TypedName {
	return a.typedName
}

// Consumes declares agent identity as a required input, so configuration
// loading fails when no identity provider is enabled.
func (a *ThunderAgent) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{agentidentity.AgentIdentityKey: ""},
	}
}

// sessionID returns the session identifier published by the agent-identity
// plugin, or "" for requests carrying none. The fairness ID is not used: an
// explicit fairness header often carries a tenant rather than a session.
// Anonymous traffic is not tracked.
func sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	id, _ := fwksched.ReadRequestAttribute[string](request, agentidentity.AgentIdentityKey)
	return id
}
