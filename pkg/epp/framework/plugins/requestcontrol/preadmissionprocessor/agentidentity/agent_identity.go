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

// Package agentidentity provides a PreAdmissionProcessor plugin that resolves
// agent identity from provider-specific headers into the FairnessID field.
package agentidentity

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	PluginType = "agent-identity"

	ClaudeCodeSessionHeader = "x-claude-code-session-id"
	OpenCodeSessionHeader   = "x-session-affinity"
	CodexSessionHeader      = "session_id"
)

// priorityHeaders is the ordered list of headers to check for agent identity.
// One canonical session header per supported agent; first non-empty value wins.
var priorityHeaders = []string{
	ClaudeCodeSessionHeader,
	OpenCodeSessionHeader,
	CodexSessionHeader,
}

// PluginFactory is the factory function for the agent identity plugin.
func PluginFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
	}, nil
}

// Plugin resolves agent identity from provider-specific headers into FairnessID.
type Plugin struct {
	typedName plugin.TypedName
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) ProcessPreAdmission(_ context.Context, request *scheduling.InferenceRequest) error {
	if request.FairnessID != "" && request.FairnessID != metadata.DefaultFairnessID {
		return nil
	}

	// Check headers in priority order.
	for _, header := range priorityHeaders {
		if id := request.Headers[header]; id != "" {
			request.FairnessID = id
			return nil
		}
	}

	// TODO(#1031): OpenAI Responses API `previous_response_id` is intentionally
	// not used here. It references the prior turn's response, not the chain, so
	// every turn would map to a different FairnessID and fall into a separate
	// fairness queue. A correct implementation needs a ResponseBody hook on this
	// plugin that records `response.id -> rootID` and folds `previous_response_id`
	// back to its root at request time. That requires exposing response body
	// bytes (or the extracted `id`) on requestcontrol.Response, which is out of
	// scope for this change.
	return nil
}