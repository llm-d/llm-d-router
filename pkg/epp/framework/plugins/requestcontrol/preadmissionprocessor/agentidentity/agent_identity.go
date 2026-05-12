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
// agent identity from provider-specific headers and body fields into the FairnessID field.
package agentidentity

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	PluginType = "agent-identity"

	AgentIDHeader               = "x-gateway-inference-agent-id"
	ClaudeCodeSessionHeader     = "x-claude-code-session-id"
	OpenCodeSessionHeader       = "x-session-affinity"
	OpenCodeParentSessionHeader = "x-parent-session-id"
)

// priorityHeaders is the ordered list of headers to check for agent identity.
// The first non-empty value wins.
var priorityHeaders = []string{
	AgentIDHeader,
	ClaudeCodeSessionHeader,
	OpenCodeSessionHeader,
	OpenCodeParentSessionHeader,
}

// PluginFactory is the factory function for the agent identity plugin.
func PluginFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
	}, nil
}

// Plugin resolves agent identity from provider-specific headers and body fields into FairnessID.
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

	// Check body for OpenAI Responses API previous_response_id.
	if id := extractPreviousResponseID(request.Body); id != "" {
		request.FairnessID = id
		return nil
	}

	return nil
}

// extractPreviousResponseID extracts the previous_response_id from the request body payload
// when available. This is used by the OpenAI Responses API to chain related requests.
func extractPreviousResponseID(body *fwkrh.InferenceRequestBody) string {
	if body == nil || body.Payload == nil {
		return ""
	}
	payloadMap, ok := body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return ""
	}
	id, ok := payloadMap["previous_response_id"]
	if !ok {
		return ""
	}
	s, ok := id.(string)
	if !ok || s == "" {
		return ""
	}
	return s
}