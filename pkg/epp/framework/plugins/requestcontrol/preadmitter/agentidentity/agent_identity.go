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

// Package agentidentity provides a PreAdmitter plugin that resolves
// agent identity from provider-specific headers into the FairnessID field.
package agentidentity

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	PluginType = "agent-identity"

	ClaudeCodeSessionHeader = "x-claude-code-session-id"
	OpenCodeSessionHeader   = "x-session-affinity"
	// CodexSessionHeader is the current (Codex >= 0.131.0) hyphenated form.
	CodexSessionHeader = "session-id"
	// CodexSessionHeaderLegacy is the underscored form used by Codex 0.130.x;
	// kept as a fallback for the brief window before users upgrade.
	CodexSessionHeaderLegacy = "session_id"
)

// defaultPriorityHeaders is the built-in ordered list, always checked. User-
// supplied headers from Parameters.AdditionalSessionHeaders are inserted
// before this list so operators can prepend higher-priority entries without
// losing the defaults.
var defaultPriorityHeaders = []string{
	ClaudeCodeSessionHeader,
	OpenCodeSessionHeader,
	CodexSessionHeader,
	CodexSessionHeaderLegacy,
}

// Parameters is the user-facing plugin configuration block. See the package
// README for a configmap example showing how additionalSessionHeaders extends
// the built-in default list.
type Parameters struct {
	// AdditionalSessionHeaders is prepended to the built-in default list.
	// Order is preserved; the request-time loop short-circuits on first match.
	AdditionalSessionHeaders []string `json:"additionalSessionHeaders,omitempty"`
}

// PluginFactory is the factory function for the agent identity plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params Parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("agent-identity: failed to decode parameters: %w", err)
		}
	}

	return &Plugin{
		typedName:       plugin.TypedName{Type: PluginType, Name: name},
		priorityHeaders: mergeHeaders(params.AdditionalSessionHeaders, defaultPriorityHeaders),
	}, nil
}

// mergeHeaders returns extras followed by defaults, lowercased. Request headers
// are stored with lowercased keys (see handlers.HandleRequestHeaders), so the
// priority list must match. Duplicates and empty strings are left in — the
// request-time loop short-circuits on first match.
func mergeHeaders(extras, defaults []string) []string {
	merged := make([]string, 0, len(extras)+len(defaults))
	for _, h := range extras {
		merged = append(merged, strings.ToLower(h))
	}
	for _, h := range defaults {
		merged = append(merged, strings.ToLower(h))
	}
	return merged
}

// Plugin resolves agent identity from provider-specific headers into FairnessID.
type Plugin struct {
	typedName       plugin.TypedName
	priorityHeaders []string
}

func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

func (p *Plugin) PreAdmit(_ context.Context, request *scheduling.InferenceRequest) error {
	if request.FairnessID != "" {
		return nil
	}

	for _, header := range p.priorityHeaders {
		if id := request.Headers[header]; id != "" {
			request.FairnessID = id
			return nil
		}
	}

	return nil
}
