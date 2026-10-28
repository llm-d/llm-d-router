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

// Package forwardtokens forwards the token IDs the router already rendered to the model server.
package forwardtokens

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	commonrequest "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	// PluginType is the registered type name of the token forwarding plugin.
	PluginType = "forward-prompt-token-ids"

	fieldPromptTokenIDs = "prompt_token_ids"
)

var _ requestcontrol.PreRequest = &Plugin{}

// Config configures token forwarding.
type Config struct {
	// MinTokens skips prompts shorter than this; a short prompt costs the server little to
	// re-tokenize, and the forwarded list grows the body. 0 forwards every prompt.
	MinTokens int `json:"minTokens,omitempty"`
}

// Plugin writes the token-producer's IDs into kv_transfer_params.prompt_token_ids, which
// vLLM's chat and completions paths read to skip their own templating and tokenization.
type Plugin struct {
	typedName plugin.TypedName
	minTokens int
}

// PluginFactory constructs a token forwarding plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := Config{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
		}
	}
	return New(name, cfg)
}

// New constructs a token forwarding plugin from validated configuration.
func New(name string, cfg Config) (*Plugin, error) {
	if cfg.MinTokens < 0 {
		return nil, fmt.Errorf("%s: minTokens must be >= 0, got %d", PluginType, cfg.MinTokens)
	}
	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		minTokens: cfg.MinTokens,
	}, nil
}

// TypedName returns the plugin's registered type and configured name.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// PreRequest adds the rendered token IDs to a single-prompt JSON request. Other
// kv_transfer_params fields are preserved, and an ID list the client sent is kept.
func (p *Plugin) PreRequest(ctx context.Context, request *scheduling.InferenceRequest,
	_ *scheduling.SchedulingResult) error {
	logger := log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEBUG)
	if request == nil || request.Body == nil || request.Body.TokenizedRequest == nil {
		return nil
	}
	body := request.Body
	// vLLM reads the IDs only on its chat path, which also serves Messages and Responses.
	// Completions and native generate already send tokens or skip templating.
	if !isChatShaped(body) || len(body.TokenizedRequest.Prompts) != 1 {
		return nil
	}
	prompt := body.TokenizedRequest.Prompts[0]
	// vLLM ignores forwarded IDs for non-text content and renders the messages instead.
	if len(prompt.TokenIDs) == 0 || len(prompt.TokenIDs) < p.minTokens || prompt.MultiModalFeatures != nil {
		return nil
	}
	payload, ok := body.Payload.(requesthandling.PayloadMap)
	if !ok {
		logger.Info("skipping token forwarding: request payload is not mutable JSON",
			"requestID", request.RequestID)
		return nil
	}
	params, ok := kvTransferParams(payload)
	if !ok {
		logger.Info("skipping token forwarding: kv_transfer_params is not an object",
			"requestID", request.RequestID)
		return nil
	}
	if _, exists := params[fieldPromptTokenIDs]; exists {
		return nil
	}
	params[fieldPromptTokenIDs] = prompt.TokenIDs
	body.MutatePayloadMap(func(m requesthandling.PayloadMap) {
		m[commonrequest.FieldKVTransferParams] = params
	})
	return nil
}

// isChatShaped reports whether vLLM renders the request through its chat path, the only path
// that honours kv_transfer_params.prompt_token_ids.
func isChatShaped(body *requesthandling.InferenceRequestBody) bool {
	return body.ChatCompletions != nil || body.Messages != nil || body.Responses != nil
}

// kvTransferParams returns the request's kv_transfer_params as a map, or a new one when absent.
func kvTransferParams(payload requesthandling.PayloadMap) (map[string]any, bool) {
	value, exists := payload[commonrequest.FieldKVTransferParams]
	if !exists || value == nil {
		return map[string]any{}, true
	}
	switch v := value.(type) {
	case map[string]any:
		return v, true
	case json.RawMessage:
		var params map[string]any
		if err := parserutil.Unmarshal(v, &params); err != nil || params == nil {
			return nil, false
		}
		return params, true
	}
	return nil, false
}
