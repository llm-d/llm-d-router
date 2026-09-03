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

package tokenizer

import (
	"context"
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const anthropicCountPath = "/v1/messages/count_tokens"

func (r *vllmHTTPRenderer) anthropicInlineSystemMode(ctx context.Context) (bool, error) {
	if r.mergeAnthropicInlineSystem != nil {
		return *r.mergeAnthropicInlineSystem, nil
	}

	r.anthropicModeMu.Lock()
	defer r.anthropicModeMu.Unlock()
	if r.detectedAnthropicMode != nil {
		return *r.detectedAnthropicMode, nil
	}

	mode, err := r.detectAnthropicInlineSystem(ctx)
	if err != nil {
		return false, err
	}
	r.detectedAnthropicMode = &mode
	return mode, nil
}

// TODO: Remove this detector and its configuration override when vLLM exposes
// an Anthropic render endpoint that returns the token IDs used for generation.
func (r *vllmHTTPRenderer) detectAnthropicInlineSystem(ctx context.Context) (bool, error) {
	probe := &fwkrh.MessagesRequest{
		System: fwkrh.AnthropicContent{Raw: "router-system-probe"},
		Messages: []fwkrh.AnthropicMessage{
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "router-user-probe-one"}},
			{Role: "system", Content: fwkrh.AnthropicContent{Raw: "router-inline-system-probe"}},
			{Role: "user", Content: fwkrh.AnthropicContent{Raw: "router-user-probe-two"}},
		},
	}
	countRequest := anthropicCountTokensRequest{
		Model:    r.modelName,
		System:   probe.System,
		Messages: probe.Messages,
	}
	var countResponse anthropicCountTokensResponse
	if err := r.postJSON(ctx, anthropicCountPath, countRequest, r.timeout, &countResponse); err != nil {
		return false, fmt.Errorf("detect Anthropic inline-system mode: %w", err)
	}

	unmergedIDs, _, unmergedErr := r.renderChatRequest(ctx, messagesToRenderChatRequest(probe, false, false))
	mergedIDs, _, mergedErr := r.renderChatRequest(ctx, messagesToRenderChatRequest(probe, true, false))
	unmergedMatches := unmergedErr == nil && len(unmergedIDs) == countResponse.InputTokens
	mergedMatches := mergedErr == nil && len(mergedIDs) == countResponse.InputTokens

	switch {
	case unmergedMatches && !mergedMatches:
		return false, nil
	case mergedMatches && !unmergedMatches:
		return true, nil
	default:
		return false, fmt.Errorf(
			"detect Anthropic inline-system mode: token count %d did not uniquely match unmerged (tokens=%d, err=%v) or merged (tokens=%d, err=%v) rendering",
			countResponse.InputTokens, len(unmergedIDs), unmergedErr, len(mergedIDs), mergedErr,
		)
	}
}

type anthropicCountTokensRequest struct {
	Model    string                   `json:"model"`
	System   fwkrh.AnthropicContent   `json:"system,omitempty"`
	Messages []fwkrh.AnthropicMessage `json:"messages"`
}

type anthropicCountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}
