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

package vllm

import (
	"encoding/json"
	"fmt"
	"io"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PreparePrefill builds a non-streaming, single-token request with the supplied
// KV and encoder-cache transfer parameters.
func (e Engine) PreparePrefill(reqCtx *pipeline.RequestContext, kvParams, ecParams map[string]any) (engine.Request, error) {
	features := buildMMFeatures(reqCtx.MultimodalEntries, true)
	format := resolveFormat(e.useOpenAIFormat, reqCtx.OriginalPath)
	var body map[string]any
	switch format {
	case gateway.FormatChatCompletions:
		body = reqcommon.SingleTokenChatRequest(reqCtx.Body)
		tokens := map[string]any{
			"token_ids": reqCtx.TokenIDs,
		}
		if features != nil {
			tokensFeatures := map[string]any{
				"mm_hashes":       features["mm_hashes"],
				"mm_placeholders": features["mm_placeholders"],
			}
			tokens["features"] = tokensFeatures
		}
		body["tokens"] = tokens
		body[reqcommon.FieldKVTransferParams] = kvParams
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}

	case gateway.FormatCompletions:
		prompt := reqCtx.Body["prompt"]
		if len(reqCtx.TokenIDs) > 0 {
			prompt = reqCtx.TokenIDs
		}
		body = reqcommon.SingleTokenCompletionRequest(reqCtx.Model, prompt)
		body["request_id"] = reqCtx.RequestID
		body[reqcommon.FieldKVTransferParams] = kvParams
		if features != nil {
			body["features"] = features
		}
		if len(ecParams) > 0 {
			body[reqcommon.FieldECTransferParams] = ecParams
		}

	case gateway.FormatGenerate:
		// The /inference/v1/generate engine reads transfer params only from
		// sampling_params.extra_args; top-level fields are ignored on input.
		sampling := map[string]any{reqcommon.FieldMaxTokens: 1}
		setGenerateTransferParams(sampling, kvParams, ecParams)
		body = map[string]any{
			"request_id":                  reqCtx.RequestID,
			"token_ids":                   reqCtx.TokenIDs,
			"model":                       reqCtx.Model,
			reqcommon.FieldSamplingParams: sampling,
		}
		capSingleTokenOutput(body, format)
		if features != nil {
			body["features"] = features
		}
	default:
		return engine.Request{}, fmt.Errorf("prefill: unsupported request format %v", format)
	}

	return engine.Request{Path: gateway.PathForFormat(format), Body: body}, nil
}

// ReadPrefillResponse extracts KV transfer parameters from a JSON response.
func (Engine) ReadPrefillResponse(body io.Reader) (any, error) {
	var response prefillResponse
	if err := json.NewDecoder(body).Decode(&response); err != nil {
		return nil, fmt.Errorf("prefill: decode response: %w", err)
	}
	return response.KVTransferParams, nil
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
}
