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
	"io"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PreparePrefill adds vLLM token, feature and transfer fields to a prefill body.
func PreparePrefill(reqCtx *pipeline.RequestContext, body, kvParams, ecParams map[string]any, format gateway.RequestFormat) {
	features := buildMMFeatures(reqCtx.MultimodalEntries, true)
	switch format {
	case gateway.FormatChatCompletions:
		tokens := map[string]any{"token_ids": reqCtx.TokenIDs}
		if features != nil {
			tokens["features"] = map[string]any{
				"mm_hashes":       features["mm_hashes"],
				"mm_placeholders": features["mm_placeholders"],
			}
		}
		body["tokens"] = tokens
	case gateway.FormatCompletions:
		body["request_id"] = reqCtx.RequestID
		if features != nil {
			body["features"] = features
		}
	case gateway.FormatGenerate:
		sampling := map[string]any{reqcommon.FieldMaxTokens: 1}
		setGenerateTransferParams(sampling, kvParams, ecParams)
		body["request_id"] = reqCtx.RequestID
		body["token_ids"] = reqCtx.TokenIDs
		body[reqcommon.FieldSamplingParams] = sampling
		if features != nil {
			body["features"] = features
		}
		return
	}
	body[reqcommon.FieldKVTransferParams] = kvParams
	if len(ecParams) > 0 {
		body[reqcommon.FieldECTransferParams] = ecParams
	}
}

// ReadPrefillResponse extracts KV transfer parameters from a JSON response.
func ReadPrefillResponse(body io.Reader) (any, error) {
	var response prefillResponse
	if err := json.NewDecoder(body).Decode(&response); err != nil {
		return nil, err
	}
	return response.KVTransferParams, nil
}

type prefillResponse struct {
	// KVTransferParams is decoded as any (not map[string]any) so a non-object
	// value does not fail the decode; coerceParamsMap coerces it.
	KVTransferParams any `json:"kv_transfer_params"`
}
