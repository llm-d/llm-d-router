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
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PreparePrefill adds vLLM token, feature and transfer fields to a prefill body.
func PreparePrefill(reqCtx *pipeline.RequestContext, body, kvParams, ecParams map[string]any, format gateway.RequestFormat) {
	switch format {
	case gateway.FormatChatCompletions:
		SetTokens(body, reqCtx.TokenIDs, reqCtx.MultimodalEntries)
	case gateway.FormatCompletions, gateway.FormatGenerate:
		body["request_id"] = reqCtx.RequestID
		if features := buildMMFeatures(reqCtx.MultimodalEntries, true); features != nil {
			body["features"] = features
		}
		if format == gateway.FormatGenerate {
			sampling := map[string]any{}
			setGenerateTransferParams(sampling, kvParams, ecParams)
			body["token_ids"] = reqCtx.TokenIDs
			body[reqcommon.FieldSamplingParams] = sampling
			return
		}
	}
	body[reqcommon.FieldKVTransferParams] = kvParams
	if len(ecParams) > 0 {
		body[reqcommon.FieldECTransferParams] = ecParams
	}
}
