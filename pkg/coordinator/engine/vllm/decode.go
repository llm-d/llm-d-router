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

// PrepareDecode adds vLLM transfer and token fields, including nested image UUIDs.
func PrepareDecode(reqCtx *pipeline.RequestContext, kvParams map[string]any, format gateway.RequestFormat) {
	injectUUIDs(reqCtx)
	switch format {
	case gateway.FormatChatCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		injectTokensField(reqCtx, reqCtx.Body)
	case gateway.FormatCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
	case gateway.FormatGenerate:
		sampling, ok := reqCtx.Body[reqcommon.FieldSamplingParams].(map[string]any)
		if !ok {
			sampling = map[string]any{}
			reqCtx.Body[reqcommon.FieldSamplingParams] = sampling
		}
		setGenerateTransferParams(sampling, kvParams, nil)
	}
}

// PrepareConditionalDecode adds rendered tokens to a vLLM chat cache probe.
func PrepareConditionalDecode(reqCtx *pipeline.RequestContext, body map[string]any, format gateway.RequestFormat) {
	if format == gateway.FormatChatCompletions && len(reqCtx.TokenIDs) > 0 {
		injectTokensField(reqCtx, body)
	}
}

func injectTokensField(reqCtx *pipeline.RequestContext, body map[string]any) {
	tokens := map[string]any{"token_ids": reqCtx.TokenIDs}
	if features := buildMMFeatures(reqCtx.MultimodalEntries, false); features != nil {
		tokens["features"] = features
	}
	body["tokens"] = tokens
}

func injectUUIDs(reqCtx *pipeline.RequestContext) {
	messages, ok := reqCtx.Body["messages"].([]any)
	if !ok {
		return
	}

	hashIdx := 0
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for _, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if partMap["type"] != "image_url" {
				continue
			}
			if hashIdx < len(reqCtx.MultimodalEntries) {
				partMap["uuid"] = reqCtx.MultimodalEntries[hashIdx].Hash
				hashIdx++
			}
		}
	}
}
