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
		SetTokens(reqCtx.Body, reqCtx.TokenIDs, reqCtx.MultimodalEntries)
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

// SetTokens writes rendered tokens and multimodal features into the vLLM chat token field.
func SetTokens(body map[string]any, tokenIDs []int, entries []pipeline.MultimodalEntry) {
	tokens := map[string]any{"token_ids": tokenIDs}
	if features := buildMMFeatures(entries, false); features != nil {
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

// setGenerateTransferParams nests the kv/ec transfer params under
// sampling_params.extra_args, the only place the /inference/v1/generate engine
// reads them (top-level kv_transfer_params/ec_transfer_params are ignored on
// input). It get-or-creates extra_args on the given sampling map so a client's
// existing generation fields survive. ecParams may be empty, in which case
// ec_transfer_params is left unset.
func setGenerateTransferParams(sampling map[string]any, kvParams any, ecParams map[string]any) {
	extraArgs, ok := sampling[reqcommon.FieldExtraArgs].(map[string]any)
	if !ok {
		extraArgs = map[string]any{}
		sampling[reqcommon.FieldExtraArgs] = extraArgs
	}
	extraArgs[reqcommon.FieldKVTransferParams] = kvParams
	if len(ecParams) > 0 {
		extraArgs[reqcommon.FieldECTransferParams] = ecParams
	}
}
