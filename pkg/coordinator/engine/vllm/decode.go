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
	"context"
	"maps"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// NewDecoder binds decode request preparation to the named KV connector.
// The returned function mutates the request body, including nested image parts;
// callers must not use those values concurrently.
func (e Engine) NewDecoder(name string) (engine.DecodeRequest, error) {
	connector, err := kv.Build(name)
	if err != nil {
		return nil, err
	}
	return func(ctx context.Context, req *pipeline.RequestContext) engine.Request {
		return e.prepareDecode(req, connector.PrepareDecodeKVParams(ctx, req))
	}, nil
}

func (e Engine) prepareDecode(reqCtx *pipeline.RequestContext, kvParams map[string]any) engine.Request {
	e.injectUUIDs(reqCtx)

	format := resolveFormat(e.useOpenAIFormat, reqCtx.OriginalPath)
	switch format {
	case gateway.FormatChatCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		e.injectTokensField(reqCtx)
	case gateway.FormatCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		if len(reqCtx.TokenIDs) > 0 {
			reqCtx.Body["prompt"] = reqCtx.TokenIDs
		}
	case gateway.FormatGenerate:
		// The /inference/v1/generate engine reads transfer params only from
		// sampling_params.extra_args; a top-level kv_transfer_params is ignored,
		// so the decode worker never pulls the prefill KV over NIXL. Merge into
		// the client's sampling_params to preserve max_tokens and other fields.
		sampling, ok := reqCtx.Body[reqcommon.FieldSamplingParams].(map[string]any)
		if !ok {
			sampling = map[string]any{}
			reqCtx.Body[reqcommon.FieldSamplingParams] = sampling
		}
		setGenerateTransferParams(sampling, kvParams, nil)
	}
	return engine.Request{Path: reqCtx.OriginalPath, Body: reqCtx.Body}
}

func (e Engine) injectTokensField(reqCtx *pipeline.RequestContext) {
	tokens := map[string]any{
		"token_ids": reqCtx.TokenIDs,
	}
	if features := buildMMFeatures(reqCtx.MultimodalEntries, false); features != nil {
		tokens["features"] = features
	}
	reqCtx.Body["tokens"] = tokens
}

func (e Engine) injectUUIDs(reqCtx *pipeline.RequestContext) {
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

// PrepareConditionalDecode builds a cache probe without mutating reqCtx.Body.
func (e Engine) PrepareConditionalDecode(reqCtx *pipeline.RequestContext) engine.Request {
	body := maps.Clone(reqCtx.Body)
	format := resolveFormat(e.useOpenAIFormat, reqCtx.OriginalPath)
	switch format {
	case gateway.FormatChatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			tokens := map[string]any{
				"token_ids": reqCtx.TokenIDs,
			}
			if features := buildMMFeatures(reqCtx.MultimodalEntries, false); features != nil {
				tokens["features"] = features
			}
			body["tokens"] = tokens
		}
	case gateway.FormatCompletions:
		if len(reqCtx.TokenIDs) > 0 {
			body["prompt"] = reqCtx.TokenIDs
		}
	}
	return engine.Request{Path: reqCtx.OriginalPath, Body: body}
}
