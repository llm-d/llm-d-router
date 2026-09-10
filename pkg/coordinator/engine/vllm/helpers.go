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

const modalityImage = "image"

// resolveFormat maps a request path to the wire format a step emits. Completions
// is always honored; otherwise OpenAI formats collapse to FormatGenerate unless
// useOpenAIFormat is set.
func resolveFormat(useOpenAIFormat bool, path string) gateway.RequestFormat {
	detected := gateway.DetectFormat(path)
	if detected == gateway.FormatCompletions {
		return gateway.FormatCompletions
	}
	if !useOpenAIFormat {
		return gateway.FormatGenerate
	}
	return detected
}

// capSingleTokenOutput rewrites body into a single-output-token, non-streaming
// request for the synthetic prefill and encode legs.
//
// Chat-completions and completions bodies carry max_tokens/max_completion_tokens/
// stream/stream_options at the top level, same as the sidecar's synthetic
// requests, so they share reqcommon.PrimeSingleTokenRequest verbatim. The
// generate format's token-limit fields live under sampling_params instead
// (vLLM's GenerateRequest schema); stream/stream_options stay top-level there
// too, and generate has no max_completion_tokens-equivalent field, so only the
// max_tokens/min_tokens capping (reqcommon.CapMaxTokensField) is reusable for it.
//
// TODO: max_output_tokens is another client-supplied output cap (Responses
// API) that a client can send instead of max_tokens/max_completion_tokens; it
// should be capped to 1 here as well so the synthetic legs stay single-token.
func capSingleTokenOutput(body map[string]any, format gateway.RequestFormat) {
	if format != gateway.FormatGenerate {
		reqcommon.PrimeSingleTokenRequest(body)
		return
	}

	sp, ok := body[reqcommon.FieldSamplingParams].(map[string]any)
	if !ok {
		sp = map[string]any{}
		body[reqcommon.FieldSamplingParams] = sp
	}
	reqcommon.CapMaxTokensField(sp)

	body[reqcommon.FieldStream] = false
	delete(body, reqcommon.FieldStreamOptions)
}

// buildMMFeatures builds the multimodal features map (mm_hashes, mm_placeholders,
// and optionally kwargs_data) from the request's multimodal entries. It returns
// nil when there are no entries.
func buildMMFeatures(entries []pipeline.MultimodalEntry, includeKwargs bool) map[string]any {
	if len(entries) == 0 {
		return nil
	}
	hashes := make([]string, len(entries))
	placeholders := make([]any, len(entries))
	kwargs := make([]string, len(entries))
	for i, entry := range entries {
		hashes[i] = entry.Hash
		placeholders[i] = map[string]any{
			"offset": entry.Placeholder.Offset,
			"length": entry.Placeholder.Length,
		}
		kwargs[i] = entry.KwargsData
	}
	features := map[string]any{
		"mm_hashes":       map[string][]string{modalityImage: hashes},
		"mm_placeholders": map[string][]any{modalityImage: placeholders},
	}
	if includeKwargs {
		features["kwargs_data"] = mmKwargsField(kwargs)
	}
	return features
}

// mmKwargsField builds the kwargs_data feature value from per-entry KwargsData
// strings. The empty string is our internal "resolve from cache" sentinel and
// MUST serialize as JSON null, not "": vLLM treats null (or an absent field) as
// a cache-hit item to fetch from the encoder cache by hash, whereas "" is decoded
// as an inline tensor and fails with "Input data was truncated". Non-empty entries
// are the base64 tensor blobs and are forwarded verbatim.
func mmKwargsField(kwargs []string) map[string][]any {
	items := make([]any, len(kwargs))
	for i, k := range kwargs {
		if k != "" {
			items[i] = k
		}
	}
	return map[string][]any{modalityImage: items}
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
