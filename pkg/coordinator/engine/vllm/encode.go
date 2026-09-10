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
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// PrepareEncode adds vLLM token and multimodal fields to a per-image request.
func PrepareEncode(body map[string]any, tokenIDs []int, entry pipeline.MultimodalEntry, format gateway.RequestFormat) {
	features := map[string]any{
		"mm_hashes":       map[string][]string{modalityImage: {entry.Hash}},
		"mm_placeholders": map[string][]any{modalityImage: {map[string]any{"offset": 1, "length": entry.Placeholder.Length}}},
	}
	if format == gateway.FormatChatCompletions {
		body["tokens"] = map[string]any{"token_ids": tokenIDs, "features": features}
	} else {
		features["kwargs_data"] = mmKwargsField([]string{entry.KwargsData})
		body["token_ids"] = tokenIDs
		body["features"] = features
	}
}
