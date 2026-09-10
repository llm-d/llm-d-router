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

// Package vllm prepares coordinator payloads for the vLLM inference protocol.
package vllm

import "github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"

const modalityImage = "image"

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
