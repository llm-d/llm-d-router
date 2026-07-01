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

package steps

import (
	"fmt"
	"io"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

// maxErrorBodySize caps how much of a non-2xx upstream response body is read
// into memory, bounding OOM exposure to an adversarial upstream pod.
const maxErrorBodySize = 8 << 10 // 8 KB

// readErrorBody reads up to maxErrorBodySize of an upstream error response body.
func readErrorBody(r io.Reader) []byte {
	body, _ := io.ReadAll(io.LimitReader(r, maxErrorBodySize))
	return body
}

// upstreamError builds a pipeline.UpstreamError tagged with the step name so the
// server can map an upstream 4xx to a client error and a 5xx to a gateway fault.
func upstreamError(step string, statusCode int, body []byte) error {
	return &pipeline.UpstreamError{Step: step, StatusCode: statusCode, Body: string(body)}
}

// parseUseOpenAIFormat reads the use_openai_format step parameter, defaulting to
// true when absent or not a bool.
func parseUseOpenAIFormat(params map[string]any) bool {
	if v, ok := params["use_openai_format"].(bool); ok {
		return v
	}
	return true
}

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
		"mm_hashes":       map[string][]string{ModalityImage: hashes},
		"mm_placeholders": map[string][]any{ModalityImage: placeholders},
	}
	if includeKwargs {
		features["kwargs_data"] = map[string][]string{ModalityImage: kwargs}
	}
	return features
}

// coerceParamsMap coerces a transfer-params value from an upstream response to a
// map: a non-object value is logged at debug and skipped (returns nil) rather
// than failing the request. A missing or null value is already nil; an empty map
// passes through so the connector's own no-metadata handling applies. label
// names the field for the debug log (e.g. "kv_transfer_params").
func coerceParamsMap(logger logr.Logger, v any, label string) map[string]any {
	switch m := v.(type) {
	case nil:
		return nil
	case map[string]any:
		return m
	default:
		logger.V(logutil.DEBUG).Info(label+" is not a JSON object; skipping",
			"type", fmt.Sprintf("%T", v))
		return nil
	}
}
