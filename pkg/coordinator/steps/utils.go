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
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
// true when absent. A present but non-bool value is a configuration error.
func parseUseOpenAIFormat(params map[string]any) (bool, error) {
	v, ok, err := paramBool(params, "use_openai_format")
	if err != nil {
		return false, err
	}
	if !ok {
		return true, nil
	}
	return v, nil
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

// capSingleTokenOutput rewrites body into a single-output-token, non-streaming
// request for the synthetic prefill and encode legs.
//
// Intentionally distinct from the sidecar's reqcommon.PrimeSingleTokenRequest
// for now.
// TODO: unify the two into one shared single-token helper in a future refactor.
func capSingleTokenOutput(body map[string]any, format gateway.RequestFormat) {
	target := body
	if format == gateway.FormatGenerate {
		sp, ok := body[reqcommon.FieldSamplingParams].(map[string]any)
		if !ok {
			sp = map[string]any{}
			body[reqcommon.FieldSamplingParams] = sp
		}
		target = sp
	}

	target[reqcommon.FieldMaxTokens] = 1
	// Strip rather than clamp min_tokens: it defaults to 0 in vLLM, so removing it
	// keeps min_tokens <= max_tokens=1 without raising the floor above the cap.
	delete(target, reqcommon.FieldMinTokens)

	if _, ok := body[reqcommon.FieldMaxCompletionTokens]; ok {
		body[reqcommon.FieldMaxCompletionTokens] = 1
	}

	// TODO: max_output_tokens is another client-supplied output cap (Responses
	// API) that a client can send instead of max_tokens/max_completion_tokens; it
	// should be capped to 1 here as well so the synthetic legs stay single-token.

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
		"mm_hashes":       map[string][]string{ModalityImage: hashes},
		"mm_placeholders": map[string][]any{ModalityImage: placeholders},
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
	return map[string][]any{ModalityImage: items}
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

// toIntSlice converts a JSON-unmarshalled []any of numeric elements to []int.
// Each element must be a non-negative integer represented as float64 or json.Number.
// The returned error identifies the offending element by index and wraps
// pipeline.ErrBadRequest.
func toIntSlice(values []any) ([]int, error) {
	out := make([]int, 0, len(values))
	for i, v := range values {
		n, err := anyToNonNegativeInt(v)
		if err != nil {
			return nil, fmt.Errorf("invalid token at index %d: %v: %w", i, err, pipeline.ErrBadRequest)
		}
		out = append(out, n)
	}
	return out, nil
}

// anyToNonNegativeInt converts a single JSON-unmarshalled numeric value to a non-negative int.
func anyToNonNegativeInt(v any) (int, error) {
	switch n := v.(type) {
	case float64:
		if n < 0 || n != math.Trunc(n) {
			return 0, fmt.Errorf("expected non-negative integer, got %v", v)
		}
		// An in-range integer-valued float64 round-trips through int; a value
		// too large to fit does not (the conversion saturates), so this rejects
		// overflow without depending on the fragile float64(MaxInt) boundary.
		i := int(n)
		if float64(i) != n {
			return 0, fmt.Errorf("expected non-negative integer, got %v", v)
		}
		return i, nil
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, err
		}
		if i < 0 || i > math.MaxInt {
			return 0, fmt.Errorf("expected non-negative integer, got %d", i)
		}
		return int(i), nil
	default:
		return 0, fmt.Errorf("expected number, got %T", v)
	}
}

// extractTokenIDs converts body["token_ids"] from a JSON-unmarshalled value to []int.
// Returns ErrBadRequest when the field is absent, not an array, empty, or contains
// non-integer or negative values.
func extractTokenIDs(raw any) ([]int, error) {
	if raw == nil {
		return nil, fmt.Errorf("token_ids is required: %w", pipeline.ErrBadRequest)
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("token_ids must be an array, got %T: %w", raw, pipeline.ErrBadRequest)
	}
	if len(arr) == 0 {
		return nil, fmt.Errorf("token_ids must not be empty: %w", pipeline.ErrBadRequest)
	}
	return toIntSlice(arr)
}

// mmImageArray reads features[field][ModalityImage] as a JSON array. present is
// false when field or its image entry is absent or null, a valid "no such
// modality" state rather than an error. A present value of the wrong type
// (field not an object, or the image entry not an array) is ErrBadRequest, so a
// malformed request fails loudly instead of being silently coerced to absent.
func mmImageArray(features map[string]any, field string) (arr []any, present bool, err error) {
	rawField, ok := features[field]
	if !ok || rawField == nil {
		return nil, false, nil
	}
	m, ok := rawField.(map[string]any)
	if !ok {
		return nil, false, fmt.Errorf("%s must be an object: %w", field, pipeline.ErrBadRequest)
	}
	rawImage, ok := m[ModalityImage]
	if !ok || rawImage == nil {
		return nil, false, nil
	}
	arr, ok = rawImage.([]any)
	if !ok {
		return nil, false, fmt.Errorf("%s[%s] must be an array: %w", field, ModalityImage, pipeline.ErrBadRequest)
	}
	return arr, true, nil
}

// extractMultimodalEntries builds []pipeline.MultimodalEntry from the parallel
// slices in a generate-format features map. Returns nil when features is nil or
// mm_hashes.image is absent or empty (text-only request).
//
// mm_hashes and mm_placeholders are required and must be the same length.
// kwargs_data is optional: an absent field means every item resolves from the
// encoder cache by hash (a cache-hit request), so each entry's KwargsData is "".
// When present, kwargs_data must be parallel to mm_hashes, but an individual
// item may be null (a cache hit within a mixed batch), which maps to "".
//
// Returns ErrBadRequest when a present field has the wrong type, required slices
// have different lengths, or any element has an unexpected type.
func extractMultimodalEntries(features map[string]any) ([]pipeline.MultimodalEntry, error) {
	if features == nil {
		return nil, nil
	}
	rawHashes, _, err := mmImageArray(features, "mm_hashes")
	if err != nil {
		return nil, err
	}
	// mmImageArray returns a nil slice whenever the field is absent, so the
	// length check subsumes the presence check here.
	if len(rawHashes) == 0 {
		return nil, nil
	}

	rawPlaceholders, present, err := mmImageArray(features, "mm_placeholders")
	if err != nil {
		return nil, err
	}
	if !present {
		return nil, fmt.Errorf("mm_placeholders[%s] is required when mm_hashes[%s] is set: %w",
			ModalityImage, ModalityImage, pipeline.ErrBadRequest)
	}

	rawKwargs, hasKwargs, err := mmImageArray(features, "kwargs_data")
	if err != nil {
		return nil, err
	}

	n := len(rawHashes)
	if len(rawPlaceholders) != n {
		return nil, fmt.Errorf("features length mismatch: mm_hashes has %d, mm_placeholders has %d: %w",
			n, len(rawPlaceholders), pipeline.ErrBadRequest)
	}
	// When present, kwargs_data is parallel to mm_hashes: full length with null
	// placeholders for cached items, never a shortened list. The whole field is
	// absent for metadata-only (cache-hit) requests.
	if hasKwargs && len(rawKwargs) != n {
		return nil, fmt.Errorf("features length mismatch: mm_hashes has %d, kwargs_data has %d: %w",
			n, len(rawKwargs), pipeline.ErrBadRequest)
	}

	entries := make([]pipeline.MultimodalEntry, n)
	for i := range entries {
		hash, ok := rawHashes[i].(string)
		if !ok {
			return nil, fmt.Errorf("mm_hashes[%d] must be a string: %w", i, pipeline.ErrBadRequest)
		}

		pMap, ok := rawPlaceholders[i].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("mm_placeholders[%d] must be an object: %w", i, pipeline.ErrBadRequest)
		}
		// The non-negative guarantee here is load-bearing, not just input
		// hygiene. EncodeStep.buildEncodeTokenIDs indexes fullTokenIDs[offset]
		// (guarded only on the upper bound) and allocates make([]int, 1+length);
		// a negative offset or length panics there. vLLM's own schema declares
		// these as plain ints and accepts negatives, so this stays stricter
		// deliberately. Do not relax it to a plain int parse.
		offset, err := anyToNonNegativeInt(pMap["offset"])
		if err != nil {
			return nil, fmt.Errorf("mm_placeholders[%d].offset: %v: %w", i, err, pipeline.ErrBadRequest)
		}
		length, err := anyToNonNegativeInt(pMap["length"])
		if err != nil {
			return nil, fmt.Errorf("mm_placeholders[%d].length: %v: %w", i, err, pipeline.ErrBadRequest)
		}

		// Empty KwargsData is the sentinel for "resolve from cache": either the
		// whole kwargs_data field is absent or this item is null.
		var kwarg string
		if hasKwargs {
			switch k := rawKwargs[i].(type) {
			case string:
				kwarg = k
			case nil:
			default:
				return nil, fmt.Errorf("kwargs_data[%d] must be a string or null: %w", i, pipeline.ErrBadRequest)
			}
		}

		entries[i] = pipeline.MultimodalEntry{
			Index:      i,
			Hash:       hash,
			KwargsData: kwarg,
			Placeholder: pipeline.PlaceholderRange{
				Offset: offset,
				Length: length,
			},
		}
	}
	return entries, nil
}

// validateSamplingParams checks that sampling_params and its nested extra_args,
// when present, are JSON objects. Both are optional. The decode step merges
// kv_transfer_params into sampling_params.extra_args; a non-object at either
// level would fall into its fallback branch and be silently replaced with an
// empty map, discarding client-requested generation parameters with no error.
// Validating once at ingestion keeps that path fail-loud, consistent with
// token_ids and features.
func validateSamplingParams(body map[string]any) error {
	raw, ok := body[reqcommon.FieldSamplingParams]
	if !ok || raw == nil {
		return nil
	}
	sampling, ok := raw.(map[string]any)
	if !ok {
		return fmt.Errorf("%s must be an object, got %T: %w",
			reqcommon.FieldSamplingParams, raw, pipeline.ErrBadRequest)
	}
	ea, ok := sampling[reqcommon.FieldExtraArgs]
	if !ok || ea == nil {
		return nil
	}
	if _, ok := ea.(map[string]any); !ok {
		return fmt.Errorf("%s.%s must be an object, got %T: %w",
			reqcommon.FieldSamplingParams, reqcommon.FieldExtraArgs, ea, pipeline.ErrBadRequest)
	}
	return nil
}

// validatePlaceholderBounds checks that every placeholder span [offset,
// offset+length) lies within a prompt of tokenCount tokens. It guards the
// generate path, where the client supplies placeholder geometry directly:
// EncodeStep.buildEncodeTokenIDs indexes token_ids[offset] and allocates
// make([]int, 1+length), so an out-of-range offset reads the wrong token and an
// unbounded length (a tiny request can claim billions) is a memory-exhaustion
// vector. vLLM declares offset/length as plain unbounded ints on the generate
// endpoint and does not enforce this, so the coordinator does. offset and
// length are already guaranteed non-negative by extractMultimodalEntries.
func validatePlaceholderBounds(entries []pipeline.MultimodalEntry, tokenCount int) error {
	for i, e := range entries {
		off := e.Placeholder.Offset
		length := e.Placeholder.Length
		if off >= tokenCount {
			return fmt.Errorf("mm_placeholders[%d].offset %d out of range for %d token_ids: %w",
				i, off, tokenCount, pipeline.ErrBadRequest)
		}
		// off < tokenCount, so tokenCount-off is positive and cannot overflow.
		if length > tokenCount-off {
			return fmt.Errorf("mm_placeholders[%d] span (offset %d + length %d) exceeds %d token_ids: %w",
				i, off, length, tokenCount, pipeline.ErrBadRequest)
		}
	}
	return nil
}
