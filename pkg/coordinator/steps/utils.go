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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
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

// gatewayHeaders returns a fresh map per call, so the caller may modify it.
func gatewayHeaders(reqCtx *pipeline.RequestContext, phase string) map[string]string {
	headers := reqCtx.ForwardedHeaders()
	headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID
	headers[reqcommon.EPPProfileHeaderKey] = phase
	return headers
}

// checkStatus returns a pipeline.UpstreamError tagged with step when the status
// of resp is other than 200. A non-200 consumes up to maxErrorBodySize of the
// body; a 200 leaves it unread. Closing stays with the caller.
func checkStatus(step string, resp *http.Response) error {
	if resp.StatusCode == http.StatusOK {
		return nil
	}
	return upstreamError(step, resp.StatusCode, readErrorBody(resp.Body))
}

// gatewayRequest is the parameter of a POST that a step sends to the gateway.
type gatewayRequest struct {
	// logMsg is the message of the DEBUG request record.
	logMsg string
	// step tags the errors.
	step string
	// upstream labels the call's metrics.
	upstream string
	path     string
	body     []byte
	headers  map[string]string
}

// postToGateway sends req to the gateway and returns the response. On success
// the caller closes the response body. An error returns a nil response, with
// the body already closed when the status was other than 200.
func postToGateway(ctx context.Context, logger logr.Logger, gwClient *gateway.Client, req gatewayRequest) (*http.Response, error) {
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info(req.logMsg, "method", "POST", "path", req.path, "bodyLen", len(req.body), "headers", httplog.RedactedHeaders(req.headers))
	}

	call := coordmetrics.StartUpstreamCall(req.upstream)
	resp, err := gwClient.Post(ctx, req.path, req.body, req.headers)
	call.Done()
	if err != nil {
		return nil, fmt.Errorf("%s: request: %w", req.step, err)
	}
	if err := checkStatus(req.step, resp); err != nil {
		_ = resp.Body.Close()
		return nil, err
	}
	return resp, nil
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

// rejectUseOpenAIFormatOverride returns an error if params sets use_openai_format.
// decode and conditional-decode derive their body format directly from the
// request's original path, so a step-level override has no effect; rejecting
// the key surfaces stale config instead of silently ignoring it.
func rejectUseOpenAIFormatOverride(step string, params map[string]any) error {
	if _, ok := params["use_openai_format"]; ok {
		return fmt.Errorf("%s: use_openai_format is not a valid parameter for this step", step)
	}
	return nil
}

// unreachableFormatError builds an error for a request format with no
// registered coordinator route (see server.go), signaling a routing bug
// rather than a client error.
func unreachableFormatError(format reqcommon.APIType) error {
	return fmt.Errorf("unsupported request format %v: no coordinator route serves it", format)
}

// resolveFormat maps a request path to the wire format a step emits. The steps
// build only Completions, Chat Completions, Responses, and generate bodies, so
// any other API collapses to APITypeVLLMGenerate; Chat Completions and
// Responses additionally require useOpenAIFormat. Generate is the fallback
// because its body carries the prompt as reqCtx.TokenIDs and does not depend
// on the client's request shape.
func resolveFormat(useOpenAIFormat bool, path string) reqcommon.APIType {
	switch detected := reqcommon.DetectAPIType(path); detected {
	case reqcommon.APITypeCompletions:
		return detected
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses:
		if useOpenAIFormat {
			return detected
		}
	}
	return reqcommon.APITypeVLLMGenerate
}

// promptItems returns the array an API carries its prompt items in: a
// chat-completions messages array, or a Responses input array. ok is false for
// an API that carries no item array, and for a body whose field is absent or
// holds something other than an array.
func promptItems(body map[string]any, apiType reqcommon.APIType) ([]any, bool) {
	var field string
	switch apiType {
	case reqcommon.APITypeChatCompletions:
		field = reqcommon.FieldMessages
	case reqcommon.APITypeResponses:
		field = reqcommon.FieldInput
	default:
		return nil, false
	}
	items, ok := body[field].([]any)
	return items, ok
}

// isImagePart reports whether a content part of type partType names an image on
// a request to apiType.
//
// A chat-completions request may name one either way: vLLM's chat parser primes
// input_image and image_url through the same content part map, so an
// input_image on a chat request reaches the model and has to be walked. A
// Responses request names it input_image only, since the Responses input union
// does not define image_url and a request carrying one fails the model server's
// input validation before any worker sees it. The sidecar's encoder fan-out
// applies the same rule.
func isImagePart(partType string, apiType reqcommon.APIType) bool {
	switch partType {
	case reqcommon.PartTypeInputImage:
		return true
	case reqcommon.PartTypeImageURL:
		return apiType != reqcommon.APITypeResponses
	}
	return false
}

// imagePart is an image content part together with the body position it was
// found at, named for error messages: "message 0 content part 2", "input item 1
// output part 0".
type imagePart struct {
	part     map[string]any
	location string
}

// collectImageParts walks a chat-completions messages array or a Responses
// input array and returns the image content parts in order.
//
// Every step that indexes reqCtx.MultimodalEntries by position walks from here:
// replace-media-urls builds the entries, encode picks the part to prime, and
// decode stamps the hash. They agree because they see the same parts in the
// same order, so a second walk elsewhere would reintroduce the chance to
// disagree. Walking once also lets the encode fan-out index by position instead
// of re-walking per image (O(N*M) -> O(N+M)).
//
// Which part types count as an image is isImagePart's rule.
func collectImageParts(items []any, apiType reqcommon.APIType) []imagePart {
	itemLabel := "message"
	if apiType == reqcommon.APITypeResponses {
		itemLabel = "input item"
	}

	var parts []imagePart
	for itemIdx, item := range items {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		for _, array := range reqcommon.ItemPartArrays(itemMap, apiType) {
			for partIdx, part := range array.Parts {
				partMap, ok := part.(map[string]any)
				if !ok {
					continue
				}
				partType, _ := partMap[reqcommon.FieldType].(string)
				if !isImagePart(partType, apiType) {
					continue
				}
				parts = append(parts, imagePart{
					part:     partMap,
					location: fmt.Sprintf("%s %d %s part %d", itemLabel, itemIdx, array.Field, partIdx),
				})
			}
		}
	}
	return parts
}

// buildPrefillMMFeatures builds features for the prefill request, deciding
// per entry whether to ship mm_metadata or kwargs_data. An entry whose hash
// has a descriptor in ecTransferParams and carries non-empty MMMetadata is
// emitted as mm_metadata[i] with a null kwargs_data[i]; every other entry is
// emitted as kwargs_data[i] with a null mm_metadata[i]. Both fields are
// always present so vLLM's per-item merge (merge_mm_kwargs_items) can
// reconstruct each entry from whichever side is non-null.
//
// The choice is per entry because a single encode sub-request can return no
// descriptor for its image (nixlEC.MergeEncodeResponse logs a warning and
// continues rather than failing the request): that entry has no EC path, so
// its tensor data stays inline. vLLM's _require_ec_for_metadata_only is a
// request-level check and does not catch partial coverage, so the coverage
// reconciliation lives here.
func buildPrefillMMFeatures(entries []pipeline.MultimodalEntry, ecTransferParams map[string]any) map[string]any {
	if len(entries) == 0 {
		return nil
	}
	hashes := make([]string, len(entries))
	placeholders := make([]any, len(entries))
	kwargs := make([]string, len(entries))
	metadata := make([]string, len(entries))
	for i, entry := range entries {
		hashes[i] = entry.Hash
		placeholders[i] = map[string]any{
			"offset": entry.Placeholder.Offset,
			"length": entry.Placeholder.Length,
		}
		if _, hasEC := ecTransferParams[entry.Hash]; hasEC && entry.MMMetadata != "" {
			metadata[i] = entry.MMMetadata
		} else {
			kwargs[i] = entry.KwargsData
		}
	}
	return map[string]any{
		"mm_hashes":       map[string][]string{ModalityImage: hashes},
		"mm_placeholders": map[string][]any{ModalityImage: placeholders},
		"kwargs_data":     mmBase64Field(kwargs),
		"mm_metadata":     mmBase64Field(metadata),
	}
}

// mmBase64Field builds a modality-keyed feature value from per-entry base64
// strings (kwargs_data or mm_metadata). The empty string is our internal
// "resolve from cache" / absent sentinel and MUST serialize as JSON null, not
// "": vLLM treats null (or an absent field) as a cache-hit item to fetch from
// the encoder cache by hash, whereas "" is decoded as an inline blob and fails
// with "Input data was truncated". Non-empty entries are forwarded verbatim.
func mmBase64Field(values []string) map[string][]any {
	items := make([]any, len(values))
	for i, v := range values {
		if v != "" {
			items[i] = v
		}
	}
	return map[string][]any{ModalityImage: items}
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
// mm_metadata is accepted and unread: this parser only runs for inbound
// generate, where EncodeStep is skipped (encode.go:99), so no entry gets EC
// coverage and prefill ships kwargs_data for all of them. On the chat leg the
// metadata comes from the render response, not the client body.
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
	// placeholders for cached items, never a shortened list.
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
