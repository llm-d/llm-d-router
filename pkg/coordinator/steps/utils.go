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
	"slices"

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

// parseOutputMode reads the output_mode step parameter, defaulting to text so a
// tokens-in pipeline answers the client with readable text. derender asks the
// worker to apply its own parsers and answer with a parsed assistant message.
//
// tokens is refused: decode streams its response straight to the client, so
// that mode would answer with token IDs and no text at all. Any other value is
// a configuration error rather than a field forwarded to the worker, which
// rejects an unknown mode with 400.
func parseOutputMode(params map[string]any) (string, error) {
	mode, err := paramString(params, reqcommon.FieldOutputMode)
	if err != nil {
		return "", err
	}
	switch mode {
	case "":
		return reqcommon.OutputModeText, nil
	case reqcommon.OutputModeText, reqcommon.OutputModeDerender:
		return mode, nil
	default:
		return "", fmt.Errorf("%s must be %q or %q, got %q",
			reqcommon.FieldOutputMode, reqcommon.OutputModeText, reqcommon.OutputModeDerender, mode)
	}
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
// build only Completions, Chat Completions, Responses, Messages, and generate
// bodies, so any other API collapses to APITypeVLLMGenerate; Chat Completions,
// Responses, and Messages additionally require useOpenAIFormat. Generate is the
// fallback because its body carries the prompt as reqCtx.TokenIDs and does not
// depend on the client's request shape.
func resolveFormat(useOpenAIFormat bool, path string) reqcommon.APIType {
	switch detected := reqcommon.DetectAPIType(path); detected {
	case reqcommon.APITypeCompletions:
		return detected
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeMessages:
		if useOpenAIFormat {
			return detected
		}
	}
	return reqcommon.APITypeVLLMGenerate
}

// buildPrefillMMFeatures builds features for the prefill request, deciding
// per entry whether to ship mm_metadata or kwargs_data. An entry whose hash
// has a descriptor in ecTransferParams and carries non-empty MMMetadata is
// emitted as mm_metadata with a null kwargs_data slot; every other entry is
// emitted as kwargs_data with a null mm_metadata slot. Both fields are
// always present so vLLM's per-item merge (merge_mm_kwargs_items) can
// reconstruct each entry from whichever side is non-null.
//
// The choice is per entry because a single encode sub-request can return no
// descriptor for its image (nixlEC.MergeEncodeResponse logs a warning and
// continues rather than failing the request): that entry has no EC path, so
// its tensor data stays inline. vLLM's _require_ec_for_metadata_only is a
// request-level check and does not catch partial coverage, so the coverage
// reconciliation lives here.
//
// Entries are grouped by Modality: a mixed-modality request has one key per
// modality in each feature map.
func buildPrefillMMFeatures(entries []pipeline.MultimodalEntry, ecTransferParams map[string]any) map[string]any {
	if len(entries) == 0 {
		return nil
	}
	hashesByMod := make(map[reqcommon.Modality][]string)
	placeholdersByMod := make(map[reqcommon.Modality][]any)
	kwargsByMod := make(map[reqcommon.Modality][]any)
	metadataByMod := make(map[reqcommon.Modality][]any)
	for _, entry := range entries {
		mod := entry.Modality
		hashesByMod[mod] = append(hashesByMod[mod], entry.Hash)
		placeholdersByMod[mod] = append(placeholdersByMod[mod], map[string]any{
			"offset": entry.Placeholder.Offset,
			"length": entry.Placeholder.Length,
		})
		if _, hasEC := ecTransferParams[entry.Hash]; hasEC && entry.MMMetadata != "" {
			kwargsByMod[mod] = append(kwargsByMod[mod], nil)
			metadataByMod[mod] = append(metadataByMod[mod], kwargsSentinel(entry.MMMetadata))
		} else {
			kwargsByMod[mod] = append(kwargsByMod[mod], kwargsSentinel(entry.KwargsData))
			metadataByMod[mod] = append(metadataByMod[mod], nil)
		}
	}
	return map[string]any{
		"mm_hashes":       hashesByMod,
		"mm_placeholders": placeholdersByMod,
		"kwargs_data":     kwargsByMod,
		"mm_metadata":     metadataByMod,
	}
}

// promptItemsField returns the body field an API carries its prompt items in:
// messages for chat completions and Messages, input for Responses. ok is false
// for an API that carries no item array.
func promptItemsField(apiType reqcommon.APIType) (string, bool) {
	switch apiType {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages:
		return reqcommon.FieldMessages, true
	case reqcommon.APITypeResponses:
		return reqcommon.FieldInput, true
	default:
		return "", false
	}
}

// promptItems returns the array an API carries its prompt items in: the
// messages array of chat completions or Messages, or a Responses input array.
// ok is false for an API that carries no item array, and for a body whose field
// is absent or holds something other than an array.
//
// The server keeps the field as a json.RawMessage so its key order survives
// re-marshaling. A step that edits the returned items stores them back with
// setPromptItems so the edit is forwarded.
func promptItems(body map[string]any, apiType reqcommon.APIType) ([]any, bool) {
	field, ok := promptItemsField(apiType)
	if !ok {
		return nil, false
	}
	switch v := body[field].(type) {
	case []any:
		return v, true
	case json.RawMessage:
		var items []any
		if err := json.Unmarshal(v, &items); err != nil {
			return nil, false
		}
		return items, items != nil
	default:
		return nil, false
	}
}

// setPromptItems stores items under the field promptItems reads them from.
func setPromptItems(body map[string]any, apiType reqcommon.APIType, items []any) {
	if field, ok := promptItemsField(apiType); ok {
		body[field] = items
	}
}

// mediaPart is a media content part together with the modality it names and the
// body position it was found at, the latter for error messages: "message 0
// content part 2", "input item 1 output part 0".
type mediaPart struct {
	part     map[string]any
	modality reqcommon.Modality
	location string
}

// collectMediaParts walks the messages array of chat completions or Messages,
// or a Responses input array, and returns the media content parts in order.
//
// Every step that pairs reqCtx.MultimodalEntries with parts by position walks
// from here -- replace-media-urls builds the entries, encode picks the part to
// prime, decode stamps the hash -- so they see the same parts in the same order
// and a second walk elsewhere could disagree. Walking once also lets the encode
// fan-out index by position instead of re-walking per entry (O(N*M) -> O(N+M)).
//
// Parts are returned whenever the type names a modality, with no check that the
// part carries usable media: replace-media-urls rejects an unusable one as it
// builds the entries (see collectMediaRefs), so filtering here would shift every
// later part of its modality onto another part's hash. Which types count is
// reqcommon.PartModality's rule, the same on chat completions and Responses, and
// reqcommon.MessagesBlockModality's on a Messages request. Beyond that, apiType
// selects the item label and which arrays an item holds.
func collectMediaParts(items []any, apiType reqcommon.APIType) []mediaPart {
	itemLabel := "message"
	if apiType == reqcommon.APITypeResponses {
		itemLabel = "input item"
	}
	modalityOf := reqcommon.PartModality
	if apiType == reqcommon.APITypeMessages {
		modalityOf = reqcommon.MessagesBlockModality
	}

	var parts []mediaPart
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
				modality, ok := modalityOf(partType)
				if !ok {
					continue
				}
				parts = append(parts, mediaPart{
					part:     partMap,
					modality: modality,
					location: fmt.Sprintf("%s %d %s part %d", itemLabel, itemIdx, array.Field, partIdx),
				})
			}
		}
	}
	return parts
}

// inlineAudioData reads the payload of an input_audio content part, whose audio
// sits in the request body rather than behind a URL: "data" holds the base64
// audio and "format" names the codec. ok is false when there is no base64 data,
// the inline counterpart of reqcommon.MediaPartURLRef reporting no URL. format
// is returned unchecked; audioFormatToMIME is what judges it.
func inlineAudioData(part map[string]any) (data, format string, ok bool) {
	inner, isObject := part[reqcommon.PartTypeInputAudio].(map[string]any)
	if !isObject {
		return "", "", false
	}
	data, _ = inner["data"].(string)
	if data == "" {
		return "", "", false
	}
	format, _ = inner["format"].(string)
	return data, format, true
}

// mediaPartCarriesPayload reports whether a media content part still carries
// the media it names: a readable URL for a URL-based part, base64 data for an
// inline input_audio part. replace-media-urls already rejects a part carrying
// neither (see collectMediaRefs), so a later check is defensive.
func mediaPartCarriesPayload(part map[string]any) bool {
	if partType, _ := part[reqcommon.FieldType].(string); partType == reqcommon.PartTypeInputAudio {
		_, _, ok := inlineAudioData(part)
		return ok
	}
	return reqcommon.MediaPartURL(part) != ""
}

// groupMediaPartsByModality indexes a collectMediaParts walk by modality, each
// list keeping the walk's order: the shape a step indexes by an entry's
// per-modality position.
func groupMediaPartsByModality(parts []mediaPart) map[reqcommon.Modality][]mediaPart {
	byMod := make(map[reqcommon.Modality][]mediaPart)
	for _, p := range parts {
		byMod[p.modality] = append(byMod[p.modality], p)
	}
	return byMod
}

// modalityLocalIndexes returns, for each entry, its position among the entries
// sharing its modality. That position is the entry's coordinate into the
// per-modality part lists and the per-modality render response slots.
func modalityLocalIndexes(entries []pipeline.MultimodalEntry) []int {
	local := make([]int, len(entries))
	counter := make(map[reqcommon.Modality]int)
	for i, entry := range entries {
		local[i] = counter[entry.Modality]
		counter[entry.Modality]++
	}
	return local
}

// validateEntryModalities rejects a MultimodalEntry with no Modality. Render is
// the only caller, as the first step that pairs on the field; the later readers
// inherit the entries it passed. Not ErrBadRequest: both producers set the
// field, so an untagged entry is a coordinator bug and should surface as a 5xx.
//
// It is a backstop -- reqcommon.PartModality returns ok == false rather than an
// empty Modality, and extractMultimodalEntries rejects an empty feature-map key
// -- against an entry defaulted to some modality, which would splice into that
// modality's index sequence and shift every later entry sharing the label.
func validateEntryModalities(entries []pipeline.MultimodalEntry) error {
	for i, entry := range entries {
		if entry.Modality == "" {
			return fmt.Errorf("multimodal entry %d has no modality (hash %q)", i, entry.Hash)
		}
	}
	return nil
}

// kwargsSentinel implements the JSON-null "resolve from cache" convention for
// one kwargs_data slot. Our sentinel is the empty string, which MUST serialize
// as null, not "": vLLM reads null (or an absent field) as a cache-hit item to
// fetch by hash, while "" is decoded as an inline tensor and fails with "Input
// data was truncated".
func kwargsSentinel(k string) any {
	if k == "" {
		return nil
	}
	return k
}

// singleEntryKwargs builds a kwargs_data value for one entry: each encode
// fanout sub-request carries exactly one entry's kwargs under its modality key.
func singleEntryKwargs(modality reqcommon.Modality, kwargs string) map[reqcommon.Modality][]any {
	return map[reqcommon.Modality][]any{modality: {kwargsSentinel(kwargs)}}
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

// mmModalityArray reads features[field][modality] as a JSON array. present is
// false when field or its per-modality entry is absent or null, a valid "no
// such modality" state rather than an error. A present value of the wrong type
// is ErrBadRequest, so a malformed request fails loudly instead of reading as
// absent.
func mmModalityArray(features map[string]any, field string, modality reqcommon.Modality) (arr []any, present bool, err error) {
	rawField, ok := features[field]
	if !ok || rawField == nil {
		return nil, false, nil
	}
	m, ok := rawField.(map[string]any)
	if !ok {
		return nil, false, fmt.Errorf("%s must be an object: %w", field, pipeline.ErrBadRequest)
	}
	// The client's features map is keyed by plain JSON strings: this is the
	// boundary between the typed vocabulary and the keys a request arrived with.
	raw, ok := m[string(modality)]
	if !ok || raw == nil {
		return nil, false, nil
	}
	arr, ok = raw.([]any)
	if !ok {
		return nil, false, fmt.Errorf("%s[%s] must be an array: %w", field, modality, pipeline.ErrBadRequest)
	}
	return arr, true, nil
}

// modalitiesInFeatures returns the modality keys present in features[field],
// sorted so entry ordering stays deterministic across map iteration.
// (nil, nil) when absent or an empty object, ErrBadRequest when present but not
// an object.
//
// A key here is whatever the client sent, so this is where the vocabulary opens
// up: a token-in request may name a modality reqcommon declares no constant for,
// and the pipeline carries it through. The one rejection is the empty key, which
// would become an untagged MultimodalEntry (see validateEntryModalities),
// whatever value it carries -- null means "modality absent" only for a named key.
func modalitiesInFeatures(features map[string]any, field string) ([]reqcommon.Modality, error) {
	raw, ok := features[field]
	if !ok || raw == nil {
		return nil, nil
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object: %w", field, pipeline.ErrBadRequest)
	}
	var out []reqcommon.Modality
	for k, v := range m {
		// Before the null skip below, so the key is judged on its own: both
		// {"": null} and {"": []} are rejected.
		if k == "" {
			return nil, fmt.Errorf("%s has an empty modality key: %w", field, pipeline.ErrBadRequest)
		}
		if v == nil {
			continue
		}
		out = append(out, reqcommon.Modality(k))
	}
	slices.Sort(out)
	return out, nil
}

// checkModalitiesHashed reports an error when mm_placeholders or kwargs_data
// carries a modality hashed does not name. Only modalities mm_hashes names
// become entries, so skipping the extra key would drop the item from the
// prefill and decode bodies while its placeholder tokens stay in token_ids,
// leaving the engine placeholders with nothing behind them.
func checkModalitiesHashed(features map[string]any, hashed []reqcommon.Modality) error {
	known := make(map[reqcommon.Modality]struct{}, len(hashed))
	for _, mod := range hashed {
		known[mod] = struct{}{}
	}
	for _, field := range []string{"mm_placeholders", "kwargs_data"} {
		mods, err := modalitiesInFeatures(features, field)
		if err != nil {
			return err
		}
		for _, mod := range mods {
			if _, ok := known[mod]; !ok {
				return fmt.Errorf("%s[%s] has no matching mm_hashes[%s]: %w",
					field, mod, mod, pipeline.ErrBadRequest)
			}
		}
	}
	return nil
}

// extractMultimodalEntries builds []pipeline.MultimodalEntry from the parallel
// slices in a generate-format features map. Each modality key under mm_hashes
// produces a run of entries, modalities visited in sorted order. Returns nil for
// a text-only request.
//
// mm_hashes names the modality set, so a modality only the other fields carry is
// ErrBadRequest (see checkModalitiesHashed). Per modality, mm_hashes and
// mm_placeholders are required and of equal length; kwargs_data is optional, and
// an absent field or null item means "resolve from the encoder cache by hash".
// A wrong type, length mismatch, or unexpected element is ErrBadRequest.
func extractMultimodalEntries(features map[string]any) ([]pipeline.MultimodalEntry, error) {
	if features == nil {
		return nil, nil
	}
	modalities, err := modalitiesInFeatures(features, "mm_hashes")
	if err != nil {
		return nil, err
	}
	if err := checkModalitiesHashed(features, modalities); err != nil {
		return nil, err
	}
	if len(modalities) == 0 {
		return nil, nil
	}

	var entries []pipeline.MultimodalEntry
	for _, mod := range modalities {
		rawHashes, _, err := mmModalityArray(features, "mm_hashes", mod)
		if err != nil {
			return nil, err
		}

		rawPlaceholders, present, err := mmModalityArray(features, "mm_placeholders", mod)
		if err != nil {
			return nil, err
		}

		rawKwargs, hasKwargs, err := mmModalityArray(features, "kwargs_data", mod)
		if err != nil {
			return nil, err
		}

		n := len(rawHashes)
		if !present && n > 0 {
			return nil, fmt.Errorf("mm_placeholders[%s] is required when mm_hashes[%s] is set: %w",
				mod, mod, pipeline.ErrBadRequest)
		}
		if len(rawPlaceholders) != n {
			return nil, fmt.Errorf("features length mismatch for %s: mm_hashes has %d, mm_placeholders has %d: %w",
				mod, n, len(rawPlaceholders), pipeline.ErrBadRequest)
		}
		// When present, kwargs_data is parallel to mm_hashes: full length with
		// nulls for cached items, never shortened. Metadata-only (cache-hit)
		// requests omit the field entirely.
		if hasKwargs && len(rawKwargs) != n {
			return nil, fmt.Errorf("features length mismatch for %s: mm_hashes has %d, kwargs_data has %d: %w",
				mod, n, len(rawKwargs), pipeline.ErrBadRequest)
		}

		for i := 0; i < n; i++ {
			hash, ok := rawHashes[i].(string)
			if !ok {
				return nil, fmt.Errorf("mm_hashes[%s][%d] must be a string: %w", mod, i, pipeline.ErrBadRequest)
			}

			pMap, ok := rawPlaceholders[i].(map[string]any)
			if !ok {
				return nil, fmt.Errorf("mm_placeholders[%s][%d] must be an object: %w", mod, i, pipeline.ErrBadRequest)
			}
			// The non-negative guarantee is load-bearing:
			// EncodeStep.buildEncodeTokenIDs indexes fullTokenIDs[offset]
			// (upper-bound guarded only) and allocates make([]int, 1+length),
			// which panics on a negative. vLLM accepts negatives here, so this
			// is deliberately stricter; do not relax it to a plain int parse.
			offset, err := anyToNonNegativeInt(pMap["offset"])
			if err != nil {
				return nil, fmt.Errorf("mm_placeholders[%s][%d].offset: %v: %w", mod, i, err, pipeline.ErrBadRequest)
			}
			length, err := anyToNonNegativeInt(pMap["length"])
			if err != nil {
				return nil, fmt.Errorf("mm_placeholders[%s][%d].length: %v: %w", mod, i, err, pipeline.ErrBadRequest)
			}

			// Empty KwargsData is the "resolve from cache" sentinel: either the
			// whole kwargs_data field is absent or this item is null.
			var kwarg string
			if hasKwargs {
				switch k := rawKwargs[i].(type) {
				case string:
					kwarg = k
				case nil:
				default:
					return nil, fmt.Errorf("kwargs_data[%s][%d] must be a string or null: %w", mod, i, pipeline.ErrBadRequest)
				}
			}

			entries = append(entries, pipeline.MultimodalEntry{
				Modality:   mod,
				Hash:       hash,
				KwargsData: kwarg,
				Placeholder: pipeline.PlaceholderRange{
					Offset: offset,
					Length: length,
				},
			})
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
