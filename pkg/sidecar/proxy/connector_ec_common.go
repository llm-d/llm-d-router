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

// This file holds the encoder fan-out scaffolding shared by every EC
// connector: deduplicated multimodal-item extraction and the parallel
// per-item encoder dispatch loop. Each EC connector
// (ec-example via fanoutEncoderPrimer, ec-nixl via fanoutEncoderCollect)
// supplies its own per-response perItem callback and otherwise reuses
// these helpers verbatim.

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"golang.org/x/sync/errgroup"
)

// mmTypeInputImage is the Responses API's equivalent of image_url.
const mmTypeInputImage = "input_image"

// Multimodal content types that need encoder processing.
var mmTypes = map[string]bool{
	"image_url":      true,
	"audio_url":      true,
	"video_url":      true,
	"input_audio":    true,
	mmTypeInputImage: true,
}

// requestInput returns the request's Responses input items, decoded the
// same way requestMessages decodes messages. Responses' input may also be a
// bare JSON string (a single text turn), which yields a nil slice and no
// error, the same as an absent field.
func requestInput(req map[string]any) ([]json.RawMessage, error) {
	switch v := req[requestFieldInput].(type) {
	case nil:
		return nil, nil
	case json.RawMessage:
		var items []json.RawMessage
		if err := json.Unmarshal(v, &items); err != nil {
			var s string
			if json.Unmarshal(v, &s) == nil {
				return nil, nil
			}
			return nil, err
		}
		return items, nil
	default:
		return nil, fmt.Errorf("input is %T, want a JSON array or string", v)
	}
}

// truncateLongStrings recursively shortens long string values for logging.
func truncateLongStrings(v any, maxLen int) any {
	switch x := v.(type) {
	case string:
		if len(x) > maxLen {
			return fmt.Sprintf("%s...(%d bytes)", x[:maxLen], len(x))
		}
		return x
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, vv := range x {
			out[k] = truncateLongStrings(vv, maxLen)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, vv := range x {
			out[i] = truncateLongStrings(vv, maxLen)
		}
		return out
	default:
		return v
	}
}

// extractMMItems extracts all multimodal content parts from the request:
// chat-completions' messages array, or a Responses input array. Which field
// to walk is gated on apiType, not field presence: a client could send a
// stray field the other format doesn't use, and presence-based sniffing
// would process the wrong one.
func extractMMItems(logger logr.Logger, requestData map[string]any, apiType reqcommon.APIType) []map[string]any {
	var items []map[string]any

	var wrapped []json.RawMessage
	var err error
	switch apiType {
	case reqcommon.APITypeResponses:
		wrapped, err = requestInput(requestData)
	default:
		wrapped, err = requestMessages(requestData)
	}
	if err != nil {
		logger.V(logging.DEBUG).Info("cannot read request content for multimodal extraction", "error", err)
		return items
	}

	for _, raw := range wrapped {
		var itemMap map[string]any
		if err := json.Unmarshal(raw, &itemMap); err != nil {
			continue
		}

		content := itemMap[requestFieldContent]
		contentList, ok := content.([]any)
		if !ok {
			continue
		}

		for _, part := range contentList {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}

			partType, ok := partMap["type"].(string)
			if !ok {
				continue
			}
			if partType == mmTypeInputImage && mmItemURL(partMap) == "" {
				// A file_id-referenced image (no image_url string) has no
				// content the encoder can fetch or receive inline.
				logger.V(logging.DEBUG).Info("skipping input_image with no fetchable URL", "hasFileID", partMap["file_id"] != nil)
				continue
			}

			if mmTypes[partType] {
				items = append(items, partMap)
			}
		}
	}

	return items
}

// buildEncoderRequest builds a per-item encoder request from scratch: model
// plus a single synthetic message wrapping mmItem, capped to one output
// token with streaming disabled. It does not copy the client's request: a
// client field with an incompatible schema on the encoder's own API (e.g.
// chat completions' tools) would otherwise reach it as-is. The encoder is
// addressed as Responses when the original request is Responses, so a
// Responses input_image part (bare-string URL, sibling detail field) is
// forwarded unmodified rather than reshaped into chat completions'
// image_url nesting, which vLLM's chat-completions engine ignores detail
// on. A Responses encoder request sets store to false: vLLM defaults an
// absent store to true, and nothing ever reads or reaps the response object
// that a stored per-item priming request would leave behind.
func buildEncoderRequest(originalRequest map[string]any, mmItem map[string]any, apiType reqcommon.APIType) map[string]any {
	if apiType != reqcommon.APITypeResponses {
		apiType = reqcommon.APITypeChatCompletions
	}

	encoderRequest := map[string]any{}
	if model, ok := originalRequest[requestFieldModel]; ok {
		encoderRequest[requestFieldModel] = model
	}
	message := map[string]any{"role": "user", "content": []map[string]any{mmItem}}
	if apiType == reqcommon.APITypeResponses {
		encoderRequest[requestFieldInput] = []map[string]any{message}
		encoderRequest[requestFieldStore] = false
	} else {
		encoderRequest[requestFieldMessages] = []map[string]any{message}
	}

	reqcommon.CapSingleToken(encoderRequest, apiType)

	return encoderRequest
}

// mmItemURL returns the URL string for a URL-based multimodal item, or
// empty string when the item carries inline data instead. A Responses
// input_image item stores its URL as a bare string directly on the item,
// unlike the other three types, which nest it under a same-named object.
func mmItemURL(item map[string]any) string {
	itemType, _ := item["type"].(string)
	switch itemType {
	case "image_url", "audio_url", "video_url":
		if m, ok := item[itemType].(map[string]any); ok {
			if u, ok := m["url"].(string); ok {
				return u
			}
		}
	case mmTypeInputImage:
		if u, ok := item["image_url"].(string); ok {
			return u
		}
	}
	return ""
}

// mmItemsForFanout extracts the multimodal items from a request body and
// deduplicates URL-based items (image_url / audio_url / video_url /
// input_image). Non-URL items (e.g. inline input_audio) are kept verbatim.
// Returns nil when there is no multimodal content. The caller should skip
// the encoder stage in that case.
func (s *Server) mmItemsForFanout(originalRequest map[string]any, requestID string, apiType reqcommon.APIType) []map[string]any {
	raw := extractMMItems(s.logger, originalRequest, apiType)
	if len(raw) == 0 {
		return nil
	}
	seenURLs := make(map[string]struct{})
	items := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if url := mmItemURL(item); url != "" {
			if _, seen := seenURLs[url]; seen {
				s.logger.V(logging.DEBUG).Info("skipping duplicate multimodal URL", "url", url, "requestID", requestID)
				continue
			}
			seenURLs[url] = struct{}{}
		}
		items = append(items, item)
	}
	return items
}

// fanoutEncoder fans out one encoder request per item, in parallel, with
// round-robin over encoderHostPorts. perItem is invoked once per item AFTER
// the encoder has returned a 2xx response; it receives the item's
// positional index (post-dedup) and the buffered encoder response. The
// callback may return an error to fail the whole fan-out, or nil to
// accept. perItem may be nil for fire-and-forget primer-style usage.
//
// The first goroutine to fail cancels ctx so sibling encoder requests are
// aborted at the transport layer. Every failure is logged before propagating;
// grp.Wait returns the first non-nil error.
func (s *Server) fanoutEncoder(
	ctx context.Context,
	originalRequest map[string]any,
	items []map[string]any,
	encoderHostPorts []string,
	requestID string,
	apiType reqcommon.APIType,
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	if len(encoderHostPorts) == 0 {
		return fmt.Errorf("fanoutEncoder: no encoder hostPorts provided (requestID=%s)", requestID)
	}

	encoderPath := reqcommon.PathChatCompletions
	if apiType == reqcommon.APITypeResponses {
		encoderPath = reqcommon.PathResponses
	}

	s.logger.Info("processing multimodal items", "count", len(items), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	grp, gctx := errgroup.WithContext(ctx)
	for idx, mmItem := range items {
		hostPort := encoderHostPorts[idx%len(encoderHostPorts)]
		grp.Go(func() error {
			encoderRequest := buildEncoderRequest(originalRequest, mmItem, apiType)

			body, err := json.Marshal(encoderRequest)
			if err != nil {
				err = fmt.Errorf("failed to marshal encoder request for item %d: %w", idx, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			encoderHandler, err := s.encoderProxyHandler(hostPort)
			if err != nil {
				err = fmt.Errorf("failed to get encoder proxy handler for %s: %w", hostPort, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			req, err := http.NewRequestWithContext(gctx, "POST", encoderPath, bytes.NewReader(body))
			if err != nil {
				err = fmt.Errorf("failed to create encoder request for item %d: %w", idx, err)
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(requestHeaderRequestID, fmt.Sprintf("%s-enc-%d", requestID, idx))

			s.logger.V(logging.DEBUG).Info("sending encoder request", "item", idx, "to", hostPort, "requestID", requestID)

			pw := &bufferedResponseWriter{}
			encoderHandler.ServeHTTP(pw, req)

			if isHTTPError(pw.statusCode) {
				err := fmt.Errorf("encoder request failed for item %d with status %d: %s", idx, pw.statusCode, pw.buffer.String())
				s.logger.Error(err, "encoder fanout", "item", idx, "requestID", requestID)
				return err
			}

			if perItem != nil {
				if err := perItem(idx, pw); err != nil {
					s.logger.Error(err, "encoder fanout perItem", "item", idx, "requestID", requestID)
					return err
				}
			}

			s.logger.V(logging.DEBUG).Info("encoder request completed", "item", idx, "requestID", requestID)
			return nil
		})
	}
	return grp.Wait()
}

// runPDPipeline finalizes the post-encoder request and dispatches it to the
// configured P/D connector or directly to the decoder. The caller has already
// generated requestID and merged any encoder-side metadata into
// body. On JSON-marshal failure, runPDPipeline writes the error
// response itself (matching the existing handler pattern) and returns.
func (s *Server) runPDPipeline(
	w http.ResponseWriter,
	r *http.Request,
	body map[string]any,
	prefillEndPoint string,
	requestID string,
	apiType reqcommon.APIType,
) {
	// Skip decode-first; the encoder has run and prefill must execute.
	body[requestFieldCacheHitThreshold] = 0

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	pdRequest := cloneRequestWithBody(r.Context(), r, modifiedBody)
	pdRequest.Header.Add(requestHeaderRequestID, requestID)

	destination := "decoder"
	if len(prefillEndPoint) > 0 {
		destination = "prefiller"
	}

	// Don't log the full body. Inline base64 images can be MB each.
	if v := s.logger.V(logging.DEBUG); v.Enabled() {
		kv := []any{
			"requestID", requestID,
			"destination", destination,
			"prefiller", prefillEndPoint,
			"bodyBytes", len(modifiedBody),
		}
		if ec, ok := body[requestFieldECTransferParams]; ok {
			kv = append(kv, requestFieldECTransferParams, truncateLongStrings(ec, 64))
		}
		v.Info("forwarding request after encoder", kv...)
	}

	if len(prefillEndPoint) > 0 {
		s.logger.V(logging.DEBUG).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		// The encoder path does not carry a KV cache source: the P2P prefix pull
		// is not wired through encoder disaggregation. The empty source skips the
		// p2p injection regardless of --enable-p2p-pull.
		s.handlePDConnector(w, pdRequest, prefillEndPoint, "", apiType)
		return
	}

	s.logger.V(logging.DEBUG).Info("no prefiller configured, going directly to decoder after encoder")
	if !s.forwardDataParallel || !s.dataParallelHandler(w, pdRequest) {
		s.decoderProxy.ServeHTTP(w, pdRequest)
	}
}
