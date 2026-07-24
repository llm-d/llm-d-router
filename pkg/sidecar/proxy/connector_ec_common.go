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

	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"golang.org/x/sync/errgroup"
)

// Multimodal content types that need encoder processing.
var mmTypes = map[string]bool{
	"image_url":   true,
	"audio_url":   true,
	"video_url":   true,
	"input_audio": true,
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

// extractMMItems extracts all multimodal items from the request messages.
func extractMMItems(requestData map[string]any) []map[string]any {
	var items []map[string]any

	messages, ok := requestData["messages"].([]any)
	if !ok {
		return items
	}

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}

		content := msgMap["content"]
		contentList, ok := content.([]any)
		if !ok {
			continue
		}

		for _, item := range contentList {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}

			itemType, ok := itemMap["type"].(string)
			if !ok {
				continue
			}

			if mmTypes[itemType] {
				items = append(items, itemMap)
			}
		}
	}

	return items
}

// buildEncoderRequest creates a per-item encoder request: a deep copy of the
// original chat-completions request with only the multimodal item in
// messages[0].content (text removed), capped to a single output token, and
// stream disabled.
func buildEncoderRequest(originalRequest map[string]any, mmItem map[string]any) map[string]any {
	encoderRequest := make(map[string]any)
	for k, v := range originalRequest {
		encoderRequest[k] = v
	}

	messages := []map[string]any{
		{
			"role": "user",
			"content": []map[string]any{
				mmItem,
			},
		},
	}

	encoderRequest["messages"] = messages
	reqcommon.PrimeSingleTokenRequest(encoderRequest, originalRequest)

	return encoderRequest
}

// mmItemURL returns the URL string for a URL-based multimodal item, or
// empty string when the item carries inline data instead.
func mmItemURL(item map[string]any) string {
	itemType, _ := item["type"].(string)
	switch itemType {
	case "image_url", "audio_url", "video_url":
		if m, ok := item[itemType].(map[string]any); ok {
			if u, ok := m["url"].(string); ok {
				return u
			}
		}
	}
	return ""
}

// mmItemsForFanout extracts the multimodal items from a request body and
// deduplicates URL-based items (image_url / audio_url / video_url). Non-URL
// items (e.g. inline input_audio) are kept verbatim. Returns nil when
// there is no multimodal content. The caller should skip the encoder
// stage in that case.
func (s *Server) mmItemsForFanout(originalRequest map[string]any, requestID string) []map[string]any {
	raw := extractMMItems(originalRequest)
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
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	if len(encoderHostPorts) == 0 {
		return fmt.Errorf("fanoutEncoder: no encoder hostPorts provided (requestID=%s)", requestID)
	}

	s.logger.Info("processing multimodal items", "count", len(items), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	grp, gctx := errgroup.WithContext(ctx)
	for idx, mmItem := range items {
		hostPort := encoderHostPorts[idx%len(encoderHostPorts)]
		grp.Go(func() error {
			encoderRequest := buildEncoderRequest(originalRequest, mmItem)

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

			req, err := http.NewRequestWithContext(gctx, "POST", ChatCompletionsPath, bytes.NewReader(body))
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
// completionRequest. On JSON-marshal failure, runPDPipeline writes the error
// response itself (matching the existing handler pattern) and returns.
func (s *Server) runPDPipeline(
	w http.ResponseWriter,
	r *http.Request,
	completionRequest map[string]any,
	prefillEndPoint string,
	requestID string,
) {
	// Skip decode-first; the encoder has run and prefill must execute.
	completionRequest[requestFieldCacheHitThreshold] = 0

	modifiedBody, err := json.Marshal(completionRequest)
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
		if ec, ok := completionRequest[requestFieldECTransferParams]; ok {
			kv = append(kv, requestFieldECTransferParams, truncateLongStrings(ec, 64))
		}
		v.Info("forwarding request after encoder", kv...)
	}

	if len(prefillEndPoint) > 0 {
		s.logger.V(logging.DEBUG).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		// The encoder path does not carry a KV cache source: the P2P prefix pull
		// is not wired through encoder disaggregation. The empty source skips the
		// p2p injection regardless of --enable-p2p-pull.
		s.handlePDConnector(w, pdRequest, prefillEndPoint, "", APITypeChatCompletions)
		return
	}

	s.logger.V(logging.DEBUG).Info("no prefiller configured, going directly to decoder after encoder")
	if !s.forwardDataParallel || !s.dataParallelHandler(w, pdRequest) {
		s.decoderProxy.ServeHTTP(w, pdRequest)
	}
}
