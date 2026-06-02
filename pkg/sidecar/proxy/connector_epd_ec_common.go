// This file holds the encoder fan-out scaffolding shared by every EC
// connector: deduplicated multimodal-item extraction and the parallel
// per-item encoder dispatch loop. Each EC connector
// (ec-example via fanoutEncoderPrimer, ec-epd via fanoutEncoderCollect)
// supplies its own per-response perItem callback and otherwise reuses
// these helpers verbatim.

package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

// Multimodal content types that need encoder processing.
var mmTypes = map[string]bool{
	"image_url":   true,
	"audio_url":   true,
	"video_url":   true,
	"input_audio": true,
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
// messages[0].content (text removed) and max_tokens=1, stream disabled.
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
	encoderRequest["max_tokens"] = 1
	encoderRequest["stream"] = false
	delete(encoderRequest, "stream_options")

	return encoderRequest
}

// mmItemURL returns the URL string for a URL-based multimodal item, or
// empty string when the item carries inline data instead.
func mmItemURL(item map[string]any) string {
	itemType, _ := item["type"].(string)
	switch itemType {
	case "image_url":
		if m, ok := item["image_url"].(map[string]any); ok {
			if u, ok := m["url"].(string); ok {
				return u
			}
		}
	case "audio_url":
		if m, ok := item["audio_url"].(map[string]any); ok {
			if u, ok := m["url"].(string); ok {
				return u
			}
		}
	case "video_url":
		if m, ok := item["video_url"].(map[string]any); ok {
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
// Encoder transport errors and non-2xx responses are hard-fail; the first
// error from any goroutine is returned, all others discarded.
func (s *Server) fanoutEncoder(
	originalRequest map[string]any,
	items []map[string]any,
	encoderHostPorts []string,
	requestID string,
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	s.logger.Info("processing multimodal items", "count", len(items), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	var wg sync.WaitGroup
	errChan := make(chan error, len(items))

	for idx, mmItem := range items {
		wg.Add(1)
		// We will add more sophisticated Encoder pickup option later.
		encoderHostPort := encoderHostPorts[idx%len(encoderHostPorts)]

		go func(item map[string]any, hostPort string, itemIdx int) {
			defer wg.Done()

			encoderRequest := buildEncoderRequest(originalRequest, item)

			body, err := json.Marshal(encoderRequest)
			if err != nil {
				errChan <- fmt.Errorf("failed to marshal encoder request for item %d: %w", itemIdx, err)
				return
			}

			encoderHandler, err := s.encoderProxyHandler(hostPort)
			if err != nil {
				errChan <- fmt.Errorf("failed to get encoder proxy handler for %s: %w", hostPort, err)
				return
			}

			req, err := http.NewRequest("POST", ChatCompletionsPath, bytes.NewReader(body))
			if err != nil {
				errChan <- fmt.Errorf("failed to create encoder request for item %d: %w", itemIdx, err)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(requestHeaderRequestID, fmt.Sprintf("%s-enc-%d", requestID, itemIdx))

			s.logger.V(logging.DEBUG).Info("sending encoder request", "item", itemIdx, "to", hostPort, "requestID", requestID)

			pw := &bufferedResponseWriter{}
			encoderHandler.ServeHTTP(pw, req)

			if isHTTPError(pw.statusCode) {
				errChan <- fmt.Errorf("encoder request failed for item %d with status %d: %s", itemIdx, pw.statusCode, pw.buffer.String())
				return
			}

			if perItem != nil {
				if err := perItem(itemIdx, pw); err != nil {
					errChan <- err
					return
				}
			}

			s.logger.V(4).Info("encoder request completed", "item", itemIdx, "requestID", requestID)
		}(mmItem, encoderHostPort, idx)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}
