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

package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/google/uuid"
)

// Multimodal content types that need encoder processing
var mmTypes = map[string]bool{
	"image_url":   true,
	"audio_url":   true,
	"input_audio": true,
}

// extractMMItems extracts all multimodal items from the request messages
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

// buildEncoderRequest creates a request for a single MM item with text removed
func buildEncoderRequest(originalRequest map[string]any, mmItem map[string]any) map[string]any {
	// Create a deep copy of the original request
	encoderRequest := make(map[string]any)
	for k, v := range originalRequest {
		encoderRequest[k] = v
	}

	// Build messages with only the MM item (no text)
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

// mmItemURL returns the URL string for a multimodal item, or empty string if not URL-based.
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
	}
	return ""
}

// fanoutEncoderPrimer sends concurrent requests to encoder cluster for all MM items. We assume that there is no identical MM items in the same request.
func (s *Server) fanoutEncoderPrimer(originalRequest map[string]any, encoderHostPorts []string, requestID string) error {
	allItems := extractMMItems(originalRequest)
	if len(allItems) == 0 {
		s.logger.V(4).Info("no multimodal items, skipping encoder", "requestID", requestID)
		return nil
	}

	// Deduplicate URL-based items; keep all non-URL items (e.g. inline audio).
	seenURLs := make(map[string]struct{})
	mmItems := make([]map[string]any, 0, len(allItems))
	for _, item := range allItems {
		if url := mmItemURL(item); url != "" {
			if _, seen := seenURLs[url]; seen {
				s.logger.V(4).Info("skipping duplicate multimodal URL", "url", url, "requestID", requestID)
				continue
			}
			seenURLs[url] = struct{}{}
		}
		mmItems = append(mmItems, item)
	}

	s.logger.Info("processing multimodal items", "count", len(mmItems), "requestID", requestID, "encoderHostPorts", encoderHostPorts)

	var wg sync.WaitGroup
	errChan := make(chan error, len(mmItems))

	// Round-robin over encoder servers
	for idx, mmItem := range mmItems {
		wg.Add(1)
		// We will add more sophisticated Encoder pickup option later
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

			s.logger.V(4).Info("sending encoder request", "item", itemIdx, "to", hostPort, "requestID", requestID)

			pw := &bufferedResponseWriter{}
			encoderHandler.ServeHTTP(pw, req)

			if isHTTPError(pw.statusCode) {
				errChan <- fmt.Errorf("encoder request failed for item %d with status %d: %s", itemIdx, pw.statusCode, pw.buffer.String())
				return
			}

			s.logger.V(4).Info("encoder request completed", "item", itemIdx, "requestID", requestID)
		}(mmItem, encoderHostPort, idx)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

// handleEPD handles an Encoder-Prefiller-Decoder disaggregation request
func (s *Server) handleEPD(w http.ResponseWriter, r *http.Request, prefillEndPoint string, encodeEndPoints []string) {
	s.logger.V(4).Info("running EPD protocol", "prefiller", prefillEndPoint, "encoderCount", len(encodeEndPoints))

	_, completionRequest, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	// Generate unique request UUID
	reqUUID, err := uuid.NewUUID()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	requestID := reqUUID.String()

	// Step 1: Process through Encoder cluster (if has MM input)
	if len(encodeEndPoints) > 0 {
		if err := s.fanoutEncoderPrimer(completionRequest, encodeEndPoints, requestID); err != nil {
			s.logger.Error(err, "encoder processing failed", "requestID", requestID)
			if err := errorBadGateway(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}
	}

	// Step 2 & 3: Handle Prefiller and Decoder stages
	// Set cache_hit_threshold to 0 to skip the decode-first optimization
	// since we've already processed through the encoder
	completionRequest[requestFieldCacheHitThreshold] = 0

	// Update request body with the modified completion request
	modifiedBody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Clone request with modified body and add request ID header
	pdRequest := cloneRequestWithBody(r.Context(), r, modifiedBody)
	pdRequest.Header.Add(requestHeaderRequestID, requestID)

	// If prefiller is configured, use P/D protocol; otherwise go directly to decoder
	if len(prefillEndPoint) > 0 {
		s.logger.V(4).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		// Run the configured P/D protocol (prefill + decode)
		// This will use whichever protocol is configured: shared-storage, nixlv2, or sglang
		s.handlePDConnector(w, pdRequest, prefillEndPoint, APITypeChatCompletions)
	} else {
		s.logger.V(4).Info("no prefiller configured, going directly to decoder after encoder")
		// No prefiller, go directly to decoder (Encoder-Decoder mode)
		if !s.forwardDataParallel || !s.dataParallelHandler(w, pdRequest) {
			s.decoderProxy.ServeHTTP(w, pdRequest)
		}
	}
}
