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
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/google/uuid"
)

// truncateLongStrings returns a copy of v with any string longer than
// maxLen replaced by its prefix + a length suffix. The connector treats
// ec_transfer_params as opaque, so this walks the structure generically
// rather than special-casing any field name. The original value is left
// unmodified.
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

// fanoutEncoderCollect fans out per-image encoder requests like the primer
// does and merges each encoder's ec_transfer_params response into a single
// flat map keyed by the per-image mm_hash. Returns the merged map, the
// count of items whose response actually carried ec_transfer_params
// (`contributed`), and the total item count.
//
// An encoder response missing ec_transfer_params (or carrying a non-object
// value) is warn-and-skipped — it does not fail the request (mirrors
// NIXLv2's tolerance for missing kv_transfer_params). Encoder transport
// errors, non-2xx responses, and unparsable JSON are hard-fail — same
// shape as fanoutEncoderPrimer.
//
// The fan-out scaffolding (URL dedup, parallel goroutines, encoder handler
// setup, request/response plumbing) lives in fanoutEncoder; this function
// supplies only the per-response perItem callback that does the
// ec_transfer_params extraction and merge.
func (s *Server) fanoutEncoderCollect(
	originalRequest map[string]any,
	encoderHostPorts []string,
	requestID string,
) (map[string]any, int, int, error) {
	items := s.mmItemsForFanout(originalRequest, requestID)
	if len(items) == 0 {
		s.logger.V(4).Info("no multimodal items, skipping encoder", "requestID", requestID)
		return nil, 0, 0, nil
	}

	var (
		params      = make(map[string]any)
		paramsMu    sync.Mutex
		contributed int
	)
	err := s.fanoutEncoder(originalRequest, items, encoderHostPorts, requestID, func(idx int, pw *bufferedResponseWriter) error {
		var encoderResponse map[string]any
		if err := json.Unmarshal(pw.bodyBytes(), &encoderResponse); err != nil {
			return fmt.Errorf("failed to parse encoder response for item %d: %w", idx, err)
		}
		s.logger.Info("encoder response",
			"item", idx,
			"requestID", requestID,
			requestFieldECTransferParams, truncateLongStrings(encoderResponse[requestFieldECTransferParams], 64))
		ec, ok := encoderResponse[requestFieldECTransferParams]
		if !ok || ec == nil {
			s.logger.Info("warning: missing ec_transfer_params field in encoder response",
				"item", idx, "requestID", requestID)
			return nil
		}
		ecMap, ok := ec.(map[string]any)
		if !ok {
			s.logger.Info("warning: ec_transfer_params is not a JSON object; skipping",
				"item", idx, "requestID", requestID, "type", fmt.Sprintf("%T", ec))
			return nil
		}
		if len(ecMap) == 0 {
			s.logger.Info("warning: ec_transfer_params is empty",
				"item", idx, "requestID", requestID)
			return nil
		}
		paramsMu.Lock()
		defer paramsMu.Unlock()
		for k, v := range ecMap {
			if _, exists := params[k]; exists {
				s.logger.Info("warning: duplicate ec_transfer_params key across encoder responses; overwriting",
					"item", idx, "requestID", requestID, "key", k)
			}
			params[k] = v
		}
		contributed++
		return nil
	})
	if err != nil {
		return nil, 0, 0, err
	}
	return params, contributed, len(items), nil
}

// handleECEPD implements the "ec-epd" connector: fans out per-image encoder
// requests, captures each encoder's ec_transfer_params response, aggregates
// them into the prefill request body's ec_transfer_params.
func (s *Server) handleECEPD(w http.ResponseWriter, r *http.Request, prefillEndPoint string, encodeEndPoints []string) {
	s.logger.V(4).Info("running EC-EPD protocol", "prefiller", prefillEndPoint, "encoderCount", len(encodeEndPoints))

	_, completionRequest, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	reqUUID, err := uuid.NewUUID()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	requestID := reqUUID.String()

	// Step 1: fan out to encoders, collect per-image ec_transfer_params.
	if len(encodeEndPoints) > 0 {
		params, contributed, total, err := s.fanoutEncoderCollect(completionRequest, encodeEndPoints, requestID)
		if err != nil {
			s.logger.Error(err, "encoder processing failed", "requestID", requestID)
			if err := errorBadGateway(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}
		if total > 0 {
			// If every encoder response lacked ec_transfer_params, the
			// connector silently degrades to primer-mode behaviour
			// (forward unchanged). Log a single summary warning so the
			// operator sees the regression even when each per-item
			// missing-field warning gets lost in the noise.
			if contributed == 0 {
				s.logger.Info("warning: no encoder response carried ec_transfer_params; forwarding prefill request without it",
					"requestID", requestID, "items", total)
			} else {
				completionRequest[requestFieldECTransferParams] = params
				if contributed < total {
					s.logger.Info("warning: ec_transfer_params partially populated; some items missing transfer metadata",
						"requestID", requestID, "contributed", contributed, "items", total)
				}
			}
		}
	}

	// Step 2 & 3: Handle Prefiller and Decoder stages
	// Set cache_hit_threshold to 0 to skip the decode-first optimization
	// since we've already processed through the encoder
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

	// Don't log the full body — multimodal requests carry inline base64
	// image data URIs (~MB per image). Log size + the small ec metadata.
	s.logger.V(4).Info("forwarding request to prefiller",
		"requestID", requestID,
		"prefiller", prefillEndPoint,
		"bodyBytes", len(modifiedBody),
		requestFieldECTransferParams, truncateLongStrings(completionRequest[requestFieldECTransferParams], 64))

	if len(prefillEndPoint) > 0 {
		s.logger.V(4).Info("using P/D protocol after encoder", "prefiller", prefillEndPoint)
		s.handlePDConnector(w, pdRequest, prefillEndPoint, APITypeChatCompletions)
		return
	}

	s.logger.V(4).Info("no prefiller configured, going directly to decoder after encoder")
	if !s.forwardDataParallel || !s.dataParallelHandler(w, pdRequest) {
		s.decoderProxy.ServeHTTP(w, pdRequest)
	}
}
