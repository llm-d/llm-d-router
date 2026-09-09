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
	"context"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// fanoutEncoderPrimer sends concurrent encoder requests for each multimodal
// item, discarding the responses (status check only). Used by the
// `ec-example` connector to prime the encoder cache before forwarding the
// original request to the P/D connector. The returned item count is 0 when
// there was no multimodal input to encode, so callers can tell a genuine
// no-op apart from an encoder invocation.
func (s *Server) fanoutEncoderPrimer(ctx context.Context, originalRequest map[string]any, encoderHostPorts []string, requestID string) (int, error) {
	items := s.mmItemsForFanout(originalRequest, requestID)
	if len(items) == 0 {
		s.logger.V(logging.DEBUG).Info("no multimodal items, skipping encoder", "requestID", requestID)
		return 0, nil
	}
	return len(items), s.fanoutEncoder(ctx, originalRequest, items, encoderHostPorts, requestID, nil)
}

// handleECSharedStorage handles an Encoder-Prefiller-Decoder disaggregation request
func (s *Server) handleECSharedStorage(w http.ResponseWriter, r *http.Request, prefillEndPoint string, encodeEndPoints []string) {
	s.logger.V(logging.DEBUG).Info("running EPD protocol", "prefiller", prefillEndPoint, "encoderCount", len(encodeEndPoints))

	_, body, ok := s.readJSONBody(r, w)
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
		encodeStart := time.Now()
		total, err := s.fanoutEncoderPrimer(r.Context(), body, encodeEndPoints, requestID)
		if err != nil {
			// fanoutEncoderPrimer only errors once it has actually dispatched
			// to an encoder, so the duration sample is meaningful here too.
			metrics.RecordEncodeDuration(time.Since(encodeStart))
			metrics.RecordError(metrics.StageEncode)
			s.logger.Error(err, "encoder processing failed", "requestID", requestID)
			if err := errorBadGateway(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}
		if total > 0 {
			// total == 0 means there was no multimodal input to encode, so no
			// encoder was actually invoked; skip the sample rather than skew
			// the histogram with a sub-microsecond no-op.
			metrics.RecordEncodeDuration(time.Since(encodeStart))
		}
	}

	s.runPDPipeline(w, r, body, prefillEndPoint, requestID)
}
