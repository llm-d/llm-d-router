package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// fanoutEncoderCollect fans out per-image encoder requests and merges
// each response's ec_transfer_params into a flat hash-keyed map. Returns
// the merged map, the count of items that contributed metadata, and the
// total item count.
//
// Missing/non-object/empty ec_transfer_params is warn-and-skip.
func (s *Server) fanoutEncoderCollect(
	ctx context.Context,
	originalRequest map[string]any,
	encoderHostPorts []string,
	requestID string,
) (map[string]any, int, int, error) {
	items := s.mmItemsForFanout(originalRequest, requestID)
	if len(items) == 0 {
		s.logger.V(logging.DEBUG).Info("no multimodal items, skipping encoder", "requestID", requestID)
		return nil, 0, 0, nil
	}

	var (
		params      = make(map[string]any)
		paramsMu    sync.Mutex
		contributed int
	)
	err := s.fanoutEncoder(ctx, originalRequest, items, encoderHostPorts, requestID, func(idx int, pw *bufferedResponseWriter) error {
		var encoderResponse map[string]any
		if err := json.Unmarshal(pw.bodyBytes(), &encoderResponse); err != nil {
			return fmt.Errorf("failed to parse encoder response for item %d: %w", idx, err)
		}
		if v := s.logger.V(logging.DEBUG); v.Enabled() {
			v.Info("encoder response",
				"item", idx,
				"requestID", requestID,
				requestFieldECTransferParams, truncateLongStrings(encoderResponse[requestFieldECTransferParams], 64))
		}
		ec, ok := encoderResponse[requestFieldECTransferParams]
		if !ok || ec == nil {
			s.logger.V(logging.DEBUG).Info("missing ec_transfer_params field in encoder response",
				"item", idx, "requestID", requestID)
			return nil
		}
		ecMap, ok := ec.(map[string]any)
		if !ok {
			s.logger.V(logging.DEBUG).Info("ec_transfer_params is not a JSON object; skipping",
				"item", idx, "requestID", requestID, "type", fmt.Sprintf("%T", ec))
			return nil
		}
		if len(ecMap) == 0 {
			s.logger.V(logging.DEBUG).Info("ec_transfer_params is empty",
				"item", idx, "requestID", requestID)
			return nil
		}
		paramsMu.Lock()
		defer paramsMu.Unlock()
		for k, v := range ecMap {
			if _, exists := params[k]; exists {
				s.logger.V(logging.DEBUG).Info("duplicate ec_transfer_params key across encoder responses; overwriting",
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

// handleECNIXL fans out per-image encoder requests, aggregates each
// response's ec_transfer_params into the prefill request body, and hands
// off to the configured P/D connector.
func (s *Server) handleECNIXL(w http.ResponseWriter, r *http.Request, prefillEndPoint string, encodeEndPoints []string) {
	s.logger.V(logging.DEBUG).Info("running EC-NIXL protocol", "prefiller", prefillEndPoint, "encoderCount", len(encodeEndPoints))

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
		encodeStart := time.Now()
		params, contributed, total, err := s.fanoutEncoderCollect(r.Context(), completionRequest, encodeEndPoints, requestID)
		if err != nil {
			metrics.RecordError(s.config.ECConnector, metrics.StageEncode)
			s.logger.Error(err, "encoder processing failed", "requestID", requestID)
			if err := errorBadGateway(err, w); err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
			return
		}
		metrics.RecordEncodeDuration(s.config.ECConnector, time.Since(encodeStart))
		if total > 0 {
			// All-missing degrades silently to primer-mode; warn so the
			// operator sees the regression.
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

	s.runPDPipeline(w, r, completionRequest, prefillEndPoint, requestID)
}
