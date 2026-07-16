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
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"golang.org/x/sync/errgroup"
)

// fanoutInference fans out one inference request per item, in parallel, with
// round-robin over inferenceHostPorts. perItem is invoked once per item AFTER
// the inference has returned a 2xx response; it receives the item's
// positional index (post-dedup) and the buffered inference response. The
// callback may return an error to fail the whole fan-out, or nil to
// accept. perItem may be nil for fire-and-forget primer-style usage.
//
// The first goroutine to fail cancels ctx so sibling inference requests are
// aborted at the transport layer. Every failure is logged before propagating;
// grp.Wait returns the first non-nil error.
func (s *Server) fanoutInference(
	ctx context.Context,
	inferenceRequests []map[string]any,
	inferenceHostPorts []string,
	requestID string,
	perItem func(idx int, pw *bufferedResponseWriter) error,
) error {
	if len(inferenceHostPorts) == 0 {
		return fmt.Errorf("fanoutInfernce: no inference hostPorts provided (requestID=%s)", requestID)
	}

	s.logger.V(logging.DEBUG).Info("executing inference for preprocessed request items", "count", len(inferenceRequests), "requestID", requestID, "inferenceHostPorts", inferenceHostPorts)

	grp, gctx := errgroup.WithContext(ctx)
	for idx, inferenceRequest := range inferenceRequests {
		value, exists := inferenceRequest["request_id"]
		if !exists {
			err := fmt.Errorf("failed to get request_id for item %d", idx)
			s.logger.Error(err, "inference fanout")
			return err
		}

		infReqID, ok := value.(string)
		if !ok {
			err := fmt.Errorf("request_id is not a string for item %d: %v", idx, value)
			s.logger.Error(err, "inference fanout")
			return err
		}

		hostPort := inferenceHostPorts[idx%len(inferenceHostPorts)]
		grp.Go(func() error {
			body, err := json.Marshal(inferenceRequest)
			if err != nil {
				err = fmt.Errorf("failed to marshal inference request for item %d: %w", idx, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				return err
			}

			inferenceHandler, err := s.inferenceProxyHandler(hostPort)
			if err != nil {
				err = fmt.Errorf("failed to get inference proxy handler for %s: %w", hostPort, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				return err
			}

			req, err := http.NewRequestWithContext(gctx, "POST", InferencePath, bytes.NewReader(body))
			if err != nil {
				err = fmt.Errorf("failed to create inference request for item %d: %w", idx, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(requestHeaderRequestID, infReqID)

			s.logger.V(logging.DEBUG).Info("sending inference request", "inference requestID", infReqID, "to", hostPort)

			pw := &bufferedResponseWriter{}
			inferenceHandler.ServeHTTP(pw, req)

			if isHTTPError(pw.statusCode) {
				err := fmt.Errorf("inference request failed for item %d with status %d: %s", idx, pw.statusCode, pw.buffer.String())
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				return err
			}

			if perItem != nil {
				if err := perItem(idx, pw); err != nil {
					s.logger.Error(err, "inference fanout perItem", "inference requestID", infReqID)
					return err
				}
			}

			s.logger.V(logging.DEBUG).Info("inference request completed", "inference requestID", infReqID)
			return nil
		})
	}
	return grp.Wait()
}

// fanoutInferenceCollect fans out inference requests and returns responses.
func (s *Server) fanoutInferenceCollect(
	ctx context.Context,
	inferenceRequests []map[string]any,
	encoderHostPorts []string,
	requestID string,
) ([]map[string]any, error) {
	responses := make([]map[string]any, len(inferenceRequests))
	err := s.fanoutInference(ctx, inferenceRequests, encoderHostPorts, requestID, func(idx int, pw *bufferedResponseWriter) error {
		var inferenceResponse map[string]any
		if err := json.Unmarshal(pw.bodyBytes(), &inferenceResponse); err != nil {
			return fmt.Errorf("failed to parse inference response for item %d: %w", idx, err)
		}
		if v := s.logger.V(logging.DEBUG); v.Enabled() {
			v.Info("inference response",
				"item", idx,
				"requestID", requestID,
			)
		}
		responses[idx] = inferenceResponse
		return nil
	})
	if err != nil {
		return nil, err
	}
	return responses, nil
}

func (s *Server) handleIOCommon(w http.ResponseWriter, r *http.Request, preprocessHostPort string, allowedInferenceInstances []string) {
	s.logger.V(logging.DEBUG).Info("running IO disaggregated protocol", "preprocessor", preprocessHostPort, "inferenceCount", len(allowedInferenceInstances))

	rawBody, ioProcessorRequest, ok := s.readJSONBody(r, w)
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

	model := ioProcessorRequest["model"]

	// Step 1: Send to the Preprocessor
	preprocessHandler, err := s.preprocessProxyHandler(preprocessHostPort)
	if err != nil {
		s.logger.Error(err, "failed to create preprocess handler", "requestID", requestID)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	preprocessReq, err := http.NewRequestWithContext(r.Context(), "POST", PreprocessPath, bytes.NewReader(rawBody))
	if err != nil {
		err = fmt.Errorf("failed to create preprocess request: %w", err)
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	preprocessReq.Header.Set("Content-Type", "application/json")
	preprocessReq.Header.Set(requestHeaderRequestID, requestID)

	s.logger.V(logging.DEBUG).Info("executing preprocessing", "requestID", requestID, "preprocessHostPort", preprocessHostPort)
	pw := &bufferedResponseWriter{}
	preprocessHandler.ServeHTTP(pw, preprocessReq)

	if isHTTPError(pw.statusCode) {
		err := fmt.Errorf("preprocess request failed with status %d: %s", pw.statusCode, pw.buffer.String())
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	var preprocessResponse []map[string]any
	if err := json.Unmarshal(pw.bodyBytes(), &preprocessResponse); err != nil {
		err = fmt.Errorf("failed to parse preprocess response: %w", err)
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Step 2: Process through Inference cluster
	inferenceResponses, err := s.fanoutInferenceCollect(
		r.Context(),
		preprocessResponse,
		allowedInferenceInstances,
		requestID,
	)
	if err != nil {
		s.logger.Error(err, "inference failed", "requestID", requestID)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Step 3: Send to the Postprocessor
	postProcessPoolingRequest := map[string]any{
		"request_id": fmt.Sprintf("pooling-%s", requestID),
		"model":      model,
		"responses":  inferenceResponses,
	}

	body, err := json.Marshal(postProcessPoolingRequest)
	if err != nil {
		err = fmt.Errorf("failed to marshal postprocess request: %w", err)
		s.logger.Error(err, "post process failed", "requestID", requestID)
		return
	}

	req, err := http.NewRequestWithContext(r.Context(), "POST", PostprocessPath, bytes.NewReader(body))
	if err != nil {
		err = fmt.Errorf("failed to create postprocess request%w", err)
		s.logger.Error(err, "post process failed", "requestID", requestID)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(requestHeaderRequestID, fmt.Sprintf("pooling-%s", requestID))

	// Decoder proxy in this case is the postprocessor instance. It sends the request to the vLLM server (localhost:vLLMPort)
	s.logger.V(logging.DEBUG).Info("executing postprocessing", "requestID", requestID)
	s.decoderProxy.ServeHTTP(w, req)
}
