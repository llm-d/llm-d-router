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
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/sync/errgroup"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
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

	s.logger.Info("executing inference for preprocessed request items", "count", len(inferenceRequests), "requestID", requestID, "inferenceHostPorts", inferenceHostPorts)

	tracer := tracing.Tracer(tracerScope)

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
			// Each goroutine gets its own child span so timing and target are
			// recorded independently. gctx carries the parent inference span.
			spanCtx, itemSpan := tracer.Start(gctx, "inference_item",
				trace.WithSpanKind(trace.SpanKindInternal),
			)
			itemSpan.SetAttributes(
				attribute.Int("llm_d.io_disagg_proxy.inference.item_index", idx),
				attribute.String("llm_d.io_disagg_proxy.inference.request_id", infReqID),
				attribute.String("llm_d.io_disagg_proxy.inference.target", hostPort),
			)
			itemStart := time.Now()
			defer itemSpan.End()

			body, err := json.Marshal(inferenceRequest)
			if err != nil {
				err = fmt.Errorf("failed to marshal inference request for item %d: %w", idx, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				itemSpan.SetStatus(codes.Error, err.Error())
				return err
			}

			inferenceHandler, err := s.inferenceProxyHandler(hostPort)
			if err != nil {
				err = fmt.Errorf("failed to get inference proxy handler for %s: %w", hostPort, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				itemSpan.SetStatus(codes.Error, err.Error())
				return err
			}

			req, err := http.NewRequestWithContext(spanCtx, "POST", InferencePath, bytes.NewReader(body))
			if err != nil {
				err = fmt.Errorf("failed to create inference request for item %d: %w", idx, err)
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				itemSpan.SetStatus(codes.Error, err.Error())
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(requestHeaderRequestID, infReqID)

			s.logger.Info("sending inference request", "inference requestID", infReqID, "to", hostPort)

			pw := &bufferedResponseWriter{}
			inferenceHandler.ServeHTTP(pw, req)

			itemSpan.SetAttributes(
				attribute.Int("llm_d.io_disagg_proxy.inference.status_code", pw.statusCode),
				attribute.Float64("llm_d.io_disagg_proxy.inference.duration_ms", float64(time.Since(itemStart).Milliseconds())),
			)

			if isHTTPError(pw.statusCode) {
				err := fmt.Errorf("inference request failed for item %d with status %d: %s", idx, pw.statusCode, pw.buffer.String())
				s.logger.Error(err, "inference fanout", "inference requestID", infReqID)
				itemSpan.SetStatus(codes.Error, "inference request failed")
				return err
			}

			if perItem != nil {
				if err := perItem(idx, pw); err != nil {
					s.logger.Error(err, "inference fanout perItem", "inference requestID", infReqID)
					itemSpan.SetStatus(codes.Error, err.Error())
					return err
				}
			}

			s.logger.Info("inference request completed", "inference requestID", infReqID)
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
		if v := s.logger; v.Enabled() {
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
	s.logger.Info("running IO disaggregated protocol", "preprocessor", preprocessHostPort, "inferenceCount", len(allowedInferenceInstances))

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

	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	// Step 1: Send to the Preprocessor
	ctx, preprocessSpan := tracer.Start(ctx, "preprocess",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	preprocessSpan.SetAttributes(
		attribute.String("llm_d.io_disagg_proxy.request_id", requestID),
		attribute.String("llm_d.io_disagg_proxy.preprocess_target", preprocessHostPort),
	)
	preprocessStart := time.Now()

	preprocessHandler, err := s.preprocessProxyHandler(preprocessHostPort)
	if err != nil {
		s.logger.Error(err, "failed to create preprocess handler", "requestID", requestID)
		preprocessSpan.SetStatus(codes.Error, "failed to create preprocess handler")
		preprocessSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	preprocessReq, err := http.NewRequestWithContext(ctx, "POST", PreprocessPath, bytes.NewReader(rawBody))
	if err != nil {
		err = fmt.Errorf("failed to create preprocess request: %w", err)
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		preprocessSpan.SetStatus(codes.Error, err.Error())
		preprocessSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	preprocessReq.Header.Set("Content-Type", "application/json")
	preprocessReq.Header.Set(requestHeaderRequestID, requestID)

	s.logger.Info("executing preprocessing", "requestID", requestID, "preprocessHostPort", preprocessHostPort)
	pw := &bufferedResponseWriter{}
	preprocessHandler.ServeHTTP(pw, preprocessReq)

	preprocessSpan.SetAttributes(
		attribute.Int("llm_d.io_disagg_proxy.preprocess.status_code", pw.statusCode),
		attribute.Float64("llm_d.io_disagg_proxy.preprocess.duration_ms", float64(time.Since(preprocessStart).Milliseconds())),
	)

	if isHTTPError(pw.statusCode) {
		err := fmt.Errorf("preprocess request failed with status %d: %s", pw.statusCode, pw.buffer.String())
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		preprocessSpan.SetStatus(codes.Error, "preprocess request failed")
		preprocessSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	var preprocessResponse []map[string]any
	if err := json.Unmarshal(pw.bodyBytes(), &preprocessResponse); err != nil {
		err = fmt.Errorf("failed to parse preprocess response: %w", err)
		s.logger.Error(err, "preprocess failed", "requestID", requestID)
		preprocessSpan.SetStatus(codes.Error, "failed to parse preprocess response")
		preprocessSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	preprocessSpan.End()

	// Step 2: Process through Inference cluster
	ctx, inferenceSpan := tracer.Start(ctx, "inference",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	inferenceSpan.SetAttributes(
		attribute.String("llm_d.io_disagg_proxy.request_id", requestID),
		attribute.Int("llm_d.io_disagg_proxy.inference.item_count", len(preprocessResponse)),
		attribute.Int("llm_d.io_disagg_proxy.inference.host_count", len(allowedInferenceInstances)),
	)
	inferenceStart := time.Now()

	inferenceResponses, err := s.fanoutInferenceCollect(
		ctx,
		preprocessResponse,
		allowedInferenceInstances,
		requestID,
	)

	inferenceSpan.SetAttributes(
		attribute.Float64("llm_d.io_disagg_proxy.inference.duration_ms", float64(time.Since(inferenceStart).Milliseconds())),
	)

	if err != nil {
		s.logger.Error(err, "inference failed", "requestID", requestID)
		inferenceSpan.SetStatus(codes.Error, "inference fanout failed")
		inferenceSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	inferenceSpan.End()

	// Step 3: Send to the Postprocessor
	_, postprocessSpan := tracer.Start(ctx, "postprocess",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer postprocessSpan.End()

	postprocessSpan.SetAttributes(
		attribute.String("llm_d.io_disagg_proxy.request_id", requestID),
		attribute.Int("llm_d.io_disagg_proxy.postprocess.response_count", len(inferenceResponses)),
	)
	postprocessStart := time.Now()

	postProcessPoolingRequest := map[string]any{
		"request_id": fmt.Sprintf("pooling-%s", requestID),
		"model":      model,
		"responses":  inferenceResponses,
	}

	body, err := json.Marshal(postProcessPoolingRequest)
	if err != nil {
		err = fmt.Errorf("failed to marshal postprocess request: %w", err)
		s.logger.Error(err, "post process failed", "requestID", requestID)
		postprocessSpan.SetStatus(codes.Error, err.Error())
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", PostprocessPath, bytes.NewReader(body))
	if err != nil {
		err = fmt.Errorf("failed to create postprocess request%w", err)
		s.logger.Error(err, "post process failed", "requestID", requestID)
		postprocessSpan.SetStatus(codes.Error, err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(requestHeaderRequestID, fmt.Sprintf("pooling-%s", requestID))

	// Decoder proxy in this case is the postprocessor instance. It sends the request to the vLLM server (localhost:vLLMPort)
	s.logger.Info("executing postprocessing", "requestID", requestID)
	s.decoderProxy.ServeHTTP(w, req)

	postprocessSpan.SetAttributes(
		attribute.Float64("llm_d.io_disagg_proxy.postprocess.duration_ms", float64(time.Since(postprocessStart).Milliseconds())),
	)
}
