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

package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"time"

	"github.com/google/uuid"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// validRequestID bounds a client-supplied x-request-id to alphanumerics and
// dashes, at most 128 characters. handleInference replaces a header that fails
// the match with a generated UUID before it reaches error responses; the
// request-logging middleware redacts it. This keeps attacker-controlled content
// out of both surfaces.
var validRequestID = regexp.MustCompile(`^[a-zA-Z0-9\-]{1,128}$`)

func (s *Server) handleInference(w http.ResponseWriter, r *http.Request) {
	receivedAt := time.Now()
	cw := newCountingResponseWriter(w)
	route := inferenceRoute(r.URL.Path)
	// model is the raw name from the request body, or "" when the client
	// omitted it. Metric emitters normalize "" to ModelUnknown via
	// boundModel; the pipeline sees the raw value so upstream requests
	// carry what the client actually sent.
	var model string
	// inflightModel tracks the request_running label under which this
	// request currently counts. The gauge is incremented at entry (before
	// body read) so slow uploads and 413/parse rejections are visible in
	// "requests being processed"; after JSON parse the label is swapped
	// from "unknown" to the parsed model when they differ.
	inflightModel := ""
	coordmetrics.IncRequestRunning(inflightModel)
	// body is declared before the defer so RecordRequestSize can observe its
	// final length even when an early validation branch returns. Read
	// failures land here with the partial length; 413/invalid-JSON/null
	// branches land here with the full length. Success lands here with the
	// parsed model.
	var body []byte
	var parseDuration time.Duration
	var stream bool
	var reqCtx *pipeline.RequestContext
	defer func() {
		// A pipeline panic bypasses the err-branch call to IncRequestErrorTotal
		// below; recovering here records the panic under error_code=internal
		// and re-panics so chi Recoverer at the server edge still answers 500.
		r := recover()
		if r != nil {
			coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeInternal)
		}
		coordmetrics.IncRequestTotal(model)
		coordmetrics.RecordRequestDuration(model, time.Since(receivedAt))
		coordmetrics.RecordRequestSize(model, len(body))
		stepDuration := time.Duration(0)
		if reqCtx != nil {
			stepDuration = reqCtx.StepDuration
		}
		coordmetrics.RecordOrchestrationOverhead(route, time.Since(receivedAt)-parseDuration-stepDuration)
		coordmetrics.RecordResponseBytes(stream, cw.BytesWritten())
		coordmetrics.DecRequestRunning(inflightModel)
		if r != nil {
			panic(r)
		}
	}()

	parseStart := time.Now()
	var err error
	body, err = io.ReadAll(io.LimitReader(r.Body, s.maxRequestBodySize*config.BytesPerMB+1))
	parseDuration = time.Since(parseStart)
	if err != nil {
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		http.Error(cw, "failed to read request body", http.StatusBadRequest)
		return
	}
	if int64(len(body)) > s.maxRequestBodySize*config.BytesPerMB {
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		http.Error(cw, "request body too large", http.StatusRequestEntityTooLarge)
		return
	}

	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		parseDuration = time.Since(parseStart)
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		http.Error(cw, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if parsed == nil {
		parseDuration = time.Since(parseStart)
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		http.Error(cw, "invalid JSON body", http.StatusBadRequest)
		return
	}
	parseDuration = time.Since(parseStart)

	stream, _ = parsed[reqcommon.FieldStream].(bool)
	if m, ok := parsed["model"].(string); ok && m != "" {
		model = m
	}
	if model != inflightModel {
		coordmetrics.DecRequestRunning(inflightModel)
		coordmetrics.IncRequestRunning(model)
		inflightModel = model
	}

	requestID := r.Header.Get(reqcommon.RequestIDHeaderKey)
	clientRequestID := requestID
	requestIDReplaced := !validRequestID.MatchString(requestID)
	if requestIDReplaced {
		requestID = uuid.New().String()
	}

	reqCtx = &pipeline.RequestContext{
		RequestID:        requestID,
		OriginalPath:     r.URL.Path,
		OriginalHeaders:  r.Header.Clone(),
		OriginalBody:     body,
		Body:             parsed,
		Model:            model,
		Stream:           stream,
		KVTransferParams: make(map[string]any),
		ResponseWriter:   cw,
		ParseDuration:    parseDuration,
	}

	logger := ctrl.Log.WithName("handler").WithValues(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	ctx := log.IntoContext(r.Context(), logger)

	if requestIDReplaced && clientRequestID != "" {
		// Log the rejected length, never the raw value, to avoid reflecting
		// attacker-controlled content into the log.
		logger.V(logutil.DEFAULT).Info("replaced invalid client request ID", "rejectedLength", len(clientRequestID))
	}

	logger.V(logutil.DEFAULT).Info("received request", "path", r.URL.Path, "model", model, "stream", stream)

	if stream {
		// A streaming completion can outlast the server WriteTimeout, which
		// would cut the response mid-stream with a TCP reset and no app-layer
		// error. Clear the write deadline for streaming only; non-streaming
		// requests keep it as a slow-client guard.
		if err := http.NewResponseController(w).SetWriteDeadline(time.Time{}); err != nil && !errors.Is(err, http.ErrNotSupported) {
			logger.V(logutil.DEFAULT).Info("could not clear write deadline for streaming response", "error", err)
		}
	}

	if err := s.pipeline.Execute(ctx, reqCtx); err != nil {
		logger.Error(err, "pipeline execution failed")
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ClassifyErrorCode(err, pipeline.ClassifyOpts))
		var streamed *pipeline.UpstreamStreamedError
		if errors.As(err, &streamed) {
			// Response bytes were already streamed to the client (or 502
			// written by the reverse proxy's ErrorHandler); do not overwrite.
			return
		}
		status, msg := classifyPipelineError(err, reqCtx.RequestID)
		http.Error(cw, msg, status)
	}
}

// inferenceRoute maps an inbound path onto the bounded route label used by
// orchestration_overhead_seconds.
func inferenceRoute(path string) string {
	switch path {
	case gateway.PathChatCompletions:
		return coordmetrics.RouteChatCompletions
	case gateway.PathCompletions:
		return coordmetrics.RouteCompletions
	case gateway.DefaultGeneratePath:
		return coordmetrics.RouteGenerate
	default:
		return coordmetrics.RouteUnknown
	}
}

// classifyPipelineError maps a pipeline error to a client-facing status and
// message. Invalid client input is 400. An upstream 4xx is forwarded with its
// status (the request was the root cause); any other upstream status, and every
// other error, is a 502 gateway fault. Upstream response bodies stay in the
// server log (logged by the caller) and are never sent to the client.
func classifyPipelineError(err error, requestID string) (int, string) {
	if errors.Is(err, pipeline.ErrBadRequest) {
		return http.StatusBadRequest, fmt.Sprintf("bad request (request_id: %s)", requestID)
	}
	var upstream *pipeline.UpstreamError
	if errors.As(err, &upstream) && upstream.StatusCode >= http.StatusBadRequest && upstream.StatusCode < http.StatusInternalServerError {
		return upstream.StatusCode, fmt.Sprintf("%s rejected the request: HTTP %d (request_id: %s)", upstream.Step, upstream.StatusCode, requestID)
	}
	return http.StatusBadGateway, fmt.Sprintf("internal error (request_id: %s)", requestID)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}
