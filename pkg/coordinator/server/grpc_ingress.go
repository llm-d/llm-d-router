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
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// grpcScanResult holds the protocol-specific values extracted by a wire scan of
// an inbound unary-request gRPC Generate call.
type grpcScanResult struct {
	req                   pipeline.GRPCRequest
	stream                bool
	logStream             bool
	model                 string
	hasModel              bool
	droppedTransferParams bool
}

type grpcScanFunc func(r *http.Request, payload []byte) (grpcScanResult, error)

type grpcRouteSpec struct {
	path   string
	route  string
	stream bool
	scan   grpcScanFunc
}

var grpcRoutes = []grpcRouteSpec{
	{
		path:   sglangapiv1.SglangService_Generate_FullMethodName,
		route:  coordmetrics.RouteGenerate,
		stream: true,
		scan:   scanSGLangGenerate,
	},
	{
		path:   vllmpb.Inference_Generate_FullMethodName,
		route:  coordmetrics.RouteGenerate,
		stream: false,
		scan:   scanVLLMGenerate,
	},
	{
		path:   vllmpb.Inference_GenerateStream_FullMethodName,
		route:  coordmetrics.RouteGenerate,
		stream: true,
		scan:   scanVLLMGenerate,
	},
}

func lookupGRPCRoute(path string) (grpcRouteSpec, bool) {
	for _, gr := range grpcRoutes {
		if gr.path == path {
			return gr, true
		}
	}
	return grpcRouteSpec{}, false
}

func (s *Server) serveGRPC(w http.ResponseWriter, r *http.Request, scan grpcScanFunc) {
	receivedAt := time.Now()
	cw := newCountingResponseWriter(w)
	route := inferenceRoute(r.URL.Path)
	stream := false
	if gr, ok := lookupGRPCRoute(r.URL.Path); ok {
		stream = gr.stream
	}

	var model string
	inflightModel := ""
	coordmetrics.IncRequestRunning(inflightModel)

	var body []byte
	var parseDuration time.Duration
	var reqCtx *pipeline.RequestContext
	defer func() {
		rec := recover()
		if rec != nil {
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
		coordmetrics.RecordResponseSize(stream, cw.BytesWritten())
		coordmetrics.DecRequestRunning(inflightModel)
		if rec != nil {
			panic(rec)
		}
	}()

	parseStart := time.Now()
	parseDone := false
	defer func() {
		if !parseDone {
			parseDuration = time.Since(parseStart)
		}
	}()

	if enc := strings.TrimSpace(r.Header.Get("grpc-encoding")); enc != "" && !strings.EqualFold(enc, "identity") {
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		grpcframe.WriteError(cw, codes.Unimplemented, grpcframe.ErrCompressed.Error())
		return
	}

	// ReadMessage sizes the body from the gRPC length prefix, so a large
	// request is read into one exact-size buffer.
	var err error
	body, err = grpcframe.ReadMessage(r.Body, s.maxRequestBodySize*config.BytesPerMB)
	if err != nil {
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		switch {
		case errors.Is(err, grpcframe.ErrTooLarge):
			grpcframe.WriteError(cw, codes.ResourceExhausted, "request body too large")
		case errors.Is(err, grpcframe.ErrCompressed):
			grpcframe.WriteError(cw, codes.Unimplemented, grpcframe.ErrCompressed.Error())
		case errors.Is(err, grpcframe.ErrMalformed):
			grpcframe.WriteError(cw, codes.InvalidArgument, "invalid gRPC frame")
		default:
			grpcframe.WriteError(cw, codes.InvalidArgument, "failed to read request body")
		}
		return
	}

	scanned, err := scan(r, body[grpcframe.HeaderLen:])
	if err != nil {
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ErrorCodeBadRequest)
		grpcframe.WriteError(cw, codes.InvalidArgument, "invalid GenerateRequest payload")
		return
	}
	parseDuration = time.Since(parseStart)
	parseDone = true

	stream = scanned.stream
	model = scanned.model
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

	headers := r.Header.Clone()
	pipeline.StripInternalHeaders(headers)

	reqCtx = &pipeline.RequestContext{
		RequestID:        requestID,
		OriginalPath:     r.URL.Path,
		OriginalHeaders:  headers,
		OriginalBody:     body,
		Model:            model,
		Stream:           stream,
		Route:            route,
		KVTransferParams: make(map[string]any),
		ResponseWriter:   cw,
		ParseDuration:    parseDuration,
		GRPC:             scanned.req,
	}
	defer reqCtx.ClosePrefillStream()

	logger := ctrl.Log.WithName("handler").WithValues(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	ctx := log.IntoContext(r.Context(), logger)
	ctx = tracing.LoggerWithSpanContext(ctx, trace.SpanFromContext(ctx))

	if requestIDReplaced && clientRequestID != "" {
		logger.V(logutil.DEFAULT).Info("replaced invalid client request ID", "rejectedLength", len(clientRequestID))
	}
	if scanned.droppedTransferParams {
		logger.V(logutil.DEFAULT).Info("dropped client-supplied transfer params from gRPC GenerateRequest")
	}

	if scanned.hasModel {
		logger.V(logutil.DEFAULT).Info("received gRPC request", "path", r.URL.Path, "model", model, "stream", scanned.logStream)
	} else {
		logger.V(logutil.DEFAULT).Info("received gRPC request", "path", r.URL.Path, "stream", scanned.logStream)
	}

	if err := s.pipeline.Execute(ctx, reqCtx); err != nil {
		if pipeline.IsClientCancellation(ctx, err) {
			logger.V(logutil.VERBOSE).Info("pipeline aborted: client cancelled", "error", err)
			return
		}
		logger.Error(err, "pipeline execution failed")
		coordmetrics.IncRequestErrorTotal(model, coordmetrics.ClassifyErrorCode(err, pipeline.ClassifyOpts))
		var streamed *pipeline.UpstreamStreamedError
		var written *pipeline.ResponseWrittenError
		if errors.As(err, &streamed) || errors.As(err, &written) {
			return
		}
		code, msg := classifyGRPCPipelineError(err, reqCtx.RequestID)
		grpcframe.WriteError(cw, code, msg)
	}
}

func classifyGRPCPipelineError(err error, requestID string) (codes.Code, string) {
	if errors.Is(err, pipeline.ErrBadRequest) {
		return codes.InvalidArgument, fmt.Sprintf("bad request (request_id: %s)", requestID)
	}
	var upstream *pipeline.UpstreamError
	if errors.As(err, &upstream) && upstream.StatusCode > 0 {
		if upstream.StatusCode >= http.StatusBadRequest && upstream.StatusCode < http.StatusInternalServerError {
			return pipeline.HTTPStatusToGRPCCode(upstream.StatusCode),
				fmt.Sprintf("%s rejected the request: HTTP %d (request_id: %s)", upstream.Step, upstream.StatusCode, requestID)
		}
		return pipeline.HTTPStatusToGRPCCode(upstream.StatusCode),
			reqcommon.InternalErrorMessage(requestID)
	}
	if errors.As(err, &upstream) {
		return codes.Unavailable, reqcommon.InternalErrorMessage(requestID)
	}
	return codes.Internal, reqcommon.InternalErrorMessage(requestID)
}
