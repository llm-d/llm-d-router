/*
Copyright 2025 The Kubernetes Authors.

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

package handlers

import (
	"bytes"
	"context"
	"io"
	"strings"
	"sync"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	otelcodes "go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"strconv"

	envoy "github.com/llm-d/llm-d-router/pkg/common/envoy"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// EvictChannelLookup is an optional interface for looking up eviction channels by request ID.
// When set on the StreamingServer, the Process() loop will select on the eviction channel
// to support eviction of in-flight requests via ext_proc ImmediateResponse.
type EvictChannelLookup interface {
	Get(requestID string) chan struct{}
	GetReason(requestID string) errcommon.RequestDroppedReason
	Deregister(requestID string)
}

func NewStreamingServer(datastore Datastore, director Director, parserRegistry *ParserRegistry, maxPoolBufferSize int) *StreamingServer {
	return &StreamingServer{
		director:          director,
		datastore:         datastore,
		parserRegistry:    parserRegistry,
		maxPoolBufferSize: maxPoolBufferSize,
		bufferPool: sync.Pool{
			New: func() any {
				return new(bytes.Buffer)
			},
		},
	}
}

// SetEvictChannelLookup sets the eviction channel lookup for eviction support.
func (s *StreamingServer) SetEvictChannelLookup(lookup EvictChannelLookup) {
	s.evictionLookup = lookup
}

// SetEmitEndpointScores controls whether the per-endpoint scheduler scores are emitted in the
// request-path dynamic metadata under metadata.DestinationEndpointScoresKey. Off by default.
func (s *StreamingServer) SetEmitEndpointScores(enabled bool) {
	s.emitEndpointScores = enabled
}

type Director interface {
	HandleRequest(ctx context.Context, reqCtx *RequestContext, inferenceRequestBody *fwkrh.InferenceRequestBody) (*RequestContext, error)
	HandleResponseHeader(ctx context.Context, reqCtx *RequestContext) *RequestContext
	HandleResponseBody(ctx context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext
	GetRandomEndpoint() *fwkdl.EndpointMetadata
}

type Datastore interface {
	PoolGet() (*datalayer.EndpointPool, error)
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type StreamingServer struct {
	datastore         Datastore
	director          Director
	parserRegistry    *ParserRegistry
	evictionLookup    EvictChannelLookup // optional, set for eviction support
	bufferPool        sync.Pool
	maxPoolBufferSize int
	// emitEndpointScores enables emitting per-endpoint scheduler scores in the request-path
	// dynamic metadata. Off by default; set via SetEmitEndpointScores.
	emitEndpointScores bool
}

// RequestContext stores context information during the life time of an HTTP request.
//
// Exported fields are read and written by the request-control layers (director and
// admission control). Unexported fields are private to this package.
type RequestContext struct {
	TargetPod      *fwkdl.EndpointMetadata
	TargetEndpoint string
	// TargetEndpointScores maps endpoint address to the scheduler's score for it, covering
	// every endpoint the primary profile scored rather than only those in TargetEndpoint.
	// Nil when the primary profile ran no scorers.
	TargetEndpointScores     map[string]float64
	IncomingModelName        string
	TargetModelName          string
	ObjectiveKey             string
	Priority                 int
	RequestReceivedTimestamp time.Time
	RequestSize              int
	Usage                    fwkrh.Usage
	StreamedEvents           int
	// ResponseBodyStarted is maintained by the director for start-of-stream detection.
	ResponseBodyStarted bool
	Request             *Request
	Response            *Response
	Parser              fwkrh.Parser
	SchedulingRequest   *fwksched.InferenceRequest

	// TerminationCause is set only when the stream ends without completing; the end-of-stream
	// response record defaults an unset cause to a natural completion.
	TerminationCause fwkrc.TerminationCause

	// Lifecycle bookkeeping.
	firstTokenTimestamp        time.Time
	lastChunkReceivedTimestamp time.Time
	responseCompleteTimestamp  time.Time
	responseSize               int
	responseComplete           bool
	responseStatusCode         string
	requestRunning             bool

	// responseProcessingDuration is the EPP cost of handling the response. For a
	// streamed response it is the sum of the per-chunk handler slices, since the
	// gaps between chunks are model server generation time. For a non-streaming
	// response it is the single interval from responseHeadersReceivedAt onward,
	// during which the response is entirely in EPP's hands.
	responseProcessingDuration time.Duration
	responseHeadersReceivedAt  time.Time

	// Envoy ext_proc protocol state.
	requestState         streamRequestState
	requestDroppedReason errcommon.RequestDroppedReason
	modelServerStreaming bool

	reqHeaderResp *extProcPb.ProcessingResponse
	reqBodyResp   []*extProcPb.ProcessingResponse

	respHeaderResp  *extProcPb.ProcessingResponse
	respBodyResp    []*extProcPb.ProcessingResponse
	respTrailerResp *extProcPb.ProcessingResponse
}

type Request struct {
	Headers  map[string]string
	RawBody  []byte // This field will be updated when request body is modified (e.g. model mutation in requestBody)
	Metadata map[string]any
}
type Response struct {
	Headers         map[string]string
	DynamicMetadata *structpb.Struct
}
type streamRequestState int

const (
	requestReceived streamRequestState = iota
	headerRequestResponseComplete
	bodyRequestResponsesComplete
	responseReceived
	headerResponseResponseComplete
	bodyResponseResponsesComplete
	// requestEvicted indicates the request was evicted by flow control.
	// The state machine sends an ImmediateResponse(429) to the proxy.
	requestEvicted
	// requestResponseProcessingSkipped indicates that EPP response-phase stream interception was skipped for this request.
	// The state machine sends a RequestHeadersResponse and RequestBodyResponse with the routing decision
	// from the scheduling director to the proxy, and then gracefully closes the stream to stop further external processing.
	requestResponseProcessingSkipped
)

// recvResult holds the result of a srv.Recv() call from the reader goroutine.
type recvResult struct {
	req *extProcPb.ProcessingRequest
	err error
}

func (s *StreamingServer) getOrResolveParser(ctx context.Context, reqCtx *RequestContext) (fwkrh.Parser, error) {
	if reqCtx.Parser != nil {
		return reqCtx.Parser, nil
	}

	logger := log.FromContext(ctx)
	var headers map[string]string
	if reqCtx.Request != nil {
		headers = reqCtx.Request.Headers
	}
	path := fwkrequest.GetRequestPath(headers)
	parser, err := s.parserRegistry.Resolve(path)
	if err != nil {
		logger.Error(err, "Error resolving parser for path", "path", path)
		return nil, err
	}

	reqCtx.Parser = parser
	return parser, nil
}

// extractTraceContext returns ctx augmented with the upstream trace context
// carried in the incoming Envoy request headers (e.g. the traceparent set by the
// client or the Gateway), using the globally configured text map propagator.
//
// The header wire format is the W3C Trace Context spec:
// https://www.w3.org/TR/trace-context/
// Extraction uses OpenTelemetry context propagation:
// https://opentelemetry.io/docs/concepts/context-propagation/
func extractTraceContext(ctx context.Context, req *extProcPb.ProcessingRequest_RequestHeaders) context.Context {
	carrier := make(propagation.MapCarrier)
	if req != nil && req.RequestHeaders != nil && req.RequestHeaders.Headers != nil {
		for _, header := range req.RequestHeaders.Headers.Headers {
			carrier[strings.ToLower(header.Key)] = envoy.GetHeaderValue(header)
		}
	}
	return otel.GetTextMapPropagator().Extract(ctx, carrier)
}

// terminationCause classifies a stream that ended without completing. ctxErr is the request
// context's error, which is non-nil once Envoy has torn the stream down under the EPP.
func terminationCause(reqCtx *RequestContext, ctxErr error) fwkrc.TerminationCause {
	switch {
	case reqCtx.requestState == requestEvicted:
		return fwkrc.TerminationCauseEvicted
	case ctxErr != nil:
		return fwkrc.TerminationCauseClientDisconnect
	default:
		return fwkrc.TerminationCauseError
	}
}

func extractFairnessAndPriority(reqCtx *RequestContext) (string, string) {
	if reqCtx == nil {
		return metadata.DefaultFairnessID, "0"
	}
	fairnessID := metadata.DefaultFairnessID
	if reqCtx.SchedulingRequest != nil && reqCtx.SchedulingRequest.FairnessID != "" {
		fairnessID = reqCtx.SchedulingRequest.FairnessID
	}
	priority := strconv.Itoa(reqCtx.Priority)
	return fairnessID, priority
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()

	// Start tracing span for the request
	tracer := tracing.Tracer("llm-d-router/pkg/epp/handlers")
	// The server span is started in the RequestHeaders branch, once the upstream
	// trace context carried in the incoming headers is available, so the EPP span
	// joins the caller's trace instead of starting a disconnected root.
	var span trace.Span
	defer func() {
		if span != nil {
			span.End()
		}
	}()

	logger := log.FromContext(ctx)
	loggerTrace := logger.V(logutil.TRACE)
	loggerTrace.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{
		requestState: requestReceived,
		Request: &Request{
			Headers:  make(map[string]string),
			Metadata: make(map[string]any),
		},
		Response: &Response{
			Headers: make(map[string]string),
		},
	}

	// Request-phase failures (parser resolution, body parsing, admission
	// rejection) leave the switch before the success path, so both call this.
	// Flow-control rejections carry the queue wait and are the slowest samples;
	// dropping them would bias the histogram low.
	recordRequestProcessing := sync.OnceFunc(func() {
		metrics.RecordRequestProcessingLatency(time.Since(reqCtx.RequestReceivedTimestamp))
	})

	// Record EPP response processing latency once when the stream ends. Using a
	// defer (rather than emitting on end-of-stream) ensures aborted streams
	// (client cancel or upstream error before EOS) are also recorded, so the
	// metric is not biased toward fully streamed responses. The guard skips
	// requests that never reached the response phase.
	defer func() {
		if reqCtx.responseProcessingDuration > 0 {
			metrics.RecordResponseProcessingLatency(reqCtx.responseProcessingDuration)
		}
	}()

	buf := s.bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer func() {
		// Return to pool if capacity is within limits.
		if buf.Cap() <= s.maxPoolBufferSize || s.maxPoolBufferSize == 0 {
			s.bufferPool.Put(buf)
		}
	}()
	var respBody []byte
	var evictionRequestID string

	// Start a single reader goroutine for the lifetime of the stream.
	// This avoids spawning a new goroutine per message and allows the main loop to
	// select on both incoming messages and the eviction channel.
	recvCh := make(chan recvResult, 1)
	// Capture the stream context's Done channel before ctx is reassigned in the main loop.
	// This avoids a data race between the reader goroutine reading ctx and the main loop writing it.
	streamDone := srv.Context().Done()
	go func() {
		for {
			req, err := srv.Recv()
			select {
			case recvCh <- recvResult{req: req, err: err}:
			case <-streamDone:
				return
			}
			if err != nil {
				return
			}
		}
	}()

	// evictCh starts nil — selecting on a nil channel blocks forever.
	// After scheduling, it is set to the eviction channel, dynamically
	// enabling eviction listening.
	var evictCh chan struct{}

	// Create error handling var as each request should only report once for
	// error metrics. This doesn't cover the error "Cannot receive stream request" because
	// such errors might happen even though response is processed.
	var err error
	defer func() {
		// Clean up eviction channel registration on exit.
		if s.evictionLookup != nil && evictionRequestID != "" {
			s.evictionLookup.Deregister(evictionRequestID)
		}
		fairnessID, priority := extractFairnessAndPriority(reqCtx)
		if reqCtx.responseStatusCode != "" {
			metrics.RecordRequestErrCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.responseStatusCode)
		} else if err != nil {
			metrics.RecordRequestErrCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, errcommon.CanonicalCode(err))
		}
		if span != nil {
			if err != nil {
				span.RecordError(err)
				span.SetStatus(otelcodes.Error, err.Error())
			} else if reqCtx.responseStatusCode != "" {
				span.SetStatus(otelcodes.Error, reqCtx.responseStatusCode)
			}
		}
		if reqCtx.requestRunning {
			metrics.DecRunningRequests(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority)
		}

		// If we scheduled a pod (TargetPod != nil) but never marked the response  as complete (e.g. error, disconnect,
		// panic), force the completion hooks to run.
		if reqCtx.TargetPod != nil && !reqCtx.responseComplete {
			reqCtx.TerminationCause = terminationCause(reqCtx, ctx.Err())
			// Use a fresh context as the request context might be canceled (Client Disconnect).
			// We only need logging from the original context.
			cleanupCtx := log.IntoContext(context.Background(), logger)
			s.director.HandleResponseBody(cleanupCtx, reqCtx, true)
		}
	}()

	for {
		var req *extProcPb.ProcessingRequest
		var recvErr error

		// Main select: listen for incoming messages, eviction signals, and context cancellation.
		// evictCh is nil until scheduling completes, so the eviction case blocks forever until then.
		select {
		case result := <-recvCh:
			req = result.req
			recvErr = result.err
		case <-evictCh:
			// Skip if the response already completed — sending ImmediateResponse
			// after the final body chunk would be a protocol violation.
			if reqCtx.responseComplete {
				logger.V(logutil.DEBUG).Info("Eviction signal received but response already complete, ignoring",
					"requestID", evictionRequestID)
				evictCh = nil // prevent closed channel from firing repeatedly
				continue
			}
			// Eviction triggered — transition to evicted state and let the state machine send the response.
			logger.Info("Request evicted by flow control", "requestID", evictionRequestID)
			reqCtx.requestState = requestEvicted
			if s.evictionLookup != nil {
				reqCtx.requestDroppedReason = s.evictionLookup.GetReason(evictionRequestID)
			}
			if sendErr := reqCtx.updateStateAndSendIfNeeded(srv, logger); sendErr != nil {
				return sendErr
			}
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}

		if recvErr == io.EOF || status.Code(recvErr) == codes.Canceled {
			return nil
		}
		if recvErr != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", recvErr)
		}

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			requestID := envoy.ExtractHeaderValue(v, reqcommon.RequestIDHeaderKey)
			// request ID is a must for maintaining a state per request in plugins that hold internal state and use PluginState.
			// if request id was not supplied as a header, we generate it ourselves.
			if len(requestID) == 0 {
				requestID = uuid.NewString()
				loggerTrace.Info("RequestID header is not found in the request, generated a request id")
				reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey] = requestID // update in headers so director can consume it
			}
			logger = logger.WithValues(reqcommon.RequestIDHeaderKey, requestID)
			logger.V(logutil.DEFAULT).Info("EPP received request") // Request ID will be logged too as part of logger context values.
			loggerTrace = logger.V(logutil.TRACE)
			ctx = log.IntoContext(ctx, logger)

			// Re-parent the server span to the upstream trace context (e.g. the
			// traceparent set by the client or the Gateway) carried in the incoming
			// headers, then start it. The headers are only available here, so the span
			// cannot be started at the top of Process without orphaning the trace.
			ctx = extractTraceContext(ctx, v)
			ctx, span = tracer.Start(ctx, "gateway.request", trace.WithSpanKind(trace.SpanKindServer))

			err = s.HandleRequestHeaders(ctx, reqCtx, v)
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerTrace.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			// In the stream case, we can receive multiple request bodies.
			buf.Write(v.RequestBody.Body)

			// Message is buffered, we can read and decode.
			if v.RequestBody.EndOfStream {
				loggerTrace.Info("decoding")
				reqCtx.Request.Metadata = envoy.ExtractMetadataValues(req)
				reqCtx.Request.RawBody = make([]byte, buf.Len())
				copy(reqCtx.Request.RawBody, buf.Bytes())

				// Body stream complete. Capture raw size for flow control.
				reqCtx.RequestSize = buf.Len()
				buf.Reset()

				parser, resolveErr := s.getOrResolveParser(ctx, reqCtx)
				if resolveErr != nil {
					err = errcommon.Error{Code: errcommon.BadRequest, Msg: resolveErr.Error()}
					logger.Error(err, "Error resolving parser for request body")
					break
				}
				before := time.Now()
				parseResult, parseErr := parser.ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
				metrics.RecordPluginProcessingLatency(fwkrh.RequestParsingExtensionPoint, parser.TypedName().Type, parser.TypedName().Name, time.Since(before))
				if parseErr != nil {
					err = errcommon.Error{Code: errcommon.BadRequest, Msg: parseErr.Error()}
					logger.Error(err, "Error parsing request")
					break
				}

				reqCtx, err = s.director.HandleRequest(ctx, reqCtx, parseResult.Body)
				if err != nil {
					logger.Error(err, "Error handling request")
					break
				}

				// After scheduling, look up the eviction channel for eviction support.
				// Setting evictCh from nil to a real channel dynamically enables the
				// eviction case in the main select.
				if s.evictionLookup != nil {
					evictionRequestID = reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey]
					evictCh = s.evictionLookup.Get(evictionRequestID)
				}

				if reqCtx.SchedulingRequest != nil && reqCtx.SchedulingRequest.Body != nil {
					reqCtx.modelServerStreaming = reqCtx.SchedulingRequest.Body.Stream
				}

				reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(ctx, reqCtx)
				reqCtx.reqBodyResp = envoy.GenerateRequestBodyResponses(reqCtx.Request.RawBody)
				fairnessID, priority := extractFairnessAndPriority(reqCtx)
				metrics.RecordRequestCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, reqCtx.Priority)
				metrics.RecordRequestSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, fairnessID, priority, reqCtx.RequestSize)

				if parseResult.SkipResponseProcessing {
					reqCtx.requestState = requestResponseProcessingSkipped
				}

				recordRequestProcessing()
			}
		case *extProcPb.ProcessingRequest_RequestTrailers:
			// This is currently unused.
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			// Overwrites the request-phase value on purpose. Response-received plugins
			// read this through Response.ReqMetadata to learn which endpoint actually
			// served the request, and Envoy only reports that at the response phase.
			reqCtx.Request.Metadata = envoy.ExtractMetadataValues(req)
			respHeadersReceivedAt := time.Now()
			// The traceEnabled guard is intentional: passing arguments to a disabled
			// logger still boxes them into a heap-allocated slice, and this loop runs
			// per header. string(header.RawValue) in the status comparison does not
			// allocate; the content-type check runs at most once per response.
			traceEnabled := loggerTrace.Enabled()
			for _, header := range v.ResponseHeaders.Headers.GetHeaders() {
				if traceEnabled {
					loggerTrace.Info("header", "key", header.Key, "value", string(header.RawValue))
				}
				if header.Key == "status" && string(header.RawValue) != "200" {
					reqCtx.responseStatusCode = errcommon.ModelServerError
				} else if header.Key == "content-type" && strings.Contains(string(header.RawValue), "text/event-stream") {
					reqCtx.modelServerStreaming = true
					if traceEnabled {
						loggerTrace.Info("model server is streaming response")
					}
				}
			}
			reqCtx.requestState = responseReceived
			reqCtx = s.HandleResponseHeaders(ctx, reqCtx, v)
			reqCtx.respHeaderResp = s.generateResponseHeaderResponse(reqCtx)
			reqCtx.responseHeadersReceivedAt = respHeadersReceivedAt
			reqCtx.responseProcessingDuration += time.Since(respHeadersReceivedAt)

		case *extProcPb.ProcessingRequest_ResponseBody:
			endOfStream := v.ResponseBody.EndOfStream
			chunk := v.ResponseBody.Body

			if reqCtx.modelServerStreaming {
				respBodyStart := time.Now()
				if endOfStream {
					reqCtx.responseComplete = true
					reqCtx.responseCompleteTimestamp = time.Now()
				}
				s.HandleResponseBody(ctx, reqCtx, chunk, endOfStream)
				// Rewrite the model name in response body back to the original client-facing name.
				chunk, _ = rewriteModelName(chunk, reqCtx.TargetModelName, reqCtx.IncomingModelName)
				// For streaming response, we send response chunk back to envoy every time we received it.
				reqCtx.respBodyResp = generateResponseBodyResponses(chunk, endOfStream, reqCtx.Response.DynamicMetadata)
				reqCtx.responseProcessingDuration += time.Since(respBodyStart)
			} else {
				respBody = append(respBody, chunk...)
				if endOfStream {
					s.finishResponse(ctx, reqCtx, respBody, reqCtx.modelServerStreaming, true)
				}
			}
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			// For HTTP, the response trailer is not sent. Thus, this case will not be triggered.
			// For gRPC(over HTTP2), the protocol relies on responseTrailers to determine whether a response is complete.
			// More info: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/doc/PROTOCOL-HTTP2.md#responses
			s.finishResponse(ctx, reqCtx, respBody, reqCtx.modelServerStreaming, false)
			reqCtx.respTrailerResp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseTrailers{
					ResponseTrailers: &extProcPb.TrailersResponse{},
				},
			}
		}

		// Handle the err and fire an immediate response.
		if err != nil {
			recordRequestProcessing()
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Failed to process request", "request", req)
			} else {
				logger.Error(err, "Failed to process request")
			}
			resp, err := errcommon.BuildErrResponse(err)
			if err != nil {
				return err
			}
			if err := srv.Send(resp); err != nil {
				logger.Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
			return nil
		}
		loggerTrace.Info("checking", "request state", reqCtx.requestState)
		if err := reqCtx.updateStateAndSendIfNeeded(srv, logger); err != nil {
			return err
		}
		if reqCtx.requestState == requestResponseProcessingSkipped {
			logger.V(logutil.DEFAULT).Info("EPP skipped response interception, routed request",
				"targetEndpoint", reqCtx.TargetEndpoint,
				"targetModel", reqCtx.TargetModelName)
			// Gracefully close the gRPC stream to stop external processing for this request.
			// This ensures Envoy continues with the request without calling further phases.
			// See: https://github.com/envoyproxy/envoy/blob/0533de0acca281110945e5726bbb306fbb12bde5/api/envoy/service/ext_proc/v3/external_processor.proto#L40-L41
			return nil
		}
	}
}

// finishResponse ensures all post-response logic, such as metric recording
// and state updates, is executed exactly once for the request lifecycle.
func (s *StreamingServer) finishResponse(ctx context.Context, reqCtx *RequestContext, body []byte, modelStreaming bool, setEos bool) {
	// Return early if the response has already been finished to prevent
	// duplicate execution of side effects and metrics.
	if reqCtx.responseComplete {
		return
	}

	start := time.Now()
	reqCtx.responseComplete = true
	reqCtx.responseCompleteTimestamp = time.Now()
	reqCtx = s.HandleResponseBody(ctx, reqCtx, body, true)
	if !modelStreaming {
		// Rewrite the model name in response body back to the original client-facing name.
		body, _ = rewriteModelName(body, reqCtx.TargetModelName, reqCtx.IncomingModelName)
		// For non-streaming response, we send response back to envoy after receiving all the response body.
		reqCtx.respBodyResp = generateResponseBodyResponses(body, setEos, reqCtx.Response.DynamicMetadata)
	}
	if modelStreaming || reqCtx.responseHeadersReceivedAt.IsZero() {
		reqCtx.responseProcessingDuration += time.Since(start)
	} else {
		// Supersedes the header slice already accumulated: the interval since the
		// response headers arrived covers it and the body wait in between.
		reqCtx.responseProcessingDuration = time.Since(reqCtx.responseHeadersReceivedAt)
	}
}

// rewriteModelName replaces occurrences of the target (internal) model name with the
// incoming (client-facing) model name in the response body bytes. This ensures clients
// see the model name they originally requested, not the internal backend model name.
// It is a no-op when the names are identical or either is empty.
// rewriteModelName replaces the target model name with the client-facing incoming model
// name in body. It reports whether it mutated body, so callers can skip re-sending a copy
// when nothing changed.
func rewriteModelName(body []byte, targetModel, incomingModel string) ([]byte, bool) {
	if targetModel == "" || incomingModel == "" || targetModel == incomingModel {
		return body, false
	}
	old := []byte(`"model":"` + targetModel + `"`)
	new := []byte(`"model":"` + incomingModel + `"`)
	if bytes.Contains(body, old) {
		return bytes.ReplaceAll(body, old, new), true
	}
	// Also handle the case where JSON has spaces after the colon: "model": "..."
	old = []byte(`"model": "` + targetModel + `"`)
	new = []byte(`"model": "` + incomingModel + `"`)
	if bytes.Contains(body, old) {
		return bytes.ReplaceAll(body, old, new), true
	}
	return body, false
}

// updateStateAndSendIfNeeded checks state and can send multiple responses in a single pass, but only if ordered properly.
// Order of requests matter in FULL_DUPLEX_STREAMING. For both request and response, the order of response sent back MUST be: Header->Body->Trailer, with trailer being optional.
func (r *RequestContext) updateStateAndSendIfNeeded(srv extProcPb.ExternalProcessor_ProcessServer, logger logr.Logger) error {
	loggerTrace := logger.V(logutil.TRACE)

	// Handle eviction — send ImmediateResponse(429) to Envoy to reset the upstream connection.
	if r.requestState == requestEvicted {
		loggerTrace.Info("Sending ImmediateResponse for evicted request")
		ir := &extProcPb.ImmediateResponse{
			Status: &envoyTypePb.HttpStatus{
				Code: envoyTypePb.StatusCode_TooManyRequests,
			},
			Body: []byte("request evicted by flow control"),
		}
		if r.requestDroppedReason != "" {
			ir.Headers = &extProcPb.HeaderMutation{
				SetHeaders: []*configPb.HeaderValueOption{
					{
						Header: &configPb.HeaderValue{
							Key:      errcommon.RequestDroppedReasonHeaderKey,
							RawValue: []byte(r.requestDroppedReason),
						},
					},
				},
			}
		}
		return srv.Send(&extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: ir,
			},
		})
	}

	// Handle skip — send response with the director's routing decision to the proxy.
	if r.requestState == requestResponseProcessingSkipped {
		if r.reqHeaderResp != nil {
			if err := srv.Send(r.reqHeaderResp); err != nil {
				logger.Error(err, "error sending response")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
		if r.reqBodyResp != nil {
			for _, response := range r.reqBodyResp {
				if err := srv.Send(response); err != nil {
					logger.Error(err, "error sending response")
					return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
				}
			}
		}
		return nil
	}

	// No switch statement as we could send multiple responses in one pass.
	if r.requestState == requestReceived && r.reqHeaderResp != nil {
		loggerTrace.Info("Sending request header response", "obj", r.reqHeaderResp)
		if err := srv.Send(r.reqHeaderResp); err != nil {
			logger.Error(err, "error sending response")
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.requestState = headerRequestResponseComplete
	}
	if r.requestState == headerRequestResponseComplete && len(r.reqBodyResp) > 0 {
		loggerTrace.Info("Sending request body response(s)")

		for _, response := range r.reqBodyResp {
			if err := srv.Send(response); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
		logger.V(logutil.DEFAULT).Info("EPP sent request body response(s) to proxy", "modelName", r.IncomingModelName, "targetModelName", r.TargetModelName)
		r.requestState = bodyRequestResponsesComplete
		fairnessID, priority := extractFairnessAndPriority(r)
		metrics.IncRunningRequests(r.IncomingModelName, r.TargetModelName, fairnessID, priority)
		r.requestRunning = true
		// Dump the response so a new stream message can begin
		r.reqBodyResp = nil
	}
	if r.requestState == responseReceived && r.respHeaderResp != nil {
		loggerTrace.Info("Sending response header response", "obj", r.respHeaderResp)
		if err := srv.Send(r.respHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.requestState = headerResponseResponseComplete
	}
	if r.requestState == headerResponseResponseComplete {
		loggerTrace.Info("Sending response body response(s)")
		for _, response := range r.respBodyResp {
			if err := srv.Send(response); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
		if r.responseComplete {
			logger.V(logutil.DEFAULT).Info("EPP sent response body back to proxy")
			r.requestState = bodyResponseResponsesComplete
		}
		// Dump the response so a new stream message can begin
		r.respBodyResp = nil
	}
	if r.requestState == bodyResponseResponsesComplete && r.respTrailerResp != nil {
		// Trailers in responses are not guaranteed
		if err := srv.Send(r.respTrailerResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		logger.V(logutil.DEBUG).Info("EPP sent trailer back to proxy")
	}
	return nil
}
