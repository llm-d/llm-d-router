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
	"fmt"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

const (
	// requestHeaderDataParallelRank pins a request to a specific vLLM
	// data-parallel rank, set on both requests of a disagg pair (see pickDPRank).
	requestHeaderDataParallelRank = "x-data-parallel-rank"

	requestFieldTransferID = "transfer_id"
)

// concurrentPrefillGrace limits how long prefill may continue after a
// successful decode returns. A variable so tests can shorten it.
var concurrentPrefillGrace = 30 * time.Second

// runConcurrentPD fires the prefill and decode requests of a concurrent-dispatch
// P/D protocol (Mooncake, SGLang) in parallel: prefill runs in a goroutine
// and its response is discarded (only status and duration are recorded on
// its span), while decode runs on the calling goroutine and streams its
// response to w. connector is recorded on both spans to identify the
// protocol.
func (s *Server) runConcurrentPD(
	w http.ResponseWriter,
	r *http.Request,
	prefillBody, decodeBody []byte,
	prefillHost, connector string,
	// prepareRequests lets a connector mutate the cloned requests before dispatch,
	// e.g. Mooncake sets a DP-rank header so prefill lands on the same rank decode reads from.
	prepareRequests func(prefillReq, decodeReq *http.Request),
	toolFields *toolCallingRequest,
) {
	defer toolFields.report(r.Context())
	tracer := tracing.Tracer(tracerScope)
	ctx, cancelDecode := context.WithCancelCause(r.Context())
	defer cancelDecode(nil)

	// WithoutCancel for prefill so it isn't aborted when the decode response finishes first.
	prefillCtx, cancelPrefill := context.WithCancel(context.WithoutCancel(ctx))
	// Client disconnect aborts prefill too. Stop on return: net/http also cancels
	// r.Context() once the handler finishes, which must not abort prefill.
	defer context.AfterFunc(r.Context(), cancelPrefill)()
	prefillReq := cloneRequestWithBody(prefillCtx, r, prefillBody)
	decodeReq := cloneRequestWithBody(ctx, r, decodeBody)
	if prepareRequests != nil {
		prepareRequests(prefillReq, decodeReq)
	}

	// Prefill runs in a goroutine: response is discarded (status/duration only).
	// Decode runs on the calling goroutine: writes the actual response back via w.
	// Both spans start from the request context so they come out as siblings, and
	// each outbound request carries its own stage span.
	_, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillReq = prefillReq.WithContext(trace.ContextWithSpan(prefillReq.Context(), prefillSpan))
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillTarget(prefillHost),
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyPrefillAsync(true),
	)
	// Lets a consumer of the span (for example the scheduler, which cannot see the prefill response) tie it to the
	// request it routed. The nixlv2 connector sets the same attribute on its own prefill span.
	if requestID := r.Header.Get(reqcommon.RequestIDHeaderKey); requestID != "" {
		prefillSpan.SetAttributes(semconv.LLMDPDProxyRequestID(requestID))
	}
	prefillStart := time.Now()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		log.FromContext(r.Context()).Error(err, "failed to create prefill handler")
		if err := errorBadGateway(w, r); err != nil {
			log.FromContext(r.Context()).Error(err, "failed to send error response to client")
		}
		return
	}

	// Aggregate before starting the worker so reporting does not wait for prefill.
	toolFields.compareDispatchedForSpan(prefillBody, prefillSpan)
	go func() {
		defer cancelPrefill()
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				log.FromContext(prefillCtx).Error(fmt.Errorf("prefill handler panicked (%T)", rec), "panic in prefill request")
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillErr := servePrefill(prefillHandler, pw, prefillReq)
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
			semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
		)
		// A prefill the sidecar cancelled is not a prefill error.
		if prefillCtx.Err() == nil {
			if failed, relay := s.concurrentPrefillFailed(prefillReq, pw, prefillErr); failed {
				observeStageError(r.Context(), metrics.StagePrefill)
				prefillSpan.SetStatus(codes.Error, "prefill request failed")
				cancelDecode(&prefillFailedError{response: relay})
			}
		}
		observeStageDuration(r.Context(), metrics.StagePrefill, prefillDuration)
		log.FromContext(prefillCtx).V(logging.DEBUG).Info("concurrent-dispatch prefill request completed", "connector", connector, "status", pw.statusCode)
	}()

	// Decode Stage
	dCtx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyConnector(connector),
		semconv.LLMDPDProxyDecodeConcurrentWithPrefill(true),
	)
	decodeStart := time.Now()

	decodeReq = decodeReq.WithContext(dCtx)
	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeFailed := func() bool {
		return decodeStatus.failed() || grpcframe.HasErrorStatus(decodeWriter.Header())
	}
	decodeReturned := false
	defer recordDecodeAbort(r.Context(), &decodeReturned, decodeStart)
	defer func() {
		if prefillCtx.Err() != nil {
			return
		}
		// The request is over once decode has failed; a prefill still running
		// would compute KV nobody consumes.
		if !decodeReturned || decodeFailed() {
			cancelPrefill()
			return
		}
		// Prefill has no deadline of its own; one that never responds would
		// hold the goroutine and connection indefinitely.
		// Not awaited: the client's response must not wait on the prefill response.
		graceTimer := time.AfterFunc(concurrentPrefillGrace, func() {
			log.FromContext(prefillCtx).Error(nil, "concurrent-dispatch prefill did not complete within the grace period after decode; cancelled",
				"connector", connector)
			cancelPrefill()
		})
		context.AfterFunc(prefillCtx, func() { graceTimer.Stop() })
	}()
	toolFields.compareDispatchedForSpan(decodeBody, decodeSpan)
	s.decoderProxy.ServeHTTP(decodeWriter, decodeReq)
	decodeReturned = true

	decodeDuration := time.Since(decodeStart)
	observeStageDuration(r.Context(), metrics.StageDecode, decodeDuration)
	if decodeFailed() {
		observeStageError(r.Context(), metrics.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())),
		semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host),
	)

	// End-to-end P/D timing. True TTFT captures time from gateway request start
	// to decode start; prefill duration is tracked in the async prefill span.
	// Set on the decode span explicitly, since ctx is the forward_request context.
	if decodeSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		decodeSpan.SetAttributes(
			semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
			semconv.LLMDPDProxyTrueTTFTMs(float64(trueTTFT.Milliseconds())),
			semconv.LLMDPDProxyDecodeDurationMsSummary(float64(decodeDuration.Milliseconds())),
			semconv.LLMDPDProxyConcurrentPD(true),
		)
	}
}

// prefillFailedError is the cause a concurrent decode request is cancelled
// with when prefill fails. The decoder proxy returns its response to the client.
type prefillFailedError struct {
	response *bufferedResponseWriter
}

func (s *Server) concurrentPrefillFailed(prefillReq *http.Request, pw *bufferedResponseWriter, prefillErr error) (bool, *bufferedResponseWriter) {
	if isHTTPError(pw.statusCode) || grpcframe.HasErrorStatus(pw.headers) {
		return true, pw
	}
	if !isVLLMGRPCRequest(prefillReq) {
		return false, nil
	}
	if _, err := extractVLLMPrefillFinishInfo(pw, prefillErr); err != nil {
		log.FromContext(prefillReq.Context()).Error(err, "concurrent-dispatch gRPC prefill stream incomplete")
		synth := &bufferedResponseWriter{}
		_ = errorBadGateway(synth, prefillReq)
		return true, synth
	}
	return false, nil
}

// recordSequentialPDTimingAttrs records end-to-end sequential P/D timing
// attributes on decodeSpan. True TTFT measures gateway request start to decode
// start, including scheduling, prefill, and coordination overhead.
func recordSequentialPDTimingAttrs(
	ctx context.Context,
	decodeSpan trace.Span,
	prefillStart time.Time,
	prefillDuration time.Duration,
	decodeStart time.Time,
	decodeDuration time.Duration,
) {
	if !decodeSpan.SpanContext().IsValid() {
		return
	}
	var totalDuration, trueTTFT time.Duration
	if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
		if requestStart, ok := requestStartValue.(time.Time); ok {
			totalDuration = time.Since(requestStart)
			trueTTFT = decodeStart.Sub(requestStart)
		}
	}
	coordinatorOverhead := decodeStart.Sub(prefillStart.Add(prefillDuration))
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
		semconv.LLMDPDProxyTrueTTFTMs(float64(trueTTFT.Milliseconds())),
		semconv.LLMDPDProxyPrefillDurationMsSummary(float64(prefillDuration.Milliseconds())),
		semconv.LLMDPDProxyDecodeDurationMsSummary(float64(decodeDuration.Milliseconds())),
		semconv.LLMDPDProxyCoordinatorOverheadMs(float64(coordinatorOverhead.Milliseconds())),
		semconv.LLMDPDProxyConcurrentPD(false),
	)
}

func (e *prefillFailedError) Error() string {
	if grpcframe.HasErrorStatus(e.response.headers) {
		code, _ := grpcframe.Status(e.response.headers)
		return fmt.Sprintf("prefill returned %d (grpc-status %s)", e.response.statusCode, code)
	}
	return fmt.Sprintf("prefill returned %d", e.response.statusCode)
}

// recordDecodeAbort records decode duration and a decode error when decode
// dispatch did not return normally. The reverse proxy panics with
// http.ErrAbortHandler when a stream breaks after headers were sent (client
// disconnect, load balancer timeout), which skips the metrics recorded after
// dispatch. Deferred without recover, so the panic still propagates.
func recordDecodeAbort(ctx context.Context, returned *bool, start time.Time) {
	if !*returned {
		observeStageDuration(ctx, metrics.StageDecode, time.Since(start))
		observeStageError(ctx, metrics.StageDecode)
	}
}
