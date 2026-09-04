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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

// runConcurrentPD fires the prefill and decode legs of a concurrent-dispatch
// P/D protocol (Mooncake, OffloadingConnector P2P, SGLang) in parallel:
// prefill runs in a goroutine and its response is discarded (only status and
// duration are recorded on its span), while decode runs on the calling
// goroutine and streams its response to w. connector is recorded on both
// spans to identify the protocol. The per-request "prefill request
// completed" log line is unified at DEBUG for all three connectors,
// deliberately dropping the prior per-connector split (Mooncake/SGLang at
// TRACE, P2P at DEBUG): three verbosity levels for structurally identical
// code was the inconsistency worth fixing, not a behavior worth preserving.
// prepareRequests, if non-nil, is called after both requests are built so a
// caller can attach protocol-specific headers (e.g. Mooncake's DP-rank
// header) before dispatch.
func (s *Server) runConcurrentPD(
	w http.ResponseWriter,
	r *http.Request,
	prefillBody, decodeBody []byte,
	prefillHost, connector string,
	prepareRequests func(prefillReq, decodeReq *http.Request),
) {
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	// WithoutCancel for prefill so it isn't aborted when the decode response finishes first.
	prefillReq := cloneRequestWithBody(context.WithoutCancel(ctx), r, prefillBody)
	decodeReq := cloneRequestWithBody(ctx, r, decodeBody)
	if prepareRequests != nil {
		prepareRequests(prefillReq, decodeReq)
	}

	// Prefill runs in a goroutine: response is discarded (status/duration only).
	// Decode runs on the calling goroutine: writes the actual response back via w.
	ctx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.prefill_target", prefillHost),
		attribute.String("llm_d.pd_proxy.connector", connector),
		attribute.Bool("llm_d.pd_proxy.prefill.async", true),
	)
	prefillStart := time.Now()

	prefillHandler, err := s.prefillerProxyHandler(prefillHost)
	if err != nil {
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	go func() {
		defer prefillSpan.End()
		defer func() {
			if rec := recover(); rec != nil && rec != http.ErrAbortHandler {
				s.logger.Error(fmt.Errorf("panic: %v", rec), "panic in prefill request")
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, prefillReq)
		prefillDuration := time.Since(prefillStart)
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
		)
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
		}
		s.logger.V(logging.DEBUG).Info("concurrent-dispatch prefill request completed", "connector", connector, "status", pw.statusCode)
	}()

	// Decode Stage
	ctx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.connector", connector),
		attribute.Bool("llm_d.pd_proxy.decode.concurrent_with_prefill", true),
	)
	decodeStart := time.Now()

	decodeReq = decodeReq.WithContext(ctx)
	s.decoderProxy.ServeHTTP(w, decodeReq)

	decodeDuration := time.Since(decodeStart)
	decodeSpan.SetAttributes(
		attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(decodeDuration.Milliseconds())),
		attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host),
	)

	// End-to-end P/D timing. True TTFT captures time from gateway request start
	// to decode start; prefill duration is tracked in the async prefill span.
	if currentSpan := trace.SpanFromContext(ctx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		currentSpan.SetAttributes(
			attribute.Float64("llm_d.pd_proxy.total_duration_ms", float64(totalDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.true_ttft_ms", float64(trueTTFT.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.decode_duration_ms", float64(decodeDuration.Milliseconds())),
			attribute.Bool("llm_d.pd_proxy.concurrent_pd", true),
		)
	}
}
