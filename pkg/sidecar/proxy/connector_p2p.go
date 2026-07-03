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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

// handleP2P implements the vLLM P2PConnector orchestration contract. The
// prefiller stores KV under a kv_request_id with no peer address; the decoder
// pulls it using the prefiller's P2PConnector host/port. Both legs are
// dispatched concurrently: the connector parks any KV blocks stored before the
// decoder's fetch binds the session, so ordering between the legs is safe.
func (s *Server) handleP2P(w http.ResponseWriter, r *http.Request, prefillPodHostPort string) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		if err := errorJSONInvalid(fmt.Errorf("failed to read request body: %w", err), w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	var requestData map[string]any
	if err := json.Unmarshal(body, &requestData); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	kvRequestID := newUUID()
	s.logger.Info("running P2P protocol",
		"prefill_host", extractHost(prefillPodHostPort),
		"kv_request_id", kvRequestID,
		"p2p_connector_port", s.config.P2PConnectorPort)

	// Prefill leg: store KV under kv_request_id, no peer address. Capped to a
	// single output token so the prefiller returns as soon as KV is stored.
	prefillData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		prefillData[k] = v
	}
	prefillData[requestFieldKVTransferParams] = map[string]any{
		requestFieldP2PDecodeParams: map[string]any{
			requestFieldKVRequestID: kvRequestID,
		},
	}
	prefillData[requestFieldStream] = false
	delete(prefillData, requestFieldStreamOptions)
	prefillData[requestFieldMaxTokens] = 1
	if _, ok := prefillData[requestFieldMaxCompletionTokens]; ok {
		prefillData[requestFieldMaxCompletionTokens] = 1
	}

	prefillBody, err := json.Marshal(prefillData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("prefill request body", "body", string(prefillBody))
	}

	// Decode leg: pull KV from the prefiller's P2PConnector. Original body
	// (streaming, token limits) is preserved.
	decodeData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		decodeData[k] = v
	}
	decodeData[requestFieldKVTransferParams] = map[string]any{
		requestFieldP2PPrefillParams: map[string]any{
			requestFieldKVRequestID: kvRequestID,
			requestFieldRemoteHost:  extractHost(prefillPodHostPort),
			requestFieldRemotePort:  s.config.P2PConnectorPort,
		},
	}

	decodeBody, err := json.Marshal(decodeData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decode request body", "body", string(decodeBody))
	}

	s.handleP2PConcurrentRequests(w, r, prefillBody, decodeBody, prefillPodHostPort)
}

func (s *Server) handleP2PConcurrentRequests(w http.ResponseWriter, r *http.Request, prefillBody, decodeBody []byte, prefillHost string) {
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	// WithoutCancel for prefill so it isn't aborted when the decode response finishes first.
	prefillReq := cloneRequestWithBody(context.WithoutCancel(ctx), r, prefillBody)
	decodeReq := cloneRequestWithBody(ctx, r, decodeBody)

	// Prefill runs in a goroutine: only stores KV, response is discarded.
	// Decode runs on the main thread: writes the actual response back via w.
	ctx, prefillSpan := tracer.Start(ctx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.prefill_target", prefillHost),
		attribute.String("llm_d.pd_proxy.connector", KVConnectorP2P),
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
		s.logger.V(logging.DEBUG).Info("p2p prefill request completed", "status", pw.statusCode)
	}()

	// Decode Stage
	ctx, decodeSpan := tracer.Start(ctx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.connector", KVConnectorP2P),
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
