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
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	logging "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

// handleP2P implements the vLLM OffloadingConnector P2P orchestration contract. The
// prefiller stores KV under a kv_request_id with no peer address; the decoder
// pulls it using the prefiller's OffloadingConnector P2P tier host/port. Both legs are
// dispatched concurrently: the connector parks any KV blocks stored before the
// decoder's fetch binds the session, so ordering between the legs is safe.
func (s *Server) handleP2P(w http.ResponseWriter, r *http.Request, prefillPodHostPort, kvCacheSource string) {
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
	prefillP2PPort := s.p2pPortFor(prefillPodHostPort)
	s.logger.Info("running P2P protocol",
		"prefill_host", extractHost(prefillPodHostPort),
		"kv_request_id", kvRequestID,
		"p2p_connector_port", prefillP2PPort)

	// Prefill leg: store KV under kv_request_id, no peer address. Capped to a
	// single output token so the prefiller returns as soon as KV is stored.
	prefillData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		prefillData[k] = v
	}
	prefillKVParams := map[string]any{
		requestFieldP2PDecodeParams: map[string]any{
			requestFieldKVRequestID: kvRequestID,
		},
	}
	s.addP2PPullToPrefill(prefillKVParams, kvCacheSource, prefillPodHostPort)
	prefillData[requestFieldKVTransferParams] = prefillKVParams
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

	// Decode leg: pull KV from the prefiller's OffloadingConnector P2P tier. Original body
	// (streaming, token limits) is preserved.
	decodeData := make(map[string]any, len(requestData)+1)
	for k, v := range requestData {
		decodeData[k] = v
	}
	decodeData[requestFieldKVTransferParams] = map[string]any{
		requestFieldP2PPrefillParams: map[string]any{
			requestFieldKVRequestID: kvRequestID,
			requestFieldRemoteHost:  extractHost(prefillPodHostPort),
			requestFieldRemotePort:  prefillP2PPort,
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
		attribute.String("llm_d.pd_proxy.connector", KVConnectorOffloading),
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
		attribute.String("llm_d.pd_proxy.connector", KVConnectorOffloading),
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

// p2pPullAvailable reports whether this deployment can pull cached prefix over
// the OffloadingConnector P2P tier. That tier is the PD connector itself when
// KVConnector is offloading, or is composed alongside NIXL via MultiConnector
// (declared with --enable-p2p-pull) when the PD connector is NIXLv2. On any
// other connector --enable-p2p-pull has no effect, since no MultiConnector
// routes the p2p params to an OffloadingConnector.
func (s *Server) p2pPullAvailable() bool {
	return s.config.KVConnector == KVConnectorOffloading ||
		(s.config.EnableP2PPull && s.config.KVConnector == KVConnectorNIXLV2)
}

// addP2PPullToPrefill adds the OffloadingConnector p2p pull block to a prefill
// leg's kv_transfer_params so the prefiller pulls cached prefix from
// kvCacheSource while keeping its own computed blocks available for the
// decoder. It is a no-op when no source is set or the source resolves to the
// prefiller itself, since there is nothing to pull from oneself. The p2p key
// composes with NIXL params: vLLM's MultiConnector routes it to the
// OffloadingConnector and the NIXL fields to the NixlConnector.
func (s *Server) addP2PPullToPrefill(prefillKVParams map[string]any, kvCacheSource, prefillPodHostPort string) {
	if kvCacheSource != "" && extractHost(kvCacheSource) != extractHost(prefillPodHostPort) {
		prefillKVParams[requestFieldP2PParams] = s.p2pSourceParams(kvCacheSource)
	}
}

// p2pSourceParams builds the kv_transfer_params.p2p block for a pull from
// sourceHostPort's OffloadingConnector P2P tier. The kv_request_id is its
// own fresh UUID: in P2P mode it is consumer-side only.
func (s *Server) p2pSourceParams(sourceHostPort string) map[string]any {
	return map[string]any{
		requestFieldKVRequestID: newUUID(),
		requestFieldRemoteHost:  extractHost(sourceHostPort),
		requestFieldRemotePort:  s.p2pPortFor(sourceHostPort),
	}
}

// p2pPortFor resolves the P2P tier control port on the target endpoint. The
// sidecar serves rank r on <base port>+r (data_parallel.go), so the routed
// endpoint's port encodes the pod-local rank. vLLM binds the tier at
// <p2p-connector-port> + the global data_parallel_index, so the mapping is
// correct when each pod is its own DP group (local rank == global index).
// Multi-pod DP groups (e.g. LWS wide-EP) need the source's global rank
// supplied per request instead; this derivation is the per-pod fallback. A
// port outside the rank range (or unparsable) falls back to the base P2P
// port, which is rank 0's tier.
func (s *Server) p2pPortFor(targetHostPort string) int {
	base := s.config.P2PConnectorPort
	if s.config.DataParallelSize <= 1 || s.dpBasePort == 0 {
		return base
	}
	// Backward compatible behavior: trim `http:` prefix (see createProxyHandler).
	targetHostPort, _ = strings.CutPrefix(targetHostPort, "http://")
	_, portStr, err := net.SplitHostPort(targetHostPort)
	if err != nil {
		return base
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return base
	}
	rank := port - s.dpBasePort
	if rank < 0 || rank >= s.config.DataParallelSize {
		return base
	}
	return base + rank
}

// decodeWithP2PSource serves a decoder-only request through the local vLLM
// with kv_transfer_params.p2p injected, so the engine looks up and pulls
// cached prefix blocks from the peer at sourceHostPort instead of recomputing
// them. It replaces any client-supplied kv_transfer_params (the sidecar owns
// that field) and routes through dispatchDecode so chunked decode still
// applies. When sourceHostPort resolves to this pod, injecting would tell the
// engine to pull the prefix it is already computing, so it decodes normally.
func (s *Server) decodeWithP2PSource(w http.ResponseWriter, r *http.Request, sourceHostPort string) {
	raw, requestData, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	if extractHost(sourceHostPort) == os.Getenv("POD_IP") {
		s.logger.V(logging.DEBUG).Info("KV cache source is the local pod, skipping p2p injection",
			"source", sourceHostPort)
		s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, raw), requestData)
		return
	}

	p2pParams := s.p2pSourceParams(sourceHostPort)
	// Rebuild kv_transfer_params from scratch: the sidecar owns this field, so
	// client-supplied keys are dropped rather than forwarded to vLLM.
	requestData[requestFieldKVTransferParams] = map[string]any{requestFieldP2PParams: p2pParams}

	s.logger.Info("running P2P source protocol",
		"source_host", extractHost(sourceHostPort),
		"kv_request_id", p2pParams[requestFieldKVRequestID],
		"p2p_connector_port", p2pParams[requestFieldRemotePort])

	newBody, err := json.Marshal(requestData)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if v := s.logger.V(logging.TRACE); v.Enabled() {
		v.Info("decoder request body with p2p source", "body", string(newBody))
	}

	s.dispatchDecode(w, cloneRequestWithBody(r.Context(), r, newBody), requestData)
}
