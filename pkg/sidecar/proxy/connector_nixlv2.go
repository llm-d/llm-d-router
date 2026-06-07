/*
Copyright 2025 The llm-d Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

func (s *Server) handleNIXLV2(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, apiType APIType) {
	tokenLimitFields := tokenLimitFieldsForAPIType(apiType)
	s.logger.V(4).Info("running NIXL protocol V2", "url", prefillPodHostPort, "tokenLimitFields", tokenLimitFields)

	original, completionRequest, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	// Generate unique request UUID
	uuid, err := uuid.NewUUID()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	uuidStr := uuid.String()
	// MoRI-IO wire protocol requires transfer_id values to start with
	// MoRIIOConstants.TRANSFER_PREFIX ("tx") so MoRIIOEngine._handle_message
	// can route them to _handle_completion_message.  We keep uuidStr (no
	// prefix) for sidecar logging and use transferID (prefixed) on the wire.
	transferID := "tx" + uuidStr

	// MoRI-IO WRITE-mode concurrent-dispatch path: fires prefill and
	// decode HTTP legs in parallel goroutines and synthesises decode's
	// kv_transfer_params from sidecar config instead of reading them
	// from prefill's response.  See Config.MoRIIOParallelDispatch for
	// rationale.  When the flag is off this branch is skipped and the
	// strictly-serial path (rest of this function) runs, producing the
	// same wire shape as the upstream NIXLv2 connector.
	if s.config.MoRIIOParallelDispatch && s.config.MoRIIOWriteMode {
		s.runNIXLProtocolV2WriteParallel(w, r, original, completionRequest, uuidStr, transferID, prefillPodHostPort)
		return
	}

	// Prefill Stage
	tracer := telemetry.Tracer()
	ctx := r.Context()

	ctx, prefillSpan := tracer.Start(ctx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.prefill_target", prefillPodHostPort),
		attribute.String("llm_d.pd_proxy.connector", KVConnectorNIXLV2),
	)
	prefillStart := time.Now()

	// 1. Prepare prefill request
	preq := r.Clone(ctx)

	preq.Header.Add(requestHeaderRequestID, uuidStr)

	// Pin both legs of the disagg pair to the same DP rank (header on
	// the HTTP leg, remote_dp_rank on the kv_transfer_params wire).
	// Computed once so the same rank is used everywhere the request
	// fans out below.  Single-DP deployments (MoRIIODPSize<=1) skip
	// the header emission; the wire shape stays unchanged for
	// non-Wide-EP setups.
	dpRank := pickDPRank(uuidStr, s.config.MoRIIODPSize)
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}

	// Save original values based on API type
	streamValue, streamOk := completionRequest[requestFieldStream]
	streamOptionsValue, streamOptionsOk := completionRequest[requestFieldStreamOptions]

	// Save and override token limit fields for prefill
	type savedField struct {
		field   string
		val     any
		present bool
	}
	var savedTokenValues [2]savedField
	for i, field := range tokenLimitFields {
		if v, ok := completionRequest[field]; ok {
			savedTokenValues[i] = savedField{field: field, val: v, present: true}
		} else {
			savedTokenValues[i] = savedField{field: field}
		}
	}

	// Snapshot the original request map before prefill mutations so the
	// fallback-to-decode path can dispatch with the correct original fields.
	originalRequest := maps.Clone(completionRequest)

	// In MoRI-IO WRITE mode the prefill engine pushes KV directly to decode via
	// RDMA Write. For that to work, the prefill connector's update_state_after_alloc
	// needs remote_host, remote_notify_port, and transfer_id up front -- fields
	// that the standard NIXLv2 contract leaves nil.  When --moriio-write-mode is
	// set we populate them from the sidecar's static config: remote_host is
	// decode's pod IP (s.config.MoRIIODecodePodIP from POD_IP env via downward API)
	// plus a transfer_id derived from the request UUID.  READ-mode deployments
	// fall through to the unchanged nil-fields path.
	if s.config.MoRIIOWriteMode {
		// remote_host MUST be decode's routable pod IP (not localhost) -- the
		// producer/prefill side reads this as the destination for its RDMA
		// handshake (moriio_connector.py:1330 synthesises
		// remote_engine_id = remote_host + ":" + remote_handshake_port).
		// See proxy.go::Config.MoRIIODecodePodIP for the failure mode if this
		// is "localhost".
		completionRequest[requestFieldKVTransferParams] = map[string]any{
			requestFieldDoRemoteDecode:       true,
			requestFieldDoRemotePrefill:      false,
			requestFieldRemoteEngineID:       nil,
			requestFieldRemoteBlockIDs:       nil,
			requestFieldRemoteHost:           s.config.MoRIIODecodePodIP,
			requestFieldRemotePort:           nil,
			requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
			requestFieldRemoteDPRank:         dpRank,
			requestFieldRemoteDPRankOverride: true,
			requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
			requestFieldTransferID:           transferID,
		}
	} else {
		completionRequest[requestFieldKVTransferParams] = map[string]any{
			requestFieldDoRemoteDecode:  true,
			requestFieldDoRemotePrefill: false,
			requestFieldRemoteEngineID:  nil,
			requestFieldRemoteBlockIDs:  nil,
			requestFieldRemoteHost:      nil,
			requestFieldRemotePort:      nil,
		}
	}

	completionRequest[requestFieldStream] = false
	delete(completionRequest, requestFieldStreamOptions)

	for _, field := range tokenLimitFields {
		completionRequest[field] = 1
	}

	pbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	preq.Body = io.NopCloser(bytes.NewReader(pbody))
	preq.ContentLength = int64(len(pbody))

	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// 2. Forward request to prefiller
	s.logger.V(4).Info("sending prefill request", "to", prefillPodHostPort)
	s.logger.V(5).Info("Prefill request", "body", string(pbody))
	pw := &bufferedResponseWriter{}
	prefillHandler.ServeHTTP(pw, preq)

	prefillDuration := time.Since(prefillStart)
	prefillSpan.SetAttributes(
		attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
		attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
	)

	if isHTTPError(pw.statusCode) {
		s.logger.Error(err, "request failed", "code", pw.statusCode, "body", pw.buffer.String())
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		prefillSpan.End()

		if shouldFallbackToDecode(pw) {
			s.logger.Info("fallback to decode", "request_id", uuidStr)
			fallbackReq := cloneRequestWithBody(r.Context(), r, original)
			s.dispatchDecode(w, fallbackReq, originalRequest)
		} else {
			for key, values := range pw.Header() {
				for _, v := range values {
					w.Header().Add(key, v)
				}
			}
			w.WriteHeader(pw.statusCode)
			_, err := w.Write(pw.bodyBytes())
			if err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
		}
		return
	}
	prefillSpan.End()

	// Process response - extract p/d fields
	var prefillerResponse map[string]any
	if err := json.Unmarshal(pw.bodyBytes(), &prefillerResponse); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// 3. Verify response

	pKVTransferParams, ok := prefillerResponse[requestFieldKVTransferParams]
	if !ok {
		s.logger.Info("warning: missing 'kv_transfer_params' field in prefiller response")
	}
	pCachedTokens, hasPCachedTokens := extractCachedTokens(prefillerResponse)
	if !hasPCachedTokens {
		// vLLM returns prompt_tokens_details as null when cached_tokens is 0,
		// so treat a missing prefiller cached_tokens value as zero.
		pCachedTokens = 0
	}

	s.logger.V(5).Info("received prefiller response", requestFieldKVTransferParams, pKVTransferParams)

	// Decode Stage

	ctx, decodeSpan := tracer.Start(ctx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.connector", KVConnectorNIXLV2),
	)
	decodeStart := time.Now()

	// 1. Prepare decode request
	dreq := r.Clone(ctx)

	dreq.Header.Add(requestHeaderRequestID, uuidStr)

	// Stamp the same DP-rank pin on the decode HTTP leg.  Recompute
	// rather than reuse the prefill-leg local variable: the two legs
	// are far apart in this function and a future refactor that
	// splits them must not silently lose the binding.
	if s.config.MoRIIODPSize > 1 {
		dreq.Header.Set(
			requestHeaderDataParallelRank,
			strconv.Itoa(pickDPRank(uuidStr, s.config.MoRIIODPSize)),
		)
	}

	delete(completionRequest, requestFieldStream)
	streamingEnabled := false
	if streamOk {
		completionRequest[requestFieldStream] = streamValue
		if streamBool, ok := streamValue.(bool); ok {
			streamingEnabled = streamBool
		}
	}
	decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.streaming", streamingEnabled))
	if streamOptionsOk {
		completionRequest[requestFieldStreamOptions] = streamOptionsValue
	}

	for i := range savedTokenValues[:len(tokenLimitFields)] {
		sv := &savedTokenValues[i]
		delete(completionRequest, sv.field)
		if sv.present {
			completionRequest[sv.field] = sv.val
		}
	}
	// In WRITE mode, fill in the kv_transfer_params fields that vLLM's
	// MoRIIOConnector.update_state_after_alloc -> send_notify_block path reads
	// on the decode (consumer) side. vLLM's MoRIIOConnector.request_finished
	// only echoes back {do_remote_prefill, do_remote_decode, remote_block_ids,
	// remote_engine_id, remote_host, remote_port, tp_size}; it does NOT echo
	// transfer_id, remote_notify_port, or remote_dp_rank.  Without those the
	// decode side either hits KeyError("remote_notify_port") on send_notify_block
	// or silently fails to bind notifications to the right request and the chat
	// request hangs (see moriio_connector.py around line 309 + 427).
	//
	// The values we inject here come from the sidecar's static config because
	// they are pod-local (the decode pod is co-located with this sidecar).
	if s.config.MoRIIOWriteMode {
		if dKVParams, ok := pKVTransferParams.(map[string]any); ok {
			if _, present := dKVParams[requestFieldTransferID]; !present {
				dKVParams[requestFieldTransferID] = transferID
			}
			if _, present := dKVParams[requestFieldRemoteNotifyPort]; !present {
				dKVParams[requestFieldRemoteNotifyPort] = s.config.MoRIIODecodeNotifyPort
			}
			if _, present := dKVParams[requestFieldRemoteDPRank]; !present {
				// Per-request DP-rank pin via blake2s(uuid)%dpSize.
				// The override sentinel tells decode-side vLLM to
				// use this value verbatim and skip its own blake2s
				// recomputation (avoids cross-language hash drift).
				dKVParams[requestFieldRemoteDPRank] = pickDPRank(uuidStr, s.config.MoRIIODPSize)
				dKVParams[requestFieldRemoteDPRankOverride] = true
			}
			// remote_handshake_port is required by the decode-side
			// MoRIIOConnector.build_connector_meta -> add_new_req path
			// (moriio_common.py:299).  Prefill's request_finished response
			// does not include it (only remote_host + remote_port).  Since
			// both prefill and decode pods use the same MoRI-IO default
			// handshake port (6301), backfilling from sidecar config is
			// always correct.
			if _, present := dKVParams[requestFieldRemoteHandshakePort]; !present {
				dKVParams[requestFieldRemoteHandshakePort] = s.config.MoRIIODecodeHandshakePort
			}
		}
	}
	completionRequest[requestFieldKVTransferParams] = pKVTransferParams

	dbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))

	// 2. Forward to local decoder.

	s.logger.V(5).Info("sending request to decoder", "body", string(dbody))
	decodeWriter, finalizeDecodeWriter := newCachedTokensResponseWriterWithFinalize(w, pCachedTokens)
	dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(decodeWriter, dreq)
	decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.data_parallel", dataParallelUsed))

	if !dataParallelUsed {
		s.logger.V(4).Info("sending request to decoder", "to", s.config.DecoderURL.Host)
		decodeSpan.SetAttributes(attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host))
		s.dispatchDecode(decodeWriter, dreq, completionRequest)
	}
	if err := finalizeDecodeWriter(); err != nil {
		s.logger.Error(err, "failed to flush cached token response writer")
		decodeSpan.SetStatus(codes.Error, "failed to flush cached token response writer")
		return
	}

	decodeDuration := time.Since(decodeStart)
	decodeSpan.SetAttributes(attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(decodeDuration.Milliseconds())))

	// Calculate end-to-end P/D timing metrics.
	// True TTFT captures time from gateway request start to decode start, including
	// gateway routing, scheduling, prefill, and coordination overhead that
	// per-instance vLLM metrics miss.
	if currentSpan := trace.SpanFromContext(ctx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		coordinatorOverhead := decodeStart.Sub(prefillStart.Add(prefillDuration))

		currentSpan.SetAttributes(
			attribute.Float64("llm_d.pd_proxy.total_duration_ms", float64(totalDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.true_ttft_ms", float64(trueTTFT.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.prefill_duration_ms", float64(prefillDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.decode_duration_ms", float64(decodeDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.coordinator_overhead_ms", float64(coordinatorOverhead.Milliseconds())),
		)
	}
}

// runNIXLProtocolV2WriteParallel runs the concurrent-dispatch path for
// MoRI-IO WRITE mode.  Both the prefill and decode HTTP requests are
// built up-front (decode's kv_transfer_params is synthesised from sidecar
// config + the EPP-supplied prefillPodHostPort rather than being read out
// of prefill's response) and the two upstream calls fire in parallel
// goroutines.  This lets the decode-side MoRIIOConnector.
// update_state_after_alloc / block allocation overlap with prefill's
// forward pass instead of waiting for prefill to flush its response body
// (the strictly-serial behaviour in runNIXLProtocolV2 above).
//
// `original` is the unmodified request body the gateway sent us; `completionRequest`
// is its parsed map (caller retains shared ownership -- this function only
// mutates a *local copy* via Go's by-value copy semantics for maps would NOT
// suffice, so we explicitly snapshot the relevant scalar fields before
// emitting two divergent bodies).
func (s *Server) runNIXLProtocolV2WriteParallel(
	w http.ResponseWriter, r *http.Request, original []byte,
	completionRequest map[string]any, uuidStr, transferID, prefillPodHostPort string,
) {
	s.logger.V(4).Info("running NIXL protocol V2 (concurrent dispatch)",
		"url", prefillPodHostPort, "request_id", uuidStr)

	tracer := telemetry.Tracer()
	parentCtx := r.Context()
	requestStartedAt := time.Now()

	// Snapshot client-supplied fields *before* we mutate completionRequest
	// to build the prefill body.  These need to be put back when constructing
	// the decode body.
	streamValue, streamOk := completionRequest[requestFieldStream]
	streamOptionsValue, streamOptionsOk := completionRequest[requestFieldStreamOptions]
	maxTokensValue, maxTokensOk := completionRequest[requestFieldMaxTokens]
	maxCompletionTokensValue, maxCompletionTokensOk := completionRequest[requestFieldMaxCompletionTokens]

	// Pin both legs of this disagg pair to the same DP rank.  Computed
	// once and stamped on (a) the prefill kv_transfer_params
	// remote_dp_rank, (b) the decode kv_transfer_params remote_dp_rank,
	// and (c) the X-Data-Parallel-Rank header on both HTTP legs below.
	dpRank := pickDPRank(uuidStr, s.config.MoRIIODPSize)

	// ---------- Build prefill body ----------
	// remote_host points at the decode pod (this sidecar's pod) so
	// prefill can RDMA-Write KV there; notify/handshake/dp ports come
	// from this sidecar's local MoRI-IO config.
	completionRequest[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemoteDecode:       true,
		requestFieldDoRemotePrefill:      false,
		requestFieldRemoteEngineID:       nil,
		requestFieldRemoteBlockIDs:       nil,
		requestFieldRemoteHost:           s.config.MoRIIODecodePodIP,
		requestFieldRemotePort:           nil,
		requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
		requestFieldTransferID:           transferID,
		// Reference-proxy alignment (matches moriio_toy_proxy_server.py):
		// remote_tp_size is set alongside tp_size because the upstream
		// reference proxy passes it on both legs (the field is read by
		// add_new_req via .get("tp_size",1) -- both names default to 1
		// if absent, so duplicating is harmless and keeps wire-shape
		// parity).
		// `remote_dp_size` IS consumed (moriio_connector.py:1093) and gates
		// the per-DP-rank handshake loop; defaulting to 1 is correct for
		// our 2P2D DP=1 deployment but we set it explicitly for forward
		// compat with multi-DP.
		"tp_size":        s.config.MoRIIOTPSize,
		"remote_tp_size": s.config.MoRIIOTPSize,
		// Wide-EP support: with TP=1/DP>1 the decode-side MoRIIOConnector
		// must iterate handshake registration over all DP ranks, so we
		// pass the configured DP world size through here instead of
		// emitting a hardcoded 1.
		"remote_dp_size": s.config.MoRIIODPSize,
	}
	completionRequest[requestFieldStream] = false
	delete(completionRequest, requestFieldStreamOptions)
	completionRequest[requestFieldMaxTokens] = 1
	completionRequest[requestFieldMaxCompletionTokens] = 1

	pbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal P)")
		}
		return
	}

	// ---------- Build decode body ----------
	// Restore the client's streaming flags and max-token caps.
	delete(completionRequest, requestFieldStream)
	if streamOk {
		completionRequest[requestFieldStream] = streamValue
	}
	if streamOptionsOk {
		completionRequest[requestFieldStreamOptions] = streamOptionsValue
	}
	delete(completionRequest, requestFieldMaxTokens)
	if maxTokensOk {
		completionRequest[requestFieldMaxTokens] = maxTokensValue
	}
	delete(completionRequest, requestFieldMaxCompletionTokens)
	if maxCompletionTokensOk {
		completionRequest[requestFieldMaxCompletionTokens] = maxCompletionTokensValue
	}

	// Synthesise decode-leg kv_transfer_params.  In the strictly-serial
	// path these come from the prefill HTTP response
	// (request_finished -> kv_transfer_params):
	//
	//   return delay_free_blocks, dict(
	//     do_remote_prefill=True,           # <-- flipped from False by prefill
	//     do_remote_decode=False,
	//     remote_block_ids=computed_block_ids,
	//     remote_engine_id=self.engine_id,
	//     remote_host=self.host_ip,
	//     remote_port=self.handshake_port,
	//     tp_size=...
	//   )
	//
	// (vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
	// :614-622).  The do_remote_prefill=True flag is critical -- it is what
	// gates the decode-side update_state_after_alloc -> send_notify_block path
	// (moriio_connector.py:396), which sends decode's block addresses to the
	// prefill engine.  Without it, decode silently skips the notify, prefill's
	// WriteTask hangs waiting for it, eventually times out after 300s, and
	// the scheduler state gets corrupted (subsequent requests KeyError).
	//
	// We synthesise the same shape here from sidecar config + EPP's
	// prefillPodHostPort so decode can start its block-allocation handshake
	// immediately (in parallel with prefill's forward pass), instead of
	// waiting for prefill's HTTP response.
	prefillHost, _, splitErr := net.SplitHostPort(prefillPodHostPort)
	if splitErr != nil {
		// EPP didn't give us a host:port; fall back to the raw value as host.
		prefillHost = prefillPodHostPort
	}

	completionRequest[requestFieldKVTransferParams] = map[string]any{
		// CRITICAL: do_remote_prefill MUST be true on the decode leg to gate
		// MoRIIOConnector.update_state_after_alloc -> send_notify_block.
		requestFieldDoRemotePrefill: true,
		requestFieldDoRemoteDecode:  false,
		requestFieldRemoteEngineID:  fmt.Sprintf("%s:%d", prefillHost, s.config.MoRIIOPrefillHandshakePort),
		// remote_block_ids is unused in WRITE mode (decode allocates its own
		// blocks and tells prefill via send_notify_block).  We emit an empty
		// list rather than nil to avoid any iteration-over-None surprises in
		// downstream MoRIIO code paths.
		requestFieldRemoteBlockIDs:       []any{},
		requestFieldRemoteHost:           prefillHost,
		requestFieldRemotePort:           s.config.MoRIIOPrefillHandshakePort,
		requestFieldRemoteNotifyPort:     s.config.MoRIIOPrefillNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIOPrefillHandshakePort,
		requestFieldTransferID:           transferID,
		"tp_size":                        s.config.MoRIIOTPSize,
		// Reference-proxy alignment: see prefill body above for rationale.
		"remote_tp_size": s.config.MoRIIOTPSize,
		// Wide-EP support: see prefill body above for rationale.
		"remote_dp_size": s.config.MoRIIODPSize,
	}

	dbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal D)")
		}
		return
	}

	// ---------- Fire prefill + decode in parallel ----------
	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client (concurrent-dispatch P handler init)")
		}
		return
	}

	// Build cloned requests under separate contexts so each carries its own
	// span lineage and either side can be observed/cancelled independently
	// without affecting the other.
	pCtx, prefillSpan := tracer.Start(parentCtx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.prefill_target", prefillPodHostPort),
		attribute.String("llm_d.pd_proxy.connector", "nixlv2"),
		attribute.Bool("llm_d.pd_proxy.parallel_dispatch", true),
	)
	preq := r.Clone(pCtx)
	preq.Header.Set(requestHeaderRequestID, uuidStr)
	// Pin prefill HTTP leg to DP rank H.  Same dpRank var computed at
	// the top of this function -- both legs use it so vLLM dispatches
	// them to the same ApiServer process / DP rank despite
	// SO_REUSEPORT. Header is harmless when MoRIIODPSize<=1 but we
	// still gate to keep the wire byte-identical to pre-patch for
	// non-Wide-EP setups.
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	preq.Body = io.NopCloser(strings.NewReader(string(pbody)))
	preq.ContentLength = int64(len(pbody))

	dCtx, decodeSpan := tracer.Start(parentCtx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()
	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.connector", "nixlv2"),
		attribute.Bool("llm_d.pd_proxy.parallel_dispatch", true),
	)
	dreq := r.Clone(dCtx)
	dreq.Header.Set(requestHeaderRequestID, uuidStr)
	// Pin decode HTTP leg to the same DP rank H as prefill.
	if s.config.MoRIIODPSize > 1 {
		dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	dreq.Body = io.NopCloser(strings.NewReader(string(dbody)))
	dreq.ContentLength = int64(len(dbody))

	s.logger.V(5).Info("concurrent-dispatch prefill request body", "body", string(pbody))
	s.logger.V(5).Info("concurrent-dispatch decode request body", "body", string(dbody))

	var wg sync.WaitGroup
	wg.Add(2)

	// Prefill goroutine: response body is discarded; we only observe the
	// status code for telemetry / fallback decisions.
	var prefillStatus int
	var prefillBody string
	prefillStartedAt := time.Now()
	go func() {
		defer wg.Done()
		defer prefillSpan.End()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, preq)
		prefillStatus = pw.statusCode
		prefillBody = pw.buffer.String()
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(time.Since(prefillStartedAt).Milliseconds())),
		)
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
			s.logger.Error(nil, "concurrent-dispatch prefill returned error status",
				"status", pw.statusCode, "request_id", uuidStr, "body", pw.buffer.String())
		}
	}()

	// Decode goroutine: streams directly to the client's ResponseWriter.
	// dataParallelHandler may steal the request and dispatch to another
	// data-parallel replica; preserve that semantics.
	decodeStartedAt := time.Now()
	go func() {
		defer wg.Done()
		dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(w, dreq)
		decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.data_parallel", dataParallelUsed))
		if !dataParallelUsed {
			decodeSpan.SetAttributes(attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host))
			s.decoderProxy.ServeHTTP(w, dreq)
		}
		decodeSpan.SetAttributes(attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(time.Since(decodeStartedAt).Milliseconds())))
	}()

	wg.Wait()

	// If prefill failed *and* decode produced no response, surface the
	// prefill error.  In practice prefill failures typically also cause
	// decode to hang (no KV ever arrives) so the client sees a timeout
	// rather than a clean 502 -- but if decode happens to short-circuit
	// (e.g. dataParallelHandler error), we want to be loud about why.
	if isHTTPError(prefillStatus) {
		s.logger.Info("concurrent-dispatch: prefill failed -- decode may have streamed an error or hung",
			"request_id", uuidStr, "p_status", prefillStatus, "p_body_snippet", truncate(prefillBody, 256))
	}

	// End-to-end timing metrics, mirroring the strictly-serial path.
	if currentSpan := trace.SpanFromContext(parentCtx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		if requestStartValue := parentCtx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
			}
		}
		currentSpan.SetAttributes(
			attribute.Float64("llm_d.pd_proxy.total_duration_ms", float64(totalDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.parallel_window_ms", float64(time.Since(requestStartedAt).Milliseconds())),
			attribute.Bool("llm_d.pd_proxy.parallel_dispatch", true),
		)
	}
	_ = original // kept for signature symmetry with the strictly-serial path
}

// truncate returns s shortened to at most n characters, with an "..." suffix
// if truncation occurred.  Used only for log snippets.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
