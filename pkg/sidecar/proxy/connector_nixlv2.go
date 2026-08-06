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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
)

// tokenLimitMap returns the map holding the token-limit fields: sampling_params
// for the generate API (created if absent), or the request itself otherwise.
// The second return value reports whether an empty sampling_params map was
// synthesized; callers must drop it before dispatching downstream if it stays empty.
func tokenLimitMap(req map[string]any, apiType APIType) (map[string]any, bool) {
	if apiType != APITypeGenerate {
		return req, false
	}
	if sp, ok := req[requestFieldSamplingParams].(map[string]any); ok {
		return sp, false
	}
	sp := map[string]any{}
	req[requestFieldSamplingParams] = sp
	return sp, true
}

func (s *Server) handleNIXLV2(w http.ResponseWriter, r *http.Request, prefillPodHostPort, kvCacheSource string, apiType APIType) {
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

	// Parallel-dispatch path synthesises decode's kv_transfer_params from config
	// instead of the prefill response. The serial path below is unchanged when off.
	if s.config.MoRIIOParallelDispatch && s.config.MoRIIOWriteMode {
		// MoRI-IO requires transfer_id to carry the "tx" prefix for message routing.
		transferID := "tx" + uuidStr
		s.runNIXLProtocolV2WriteParallel(w, r, original, completionRequest, uuidStr, transferID, prefillPodHostPort, kvCacheSource)
		return
	}

	// Prefill Stage
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	ctx, prefillSpan := tracer.Start(ctx, "prefill",
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

	// Pin both legs to the same DP rank; the header is skipped for single-DP.
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
	tokenMap, createdSamplingParams := tokenLimitMap(completionRequest, apiType)
	savedTokenValues := make([]savedField, len(tokenLimitFields))
	for i, field := range tokenLimitFields {
		if v, ok := tokenMap[field]; ok {
			savedTokenValues[i] = savedField{field: field, val: v, present: true}
		} else {
			savedTokenValues[i] = savedField{field: field}
		}
	}

	// WRITE mode populates the destination fields the prefill engine needs for
	// its RDMA Write; READ mode leaves them nil per the standard NIXLv2 contract.
	if s.config.MoRIIOWriteMode {
		// MoRI-IO requires transfer_id to carry the "tx" prefix for message routing.
		transferID := "tx" + uuidStr
		completionRequest[requestFieldKVTransferParams] = map[string]any{
			requestFieldDoRemoteDecode:       true,
			requestFieldDoRemotePrefill:      false,
			requestFieldRemoteEngineID:       nil,
			requestFieldRemoteBlockIDs:       nil,
			requestFieldRemoteHost:           s.currentDecodePodIP(ctx),
			requestFieldRemotePort:           nil,
			requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
			requestFieldRemoteDPRank:         dpRank,
			requestFieldRemoteDPRankOverride: true,
			requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
			requestFieldTransferID:           transferID,
			"tp_size":                        s.config.MoRIIOTPSize,
			"remote_dp_size":                 s.config.MoRIIODPSize,
		}
		// Wide-EP fan-out (prefill leg, serial path): remote_hosts must be the
		// DECODE-side pod IPs so prefill handshakes the right pods. Re-resolved
		// per request so peer restarts (new IP) are picked up within the TTL.
		if decodeHosts := s.currentDecodeHosts(ctx); len(decodeHosts) > 0 {
			pkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
			hosts := make([]any, len(decodeHosts))
			for i, h := range decodeHosts {
				hosts[i] = h
			}
			pkv["remote_hosts"] = hosts
			if s.config.MoRIIODPSizeLocal > 0 {
				pkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
			}
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

	// Compose the OffloadingConnector p2p pull onto the NIXL prefill leg.
	s.addP2PPullToPrefill(completionRequest[requestFieldKVTransferParams].(map[string]any), kvCacheSource, prefillPodHostPort)

	completionRequest[requestFieldStream] = false
	delete(completionRequest, requestFieldStreamOptions)

	for _, field := range tokenLimitFields {
		tokenMap[field] = 1
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

	// Retry on transient 5xx (502/503/504): these failures (e.g. connection
	// reset → 502) are common when the prefill pod's accept queue overflows
	// under load. Retrying the same host avoids expensive local prefill on
	// decode. Non-transient errors (500/501) fail immediately.
	var pw *bufferedResponseWriter
retryLoop:
	for attempt := 0; ; attempt++ {
		pw = &bufferedResponseWriter{}
		preq.Body = io.NopCloser(bytes.NewReader(pbody))
		preq.ContentLength = int64(len(pbody))
		prefillHandler.ServeHTTP(pw, preq)

		if !isHTTPError(pw.statusCode) {
			break
		}
		if !isRetryableStatus(pw.statusCode) {
			break
		}
		if attempt >= s.config.PrefillMaxRetries {
			break
		}

		s.logger.Info("retrying prefill request",
			"attempt", attempt+1,
			"target", prefillPodHostPort,
			"request_id", uuidStr,
			"previous_code", pw.statusCode)

		select {
		case <-time.After(s.config.PrefillRetryBackoff):
		case <-preq.Context().Done():
			break retryLoop
		}
	}

	prefillDuration := time.Since(prefillStart)
	prefillSpan.SetAttributes(
		attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
		attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
	)

	if isHTTPError(pw.statusCode) {
		s.logger.Error(fmt.Errorf("prefill returned %d", pw.statusCode), "prefill request failed",
			"request_id", uuidStr,
			"body", pw.buffer.String())
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		prefillSpan.End()

		for key, values := range pw.Header() {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
		w.WriteHeader(pw.statusCode)
		if _, writeErr := w.Write(pw.bodyBytes()); writeErr != nil {
			s.logger.Error(writeErr, "failed to send error response to client")
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

	ctx, decodeSpan := tracer.Start(ctx, "decode",
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

	// Decode's DP rank is PROPAGATED from the prefill leg's returned
	// kv_transfer_params (remote_dp_rank = the rank prefill actually ran on),
	// not independently re-derived here. This is the router-applies-the-
	// connector-returned-rank model (PR #45043 review, njhill): the prefill
	// connector returns the rank, the router pins the decode leg to it, so both
	// legs agree without each hashing the request id. The returned rank is
	// validated to be in [0, dp_size); an omitted, non-numeric, or out-of-range
	// value falls back to the deterministic hash. The header AND the decode
	// body's remote_dp_rank are then pinned to the SAME validated value so they
	// cannot target different ranks and hang the transfer.
	// DP-rank propagation is a MoRI-IO WRITE-mode concern only. In standard
	// NIXLv2 READ mode the decode body's remote_dp_rank / remote_dp_rank_override
	// and the x-data-parallel-rank header are left untouched, matching the legacy
	// wire shape.
	if s.config.MoRIIOWriteMode {
		decodeDPRank, usedReturned := resolveDecodeDPRank(pKVTransferParams, uuidStr, s.config.MoRIIODPSize)
		if pkv, ok := pKVTransferParams.(map[string]any); ok {
			if rv, present := pkv[requestFieldRemoteDPRank]; present && !usedReturned && s.config.MoRIIODPSize > 1 {
				s.logger.V(1).Info("prefill returned invalid/out-of-range remote_dp_rank; using hash fallback",
					"request_id", uuidStr, "returned", rv,
					"dp_size", s.config.MoRIIODPSize, "rank", decodeDPRank)
			}
			pkv[requestFieldRemoteDPRank] = decodeDPRank
			pkv[requestFieldRemoteDPRankOverride] = true
		}
		if s.config.MoRIIODPSize > 1 {
			dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(decodeDPRank))
		}
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

	for i := range savedTokenValues {
		sv := &savedTokenValues[i]
		delete(tokenMap, sv.field)
		if sv.present {
			tokenMap[sv.field] = sv.val
		}
	}
	// Drop the sampling_params map synthesized for prefill capping if it ended up
	// empty, so the decode request matches the caller's original (which omitted it).
	if createdSamplingParams && len(tokenMap) == 0 {
		delete(completionRequest, requestFieldSamplingParams)
	}

	// WRITE mode: backfill the decode-side kv_transfer_params fields that
	// vLLM's request_finished response does not echo back, sourcing the
	// pod-local values from sidecar config.
	if s.config.MoRIIOWriteMode {
		if dKVParams, ok := pKVTransferParams.(map[string]any); ok {
			if _, present := dKVParams[requestFieldTransferID]; !present {
				// MoRI-IO requires transfer_id to carry the "tx" prefix.
				dKVParams[requestFieldTransferID] = "tx" + uuidStr
			}
			if _, present := dKVParams[requestFieldRemoteNotifyPort]; !present {
				dKVParams[requestFieldRemoteNotifyPort] = s.config.MoRIIODecodeNotifyPort
			}
			if _, present := dKVParams[requestFieldRemoteDPRank]; !present {
				dKVParams[requestFieldRemoteDPRank] = pickDPRank(uuidStr, s.config.MoRIIODPSize)
				dKVParams[requestFieldRemoteDPRankOverride] = true
			}
			if _, present := dKVParams[requestFieldRemoteHandshakePort]; !present {
				dKVParams[requestFieldRemoteHandshakePort] = s.config.MoRIIODecodeHandshakePort
			}
			// Wide-EP fields for decode-side handshake loop
			if _, present := dKVParams["remote_dp_size"]; !present {
				dKVParams["remote_dp_size"] = s.config.MoRIIODPSize
			}
			// Wide-EP fan-out (decode leg, serial path): remote_hosts must be the
			// PREFILL-side pod IPs so decode fans out handshakes across prefill pods.
			// Re-resolved per request so peer restarts (new IP) are picked up.
			if remoteHosts := s.currentRemoteHosts(ctx); len(remoteHosts) > 0 {
				if _, present := dKVParams["remote_hosts"]; !present {
					hosts := make([]any, len(remoteHosts))
					for i, h := range remoteHosts {
						hosts[i] = h
					}
					dKVParams["remote_hosts"] = hosts
				}
				if s.config.MoRIIODPSizeLocal > 0 {
					if _, present := dKVParams["remote_dp_size_local"]; !present {
						dKVParams["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
					}
				}
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

// runNIXLProtocolV2WriteParallel is the MoRI-IO WRITE-mode concurrent-dispatch
// path: it builds both the prefill and decode bodies up front (synthesising
// decode's kv_transfer_params from config and prefillPodHostPort) and fires the
// two upstream calls in parallel so decode's block allocation overlaps prefill.
func (s *Server) runNIXLProtocolV2WriteParallel(
	w http.ResponseWriter, r *http.Request, original []byte,
	completionRequest map[string]any, uuidStr, transferID, prefillPodHostPort, kvCacheSource string,
) {
	s.logger.V(4).Info("running NIXL protocol V2 (concurrent dispatch)",
		"url", prefillPodHostPort, "request_id", uuidStr)

	tracer := tracing.Tracer()
	parentCtx := r.Context()
	requestStartedAt := time.Now()

	// Snapshot client fields before mutating completionRequest into the prefill
	// body; they are restored when building the decode body.
	streamValue, streamOk := completionRequest[requestFieldStream]
	streamOptionsValue, streamOptionsOk := completionRequest[requestFieldStreamOptions]
	maxTokensValue, maxTokensOk := completionRequest[requestFieldMaxTokens]
	maxCompletionTokensValue, maxCompletionTokensOk := completionRequest[requestFieldMaxCompletionTokens]
	maxOutputTokensValue, maxOutputTokensOk := completionRequest[requestFieldMaxOutputTokens]
	minTokensValue, minTokensOk := completionRequest[requestFieldMinTokens]

	// Pin both legs to the same DP rank (kv_transfer_params + HTTP header).
	dpRank := pickDPRank(uuidStr, s.config.MoRIIODPSize)

	// Build prefill body. remote_host points at the decode pod so prefill can
	// RDMA-Write KV there; remote_dp_size gates the decode-side per-DP-rank
	// handshake loop for Wide-EP.
	completionRequest[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemoteDecode:       true,
		requestFieldDoRemotePrefill:      false,
		requestFieldRemoteEngineID:       nil,
		requestFieldRemoteBlockIDs:       nil,
		requestFieldRemoteHost:           s.currentDecodePodIP(parentCtx),
		requestFieldRemotePort:           nil,
		requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
		requestFieldTransferID:           transferID,
		"tp_size":                        s.config.MoRIIOTPSize,
		"remote_dp_size":                 s.config.MoRIIODPSize,
	}
	// Wide-EP fan-out (prefill leg): remote_hosts must be the DECODE-side pod
	// IPs so prefill handshakes the right pods. Omitted when unset, falling back
	// to the single-host remote_host path. Re-resolved per request so peer
	// restarts (new IP) are picked up within the TTL.
	if decodeHosts := s.currentDecodeHosts(parentCtx); len(decodeHosts) > 0 {
		pkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
		hosts := make([]any, len(decodeHosts))
		for i, h := range decodeHosts {
			hosts[i] = h
		}
		pkv["remote_hosts"] = hosts
		if s.config.MoRIIODPSizeLocal > 0 {
			pkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
	}
	// Compose the OffloadingConnector p2p pull onto the NIXL prefill leg.
	s.addP2PPullToPrefill(completionRequest[requestFieldKVTransferParams].(map[string]any), kvCacheSource, prefillPodHostPort)

	completionRequest[requestFieldStream] = false
	delete(completionRequest, requestFieldStreamOptions)
	completionRequest[requestFieldMaxTokens] = 1
	completionRequest[requestFieldMaxCompletionTokens] = 1
	completionRequest[requestFieldMaxOutputTokens] = 1
	completionRequest[requestFieldMinTokens] = 1

	pbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal P)")
		}
		return
	}

	// ---------- Build decode body ----------
	// Restore the client's streaming flags and token-limit fields.
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
	delete(completionRequest, requestFieldMaxOutputTokens)
	if maxOutputTokensOk {
		completionRequest[requestFieldMaxOutputTokens] = maxOutputTokensValue
	}
	delete(completionRequest, requestFieldMinTokens)
	if minTokensOk {
		completionRequest[requestFieldMinTokens] = minTokensValue
	}

	// Synthesise decode-leg kv_transfer_params that the serial path would
	// otherwise read from the prefill response. do_remote_prefill must be true:
	// it gates the decode-side send_notify_block that prefill's RDMA Write waits on.
	prefillHost, _, splitErr := net.SplitHostPort(prefillPodHostPort)
	if splitErr != nil {
		prefillHost = prefillPodHostPort
	}

	completionRequest[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemotePrefill: true,
		requestFieldDoRemoteDecode:  false,
		requestFieldRemoteEngineID:  fmt.Sprintf("%s:%d", prefillHost, s.config.MoRIIOPrefillHandshakePort),
		// Empty (not nil) since decode allocates its own blocks in WRITE mode.
		requestFieldRemoteBlockIDs:       []any{},
		requestFieldRemoteHost:           prefillHost,
		requestFieldRemotePort:           s.config.MoRIIOPrefillHandshakePort,
		requestFieldRemoteNotifyPort:     s.config.MoRIIOPrefillNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIOPrefillHandshakePort,
		requestFieldTransferID:           transferID,
		"tp_size":                        s.config.MoRIIOTPSize,
		"remote_dp_size":                 s.config.MoRIIODPSize,
	}
	// Wide-EP fan-out (decode leg): the opposite host list, the PREFILL-side
	// pod IPs. A multi-pod deployment must set both host flags. Re-resolved per
	// request so peer restarts (new IP) are picked up within the TTL.
	if remoteHosts := s.currentRemoteHosts(parentCtx); len(remoteHosts) > 0 {
		dkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
		hosts := make([]any, len(remoteHosts))
		for i, h := range remoteHosts {
			hosts[i] = h
		}
		dkv["remote_hosts"] = hosts
		if s.config.MoRIIODPSizeLocal > 0 {
			dkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
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

	// Fire prefill and decode concurrently, but DEFER committing decode's
	// response to the client until prefill's outcome is known (the commit
	// point). Both legs share a cancelable context so a failed or hung prefill
	// aborts decode's in-flight request / KV wait immediately instead of
	// letting it hang, and decode's output is buffered so a failed prefill can
	// never surface a bogus 200 to the client.
	dispatchCtx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	pCtx, prefillSpan := tracer.Start(dispatchCtx, "llm_d.pd_proxy.prefill",
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
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	preq.Body = io.NopCloser(bytes.NewReader(pbody))
	preq.ContentLength = int64(len(pbody))

	dCtx, decodeSpan := tracer.Start(dispatchCtx, "llm_d.pd_proxy.decode",
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
	if s.config.MoRIIODPSize > 1 {
		dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))

	s.logger.V(5).Info("concurrent-dispatch prefill request body", "body", string(pbody))
	s.logger.V(5).Info("concurrent-dispatch decode request body", "body", string(dbody))

	// Decode writes into a deferred writer that buffers everything until we
	// commit() (prefill succeeded -> flush + stream on) or abort() (prefill
	// failed -> discard); it never writes to the client directly.
	dcw := newDeferredCommitWriter(w)

	// Prefill goroutine: body is buffered so it can be returned to the client
	// verbatim on failure. On ANY non-2xx status (transport errors surface as
	// 502 from the reverse proxy) it cancels the shared context so decode's
	// KV wait aborts immediately.
	var prefillResp *bufferedResponseWriter
	prefillDone := make(chan struct{})
	prefillStartedAt := time.Now()
	go func() {
		defer close(prefillDone)
		defer prefillSpan.End()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, preq)
		prefillResp = pw
		prefillSpan.SetAttributes(
			attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
			attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(time.Since(prefillStartedAt).Milliseconds())),
		)
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
			cancel() // KV will never arrive -> abort decode instead of hanging
			s.logger.Error(nil, "concurrent-dispatch prefill returned error status",
				"status", pw.statusCode, "request_id", uuidStr, "body", pw.buffer.String())
		}
	}()

	// Decode goroutine: buffers into dcw. dataParallelHandler may steal the
	// request and dispatch to another data-parallel replica; preserve that
	// semantics but still route through the deferred writer.
	decodeDone := make(chan struct{})
	decodeStartedAt := time.Now()
	go func() {
		defer close(decodeDone)
		dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(dcw, dreq)
		decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.data_parallel", dataParallelUsed))
		if !dataParallelUsed {
			decodeSpan.SetAttributes(attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host))
			s.decoderProxy.ServeHTTP(dcw, dreq)
		}
		decodeSpan.SetAttributes(attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(time.Since(decodeStartedAt).Milliseconds())))
	}()

	// Commit point: wait for prefill's outcome, bounded by a KV-wait backstop
	// so a hung/unreachable prefill cannot make decode wait for KV forever.
	waitTimeout := s.config.MoRIIOParallelDecodeWaitTimeout
	if waitTimeout <= 0 {
		waitTimeout = defaultMoRIIOParallelDecodeWaitTimeout
	}
	timer := time.NewTimer(waitTimeout)
	defer timer.Stop()

	select {
	case <-prefillDone:
		if prefillResp != nil && !isHTTPError(prefillResp.statusCode) {
			// Prefill succeeded: commit decode's buffered response and stream on.
			if !dcw.commit() {
				s.logger.Error(nil, "concurrent-dispatch: decode aborted before prefill-success commit",
					"request_id", uuidStr)
			}
			break
		}
		// Prefill failed: abort decode and return the prefill error verbatim.
		cancel()
		dcw.abort()
		status := http.StatusBadGateway
		if prefillResp != nil {
			status = prefillResp.statusCode
			for key, values := range prefillResp.Header() {
				for _, v := range values {
					w.Header().Add(key, v)
				}
			}
		}
		bodySnippet := ""
		if prefillResp != nil {
			bodySnippet = truncate(prefillResp.buffer.String(), 256)
		}
		s.logger.Info("concurrent-dispatch: prefill failed; returning prefill error and aborting decode",
			"request_id", uuidStr, "p_status", status, "p_body_snippet", bodySnippet)
		w.WriteHeader(status)
		if prefillResp != nil {
			if _, writeErr := w.Write(prefillResp.bodyBytes()); writeErr != nil {
				s.logger.Error(writeErr, "failed to send prefill error to client (concurrent-dispatch)")
			}
		}
	case <-timer.C:
		// Prefill did not resolve within the backstop window: cancel both legs
		// and fail rather than hang waiting for KV that may never arrive.
		cancel()
		dcw.abort()
		s.logger.Error(nil, "concurrent-dispatch: prefill did not complete within KV-wait timeout; aborting",
			"request_id", uuidStr, "timeout", waitTimeout.String())
		w.WriteHeader(http.StatusGatewayTimeout)
		if _, writeErr := w.Write([]byte(`{"error":"decode aborted: prefill did not complete within the MoRI-IO parallel-dispatch KV-wait timeout"}`)); writeErr != nil {
			s.logger.Error(writeErr, "failed to send timeout error to client (concurrent-dispatch)")
		}
		<-prefillDone // let the cancelled prefill goroutine finish to avoid a leak
	}

	// Wait for decode to finish (streamed on success, or promptly aborted) so
	// we never leak the decode goroutine or its response body.
	<-decodeDone

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

// truncate shortens s to at most n characters, appending "..." if truncated.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
