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
	"net"
	"net/http"
	"strconv"
	"sync"
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
			requestFieldRemoteHost:           s.config.MoRIIODecodePodIP,
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
		// DECODE-side pod IPs so prefill handshakes the right pods.
		if len(s.config.MoRIIODecodeHosts) > 0 {
			pkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
			hosts := make([]any, len(s.config.MoRIIODecodeHosts))
			for i, h := range s.config.MoRIIODecodeHosts {
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

	// Stamp the same DP-rank pin on the decode leg.
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
			if len(s.config.MoRIIORemoteHosts) > 0 {
				if _, present := dKVParams["remote_hosts"]; !present {
					hosts := make([]any, len(s.config.MoRIIORemoteHosts))
					for i, h := range s.config.MoRIIORemoteHosts {
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
		requestFieldRemoteHost:           s.config.MoRIIODecodePodIP,
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
	// to the single-host remote_host path.
	if len(s.config.MoRIIODecodeHosts) > 0 {
		pkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
		hosts := make([]any, len(s.config.MoRIIODecodeHosts))
		for i, h := range s.config.MoRIIODecodeHosts {
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
	// pod IPs. A multi-pod deployment must set both host flags.
	if len(s.config.MoRIIORemoteHosts) > 0 {
		dkv := completionRequest[requestFieldKVTransferParams].(map[string]any)
		hosts := make([]any, len(s.config.MoRIIORemoteHosts))
		for i, h := range s.config.MoRIIORemoteHosts {
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
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	preq.Body = io.NopCloser(bytes.NewReader(pbody))
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
	if s.config.MoRIIODPSize > 1 {
		dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
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

	// A failed prefill usually also hangs decode; log it so the cause is visible.
	if isHTTPError(prefillStatus) {
		s.logger.Info("concurrent-dispatch: prefill failed -- decode may have streamed an error or hung",
			"request_id", uuidStr, "p_status", prefillStatus, "p_body_snippet", truncate(prefillBody, 256))
	}

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
