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
	"maps"
	"net"
	"net/http"
	"slices"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// MoRI-IO WRITE-mode kv_transfer_params fields, populated by the sidecar
// so the prefill engine can push KV to decode via RDMA Write.
const (
	requestFieldRemoteNotifyPort = "remote_notify_port"
	requestFieldRemoteDPRank     = "remote_dp_rank"
	// requestFieldRemoteDPRankOverride tells the decode-side connector to use
	// the sidecar's remote_dp_rank verbatim rather than recomputing its own hash.
	requestFieldRemoteDPRankOverride = "remote_dp_rank_override"
	requestFieldRemoteHandshakePort  = "remote_handshake_port"
)

// requestFieldRemoteRequestID must be present on a NIXL push decode request:
// vLLM reads it without a default.
const requestFieldRemoteRequestID = "remote_request_id"

func newNIXLV2RequestID() (string, error) {
	id, err := uuid.NewUUID()
	if err != nil {
		return "", err
	}
	return id.String(), nil
}

func (s *Server) handleNIXLV2(w http.ResponseWriter, r *http.Request, prefillPodHostPort, kvCacheSource string, apiType reqcommon.APIType) {
	s.logger.V(logging.DEBUG).Info("running NIXL protocol V2", "url", prefillPodHostPort, "api", apiType.String())

	original, body, ok := s.readJSONBody(r, w)
	if !ok {
		return
	}

	// Generate unique request UUID
	uuidStr, err := s.nixlRequestIDFn()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Parallel-dispatch path synthesises decode's kv_transfer_params from config
	// instead of the prefill response. The serial path below is unchanged when off.
	if s.config.MoRIIOParallelDispatch && s.config.MoRIIOWriteMode {
		// MoRI-IO requires transfer_id to carry the "tx" prefix for message routing.
		transferID := "tx" + uuidStr
		s.runNIXLProtocolV2WriteParallel(w, r, original, body, uuidStr, transferID, prefillPodHostPort, kvCacheSource, apiType)
		return
	}

	// A parallel dispatch that hands the request to the serial path below for a
	// retry used the first prefill attempt. The prefill stage starts with it.
	firstAttempt := 0
	prefillStart := time.Now()
	if identity, ok := s.nixlPushParallelIdentity(r.Context(), prefillPodHostPort); ok {
		if !s.runNIXLProtocolV2PushParallel(w, r, body, uuidStr, prefillPodHostPort, kvCacheSource, apiType, identity) {
			return
		}
		recordNIXLPushDispatch(r.Context(), metrics.NIXLPushReasonPrefillRetry)
		firstAttempt = 1
	}

	// Prefill Stage
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	ctx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyPrefillTarget(prefillPodHostPort),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
	)

	// 1. Prepare prefill request
	preq := r.Clone(ctx)

	preq.Header.Add(reqcommon.RequestIDHeaderKey, uuidStr)

	// KV metadata uses global ranks; HTTP dispatch uses pod-local ranks.
	globalDPRank, localDPRank := pickDPRanks(
		uuidStr,
		s.config.MoRIIODPSize,
		s.config.MoRIIODPSizeLocal,
	)
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(localDPRank))
	}

	// Keeps the client's body intact for the decode request below.
	prefillRequest := maps.Clone(body)

	// transfer_id of the current prefill attempt in NIXL push mode, else empty.
	var pushTransferID string

	// WRITE mode populates the destination fields the prefill engine needs for
	// its RDMA Write; READ mode leaves them nil per the standard NIXLv2 contract.
	if s.config.MoRIIOWriteMode {
		// MoRI-IO requires transfer_id to carry the "tx" prefix for message routing.
		transferID := "tx" + uuidStr
		prefillRequest[reqcommon.FieldKVTransferParams] = map[string]any{
			reqcommon.FieldDoRemoteDecode:    true,
			reqcommon.FieldDoRemotePrefill:   false,
			reqcommon.FieldRemoteEngineID:    nil,
			reqcommon.FieldRemoteBlockIDs:    nil,
			reqcommon.FieldRemoteHost:        s.currentDecodePodIP(ctx),
			reqcommon.FieldRemotePort:        nil,
			requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
			requestFieldRemoteDPRank:         globalDPRank,
			requestFieldRemoteDPRankOverride: true,
			requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
			requestFieldTransferID:           transferID,
			"tp_size":                        s.config.MoRIIOTPSize,
			"remote_dp_size":                 s.config.MoRIIODPSize,
		}
		// Wide-EP fan-out (prefill request, serial path): remote_hosts must be the
		// DECODE-side pod IPs so prefill handshakes the right pods. Re-resolved
		// per request so peer restarts (new IP) are picked up within the TTL.
		if decodeHosts := s.currentDecodeHosts(ctx); len(decodeHosts) > 0 {
			pkv := prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any)
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
		prefillRequest[reqcommon.FieldKVTransferParams] = map[string]any{
			reqcommon.FieldDoRemoteDecode:  true,
			reqcommon.FieldDoRemotePrefill: false,
			reqcommon.FieldRemoteEngineID:  nil,
			reqcommon.FieldRemoteBlockIDs:  nil,
			reqcommon.FieldRemoteHost:      nil,
			reqcommon.FieldRemotePort:      nil,
		}
		if s.config.NIXLPushMode {
			pushTransferID = newTransferID()
			prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any)[requestFieldTransferID] = pushTransferID
		}
	}

	// Compose the OffloadingConnector p2p pull onto the NIXL prefill request.
	s.addP2PPullToPrefill(prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any), kvCacheSource, prefillPodHostPort)

	reqcommon.CapSingleToken(prefillRequest, apiType)

	pbody, err := json.Marshal(prefillRequest)
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
	s.logger.V(logging.DEBUG).Info("sending prefill request", "to", prefillPodHostPort)
	// Guarded: stringifying the body allocates a copy per request even when
	// TRACE is disabled.
	if trace := s.logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("Prefill request", logging.HTTPBodyKey, string(pbody))
	}

	// Retry on transient 5xx (502/503/504): these failures (e.g. connection
	// reset → 502) are common when the prefill pod's accept queue overflows
	// under load. Retrying the same host avoids expensive local prefill on
	// decode. Non-transient errors (500/501) fail immediately.
	var pw *bufferedResponseWriter
retryLoop:
	for attempt := firstAttempt; ; attempt++ {
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

		// A failed attempt may still finish on P, so each attempt gets its own
		// transfer_id and D pairs only with the attempt that succeeded.
		if pushTransferID != "" {
			pushTransferID = newTransferID()
			prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any)[requestFieldTransferID] = pushTransferID
			if pbody, err = json.Marshal(prefillRequest); err != nil {
				if err := errorJSONInvalid(err, w); err != nil {
					s.logger.Error(err, "failed to send error response to client")
				}
				return
			}
		}
	}

	prefillDuration := time.Since(prefillStart)
	metrics.RecordPrefillDuration(prefillDuration)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
		semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
	)

	if isHTTPError(pw.statusCode) {
		metrics.RecordError(metrics.StagePrefill)
		s.logger.Error(fmt.Errorf("prefill returned %d", pw.statusCode), "prefill request failed",
			"request_id", uuidStr,
			logging.HTTPBodyKey, pw.buffer.String())
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

	if s.config.NIXLPushMode {
		s.storeNIXLPushIdentity(prefillPodHostPort, prefillerResponse[reqcommon.FieldKVTransferParams])
	}

	s.runNIXLProtocolV2Decode(ctx, w, r, body, prefillerResponse, uuidStr, prefillPodHostPort, localDPRank,
		pushTransferID, prefillStart, prefillDuration)
}

// runNIXLProtocolV2Decode is the decode stage of a NIXL v2 dispatch. It sends
// body to the decoder with the kv_transfer_params of prefillerResponse, the
// parsed prefill response, and writes the decoder's response to w. A non-empty
// pushTransferID is added to those kv_transfer_params. prefillPodHostPort and
// localDPRank are used only in MoRI-IO WRITE mode.
func (s *Server) runNIXLProtocolV2Decode(
	ctx context.Context, w http.ResponseWriter, r *http.Request,
	body, prefillerResponse map[string]any,
	uuidStr, prefillPodHostPort string, localDPRank int,
	pushTransferID string, prefillStart time.Time, prefillDuration time.Duration,
) {
	// 3. Verify response

	pKVTransferParams, ok := prefillerResponse[reqcommon.FieldKVTransferParams]
	if !ok {
		s.logger.Info("warning: missing 'kv_transfer_params' field in prefiller response")
	}
	pCachedTokens, hasPCachedTokens := extractCachedTokens(prefillerResponse)
	if !hasPCachedTokens {
		// vLLM returns prompt_tokens_details as null when cached_tokens is 0,
		// so treat a missing prefiller cached_tokens value as zero.
		pCachedTokens = 0
	}

	s.logger.V(logging.TRACE).Info("received prefiller response",
		reqcommon.FieldKVTransferParams, pKVTransferParams,
		"cachedTokens", pCachedTokens,
		"hasCachedTokens", hasPCachedTokens)

	// Decode Stage

	ctx, decodeSpan := tracing.Tracer(tracerScope).Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
	)
	decodeStart := time.Now()

	// 1. Prepare decode request
	dreq := r.Clone(ctx)

	dreq.Header.Add(reqcommon.RequestIDHeaderKey, uuidStr)

	// Preserve prefill's global rank for notify routing and dispatch decode to
	// its pod-local equivalent. Fallback ranks use the selected prefill pod.
	if s.config.MoRIIOWriteMode {
		decodeDPRank, usedReturned := resolveDecodeDPRank(pKVTransferParams, uuidStr, s.config.MoRIIODPSize)
		if pkv, ok := pKVTransferParams.(map[string]any); ok {
			if !usedReturned && s.config.MoRIIODPSizeLocal > 0 && s.config.MoRIIODPSize > s.config.MoRIIODPSizeLocal {
				prefillHost := s.resolver().resolveOne(ctx, extractHost(prefillPodHostPort))
				podIndex := slices.Index(s.currentRemoteHosts(ctx), prefillHost)
				if podIndex < 0 {
					err := fmt.Errorf("cannot determine prefill pod index for %q", prefillHost)
					s.logger.Error(err, "failed to route MoRI-IO decode notify", "request_id", uuidStr)
					if err := errorBadGateway(err, w); err != nil {
						s.logger.Error(err, "failed to send error response to client")
					}
					return
				}
				decodeDPRank = podIndex*s.config.MoRIIODPSizeLocal + localDPRank
			}
			if rv, present := pkv[requestFieldRemoteDPRank]; present && !usedReturned && s.config.MoRIIODPSize > 1 {
				s.logger.Info("prefill returned invalid/out-of-range remote_dp_rank; using fallback DP rank",
					"request_id", uuidStr, "returned", rv,
					"dp_size", s.config.MoRIIODPSize, "rank", decodeDPRank)
			}
			pkv[requestFieldRemoteDPRank] = decodeDPRank
			pkv[requestFieldRemoteDPRankOverride] = true
			if s.config.MoRIIODPSize > 1 {
				// Decode can execute at a different global rank from prefill.
				pkv["is_request_leader"] = true
			}
		}
		if s.config.MoRIIODPSize > 1 {
			decodeLocalDPRank := foldDPRankToLocal(
				decodeDPRank,
				s.config.MoRIIODPSize,
				s.config.MoRIIODPSizeLocal,
			)
			dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(decodeLocalDPRank))
		}
	}

	streamingEnabled, _ := body[reqcommon.FieldStream].(bool)
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeStreaming(streamingEnabled))

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
			// Wide-EP fan-out (decode request, serial path): remote_hosts must be the
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
	// P's response carries no transfer_id.
	if dKVParams, ok := pKVTransferParams.(map[string]any); ok && pushTransferID != "" {
		dKVParams[requestFieldTransferID] = pushTransferID
	}
	body[reqcommon.FieldKVTransferParams] = pKVTransferParams

	dbody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))

	// 2. Forward to local decoder.

	if trace := s.logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("sending request to decoder", logging.HTTPBodyKey, string(dbody))
	}
	statusWriter, decodeStatus := captureResponseStatus(w)
	decodeWriter, finalizeDecodeWriter := newCachedTokensResponseWriterWithFinalize(statusWriter, pCachedTokens, streamingEnabled)
	decodeReturned := false
	defer recordDecodeAbort(&decodeReturned, decodeStart)
	dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(decodeWriter, dreq)
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDataParallel(dataParallelUsed))

	if !dataParallelUsed {
		s.logger.V(logging.DEBUG).Info("sending request to decoder", "to", s.config.DecoderURL.Host)
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
		s.dispatchDecode(decodeWriter, dreq, body)
	}
	decodeReturned = true
	if err := finalizeDecodeWriter(); err != nil {
		metrics.RecordDecodeDuration(time.Since(decodeStart))
		metrics.RecordError(metrics.StageDecode)
		s.logger.Error(err, "failed to flush cached token response writer")
		decodeSpan.SetStatus(codes.Error, "failed to flush cached token response writer")
		return
	}

	decodeDuration := time.Since(decodeStart)
	metrics.RecordDecodeDuration(decodeDuration)
	if decodeStatus.failed() {
		metrics.RecordError(metrics.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())))

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
			semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
			semconv.LLMDPDProxyTrueTTFTMs(float64(trueTTFT.Milliseconds())),
			semconv.LLMDPDProxyPrefillDurationMsSummary(float64(prefillDuration.Milliseconds())),
			semconv.LLMDPDProxyDecodeDurationMsSummary(float64(decodeDuration.Milliseconds())),
			semconv.LLMDPDProxyCoordinatorOverheadMs(float64(coordinatorOverhead.Milliseconds())),
		)
	}
}

// runNIXLProtocolV2WriteParallel is the MoRI-IO WRITE-mode concurrent-dispatch
// path: it builds both the prefill and decode bodies up front (synthesising
// decode's kv_transfer_params from config and prefillPodHostPort) and fires the
// two upstream calls in parallel so decode's block allocation overlaps prefill.
func (s *Server) runNIXLProtocolV2WriteParallel(
	w http.ResponseWriter, r *http.Request, original []byte,
	body map[string]any, uuidStr, transferID, prefillPodHostPort, kvCacheSource string,
	apiType reqcommon.APIType,
) {
	s.logger.V(logging.DEBUG).Info("running NIXL protocol V2 (concurrent dispatch)",
		"url", prefillPodHostPort, "request_id", uuidStr)

	tracer := tracing.Tracer()
	parentCtx := r.Context()
	requestStartedAt := time.Now()

	// Keeps the client's body intact for the decode request built below.
	prefillRequest := maps.Clone(body)

	// Pin both requests to the same DP rank (kv_transfer_params + HTTP header).
	// The header and remote_dp_rank must be in [0, dp_size_local).
	dpLocal := s.config.MoRIIODPSizeLocal
	if dpLocal <= 0 {
		dpLocal = s.config.MoRIIODPSize
	}
	if dpLocal <= 0 {
		dpLocal = 1
	}
	_, dpRank := pickDPRanks(
		uuidStr,
		s.config.MoRIIODPSize,
		s.config.MoRIIODPSizeLocal,
	)

	decodePodIP := s.currentDecodePodIP(parentCtx)
	decodeHosts := s.currentDecodeHosts(parentCtx)

	// Build prefill body. remote_host points at the decode pod so prefill can
	// RDMA-Write KV there; remote_dp_size stays global, gating the decode-side
	// per-DP-rank handshake loop for Wide-EP.
	prefillRequest[reqcommon.FieldKVTransferParams] = map[string]any{
		reqcommon.FieldDoRemoteDecode:    true,
		reqcommon.FieldDoRemotePrefill:   false,
		reqcommon.FieldRemoteEngineID:    nil,
		reqcommon.FieldRemoteBlockIDs:    nil,
		reqcommon.FieldRemoteHost:        decodePodIP,
		reqcommon.FieldRemotePort:        nil,
		requestFieldRemoteNotifyPort:     s.config.MoRIIODecodeNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIODecodeHandshakePort,
		requestFieldTransferID:           transferID,
		"tp_size":                        s.config.MoRIIOTPSize,
		"remote_dp_size":                 s.config.MoRIIODPSize,
	}
	// Wide-EP fan-out (prefill request): remote_hosts must be the DECODE-side pod
	// IPs so prefill handshakes the right pods. Omitted when unset, falling back
	// to the single-host remote_host path. Re-resolved per request so peer
	// restarts (new IP) are picked up within the TTL.
	if len(decodeHosts) > 0 {
		pkv := prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any)
		hosts := make([]any, len(decodeHosts))
		for i, h := range decodeHosts {
			hosts[i] = h
		}
		pkv["remote_hosts"] = hosts
		if s.config.MoRIIODPSizeLocal > 0 {
			pkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
	}
	// Compose the OffloadingConnector p2p pull onto the NIXL prefill request.
	s.addP2PPullToPrefill(prefillRequest[reqcommon.FieldKVTransferParams].(map[string]any), kvCacheSource, prefillPodHostPort)

	reqcommon.CapSingleToken(prefillRequest, apiType)

	pbody, err := json.Marshal(prefillRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal P)")
		}
		return
	}

	// ---------- Build decode body ----------
	// Synthesise decode-request kv_transfer_params that the serial path would
	// otherwise read from the prefill response. do_remote_prefill must be true:
	// it gates the decode-side send_notify_block that prefill's RDMA Write waits on.
	prefillHost, _, splitErr := net.SplitHostPort(prefillPodHostPort)
	if splitErr != nil {
		prefillHost = prefillPodHostPort
	}
	remoteHosts := s.currentRemoteHosts(parentCtx)

	// Decode: one prefill host; leader bit for follower global ranks.
	body[reqcommon.FieldKVTransferParams] = map[string]any{
		reqcommon.FieldDoRemotePrefill: true,
		reqcommon.FieldDoRemoteDecode:  false,
		reqcommon.FieldRemoteEngineID:  net.JoinHostPort(prefillHost, strconv.Itoa(s.config.MoRIIOPrefillHandshakePort)),
		// Empty (not nil) since decode allocates its own blocks in WRITE mode.
		reqcommon.FieldRemoteBlockIDs:    []any{},
		reqcommon.FieldRemoteHost:        prefillHost,
		reqcommon.FieldRemotePort:        s.config.MoRIIOPrefillHandshakePort,
		requestFieldRemoteNotifyPort:     s.config.MoRIIOPrefillNotifyPort,
		requestFieldRemoteDPRank:         dpRank,
		requestFieldRemoteDPRankOverride: true,
		requestFieldRemoteHandshakePort:  s.config.MoRIIOPrefillHandshakePort,
		requestFieldTransferID:           transferID,
		"tp_size":                        s.config.MoRIIOTPSize,
		"remote_dp_size":                 dpLocal,
		"is_request_leader":              true,
	}
	// Wide-EP fan-out (decode request): the opposite host list, the PREFILL-side
	// pod IPs. A multi-pod deployment must set both host flags. Re-resolved per
	// request so peer restarts (new IP) are picked up within the TTL.
	if len(remoteHosts) > 0 {
		dkv := body[reqcommon.FieldKVTransferParams].(map[string]any)
		dkv["remote_hosts"] = []any{prefillHost}
		if s.config.MoRIIODPSizeLocal > 0 {
			dkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
	}

	dbody, err := json.Marshal(body)
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
	// point). Both requests share a cancelable context so a failed or hung prefill
	// aborts decode's in-flight request / KV wait immediately instead of
	// letting it hang, and decode's output is buffered so a failed prefill can
	// never surface a bogus 200 to the client.
	dispatchCtx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	pCtx, prefillSpan := tracer.Start(dispatchCtx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyPrefillTarget(prefillPodHostPort),
		semconv.LLMDPDProxyConnector("nixlv2"),
		semconv.LLMDPDProxyParallelDispatch(true),
	)
	preq := r.Clone(pCtx)
	preq.Header.Set(reqcommon.RequestIDHeaderKey, uuidStr)
	if s.config.MoRIIODPSize > 1 {
		preq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	preq.Body = io.NopCloser(bytes.NewReader(pbody))
	preq.ContentLength = int64(len(pbody))

	dCtx, decodeSpan := tracer.Start(dispatchCtx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyConnector("nixlv2"),
		semconv.LLMDPDProxyParallelDispatch(true),
	)
	dreq := r.Clone(dCtx)
	dreq.Header.Set(reqcommon.RequestIDHeaderKey, uuidStr)
	if s.config.MoRIIODPSize > 1 {
		dreq.Header.Set(requestHeaderDataParallelRank, strconv.Itoa(dpRank))
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))

	if trace := s.logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("concurrent-dispatch prefill request body", logging.HTTPBodyKey, string(pbody))
		trace.Info("concurrent-dispatch decode request body", logging.HTTPBodyKey, string(dbody))
	}

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
		// ErrAbortHandler is only recovered on the request goroutine.
		defer func() {
			if rec := recover(); rec != nil {
				if rec != http.ErrAbortHandler {
					panic(rec)
				}
				prefillSpan.SetStatus(codes.Error, "prefill handler aborted")
				cancel()
				s.logger.Error(nil, "concurrent-dispatch prefill handler aborted",
					"request_id", uuidStr)
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, preq)
		prefillResp = pw
		prefillDuration := time.Since(prefillStartedAt)
		metrics.RecordPrefillDuration(prefillDuration)
		prefillSpan.SetAttributes(
			semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
			semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
		)
		if isHTTPError(pw.statusCode) {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
			cancel() // KV will never arrive -> abort decode instead of hanging
			s.logger.Error(nil, "concurrent-dispatch prefill returned error status",
				"status", pw.statusCode, "request_id", uuidStr, logging.HTTPBodyKey, pw.buffer.String())
		}
	}()

	// Decode goroutine: buffers into dcw. dataParallelHandler may steal the
	// request and dispatch to another data-parallel replica; preserve that
	// semantics but still route through the deferred writer.
	decodeDone := make(chan struct{})
	decodeStartedAt := time.Now()
	// Swallowing the abort here keeps the process alive but hides the failure
	// from the client, so record it and replay it on the request goroutine.
	var decodeAborted atomic.Bool
	// Written by the decode goroutine, read after decodeDone is closed.
	var decodeDuration time.Duration
	go func() {
		defer close(decodeDone)
		defer func() { decodeDuration = time.Since(decodeStartedAt) }()
		// Same recover: abort this request, do not kill the process.
		defer func() {
			if rec := recover(); rec != nil {
				if rec != http.ErrAbortHandler {
					panic(rec)
				}
				decodeSpan.SetStatus(codes.Error, "decode handler aborted")
				decodeAborted.Store(true)
				dcw.abort()
				s.logger.Error(nil, "concurrent-dispatch decode handler aborted",
					"request_id", uuidStr)
			}
		}()
		dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(dcw, dreq)
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDataParallel(dataParallelUsed))
		if !dataParallelUsed {
			decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
			s.decoderProxy.ServeHTTP(dcw, dreq)
		}
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(time.Since(decodeStartedAt).Milliseconds())))
	}()

	// Commit point: wait for prefill's outcome, bounded by a KV-wait backstop
	// so a hung/unreachable prefill cannot make decode wait for KV forever.
	waitTimeout := s.config.MoRIIOParallelDecodeWaitTimeout
	if waitTimeout <= 0 {
		waitTimeout = defaultMoRIIOParallelDecodeWaitTimeout
	}
	timer := time.NewTimer(waitTimeout)
	defer timer.Stop()

	// Set once this goroutine has written a terminal response of its own, so an
	// aborted decode cannot write a second status over it.
	clientResponded := false
	// Set when the KV-wait timer fired; decode then ran for the full timeout.
	decodeTimedOut := false

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
		metrics.RecordError(metrics.StagePrefill)
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
		clientResponded = true
		w.WriteHeader(status)
		if prefillResp != nil {
			if _, writeErr := w.Write(prefillResp.bodyBytes()); writeErr != nil {
				s.logger.Error(writeErr, "failed to send prefill error to client (concurrent-dispatch)")
			}
		}
	case <-timer.C:
		// Prefill did not resolve within the backstop window: cancel both requests
		// and fail rather than hang waiting for KV that may never arrive.
		cancel()
		dcw.abort()
		// Counted here rather than from prefill's own outcome: the cancelled
		// prefill may still come back 2xx if it completed as the timer fired.
		metrics.RecordError(metrics.StagePrefill)
		decodeTimedOut = true
		s.logger.Error(nil, "concurrent-dispatch: prefill did not complete within KV-wait timeout; aborting",
			"request_id", uuidStr, "timeout", waitTimeout.String())
		clientResponded = true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusGatewayTimeout)
		if _, writeErr := w.Write([]byte(`{"error":"decode aborted: prefill did not complete within the MoRI-IO parallel-dispatch KV-wait timeout"}`)); writeErr != nil {
			s.logger.Error(writeErr, "failed to send timeout error to client (concurrent-dispatch)")
		}
		<-prefillDone // let the cancelled prefill goroutine finish to avoid a leak
	}

	// Wait for decode to finish (streamed on success, or promptly aborted) so
	// we never leak the decode goroutine or its response body.
	<-decodeDone

	// Decode errors are attributed only when the commit point let decode's
	// response reach the client. When the commit point wrote the response
	// itself (prefill error or KV-wait timeout), decode's output was discarded
	// and the failure is counted as a prefill error. Decode duration is also
	// sampled on the KV-wait timeout, where decode waited the full timeout.
	if !clientResponded {
		metrics.RecordDecodeDuration(decodeDuration)
		if decodeAborted.Load() || dcw.failed() {
			metrics.RecordError(metrics.StageDecode)
		}
	} else if decodeTimedOut {
		metrics.RecordDecodeDuration(decodeDuration)
	}

	if currentSpan := trace.SpanFromContext(parentCtx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		if requestStartValue := parentCtx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
			}
		}
		currentSpan.SetAttributes(
			semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
			semconv.LLMDPDProxyParallelWindowMs(float64(time.Since(requestStartedAt).Milliseconds())),
			semconv.LLMDPDProxyParallelDispatch(true),
		)
	}
	_ = original // kept for signature symmetry with the strictly-serial path

	// Replay the decode abort here, on the request goroutine, so it reaches the
	// client the way it does on the strictly-serial path. Skipped when the
	// commit point already wrote the prefill error or the KV-wait timeout,
	// which own the response.
	if decodeAborted.Load() && !clientResponded {
		if dcw.responseStarted() {
			// Decode's response is already on the wire, so the status cannot be
			// changed; net/http recovers this and drops the connection, leaving
			// the client a truncated stream rather than a clean terminator.
			panic(http.ErrAbortHandler)
		}
		// Nothing was relayed, so the response is still ours to write. Returning
		// without one would let net/http synthesise an empty 200.
		if err := errorBadGateway(errDecodeAborted, w); err != nil {
			s.logger.Error(err, "failed to send decode abort error to client (concurrent-dispatch)")
		}
	}
}

// nixlPushParallelIdentity returns the cached NIXL push identity of the
// prefill endpoint when the request can send its prefill and decode requests
// at once. On a cache miss the serial path runs and learns the identity; so
// does a request to an endpoint marked serial-only. In NIXL push mode it
// records the reason for the dispatch mode.
func (s *Server) nixlPushParallelIdentity(ctx context.Context, prefillPodHostPort string) (nixlPushIdentity, bool) {
	if !s.config.NIXLPushMode {
		return nil, false
	}
	if s.nixlPushIdentities.serialOnly(prefillPodHostPort) {
		recordNIXLPushDispatch(ctx, metrics.NIXLPushReasonSerialOnly)
		return nil, false
	}
	identity, ok := s.nixlPushIdentities.get(prefillPodHostPort)
	if !ok {
		recordNIXLPushDispatch(ctx, metrics.NIXLPushReasonCacheMiss)
		return nil, false
	}
	recordNIXLPushDispatch(ctx, metrics.NIXLPushReasonCacheHit)
	return identity, true
}

// recordNIXLPushDispatch counts a NIXL push dispatch and sets its reason on the
// request span, where the reason of a later dispatch of the same request
// replaces it.
func recordNIXLPushDispatch(ctx context.Context, reason string) {
	metrics.RecordNIXLPushDispatch(reason)
	trace.SpanFromContext(ctx).SetAttributes(semconv.LLMDPDProxyNIXLPushDispatchReason(reason))
}

// runNIXLProtocolV2PushParallel is the NIXL push-mode concurrent-dispatch path
// for a prefill endpoint with a cached identity. It builds the decode request's
// kv_transfer_params from identity and sends the prefill and decode requests at
// once, so decode registers its KV blocks while prefill runs. Decode's response
// reaches the client only after a successful prefill response that carries
// identity. A successful prefill response with another identity cancels decode
// and sends it again the way the serial path does. When prefill answers a
// retryable status and --prefill-max-retries allows another attempt, it cancels
// decode and reports true without writing a response; the caller then retries
// on the serial path.
func (s *Server) runNIXLProtocolV2PushParallel(
	w http.ResponseWriter, r *http.Request, body map[string]any,
	uuidStr, prefillPodHostPort, kvCacheSource string,
	apiType reqcommon.APIType, identity nixlPushIdentity,
) bool {
	s.logger.V(logging.DEBUG).Info("running NIXL protocol V2 (NIXL push concurrent dispatch)",
		"url", prefillPodHostPort, "request_id", uuidStr)

	tracer := tracing.Tracer(tracerScope)
	parentCtx := r.Context()
	requestStartedAt := time.Now()
	transferID := newTransferID()

	// Keeps the client's body intact for the decode request built below.
	prefillRequest := maps.Clone(body)
	prefillKVParams := map[string]any{
		reqcommon.FieldDoRemoteDecode:  true,
		reqcommon.FieldDoRemotePrefill: false,
		reqcommon.FieldRemoteEngineID:  nil,
		reqcommon.FieldRemoteBlockIDs:  nil,
		reqcommon.FieldRemoteHost:      nil,
		reqcommon.FieldRemotePort:      nil,
		requestFieldTransferID:         transferID,
	}
	prefillRequest[reqcommon.FieldKVTransferParams] = prefillKVParams
	// Compose the OffloadingConnector p2p pull onto the NIXL prefill request.
	s.addP2PPullToPrefill(prefillKVParams, kvCacheSource, prefillPodHostPort)

	reqcommon.CapSingleToken(prefillRequest, apiType)

	pbody, err := json.Marshal(prefillRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return false
	}

	// Decode sets remote_block_ids itself and reads remote_num_tokens as
	// optional, so neither is sent.
	decodeKVParams := map[string]any(maps.Clone(identity))
	decodeKVParams[reqcommon.FieldDoRemotePrefill] = true
	decodeKVParams[reqcommon.FieldDoRemoteDecode] = false
	decodeKVParams[requestFieldRemoteRequestID] = uuidStr
	decodeKVParams[requestFieldTransferID] = transferID
	body[reqcommon.FieldKVTransferParams] = decodeKVParams

	dbody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return false
	}
	// Chunked decode appends its output to the body it is given, while body is
	// read during decode and reused when decode is sent again or prefill is
	// retried.
	decodeBody := maps.Clone(body)

	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return false
	}

	// One context for both requests: cancelling decode must also cancel
	// prefill, which could otherwise write KV into blocks decode already freed.
	dispatchCtx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	pCtx, prefillSpan := tracer.Start(dispatchCtx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyPrefillTarget(prefillPodHostPort),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
		semconv.LLMDPDProxyParallelDispatch(true),
	)
	preq := cloneRequestWithBody(pCtx, r, pbody)
	preq.Header.Set(reqcommon.RequestIDHeaderKey, uuidStr)

	dCtx, decodeSpan := tracer.Start(dispatchCtx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
		semconv.LLMDPDProxyParallelDispatch(true),
	)
	dreq := cloneRequestWithBody(dCtx, r, dbody)
	dreq.Header.Set(reqcommon.RequestIDHeaderKey, uuidStr)

	if trace := s.logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("concurrent-dispatch prefill request body", logging.HTTPBodyKey, string(pbody))
		trace.Info("concurrent-dispatch decode request body", logging.HTTPBodyKey, string(dbody))
	}

	// Holds decode's response until prefill's outcome is known.
	dcw := newDeferredCommitWriter(w)

	var prefillResp *bufferedResponseWriter
	// Written by the prefill goroutine, read after prefillDone is closed.
	var prefillDuration time.Duration
	prefillDone := make(chan struct{})
	prefillStartedAt := time.Now()
	go func() {
		defer close(prefillDone)
		defer prefillSpan.End()
		// ErrAbortHandler is only recovered on the request goroutine.
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			if rec != http.ErrAbortHandler {
				panic(rec)
			}
			prefillSpan.SetStatus(codes.Error, "prefill handler aborted")
			if pCtx.Err() != nil {
				s.logger.V(logging.DEBUG).Info("concurrent-dispatch prefill handler aborted after the dispatch was cancelled",
					"request_id", uuidStr)
				return
			}
			cancel()
			s.logger.Error(nil, "concurrent-dispatch prefill handler aborted", "request_id", uuidStr)
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, preq)
		prefillResp = pw
		prefillDuration = time.Since(prefillStartedAt)
		prefillSpan.SetAttributes(
			semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
			semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
		)
		if !isHTTPError(pw.statusCode) {
			return
		}
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		// A prefill cancelled with the dispatch fails because of decode, the
		// prefill timeout or the client, which the request goroutine handles.
		if pCtx.Err() != nil {
			s.logger.V(logging.DEBUG).Info("concurrent-dispatch prefill cancelled with the dispatch",
				"status", pw.statusCode, "request_id", uuidStr)
			return
		}
		cancel()
		s.logger.Error(nil, "concurrent-dispatch prefill returned error status",
			"status", pw.statusCode, "request_id", uuidStr, logging.HTTPBodyKey, pw.buffer.String())
	}()

	decodeDone := make(chan struct{})
	// Closed when decode fails before the dispatch is cancelled.
	decodeFailed := make(chan struct{})
	decodeStartedAt := time.Now()
	// Swallowing the abort here keeps the process alive but hides the failure
	// from the client, so record it and replay it on the request goroutine.
	var decodeAborted atomic.Bool
	// Written by the decode goroutine, read after decodeDone is closed.
	var decodeDuration time.Duration
	go func() {
		defer close(decodeDone)
		defer func() { decodeDuration = time.Since(decodeStartedAt) }()
		// A decode cancelled with the dispatch fails because of prefill, the
		// prefill timeout or the client, which the request goroutine handles.
		defer func() {
			if dCtx.Err() == nil && (decodeAborted.Load() || dcw.failed()) {
				close(decodeFailed)
			}
		}()
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			if rec != http.ErrAbortHandler {
				panic(rec)
			}
			decodeSpan.SetStatus(codes.Error, "decode handler aborted")
			decodeAborted.Store(true)
			dcw.abort()
			if dCtx.Err() != nil {
				s.logger.V(logging.DEBUG).Info("concurrent-dispatch decode handler aborted after the dispatch was cancelled",
					"request_id", uuidStr)
				return
			}
			s.logger.Error(nil, "concurrent-dispatch decode handler aborted", "request_id", uuidStr)
		}()
		dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(dcw, dreq)
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDataParallel(dataParallelUsed))
		if !dataParallelUsed {
			decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
			s.dispatchDecode(dcw, dreq, decodeBody)
		}
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(time.Since(decodeStartedAt).Milliseconds())))
	}()

	prefillTimeout := s.config.NIXLPushPrefillTimeout
	if prefillTimeout <= 0 {
		prefillTimeout = defaultNIXLPushPrefillTimeout
	}
	prefillTimer := time.NewTimer(prefillTimeout)
	defer prefillTimer.Stop()

	// Set once this goroutine has written a terminal response of its own, so an
	// aborted decode cannot write a second status over it.
	clientResponded := false
	// Set when the prefill timeout cancelled the dispatch; decode ran until it fired.
	timedOut := false
	// Set when prefill failed or timed out or when decode failed.
	dispatchFailed := false
	// Set to the prefill response when it does not carry identity; decode is
	// then sent again with it.
	var resendWith map[string]any
	// Set when decode's response is committed through a cached-tokens rewriter,
	// which may hold back the end of the response until it is called.
	var finalizeDecodeWriter func() error
	// Set when prefill answered a retryable status and the request is retried
	// on the serial path.
	retrySerially := false

	select {
	case <-prefillDone:
		if prefillResp != nil && !isHTTPError(prefillResp.statusCode) {
			var prefillerResponse map[string]any
			if err := json.Unmarshal(prefillResp.bodyBytes(), &prefillerResponse); err != nil {
				s.logger.Error(err, "concurrent-dispatch: failed to parse prefill response; keeping the cached NIXL push identity",
					"request_id", uuidStr)
			} else if answered, ok := s.storeNIXLPushIdentity(prefillPodHostPort, prefillerResponse[reqcommon.FieldKVTransferParams]); !ok || !answered.equal(identity) {
				resendWith = prefillerResponse
			}
			if resendWith != nil {
				// Decode registered with an engine that did not run this prefill,
				// so no KV is written into its blocks once it is cancelled.
				cancel()
				dcw.abort()
				clientResponded = true
				metrics.RecordNIXLPushIdentityMismatch()
				s.logger.Info("concurrent-dispatch: prefill response does not carry the NIXL push identity decode was given; sending decode again",
					"request_id", uuidStr, "target", prefillPodHostPort)
				break
			}
			// As in the serial decode stage, a prefill response without cached
			// tokens reports zero.
			pCachedTokens, _ := extractCachedTokens(prefillerResponse)
			streamingEnabled, _ := body[reqcommon.FieldStream].(bool)
			var decodeWriter http.ResponseWriter
			decodeWriter, finalizeDecodeWriter = newCachedTokensResponseWriterWithFinalize(w, pCachedTokens, streamingEnabled)
			if !dcw.commitThrough(decodeWriter) {
				s.logger.Error(nil, "concurrent-dispatch: decode aborted before prefill-success commit",
					"request_id", uuidStr)
			}
			break
		}
		// Prefill failed: cancel decode, then retry on the serial path or return
		// the prefill error verbatim.
		cancel()
		dcw.abort()
		if prefillResp != nil && isRetryableStatus(prefillResp.statusCode) && s.config.PrefillMaxRetries > 0 {
			s.logger.Info("retrying prefill request",
				"attempt", 1,
				"target", prefillPodHostPort,
				"request_id", uuidStr,
				"previous_code", prefillResp.statusCode)
			select {
			case <-time.After(s.config.PrefillRetryBackoff):
				retrySerially = true
			case <-parentCtx.Done():
			}
		}
		if retrySerially {
			break
		}
		metrics.RecordError(metrics.StagePrefill)
		dispatchFailed = true
		status := http.StatusBadGateway
		var prefillBody []byte
		if prefillResp != nil {
			status = prefillResp.statusCode
			prefillBody = prefillResp.bodyBytes()
			for key, values := range prefillResp.Header() {
				for _, v := range values {
					w.Header().Add(key, v)
				}
			}
		}
		s.logger.Info("concurrent-dispatch: prefill failed; returning prefill error and aborting decode",
			"request_id", uuidStr, "p_status", status, "p_body_snippet", truncate(string(prefillBody), 256))
		clientResponded = true
		w.WriteHeader(status)
		if _, writeErr := w.Write(prefillBody); writeErr != nil {
			s.logger.Error(writeErr, "failed to send prefill error to client (concurrent-dispatch)")
		}
	case <-decodeFailed:
		// A prefill left running could write KV into the blocks decode freed.
		// Decode's error says why the request failed, so it reaches the client.
		cancel()
		<-prefillDone
		s.logger.Info("concurrent-dispatch: decode failed before prefill answered; cancelled prefill",
			"request_id", uuidStr)
		dcw.commit()
	case <-prefillTimer.C:
		cancel()
		dcw.abort()
		// Counted here rather than from prefill's own outcome: the cancelled
		// prefill may still come back 2xx if it completed as the timer fired.
		metrics.RecordError(metrics.StagePrefill)
		dispatchFailed = true
		timedOut = true
		s.logger.Error(nil, "concurrent-dispatch: prefill did not respond within the NIXL push prefill timeout; aborting",
			"request_id", uuidStr, "timeout", prefillTimeout.String())
		clientResponded = true
		if err := errorGatewayTimeout(errNIXLPushPrefillTimeout, w); err != nil {
			s.logger.Error(err, "failed to send timeout error to client (concurrent-dispatch)")
		}
		<-prefillDone // let the cancelled prefill goroutine finish to avoid a leak
	}

	// Prefill has finished. The serial path samples the prefill stage of a
	// request it retries. An aborted prefill handler left no response.
	if !retrySerially && prefillResp != nil {
		metrics.RecordPrefillDuration(prefillDuration)
	}

	// Wait for decode to finish (streamed on success, or promptly aborted) so
	// we never leak the decode goroutine or its response body.
	<-decodeDone

	// A retryable status reports an overloaded or unreachable prefill endpoint,
	// not a changed identity, so the cached identity stays.
	if retrySerially {
		return true
	}

	if resendWith != nil {
		// The resent decode request gets its own span.
		decodeSpan.End()
		s.runNIXLProtocolV2Decode(parentCtx, w, r, body, resendWith, uuidStr, prefillPodHostPort, 0,
			transferID, prefillStartedAt, prefillDuration)
	}

	// The response of an aborted decode is replaced or torn down below, so what
	// the rewriter holds is dropped.
	var finalizeErr error
	if finalizeDecodeWriter != nil && !decodeAborted.Load() {
		if finalizeErr = finalizeDecodeWriter(); finalizeErr != nil {
			s.logger.Error(finalizeErr, "failed to flush cached token response writer")
		}
	}

	// Decode errors are attributed only when the commit point let decode's
	// response reach the client. The prefill timeout samples the decode
	// duration too.
	if !clientResponded {
		metrics.RecordDecodeDuration(decodeDuration)
		if decodeAborted.Load() || dcw.failed() || finalizeErr != nil {
			metrics.RecordError(metrics.StageDecode)
			dispatchFailed = true
		}
	} else if timedOut {
		metrics.RecordDecodeDuration(decodeDuration)
	}

	// A stale identity can cause the failure, so the next request to this
	// endpoint runs serially and learns the identity again. A client that went
	// away says nothing about the identity.
	if dispatchFailed && parentCtx.Err() == nil && s.nixlPushIdentities.dropIfMatches(prefillPodHostPort, identity) {
		metrics.RecordNIXLPushIdentityDrop()
		s.logger.V(logging.TRACE).Info("dropped NIXL push identity", "target", prefillPodHostPort, "request_id", uuidStr)
	}

	if currentSpan := trace.SpanFromContext(parentCtx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		if requestStart, ok := parentCtx.Value(requestStartTimeKey).(time.Time); ok {
			totalDuration = time.Since(requestStart)
		}
		currentSpan.SetAttributes(
			semconv.LLMDPDProxyTotalDurationMs(float64(totalDuration.Milliseconds())),
			semconv.LLMDPDProxyParallelWindowMs(float64(time.Since(requestStartedAt).Milliseconds())),
			semconv.LLMDPDProxyParallelDispatch(true),
			semconv.LLMDPDProxyNIXLPushIdentityMismatch(resendWith != nil),
		)
	}

	// Replay the decode abort on the request goroutine, unless the commit point
	// already wrote the response.
	if !decodeAborted.Load() || clientResponded {
		return false
	}
	if dcw.responseStarted() {
		// Decode's response is already on the wire, so the status cannot be
		// changed; net/http recovers this and drops the connection.
		panic(http.ErrAbortHandler)
	}
	if err := errorBadGateway(errDecodeAborted, w); err != nil {
		s.logger.Error(err, "failed to send decode abort error to client (concurrent-dispatch)")
	}
	return false
}

// truncate shortens s to at most n characters, appending "..." if truncated.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
