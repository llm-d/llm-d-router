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

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/common/usage"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
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

func newNIXLV2RequestID() (string, error) {
	id, err := uuid.NewUUID()
	if err != nil {
		return "", err
	}
	return id.String(), nil
}

// nixlSerialLeg adapts one wire format (JSON or vLLM gRPC) to the serial NIXLv2 flow.
type nixlSerialLeg interface {
	report(ctx context.Context)
	preparePrefill(w http.ResponseWriter, r *http.Request, uuidStr string, prefillKV map[string]any) (setReq func(*http.Request), bodyLen int, observeSpan func(trace.Span), ok bool)
	shouldRetryPrefill(pw *bufferedResponseWriter) bool
	extractPrefill(w http.ResponseWriter, r *http.Request, uuidStr string, pw *bufferedResponseWriter, prefillErr error, prefillSpan trace.Span) (pKVTransferParams any, cachedTokens int, rewriteCachedTokens bool, ok bool)
	streamingEnabled(r *http.Request) bool
	prepareDecode(w http.ResponseWriter, r *http.Request, dreq *http.Request, uuidStr string, pKVTransferParams any) (
		wrapWriter func(dst http.ResponseWriter, cachedTokens int, rewrite, streaming bool) (http.ResponseWriter, func() error),
		observeSpan func(trace.Span),
		dispatch func(decodeWriter http.ResponseWriter, dreq *http.Request),
		bodyLen int,
		ok bool,
	)
}

type httpNIXLSerialLeg struct {
	s          *Server
	body       map[string]any
	apiType    reqcommon.APIType
	toolFields *toolCallingRequest
}

func (l *httpNIXLSerialLeg) report(ctx context.Context) {
	l.toolFields.report(ctx)
}

func (l *httpNIXLSerialLeg) preparePrefill(w http.ResponseWriter, r *http.Request, _ string, prefillKV map[string]any) (func(*http.Request), int, func(trace.Span), bool) {
	// Keeps the client's body intact for the decode request below.
	prefillRequest := maps.Clone(l.body)
	prefillRequest[reqcommon.FieldKVTransferParams] = prefillKV
	// The EPP may forward the rendered prompt as kv_transfer_params.prompt_token_ids;
	// both hops replace kv_transfer_params, so carry the IDs or both pods re-tokenize.
	setPromptTokenIDs(prefillKV, l.s.clientPromptTokenIDs(l.body))
	reqcommon.CapSingleToken(prefillRequest, l.apiType)

	pbody, err := json.Marshal(prefillRequest)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			log.FromContext(r.Context()).Error(err, "failed to send error response to client")
		}
		return nil, 0, nil, false
	}
	return func(preq *http.Request) {
			preq.Body = io.NopCloser(bytes.NewReader(pbody))
			preq.ContentLength = int64(len(pbody))
		}, len(pbody), func(span trace.Span) {
			l.toolFields.compareValidatedDispatchedForSpan(pbody, span)
		}, true
}

func (l *httpNIXLSerialLeg) shouldRetryPrefill(pw *bufferedResponseWriter) bool {
	return isHTTPError(pw.statusCode) && isRetryableStatus(pw.statusCode)
}

func (l *httpNIXLSerialLeg) extractPrefill(w http.ResponseWriter, r *http.Request, uuidStr string, pw *bufferedResponseWriter, prefillErr error, prefillSpan trace.Span) (any, int, bool, bool) {
	logger := log.FromContext(r.Context())
	if prefillErr == nil && isHTTPError(pw.statusCode) {
		observeStageError(r.Context(), routing.StagePrefill)
		logger.Error(fmt.Errorf("prefill returned %d", pw.statusCode), "prefill request failed",
			"request_id", uuidStr,
			logBodyBytesKey, pw.buffer.Len())
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		prefillSpan.End()

		for key, values := range pw.Header() {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
		w.WriteHeader(pw.statusCode)
		if _, writeErr := w.Write(pw.bodyBytes()); writeErr != nil {
			logger.Error(writeErr, "failed to send error response to client")
		}
		return nil, 0, false, false
	}

	var prefillerResponse map[string]any
	if prefillErr == nil {
		if err := json.Unmarshal(pw.bodyBytes(), &prefillerResponse); err != nil {
			prefillErr = fmt.Errorf("invalid prefill response: %w", err)
		}
	}
	if prefillErr != nil {
		observeStageError(r.Context(), routing.StagePrefill)
		logger.Error(jsonErrorForLog(prefillErr), "prefill request failed",
			"request_id", uuidStr,
			"status", pw.statusCode,
			logBodyBytesKey, pw.buffer.Len())
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		prefillSpan.End()
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return nil, 0, false, false
	}
	prefillSpan.End()

	pKVTransferParams, ok := prefillerResponse[reqcommon.FieldKVTransferParams]
	if !ok {
		logger.Info("warning: missing 'kv_transfer_params' field in prefiller response")
	}
	// The prefill count corrects the one decode reports only when prefill
	// measured the prompt cache and decode reuses the blocks it transferred.
	// Without transfer params nothing moved, so decode recomputes the prompt and
	// the count it reports is its own measurement. A usage block whose details
	// are absent or null still measured the cache: vLLM reports zero hits that
	// way.
	pUsage := prefillerResponse[reqcommon.FieldUsage]
	pKVParams, _ := pKVTransferParams.(map[string]any)
	rewriteCachedTokens := pUsage != nil && len(pKVParams) > 0
	pCachedTokens, hasPCachedTokens := usage.CachedTokens(pUsage)

	if trace := logger.V(logging.TRACE); trace.Enabled() {
		// The count is logged only when it was measured, so a logged zero is a
		// measured zero rather than an absent measurement.
		fields := []any{
			"kvTransferParamsPresent", pKVTransferParams != nil,
			"rewriteCachedTokens", rewriteCachedTokens,
			"hasCachedTokens", hasPCachedTokens,
		}
		if hasPCachedTokens {
			fields = append(fields, "cachedTokens", pCachedTokens)
		}
		trace.Info("received prefiller response", fields...)
	}
	return pKVTransferParams, pCachedTokens, rewriteCachedTokens, true
}

func (l *httpNIXLSerialLeg) streamingEnabled(_ *http.Request) bool {
	streaming, _ := l.body[reqcommon.FieldStream].(bool)
	return streaming
}

func (l *httpNIXLSerialLeg) prepareDecode(
	w http.ResponseWriter,
	r *http.Request,
	dreq *http.Request,
	_ string,
	pKVTransferParams any,
) (
	func(http.ResponseWriter, int, bool, bool) (http.ResponseWriter, func() error),
	func(trace.Span),
	func(http.ResponseWriter, *http.Request),
	int,
	bool,
) {
	// Decode behind a prefiller has no ECConnector; the prefill stage consumed the encoder cache.
	delete(l.body, reqcommon.FieldECTransferParams)
	promptTokenIDs := l.s.clientPromptTokenIDs(l.body)
	if dKVParams, ok := pKVTransferParams.(map[string]any); ok {
		setPromptTokenIDs(dKVParams, promptTokenIDs)
	} else if pKVTransferParams == nil && promptTokenIDs != nil {
		// The prefill response omitted kv_transfer_params (the decode-recompute
		// fallback): still carry the client's forwarded prompt_token_ids so the
		// decode worker can skip re-tokenizing the long prompt. vLLM pops the
		// key and treats the emptied map as falsy, so no empty KV metadata
		// reaches the connector.
		pKVTransferParams = map[string]any{reqcommon.FieldPromptTokenIDs: promptTokenIDs}
	}
	l.body[reqcommon.FieldKVTransferParams] = pKVTransferParams

	dbody, err := json.Marshal(l.body)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			log.FromContext(r.Context()).Error(err, "failed to send error response to client")
		}
		return nil, nil, nil, 0, false
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))
	return decodeResponseWriter,
		func(span trace.Span) {
			l.toolFields.compareValidatedDispatchedForSpan(dbody, span)
		},
		func(decodeWriter http.ResponseWriter, dreq *http.Request) {
			l.s.dispatchDecode(decodeWriter, dreq, l.body, dbody, l.toolFields)
		},
		len(dbody),
		true
}

func (s *Server) handleNIXLV2(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting, apiType reqcommon.APIType) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running NIXL protocol V2", "url", prefillPodHostPort, "api", apiType.String())

	original, body, ok := s.requestBody(r, w, p2p)
	if !ok {
		return
	}
	source := p2p.pullSource(logger)
	// Telemetry failure must not change routing or reject an inference request.
	toolFields, _ := captureInferenceToolCallingRequest(r, apiType, original)

	// Generate unique request UUID
	uuidStr, err := s.nixlRequestIDFn()
	if err != nil {
		logger.Error(err, "failed to generate request ID")
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}

	// Parallel-dispatch path synthesises decode's kv_transfer_params from config
	// instead of the prefill response. The serial path below is unchanged when off.
	if s.config.MoRIIOParallelDispatch && s.config.MoRIIOWriteMode {
		// MoRI-IO requires transfer_id to carry the "tx" prefix for message routing.
		transferID := "tx" + uuidStr
		// Reuse the client snapshot when the parallel helper creates its request state.
		r = r.WithContext(context.WithValue(r.Context(), toolCallingBaselineKey, toolFields))
		s.runNIXLProtocolV2WriteParallel(w, r, original, body, uuidStr, transferID, prefillPodHostPort, source, apiType)
		return
	}

	s.runNIXLV2Serial(w, r, uuidStr, prefillPodHostPort, source, &httpNIXLSerialLeg{
		s:          s,
		body:       body,
		apiType:    apiType,
		toolFields: toolFields,
	})
}

func (s *Server) runNIXLV2Serial(
	w http.ResponseWriter,
	r *http.Request,
	uuidStr, prefillPodHostPort string,
	source *kvSource,
	leg nixlSerialLeg,
) {
	defer leg.report(r.Context())
	logger := log.FromContext(r.Context())
	tracer := tracing.Tracer(tracerScope)
	ctx := r.Context()

	// prefill and decode are sequential stages of forward_request, so both
	// spans start from the request context and come out as siblings.
	pCtx, prefillSpan := tracer.Start(ctx, "prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyPrefillTarget(prefillPodHostPort),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
	)
	prefillStart := time.Now()

	preq := r.Clone(pCtx)
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

	prefillKV := s.buildNIXLV2SerialPrefillKVParams(ctx, uuidStr, globalDPRank, source)
	setPrefillReq, pbodyLen, observePrefillSpan, ok := leg.preparePrefill(w, r, uuidStr, prefillKV)
	if !ok {
		prefillSpan.End()
		return
	}

	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		logger.Error(err, "failed to create prefill handler")
		prefillSpan.SetStatus(codes.Error, "failed to create prefill handler")
		prefillSpan.End()
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}

	logger.V(logging.DEBUG).Info("sending prefill request", "to", prefillPodHostPort)
	// Guarded: stringifying the body allocates a copy per request even when
	// TRACE is disabled.
	if trace := logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("Prefill request", logBodyBytesKey, pbodyLen)
	}
	if observePrefillSpan != nil {
		observePrefillSpan(prefillSpan)
	}

	// Retry on transient 5xx (502/503/504) or trailers-only gRPC Unavailable:
	// these failures are common when the prefill pod's accept queue overflows
	// under load. A response that already streamed body bytes is not retried,
	// because prefill already ran.
	var pw *bufferedResponseWriter
	var prefillErr error
retryLoop:
	for attempt := 0; ; attempt++ {
		pw = &bufferedResponseWriter{}
		setPrefillReq(preq)
		prefillErr = servePrefill(prefillHandler, pw, preq)
		if !leg.shouldRetryPrefill(pw) || attempt >= s.config.PrefillMaxRetries {
			break
		}
		logger.Info("retrying prefill request",
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
	observeStageDuration(r.Context(), routing.StagePrefill, prefillDuration)
	prefillSpan.SetAttributes(
		semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
		semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
	)

	// The reverse proxy also aborts the prefill body copy when the client
	// disconnects, which is not a prefill failure.
	if prefillErr != nil && preq.Context().Err() != nil {
		logger.Info("client disconnected during prefill", "request_id", uuidStr)
		prefillSpan.End()
		return
	}

	pKVTransferParams, pCachedTokens, rewriteCachedTokens, ok := leg.extractPrefill(w, r, uuidStr, pw, prefillErr, prefillSpan)
	if !ok {
		return
	}

	dCtx, decodeSpan := tracer.Start(ctx, "decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	streamingEnabled := leg.streamingEnabled(r)
	decodeSpan.SetAttributes(
		semconv.LLMDPDProxyRequestID(uuidStr),
		semconv.LLMDPDProxyConnector(constants.KVConnectorNIXLV2),
		semconv.LLMDPDProxyDecodeStreaming(streamingEnabled),
	)
	decodeStart := time.Now()

	dreq := r.Clone(dCtx)
	dreq.Header.Add(reqcommon.RequestIDHeaderKey, uuidStr)

	decodeLocalRank, err := s.prepareNIXLV2SerialDecodeKVParams(ctx, logger, pKVTransferParams, uuidStr, prefillPodHostPort, localDPRank)
	if err != nil {
		logger.Error(err, "failed to route MoRI-IO decode notify", "request_id", uuidStr)
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client")
		}
		return
	}
	if decodeLocalRank != "" {
		dreq.Header.Set(requestHeaderDataParallelRank, decodeLocalRank)
	}

	wrapDecodeWriter, observeDecodeSpan, dispatchDecode, dbodyLen, ok := leg.prepareDecode(w, r, dreq, uuidStr, pKVTransferParams)
	if !ok {
		return
	}

	if trace := logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("sending request to decoder", logBodyBytesKey, dbodyLen)
	}
	statusWriter, decodeStatus := captureResponseStatus(w)
	decodeWriter, finalizeDecodeWriter := wrapDecodeWriter(statusWriter, pCachedTokens, rewriteCachedTokens, streamingEnabled)
	decodeReturned := false
	defer recordDecodeAbort(r.Context(), &decodeReturned, decodeStart)
	dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(decodeWriter, dreq, func() {
		if observeDecodeSpan != nil {
			observeDecodeSpan(decodeSpan)
		}
	})
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDataParallel(dataParallelUsed))

	if !dataParallelUsed {
		if s.config.DecoderURL != nil {
			logger.V(logging.DEBUG).Info("sending request to decoder", "to", s.config.DecoderURL.Host)
			decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
		}
		dispatchDecode(decodeWriter, dreq)
	}
	decodeReturned = true
	if err := finalizeDecodeWriter(); err != nil {
		observeStageDuration(r.Context(), routing.StageDecode, time.Since(decodeStart))
		observeStageError(r.Context(), routing.StageDecode)
		logger.Error(err, "failed to flush cached token response writer")
		decodeSpan.SetStatus(codes.Error, "failed to flush cached token response writer")
		return
	}

	decodeDuration := time.Since(decodeStart)
	observeStageDuration(r.Context(), routing.StageDecode, decodeDuration)
	if decodeStatus.failed() {
		observeStageError(r.Context(), routing.StageDecode)
		decodeSpan.SetStatus(codes.Error, "decode request failed")
	}
	decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDurationMs(float64(decodeDuration.Milliseconds())))
	recordSequentialPDTimingAttrs(ctx, decodeSpan, prefillStart, prefillDuration, decodeStart, decodeDuration)
}

// servePrefill buffers the prefill response into pw. It returns
// errPrefillAborted in place of the http.ErrAbortHandler panic the reverse
// proxy raises when the body copy fails, which net/http would answer by
// resetting the client connection.
func servePrefill(h http.Handler, pw *bufferedResponseWriter, r *http.Request) (err error) {
	defer func() {
		if rec := recover(); rec != nil {
			if rec != http.ErrAbortHandler {
				panic(rec)
			}
			err = errPrefillAborted
		}
	}()
	h.ServeHTTP(pw, r)
	return nil
}

// decodeResponseWriter returns the writer the decode response is streamed
// through, and the function that flushes whatever it buffered. Without a prefill
// count to apply the client gets the decode worker's bytes untouched.
func decodeResponseWriter(w http.ResponseWriter, cachedTokens int, rewrite, streaming bool) (http.ResponseWriter, func() error) {
	if !rewrite {
		return w, func() error { return nil }
	}
	return usage.NewCachedTokensResponseWriter(w, cachedTokens, streaming)
}

// nixlParallelLeg adapts one wire format (JSON or vLLM gRPC) to MoRI-IO parallel dispatch.
type nixlParallelLeg interface {
	preparePrefill(preq *http.Request, span trace.Span) int
	prepareDecode(dreq *http.Request) int
	observeDecodeSpan(span trace.Span)
	checkPrefill(pw *bufferedResponseWriter) (failed bool, err error)
	decodeWriter(w http.ResponseWriter, dreq *http.Request, uuidStr string, pw *bufferedResponseWriter) (http.ResponseWriter, func() error)
	relayPrefillFailure(w http.ResponseWriter, r *http.Request, uuidStr string, pw *bufferedResponseWriter, err error)
}

type httpNIXLParallelLeg struct {
	pbody      []byte
	dbody      []byte
	streaming  bool
	toolFields *toolCallingRequest
}

func (l *httpNIXLParallelLeg) preparePrefill(preq *http.Request, span trace.Span) int {
	preq.Body = io.NopCloser(bytes.NewReader(l.pbody))
	preq.ContentLength = int64(len(l.pbody))
	l.toolFields.compareValidatedDispatchedForSpan(l.pbody, span)
	return len(l.pbody)
}

func (l *httpNIXLParallelLeg) prepareDecode(dreq *http.Request) int {
	dreq.Body = io.NopCloser(bytes.NewReader(l.dbody))
	dreq.ContentLength = int64(len(l.dbody))
	return len(l.dbody)
}

func (l *httpNIXLParallelLeg) observeDecodeSpan(span trace.Span) {
	l.toolFields.compareValidatedDispatchedForSpan(l.dbody, span)
}

func (l *httpNIXLParallelLeg) checkPrefill(pw *bufferedResponseWriter) (bool, error) {
	return isHTTPError(pw.statusCode), nil
}

func (l *httpNIXLParallelLeg) decodeWriter(w http.ResponseWriter, dreq *http.Request, uuidStr string, prefillResp *bufferedResponseWriter) (http.ResponseWriter, func() error) {
	// Rewrite decode's usage with the prefill cache count, as in serial dispatch.
	var prefillerResponse map[string]any
	if err := json.Unmarshal(prefillResp.bodyBytes(), &prefillerResponse); err != nil {
		log.FromContext(dreq.Context()).Error(jsonErrorForLog(err), "concurrent-dispatch: failed to parse prefill response", "request_id", uuidStr)
	}
	pUsage := prefillerResponse[reqcommon.FieldUsage]
	pKVParams, _ := prefillerResponse[reqcommon.FieldKVTransferParams].(map[string]any)
	pCachedTokens, _ := usage.CachedTokens(pUsage)
	return decodeResponseWriter(w, pCachedTokens, pUsage != nil && len(pKVParams) > 0, l.streaming)
}

func (l *httpNIXLParallelLeg) relayPrefillFailure(w http.ResponseWriter, r *http.Request, uuidStr string, prefillResp *bufferedResponseWriter, _ error) {
	logger := log.FromContext(r.Context())
	status := http.StatusBadGateway
	if prefillResp != nil {
		status = prefillResp.statusCode
		for key, values := range prefillResp.Header() {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
	}
	bodyBytes := 0
	if prefillResp != nil {
		bodyBytes = prefillResp.buffer.Len()
	}
	logger.Info("concurrent-dispatch: prefill failed; returning prefill error and aborting decode",
		"request_id", uuidStr, "p_status", status, logBodyBytesKey, bodyBytes)
	w.WriteHeader(status)
	if prefillResp != nil {
		if _, writeErr := w.Write(prefillResp.bodyBytes()); writeErr != nil {
			logger.Error(writeErr, "failed to send prefill error to client (concurrent-dispatch)")
		}
	}
}

// runNIXLProtocolV2WriteParallel is the MoRI-IO WRITE-mode concurrent-dispatch
// path: it builds both the prefill and decode bodies up front (synthesising
// decode's kv_transfer_params from config and prefillPodHostPort) and fires the
// two upstream calls in parallel so decode's block allocation overlaps prefill.
func (s *Server) runNIXLProtocolV2WriteParallel(
	w http.ResponseWriter, r *http.Request, original []byte,
	body map[string]any, uuidStr, transferID, prefillPodHostPort string, source *kvSource,
	apiType reqcommon.APIType,
) {
	toolFields, _ := captureInferenceToolCallingRequest(r, apiType, original)
	defer toolFields.report(r.Context())

	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running NIXL protocol V2 (concurrent dispatch)",
		"url", prefillPodHostPort, "request_id", uuidStr)

	parentCtx := r.Context()
	dpRank, prefillKV, decodeKV := s.buildNIXLV2ParallelKVParams(parentCtx, uuidStr, transferID, prefillPodHostPort, source)

	// Keeps the client's body intact for the decode request built below.
	prefillRequest := maps.Clone(body)
	// The EPP may forward the rendered prompt as kv_transfer_params.prompt_token_ids;
	// both hops replace kv_transfer_params, so carry the IDs or both pods re-tokenize.
	promptTokenIDs := s.clientPromptTokenIDs(body)
	prefillRequest[reqcommon.FieldKVTransferParams] = prefillKV
	setPromptTokenIDs(prefillKV, promptTokenIDs)
	reqcommon.CapSingleToken(prefillRequest, apiType)

	pbody, err := json.Marshal(prefillRequest)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal P)")
		}
		return
	}

	// Decode behind a prefiller has no ECConnector; the prefill stage consumed the encoder cache.
	delete(body, reqcommon.FieldECTransferParams)
	body[reqcommon.FieldKVTransferParams] = decodeKV
	setPromptTokenIDs(decodeKV, promptTokenIDs)

	dbody, err := json.Marshal(body)
	if err != nil {
		if err := errorJSONInvalid(w, r, err); err != nil {
			logger.Error(err, "failed to send error response to client (concurrent-dispatch marshal D)")
		}
		return
	}

	streamingEnabled, _ := body[reqcommon.FieldStream].(bool)
	s.executeNIXLV2WriteParallel(w, r, uuidStr, prefillPodHostPort, dpRank, &httpNIXLParallelLeg{
		pbody:      pbody,
		dbody:      dbody,
		streaming:  streamingEnabled,
		toolFields: toolFields,
	})
}

func (s *Server) executeNIXLV2WriteParallel(
	w http.ResponseWriter, r *http.Request,
	uuidStr, prefillPodHostPort string, dpRank int,
	leg nixlParallelLeg,
) {
	logger := log.FromContext(r.Context())
	tracer := tracing.Tracer()
	parentCtx := r.Context()
	requestStartedAt := time.Now()

	// ---------- Fire prefill + decode in parallel ----------
	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		logger.Error(err, "failed to create prefill handler (concurrent-dispatch)")
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send error response to client (concurrent-dispatch P handler init)")
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
	pbodyLen := leg.preparePrefill(preq, prefillSpan)

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
	dbodyLen := leg.prepareDecode(dreq)

	if trace := logger.V(logging.TRACE); trace.Enabled() {
		trace.Info("concurrent-dispatch prefill request body", logBodyBytesKey, pbodyLen)
		trace.Info("concurrent-dispatch decode request body", logBodyBytesKey, dbodyLen)
	}

	// Decode writes into a deferred writer that buffers everything until we
	// commit() (prefill succeeded -> flush + stream on) or abort() (prefill
	// failed -> discard); it never writes to the client directly.
	dcw := newDeferredCommitWriter()
	var finalizeDecodeWriter func() error

	// Prefill goroutine: body is buffered so it can be returned to the client
	// verbatim on failure. On ANY non-2xx status (transport errors surface as
	// 502 from the reverse proxy) it cancels the shared context so decode's
	// KV wait aborts immediately.
	var prefillResp *bufferedResponseWriter
	var prefillFailed bool
	var prefillErr error
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
				if pCtx.Err() != nil {
					return
				}
				prefillSpan.SetStatus(codes.Error, "prefill handler aborted")
				cancel()
				log.FromContext(pCtx).Error(nil, "concurrent-dispatch prefill handler aborted",
					"request_id", uuidStr)
			}
		}()
		pw := &bufferedResponseWriter{}
		prefillHandler.ServeHTTP(pw, preq)
		prefillResp = pw
		prefillFailed, prefillErr = leg.checkPrefill(pw)
		prefillDuration := time.Since(prefillStartedAt)
		observeStageDuration(r.Context(), routing.StagePrefill, prefillDuration)
		prefillSpan.SetAttributes(
			semconv.LLMDPDProxyPrefillStatusCode(pw.statusCode),
			semconv.LLMDPDProxyPrefillDurationMs(float64(prefillDuration.Milliseconds())),
		)
		if prefillFailed && pCtx.Err() == nil {
			prefillSpan.SetStatus(codes.Error, "prefill request failed")
			cancel() // KV will never arrive -> abort decode instead of hanging
			log.FromContext(pCtx).Error(prefillErr, "concurrent-dispatch prefill returned error status",
				"status", pw.statusCode, "request_id", uuidStr, logBodyBytesKey, pw.buffer.Len())
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
				log.FromContext(dCtx).Error(nil, "concurrent-dispatch decode handler aborted",
					"request_id", uuidStr)
			}
		}()
		// The request goroutine reads the aggregate only after decodeDone closes.
		dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(dcw, dreq, func() {
			leg.observeDecodeSpan(decodeSpan)
		})
		decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeDataParallel(dataParallelUsed))
		if !dataParallelUsed {
			decodeSpan.SetAttributes(semconv.LLMDPDProxyDecodeTarget(s.config.DecoderURL.Host))
			leg.observeDecodeSpan(decodeSpan)
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

	for decodeFinished := decodeDone; ; {
		select {
		case <-decodeFinished:
			if dispatchCtx.Err() != nil || !decodeAborted.Load() && !dcw.failed() {
				// Prefill can cancel decode before prefillDone closes.
				// A successful decode also waits for prefill's outcome.
				decodeFinished = nil
				continue
			}
			cancel()
			if !decodeAborted.Load() {
				dcw.commit(w)
			}
			<-prefillDone
		case <-prefillDone:
			if prefillResp != nil && !prefillFailed {
				decodeWriter, finalize := leg.decodeWriter(w, dreq, uuidStr, prefillResp)
				finalizeDecodeWriter = finalize
				if !dcw.commit(decodeWriter) {
					logger.Error(nil, "concurrent-dispatch: decode aborted before prefill-success commit",
						"request_id", uuidStr)
				}
				break
			}
			// Prefill failed: abort decode and return the prefill error verbatim.
			cancel()
			dcw.abort()
			observeStageError(r.Context(), routing.StagePrefill)
			clientResponded = true
			leg.relayPrefillFailure(w, r, uuidStr, prefillResp, prefillErr)
		case <-timer.C:
			// Prefill did not resolve within the backstop window: cancel both requests
			// and fail rather than hang waiting for KV that may never arrive.
			cancel()
			dcw.abort()
			// Counted here rather than from prefill's own outcome: the cancelled
			// prefill may still come back 2xx if it completed as the timer fired.
			observeStageError(r.Context(), routing.StagePrefill)
			decodeTimedOut = true
			logger.Error(nil, "concurrent-dispatch: prefill did not complete within KV-wait timeout; aborting",
				"request_id", uuidStr, "timeout", waitTimeout.String())
			clientResponded = true
			if writeErr := sendError(w, r, http.StatusGatewayTimeout, "decode aborted: prefill did not complete within the MoRI-IO parallel-dispatch KV-wait timeout"); writeErr != nil {
				logger.Error(writeErr, "failed to send timeout error to client (concurrent-dispatch)")
			}
			<-prefillDone // let the cancelled prefill goroutine finish to avoid a leak
		}
		break
	}

	// Wait for decode to finish (streamed on success, or promptly aborted) so
	// we never leak the decode goroutine or its response body.
	<-decodeDone
	var finalizeErr error
	if finalizeDecodeWriter != nil && !decodeAborted.Load() {
		if finalizeErr = finalizeDecodeWriter(); finalizeErr != nil {
			logger.Error(finalizeErr, "failed to flush cached token response writer (concurrent-dispatch)",
				"request_id", uuidStr)
		}
	}

	// Decode errors are attributed only when the commit point let decode's
	// response reach the client. When the commit point wrote the response
	// itself (prefill error or KV-wait timeout), decode's output was discarded
	// and the failure is counted as a prefill error. Decode duration is also
	// sampled on the KV-wait timeout, where decode waited the full timeout.
	if !clientResponded {
		observeStageDuration(r.Context(), routing.StageDecode, decodeDuration)
		if decodeAborted.Load() || dcw.failed() || finalizeErr != nil {
			observeStageError(r.Context(), routing.StageDecode)
		}
	} else if decodeTimedOut {
		observeStageDuration(r.Context(), routing.StageDecode, decodeDuration)
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
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send decode abort error to client (concurrent-dispatch)")
		}
	}
}

func (s *Server) buildNIXLV2SerialPrefillKVParams(ctx context.Context, uuidStr string, globalDPRank int, source *kvSource) map[string]any {
	var pkv map[string]any
	if s.config.MoRIIOWriteMode {
		transferID := "tx" + uuidStr
		pkv = map[string]any{
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
		if decodeHosts := s.currentDecodeHosts(ctx); len(decodeHosts) > 0 {
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
		pkv = map[string]any{
			reqcommon.FieldDoRemoteDecode:  true,
			reqcommon.FieldDoRemotePrefill: false,
			reqcommon.FieldRemoteEngineID:  nil,
			reqcommon.FieldRemoteBlockIDs:  nil,
			reqcommon.FieldRemoteHost:      nil,
			reqcommon.FieldRemotePort:      nil,
		}
	}
	addP2PPullToPrefill(pkv, source)
	return pkv
}

func (s *Server) prepareNIXLV2SerialDecodeKVParams(
	ctx context.Context,
	logger logr.Logger,
	pKVTransferParams any,
	uuidStr, prefillPodHostPort string,
	localDPRank int,
) (string, error) {
	if !s.config.MoRIIOWriteMode {
		return "", nil
	}
	decodeDPRank, usedReturned := resolveDecodeDPRank(pKVTransferParams, uuidStr, s.config.MoRIIODPSize)
	if pkv, ok := pKVTransferParams.(map[string]any); ok {
		if !usedReturned && s.config.MoRIIODPSizeLocal > 0 && s.config.MoRIIODPSize > s.config.MoRIIODPSizeLocal {
			prefillHost := s.resolver().resolveOne(ctx, extractHost(prefillPodHostPort))
			podIndex := slices.Index(s.currentRemoteHosts(ctx), prefillHost)
			if podIndex < 0 {
				return "", fmt.Errorf("cannot determine prefill pod index for %q", prefillHost)
			}
			decodeDPRank = podIndex*s.config.MoRIIODPSizeLocal + localDPRank
		}
		if rv, present := pkv[requestFieldRemoteDPRank]; present && !usedReturned && s.config.MoRIIODPSize > 1 {
			logger.Info("prefill returned invalid/out-of-range remote_dp_rank; using fallback DP rank",
				"request_id", uuidStr, "returned", rv,
				"dp_size", s.config.MoRIIODPSize, "rank", decodeDPRank)
		}
		pkv[requestFieldRemoteDPRank] = decodeDPRank
		pkv[requestFieldRemoteDPRankOverride] = true
		if s.config.MoRIIODPSize > 1 {
			pkv["is_request_leader"] = true
		}
		if _, present := pkv[requestFieldTransferID]; !present {
			pkv[requestFieldTransferID] = "tx" + uuidStr
		}
		if _, present := pkv[requestFieldRemoteNotifyPort]; !present {
			pkv[requestFieldRemoteNotifyPort] = s.config.MoRIIODecodeNotifyPort
		}
		if _, present := pkv[requestFieldRemoteHandshakePort]; !present {
			pkv[requestFieldRemoteHandshakePort] = s.config.MoRIIODecodeHandshakePort
		}
		if _, present := pkv["remote_dp_size"]; !present {
			pkv["remote_dp_size"] = s.config.MoRIIODPSize
		}
		if remoteHosts := s.currentRemoteHosts(ctx); len(remoteHosts) > 0 {
			if _, present := pkv["remote_hosts"]; !present {
				hosts := make([]any, len(remoteHosts))
				for i, h := range remoteHosts {
					hosts[i] = h
				}
				pkv["remote_hosts"] = hosts
			}
			if s.config.MoRIIODPSizeLocal > 0 {
				if _, present := pkv["remote_dp_size_local"]; !present {
					pkv["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
				}
			}
		}
	}
	if s.config.MoRIIODPSize > 1 {
		decodeLocalDPRank := foldDPRankToLocal(
			decodeDPRank,
			s.config.MoRIIODPSize,
			s.config.MoRIIODPSizeLocal,
		)
		return strconv.Itoa(decodeLocalDPRank), nil
	}
	return "", nil
}

func (s *Server) buildNIXLV2ParallelKVParams(
	ctx context.Context,
	uuidStr, transferID, prefillPodHostPort string,
	source *kvSource,
) (dpRank int, prefillKV, decodeKV map[string]any) {
	dpLocal := s.config.MoRIIODPSizeLocal
	if dpLocal <= 0 {
		dpLocal = s.config.MoRIIODPSize
	}
	if dpLocal <= 0 {
		dpLocal = 1
	}
	_, dpRank = pickDPRanks(
		uuidStr,
		s.config.MoRIIODPSize,
		s.config.MoRIIODPSizeLocal,
	)

	decodePodIP := s.currentDecodePodIP(ctx)
	decodeHosts := s.currentDecodeHosts(ctx)

	prefillKV = map[string]any{
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
	if len(decodeHosts) > 0 {
		hosts := make([]any, len(decodeHosts))
		for i, h := range decodeHosts {
			hosts[i] = h
		}
		prefillKV["remote_hosts"] = hosts
		if s.config.MoRIIODPSizeLocal > 0 {
			prefillKV["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
	}
	addP2PPullToPrefill(prefillKV, source)

	prefillHost, _, splitErr := net.SplitHostPort(prefillPodHostPort)
	if splitErr != nil {
		prefillHost = prefillPodHostPort
	}
	remoteHosts := s.currentRemoteHosts(ctx)

	decodeKV = map[string]any{
		reqcommon.FieldDoRemotePrefill:   true,
		reqcommon.FieldDoRemoteDecode:    false,
		reqcommon.FieldRemoteEngineID:    net.JoinHostPort(prefillHost, strconv.Itoa(s.config.MoRIIOPrefillHandshakePort)),
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
		"remote_request_id":              uuidStr,
	}
	if len(remoteHosts) > 0 {
		decodeKV["remote_hosts"] = []any{prefillHost}
		if s.config.MoRIIODPSizeLocal > 0 {
			decodeKV["remote_dp_size_local"] = s.config.MoRIIODPSizeLocal
		}
	}
	return dpRank, prefillKV, decodeKV
}
