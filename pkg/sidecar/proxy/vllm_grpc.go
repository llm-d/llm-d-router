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
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	otelcodes "go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/sync/errgroup"
	grpccodes "google.golang.org/grpc/codes"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func setUnaryGeneratePath(req *http.Request) {
	req.URL.Path = vllmpb.Inference_Generate_FullMethodName
	req.URL.RawPath = ""
}

// readVLLMGRPCRequest reads a single-frame vllm.Inference GenerateRequest from
// r.Body and scans its protobuf wire structure without decoding untouched fields.
func (s *Server) readVLLMGRPCRequest(r *http.Request, w http.ResponseWriter) (*vllmwire.Request, bool) {
	defer r.Body.Close()
	logger := log.FromContext(r.Context())

	frame, err := grpcframe.ReadMessage(r.Body, grpcframe.MaxPayloadLen)
	if err != nil {
		logger.V(logging.DEBUG).Info("invalid gRPC request frame", "error", err)
		code := grpccodes.InvalidArgument
		if errors.Is(err, grpcframe.ErrTooLarge) {
			code = grpccodes.ResourceExhausted
		}
		grpcframe.WriteError(w, code, err.Error())
		return nil, false
	}

	scanned, err := vllmwire.ScanGenerateRequest(frame[grpcframe.HeaderLen:])
	if err != nil {
		logger.V(logging.DEBUG).Info("invalid vllm.Inference request payload", "error", err)
		grpcframe.WriteError(w, grpccodes.InvalidArgument, err.Error())
		return nil, false
	}
	return &scanned, true
}

func (s *Server) requestVLLMGRPC(r *http.Request, w http.ResponseWriter, p2p p2pRouting) (*vllmwire.Request, bool) {
	if p2p.grpcReq != nil {
		return p2p.grpcReq, true
	}
	return s.readVLLMGRPCRequest(r, w)
}

// prepareVLLMGRPCP2PSource extracts and validates kv_transfer_params.remote_kv_source
// from a vllm.Inference request when EPP sets x-llm-d-kv-cache-source. The
// dispatcher builds a sanitized frame only when the request bypasses P/D, EC,
// and P2P source pull.
func (s *Server) prepareVLLMGRPCP2PSource(
	w http.ResponseWriter,
	r *http.Request,
	logger logr.Logger,
	p2p p2pRouting,
) (p2pRouting, bool) {
	scanned, ok := s.readVLLMGRPCRequest(r, w)
	if !ok {
		return p2p, false
	}
	kvMap, hasKV, err := scanned.ClientKVTransferParams()
	if err != nil {
		logger.V(logging.DEBUG).Info("invalid kv_transfer_params Struct in gRPC request", "error", err)
		grpcframe.WriteError(w, grpccodes.InvalidArgument, err.Error())
		return p2p, false
	}
	if hasKV {
		raw, exists := kvMap[reqcommon.FieldRemoteKVSource]
		if !exists {
			logger.Info("warning: KV cache source header without a remote_kv_source block in the body, proceeding without a pull",
				"clientIP", r.RemoteAddr)
		} else {
			p2p.source = s.validateKVSourceBlock(logger, r.RemoteAddr, raw)
		}
	} else {
		logger.Info("warning: KV cache source header without a remote_kv_source block in the body, proceeding without a pull",
			"clientIP", r.RemoteAddr)
	}
	p2p.grpcReq = scanned
	return p2p, true
}

// rebuildSanitizedVLLMGRPCRequest clones r over a clean frame without client
// transfer params when a decoder-only request bypasses P2P source pull.
func (s *Server) rebuildSanitizedVLLMGRPCRequest(w http.ResponseWriter, r *http.Request, req *vllmwire.Request) (*http.Request, bool) {
	segs, ok := s.buildLegSegmentsOrWriteError(w, r, req, vllmwire.LegOptions{
		Media: vllmwire.MediaAll,
	})
	if !ok {
		return r, false
	}
	cloned := r.Clone(r.Context())
	grpcframe.SetSegmentedBody(cloned, segs)
	return cloned, true
}

func (s *Server) buildLegSegmentsOrWriteError(
	w http.ResponseWriter,
	r *http.Request,
	req *vllmwire.Request,
	opts vllmwire.LegOptions,
) ([][]byte, bool) {
	segs, err := req.FrameSegments(opts)
	if err != nil {
		log.FromContext(r.Context()).Error(err, "failed to build vllm.Inference leg frame")
		code := grpccodes.Internal
		if errors.Is(err, grpcframe.ErrTooLarge) {
			code = grpccodes.ResourceExhausted
		}
		grpcframe.WriteError(w, code, err.Error())
		return nil, false
	}
	return segs, true
}

func (s *Server) buildLegFrameOrWriteError(
	w http.ResponseWriter,
	r *http.Request,
	req *vllmwire.Request,
	opts vllmwire.LegOptions,
) ([]byte, bool) {
	frame, err := req.Frame(opts)
	if err != nil {
		log.FromContext(r.Context()).Error(err, "failed to build vllm.Inference leg frame")
		code := grpccodes.Internal
		if errors.Is(err, grpcframe.ErrTooLarge) {
			code = grpccodes.ResourceExhausted
		}
		grpcframe.WriteError(w, code, err.Error())
		return nil, false
	}
	return frame, true
}

func prefillMediaMode(p2p p2pRouting) vllmwire.MediaMode {
	if p2p.encodeRan {
		return vllmwire.MediaMetadataOnly
	}
	return vllmwire.MediaAll
}

func decodeOnlyMediaMode(p2p p2pRouting) vllmwire.MediaMode {
	if p2p.encodeRan {
		return vllmwire.MediaMetadataOnly
	}
	return vllmwire.MediaAll
}

// vllmPrefillOutcome holds the FinishInfo metadata or failure mode extracted
// from a buffered vllm.Inference prefill or encoder response.
type vllmPrefillOutcome struct {
	kvTransferParams map[string]any
	ecTransferParams map[string]any
	numCachedTokens  *uint32
	// relayError reports that upstream returned an HTTP error or a non-OK gRPC
	// status that the sidecar forwards instead of mapping to 502.
	relayError bool
}

// extractVLLMPrefillFinishInfo validates a buffered gRPC response from prefill
// or encoder and scans FinishInfo from the terminal GenerateResponse frame.
func extractVLLMPrefillFinishInfo(pw *bufferedResponseWriter, prefillErr error) (vllmPrefillOutcome, error) {
	if pw == nil {
		if prefillErr != nil {
			return vllmPrefillOutcome{}, prefillErr
		}
		return vllmPrefillOutcome{}, errors.New("empty upstream gRPC response")
	}
	if isHTTPError(pw.statusCode) {
		return vllmPrefillOutcome{relayError: pw.statusCode >= http.StatusContinue},
			fmt.Errorf("upstream returned HTTP status %d", pw.statusCode)
	}
	if grpcframe.HasErrorStatus(pw.headers) {
		code, _ := grpcframe.Status(pw.headers)
		return vllmPrefillOutcome{relayError: true},
			fmt.Errorf("upstream returned gRPC error status %s", code)
	}
	if prefillErr != nil {
		return vllmPrefillOutcome{}, prefillErr
	}
	// A complete gRPC response over HTTP/2 ends with grpc-status = 0.
	statusStr, hasStatus := grpcframe.Status(pw.headers)
	if !hasStatus || statusStr != grpcframe.StatusOK {
		return vllmPrefillOutcome{}, errors.New("upstream gRPC response missing OK status trailer")
	}

	raw := pw.bodyBytes()
	var lastFinish vllmwire.ResponseInfo
	foundFinish := false
	for len(raw) > 0 {
		payload, err := grpcframe.ParsePayload(raw)
		if err != nil {
			return vllmPrefillOutcome{}, fmt.Errorf("invalid gRPC frame in upstream response: %w", err)
		}
		raw = raw[grpcframe.HeaderLen+len(payload):]
		info, scanErr := vllmwire.ScanGenerateResponse(payload)
		if scanErr != nil {
			return vllmPrefillOutcome{}, scanErr
		}
		if info.HasFinishInfo {
			lastFinish = info
			foundFinish = true
		}
	}
	if !foundFinish {
		return vllmPrefillOutcome{}, errors.New("upstream gRPC response ended without FinishInfo")
	}

	kvParams, err := lastFinish.KVTransferParams()
	if err != nil {
		return vllmPrefillOutcome{}, fmt.Errorf("decode FinishInfo.kv_transfer_params: %w", err)
	}
	ecParams, err := lastFinish.ECTransferParams()
	if err != nil {
		return vllmPrefillOutcome{}, fmt.Errorf("decode FinishInfo.ec_transfer_params: %w", err)
	}
	out := vllmPrefillOutcome{
		kvTransferParams: kvParams,
		ecTransferParams: ecParams,
	}
	if lastFinish.HasCachedTokens {
		n := lastFinish.NumCachedTokens
		out.numCachedTokens = &n
	}
	return out, nil
}

// isRetryableVLLMPrefillFailure reports whether a failed gRPC prefill attempt
// returned a transient HTTP 502/503/504 or a trailers-only gRPC Unavailable.
// Any response that already streamed body bytes is not retried.
func isRetryableVLLMPrefillFailure(pw *bufferedResponseWriter) bool {
	if pw == nil {
		return false
	}
	if isRetryableStatus(pw.statusCode) {
		return true
	}
	if !isHTTPError(pw.statusCode) && pw.buffer.Len() == 0 {
		code, ok := grpcframe.Status(pw.headers)
		return ok && code == strconv.Itoa(int(grpccodes.Unavailable))
	}
	return false
}

// relayVLLMPrefillFailure sends a failed prefill response to the gRPC client.
// Upstream gRPC errors become trailers-only responses so partial internal
// prefill frames never reach unary or streaming clients.
func (s *Server) relayVLLMPrefillFailure(w http.ResponseWriter, r *http.Request, pw *bufferedResponseWriter, outcome vllmPrefillOutcome) {
	logger := log.FromContext(r.Context())
	if !outcome.relayError || pw == nil {
		if err := errorBadGateway(w, r); err != nil {
			logger.Error(err, "failed to send gRPC bad gateway error to client")
		}
		return
	}
	if isHTTPError(pw.statusCode) {
		for key, values := range pw.Header() {
			for _, v := range values {
				w.Header().Add(key, v)
			}
		}
		w.WriteHeader(pw.statusCode)
		if _, err := w.Write(pw.bodyBytes()); err != nil {
			logger.Error(err, "failed to write prefill HTTP error body to client")
		}
		return
	}

	h := w.Header()
	h.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	grpcframe.CopyStatusHeaders(h, pw.headers)
	w.WriteHeader(http.StatusOK)
}

// serveVLLMGRPCDecode proxies the decode leg through s.decoderProxy (or the
// data-parallel proxy) and rewrites FinishInfo.num_cached_tokens on the
// terminal frame when cachedTokens is set.
func (s *Server) serveVLLMGRPCDecode(
	w http.ResponseWriter,
	dreq *http.Request,
	segs [][]byte,
	cachedTokens *uint32,
) {
	grpcframe.SetSegmentedBody(dreq, segs)
	target := w
	var rw *vllmCachedTokensResponseWriter
	if cachedTokens != nil {
		rw = newVLLMCachedTokensResponseWriter(w, *cachedTokens)
		target = rw
	}
	if !s.forwardDataParallel || !s.dataParallelHandler(target, dreq, nil) {
		s.decoderProxy.ServeHTTP(target, dreq)
	}
	if rw != nil {
		rw.finish(dreq)
	}
}

// vllmCachedTokensResponseWriter rewrites FinishInfo.num_cached_tokens on the
// terminal GenerateResponse frame while streaming non-terminal frames without
// allocating.
type vllmCachedTokensResponseWriter struct {
	dst          http.ResponseWriter
	cachedTokens uint32
	statusCode   int
	wroteHeader  bool
	passthrough  bool
	buf          bytes.Buffer
}

func newVLLMCachedTokensResponseWriter(dst http.ResponseWriter, cachedTokens uint32) *vllmCachedTokensResponseWriter {
	return &vllmCachedTokensResponseWriter{dst: dst, cachedTokens: cachedTokens}
}

func (w *vllmCachedTokensResponseWriter) Header() http.Header {
	return w.dst.Header()
}

func (w *vllmCachedTokensResponseWriter) WriteHeader(statusCode int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.statusCode = statusCode
	if isHTTPError(statusCode) || grpcframe.HasErrorStatus(w.dst.Header()) {
		w.passthrough = true
		w.dst.WriteHeader(statusCode)
		return
	}
	w.dst.Header().Del("Content-Length")
	w.dst.WriteHeader(statusCode)
}

func (w *vllmCachedTokensResponseWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if w.passthrough {
		return w.dst.Write(b)
	}
	total := len(b)
	_, _ = w.buf.Write(b)
	for w.buf.Len() >= grpcframe.HeaderLen {
		data := w.buf.Bytes()
		payload, err := grpcframe.ParsePayload(data)
		if err != nil {
			if data[0] != 0 {
				w.passthrough = true
				_, writeErr := w.dst.Write(data)
				w.buf.Reset()
				return total, writeErr
			}
			// Wait for the remaining bytes of this frame.
			break
		}
		frameLen := grpcframe.HeaderLen + len(payload)
		frame := data[:frameLen]
		outFrame, rwErr := vllmwire.RewriteTerminalCachedTokens(frame, w.cachedTokens)
		if rwErr != nil {
			outFrame = frame
		}
		if _, writeErr := w.dst.Write(outFrame); writeErr != nil {
			w.buf.Next(frameLen)
			return total, writeErr
		}
		w.buf.Next(frameLen)
	}
	return total, nil
}

func (w *vllmCachedTokensResponseWriter) Flush() {
	if f, ok := w.dst.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *vllmCachedTokensResponseWriter) finish(r *http.Request) {
	if w.buf.Len() == 0 {
		return
	}
	_, _ = w.dst.Write(w.buf.Bytes())
	w.buf.Reset()
	if !grpcframe.HasErrorStatus(w.dst.Header()) {
		canonicalStatus := http.CanonicalHeaderKey(grpcframe.HeaderStatus)
		delete(w.dst.Header(), canonicalStatus)
		delete(w.dst.Header(), http.TrailerPrefix+canonicalStatus)
		delete(w.dst.Header(), http.TrailerPrefix+grpcframe.HeaderStatus)

		reqID := r.Header.Get(reqcommon.RequestIDHeaderKey)
		w.dst.Header().Set(http.TrailerPrefix+canonicalStatus, strconv.Itoa(int(grpccodes.Unavailable)))
		w.dst.Header().Set(http.TrailerPrefix+http.CanonicalHeaderKey(grpcframe.HeaderMessage), grpcframe.EncodeStatusMessage(reqcommon.InternalErrorMessage(reqID)))
	}
}

type vllmNIXLSerialLeg struct {
	s       *Server
	scanned *vllmwire.Request
	p2p     p2pRouting
}

func (l *vllmNIXLSerialLeg) report(context.Context) {}

func (l *vllmNIXLSerialLeg) preparePrefill(
	w http.ResponseWriter,
	r *http.Request,
	uuidStr string,
	prefillKV map[string]any,
) (func(*http.Request), int, func(trace.Span), bool) {
	segs, ok := l.s.buildLegSegmentsOrWriteError(w, r, l.scanned, vllmwire.LegOptions{
		RequestID:        uuidStr,
		CapSingleToken:   true,
		Media:            prefillMediaMode(l.p2p),
		KVTransferParams: prefillKV,
		ECTransferParams: l.p2p.ecParams,
	})
	if !ok {
		return nil, 0, nil, false
	}
	return func(preq *http.Request) {
		setUnaryGeneratePath(preq)
		grpcframe.SetSegmentedBody(preq, segs)
	}, int(grpcframe.SegmentsLen(segs)), nil, true
}

func (l *vllmNIXLSerialLeg) shouldRetryPrefill(pw *bufferedResponseWriter) bool {
	return isRetryableVLLMPrefillFailure(pw)
}

func (l *vllmNIXLSerialLeg) extractPrefill(
	w http.ResponseWriter,
	r *http.Request,
	uuidStr string,
	pw *bufferedResponseWriter,
	prefillErr error,
	prefillSpan trace.Span,
) (any, int, bool, bool) {
	logger := log.FromContext(r.Context())
	outcome, extractErr := extractVLLMPrefillFinishInfo(pw, prefillErr)
	if extractErr != nil {
		observeStageError(r.Context(), routing.StagePrefill)
		logger.Error(extractErr, "gRPC prefill request failed",
			"request_id", uuidStr,
			"status", pw.statusCode,
			logBodyBytesKey, pw.buffer.Len())
		prefillSpan.SetStatus(otelcodes.Error, "prefill request failed")
		prefillSpan.End()
		l.s.relayVLLMPrefillFailure(w, r, pw, outcome)
		return nil, 0, false, false
	}
	prefillSpan.End()

	if len(outcome.kvTransferParams) == 0 {
		logger.Info("warning: missing 'kv_transfer_params' in gRPC prefiller FinishInfo")
	}
	var pKVAny any
	if outcome.kvTransferParams != nil {
		pKVAny = outcome.kvTransferParams
	}
	cached := 0
	rewrite := len(outcome.kvTransferParams) > 0 && outcome.numCachedTokens != nil
	if rewrite {
		cached = int(*outcome.numCachedTokens)
	}
	return pKVAny, cached, rewrite, true
}

func (l *vllmNIXLSerialLeg) streamingEnabled(r *http.Request) bool {
	return strings.TrimSuffix(r.URL.Path, "/") == vllmpb.Inference_GenerateStream_FullMethodName
}

func (l *vllmNIXLSerialLeg) prepareDecode(
	w http.ResponseWriter,
	r *http.Request,
	dreq *http.Request,
	uuidStr string,
	pKVTransferParams any,
) (
	func(http.ResponseWriter, int, bool, bool) (http.ResponseWriter, func() error),
	func(trace.Span),
	func(http.ResponseWriter, *http.Request),
	int,
	bool,
) {
	kvParams, _ := pKVTransferParams.(map[string]any)
	segs, ok := l.s.buildLegSegmentsOrWriteError(w, r, l.scanned, vllmwire.LegOptions{
		RequestID:        uuidStr,
		CapSingleToken:   false,
		Media:            vllmwire.MediaNone,
		KVTransferParams: kvParams,
	})
	if !ok {
		return nil, nil, nil, 0, false
	}
	contentLen := grpcframe.SetSegmentedBody(dreq, segs)
	wrapWriter := func(dst http.ResponseWriter, cachedTokens int, rewrite, _ bool) (http.ResponseWriter, func() error) {
		if !rewrite || cachedTokens < 0 {
			return dst, func() error { return nil }
		}
		rw := newVLLMCachedTokensResponseWriter(dst, uint32(cachedTokens)) //#nosec G115 -- guarded by cachedTokens >= 0 and sourced from uint32
		return rw, func() error {
			rw.finish(dreq)
			return nil
		}
	}
	dispatch := func(decodeWriter http.ResponseWriter, dreq *http.Request) {
		l.s.decoderProxy.ServeHTTP(decodeWriter, dreq)
	}
	return wrapWriter, nil, dispatch, int(contentLen), true
}

type vllmNIXLParallelLeg struct {
	s       *Server
	psegs   [][]byte
	dsegs   [][]byte
	outcome vllmPrefillOutcome
}

func (l *vllmNIXLParallelLeg) preparePrefill(preq *http.Request, _ trace.Span) int {
	setUnaryGeneratePath(preq)
	return int(grpcframe.SetSegmentedBody(preq, l.psegs))
}

func (l *vllmNIXLParallelLeg) prepareDecode(dreq *http.Request) int {
	return int(grpcframe.SetSegmentedBody(dreq, l.dsegs))
}

func (l *vllmNIXLParallelLeg) observeDecodeSpan(trace.Span) {}

func (l *vllmNIXLParallelLeg) checkPrefill(pw *bufferedResponseWriter) (bool, error) {
	outcome, err := extractVLLMPrefillFinishInfo(pw, nil)
	l.outcome = outcome
	return err != nil, err
}

func (l *vllmNIXLParallelLeg) decodeWriter(w http.ResponseWriter, dreq *http.Request, _ string, _ *bufferedResponseWriter) (http.ResponseWriter, func() error) {
	if len(l.outcome.kvTransferParams) == 0 || l.outcome.numCachedTokens == nil {
		return w, nil
	}
	rw := newVLLMCachedTokensResponseWriter(w, *l.outcome.numCachedTokens)
	return rw, func() error {
		rw.finish(dreq)
		return nil
	}
}

func (l *vllmNIXLParallelLeg) relayPrefillFailure(w http.ResponseWriter, r *http.Request, uuidStr string, pw *bufferedResponseWriter, err error) {
	logger := log.FromContext(r.Context())
	status := 0
	bodyBytes := 0
	if pw != nil {
		status = pw.statusCode
		bodyBytes = pw.buffer.Len()
	}
	logger.Info("concurrent-dispatch: gRPC prefill failed, aborting decode",
		"request_id", uuidStr, "p_status", status, logBodyBytesKey, bodyBytes)
	outcome, _ := extractVLLMPrefillFinishInfo(pw, err)
	l.s.relayVLLMPrefillFailure(w, r, pw, outcome)
}

// handleNIXLV2VLLMGRPC orchestrates NIXLv2 P/D disaggregation over vllm.Inference
// using the shared serial and parallel NIXLv2 orchestrators.
func (s *Server) handleNIXLV2VLLMGRPC(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running NIXL protocol V2 over gRPC", "url", prefillPodHostPort, "path", r.URL.Path)

	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	source := p2p.pullSource(logger)

	uuidStr, err := s.nixlRequestIDFn()
	if err != nil {
		logger.Error(err, "failed to generate request ID")
		_ = errorBadGateway(w, r)
		return
	}

	if s.config.MoRIIOParallelDispatch && s.config.MoRIIOWriteMode {
		transferID := "tx" + uuidStr
		dpRank, prefillKV, decodeKV := s.buildNIXLV2ParallelKVParams(r.Context(), uuidStr, transferID, prefillPodHostPort, source)
		psegs, pok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
			RequestID:        uuidStr,
			CapSingleToken:   true,
			Media:            prefillMediaMode(p2p),
			KVTransferParams: prefillKV,
			ECTransferParams: p2p.ecParams,
		})
		if !pok {
			return
		}
		dsegs, dok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
			RequestID:        uuidStr,
			CapSingleToken:   false,
			Media:            vllmwire.MediaNone,
			KVTransferParams: decodeKV,
		})
		if !dok {
			return
		}
		s.executeNIXLV2WriteParallel(w, r, uuidStr, prefillPodHostPort, dpRank, &vllmNIXLParallelLeg{
			s:     s,
			psegs: psegs,
			dsegs: dsegs,
		})
		return
	}

	s.runNIXLV2Serial(w, r, uuidStr, prefillPodHostPort, source, &vllmNIXLSerialLeg{
		s:       s,
		scanned: scanned,
		p2p:     p2p,
	})
}

// handleMooncakeVLLMGRPC orchestrates Mooncake concurrent P/D over vllm.Inference.
func (s *Server) handleMooncakeVLLMGRPC(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running Mooncake protocol over gRPC", "url", prefillPodHostPort)

	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}

	dpRank, prefillKV, decodeKV, err := s.prepareMooncakeKVParams(r.Context(), prefillPodHostPort)
	if err != nil {
		logger.Error(jsonErrorForLog(err), "failed to query mooncake engine ID", "prefill", prefillPodHostPort)
		_ = errorBadGateway(w, r)
		return
	}

	prefillBody, pok := s.buildLegFrameOrWriteError(w, r, scanned, vllmwire.LegOptions{
		CapSingleToken:   true,
		Media:            prefillMediaMode(p2p),
		KVTransferParams: prefillKV,
		ECTransferParams: p2p.ecParams,
	})
	if !pok {
		return
	}
	decodeBody, dok := s.buildLegFrameOrWriteError(w, r, scanned, vllmwire.LegOptions{
		CapSingleToken:   false,
		Media:            vllmwire.MediaNone,
		KVTransferParams: decodeKV,
	})
	if !dok {
		return
	}

	s.runConcurrentPD(w, r, prefillBody, decodeBody, prefillPodHostPort, constants.KVConnectorMooncake, func(prefillReq, _ *http.Request) {
		setUnaryGeneratePath(prefillReq)
		prefillReq.Header.Set(requestHeaderDataParallelRank, dpRank)
	}, nil)
}

type vllmSequentialPDLeg struct {
	s            *Server
	connector    string
	psegs        [][]byte
	dsegs        [][]byte
	cachedTokens *uint32
}

func (l *vllmSequentialPDLeg) report(context.Context) {}

func (l *vllmSequentialPDLeg) runPrefill(
	prefillCtx context.Context,
	prefillHandler http.Handler,
	pw *bufferedResponseWriter,
	r *http.Request,
	_ trace.Span,
) error {
	preq := r.Clone(prefillCtx)
	setUnaryGeneratePath(preq)
	grpcframe.SetSegmentedBody(preq, l.psegs)
	return servePrefill(prefillHandler, pw, preq)
}

func (l *vllmSequentialPDLeg) completePrefill(
	prefillCtx context.Context,
	w http.ResponseWriter,
	r *http.Request,
	pw *bufferedResponseWriter,
	prefillErr error,
	prefillSpan trace.Span,
) bool {
	if prefillErr != nil && prefillCtx.Err() != nil {
		prefillSpan.End()
		return false
	}
	outcome, extractErr := extractVLLMPrefillFinishInfo(pw, prefillErr)
	if extractErr != nil {
		observeStageError(r.Context(), routing.StagePrefill)
		prefillSpan.SetStatus(otelcodes.Error, "prefill request failed")
		prefillSpan.End()
		log.FromContext(r.Context()).Error(extractErr, "sequential gRPC prefill failed",
			"connector", l.connector, "status", pw.statusCode)
		l.s.relayVLLMPrefillFailure(w, r, pw, outcome)
		return false
	}
	prefillSpan.End()
	l.cachedTokens = outcome.numCachedTokens
	return true
}

func (l *vllmSequentialPDLeg) serveDecode(
	decodeCtx context.Context,
	decodeWriter http.ResponseWriter,
	r *http.Request,
	_ trace.Span,
) {
	l.s.serveVLLMGRPCDecode(decodeWriter, r.Clone(decodeCtx), l.dsegs, l.cachedTokens)
}

// handleP2PVLLMGRPC orchestrates OffloadingConnector P2P sequential P/D over vllm.Inference.
func (s *Server) handleP2PVLLMGRPC(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting) {
	logger := log.FromContext(r.Context())
	if p2p.prefill == nil {
		err := fmt.Errorf("request header %s is absent or not a host:port with a TCP port; the EPP must run p2p-source-producer with prefillProfileName matching the disaggregation prefill profile", routing.KVCachePrefillHeader)
		logger.Error(err, "cannot build the decode request", "prefill_host", extractHost(prefillPodHostPort))
		_ = errorInternalServerError(w, r, err)
		return
	}

	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	source := p2p.pullSource(logger)
	kvRequestID := newUUID()

	prefillKV := reqcommon.RemoteDecoderParams(kvRequestID)
	addP2PPullToPrefill(prefillKV, source)
	decodeKV := reqcommon.RemotePrefillerParams(kvRequestID, p2p.prefill.host, p2p.prefill.port)

	reqIDOverride := ""
	if scanned.RequestID() == "" {
		reqIDOverride = kvRequestID
	}

	psegs, pok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
		RequestID:        reqIDOverride,
		CapSingleToken:   true,
		Media:            prefillMediaMode(p2p),
		KVTransferParams: prefillKV,
		ECTransferParams: p2p.ecParams,
	})
	if !pok {
		return
	}
	dsegs, dok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
		RequestID:        reqIDOverride,
		CapSingleToken:   false,
		Media:            vllmwire.MediaNone,
		KVTransferParams: decodeKV,
	})
	if !dok {
		return
	}

	s.runSequentialPD(w, r, prefillPodHostPort, constants.KVConnectorOffloading, &vllmSequentialPDLeg{
		s:         s,
		connector: constants.KVConnectorOffloading,
		psegs:     psegs,
		dsegs:     dsegs,
	})
}

// decodeWithP2PSourceVLLMGRPC serves a decoder-only vllm.Inference request with
// kv_transfer_params rebuilt as {remote_kv_source: source}.
func (s *Server) decodeWithP2PSourceVLLMGRPC(w http.ResponseWriter, r *http.Request, p2p p2pRouting) {
	logger := log.FromContext(r.Context())
	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	source := p2p.source
	decodeKV := map[string]any{reqcommon.FieldRemoteKVSource: source.params()}

	logger.Info("running P2P source protocol over gRPC",
		"kv_request_id", source.kvRequestID,
		"p2p_source", source.String())

	dsegs, dok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
		CapSingleToken:   false,
		Media:            decodeOnlyMediaMode(p2p),
		KVTransferParams: decodeKV,
		ECTransferParams: p2p.ecParams,
	})
	if !dok {
		return
	}
	s.serveVLLMGRPCDecode(w, r.Clone(r.Context()), dsegs, nil)
}

// handleSharedStorageVLLMGRPC orchestrates shared-storage sequential P/D over vllm.Inference.
func (s *Server) handleSharedStorageVLLMGRPC(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, p2p p2pRouting) {
	logger := log.FromContext(r.Context())
	logger.V(logging.DEBUG).Info("running Shared Storage protocol over gRPC", "url", prefillPodHostPort)

	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}

	reqIDOverride := ""
	if scanned.RequestID() == "" {
		reqIDOverride = newUUID()
	}

	psegs, pok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
		RequestID:        reqIDOverride,
		CapSingleToken:   true,
		Media:            prefillMediaMode(p2p),
		ECTransferParams: p2p.ecParams,
	})
	if !pok {
		return
	}
	dsegs, dok := s.buildLegSegmentsOrWriteError(w, r, scanned, vllmwire.LegOptions{
		RequestID:      reqIDOverride,
		CapSingleToken: false,
		Media:          vllmwire.MediaNone,
	})
	if !dok {
		return
	}

	s.runSequentialPD(w, r, prefillPodHostPort, constants.KVConnectorSharedStorage, &vllmSequentialPDLeg{
		s:         s,
		connector: constants.KVConnectorSharedStorage,
		psegs:     psegs,
		dsegs:     dsegs,
	})
}

// vllmEncoderPlan holds the validated multimodal prompt layout for EC gRPC dispatch.
type vllmEncoderPlan struct {
	items         []vllmwire.MediaItem
	hasFeatures   bool
	subTokenIDs   [][]uint32
	uuidOverrides []string
}

// prepareVLLMEncoderPlan validates multimodal prompt requirements for gRPC EC
// disaggregation. Preprocessed features items slice prompt token_ids using
// offset and length. Raw media items use one full-prompt encode request.
func prepareVLLMEncoderPlan(scanned *vllmwire.Request, uuidOverrides []string) (vllmEncoderPlan, error) {
	items, err := scanned.MediaItems()
	if err != nil {
		return vllmEncoderPlan{}, err
	}
	if len(items) == 0 {
		return vllmEncoderPlan{}, nil
	}
	if scanned.Prompt != vllmwire.PromptTokenIDs {
		return vllmEncoderPlan{}, errors.New("multimodal vllm.Inference requests require prompt token_ids")
	}
	promptTokenIDs, err := scanned.TokenIDs()
	if err != nil {
		return vllmEncoderPlan{}, err
	}
	if len(promptTokenIDs) == 0 {
		return vllmEncoderPlan{}, errors.New("multimodal vllm.Inference requests require non-empty prompt token_ids")
	}

	hasFeatures := items[0].HasFeatures
	for i := 1; i < len(items); i++ {
		if items[i].HasFeatures != hasFeatures {
			return vllmEncoderPlan{}, errors.New("cannot mix raw media and preprocessed features in media")
		}
	}

	plan := vllmEncoderPlan{
		items:         items,
		hasFeatures:   hasFeatures,
		uuidOverrides: uuidOverrides,
	}
	if !hasFeatures {
		return plan, nil
	}

	plan.subTokenIDs = make([][]uint32, len(items))
	bosToken := promptTokenIDs[0]
	promptLen := uint64(len(promptTokenIDs))
	for i, item := range items {
		if item.Length == 0 || item.Offset > promptLen || item.Length > promptLen-item.Offset {
			return vllmEncoderPlan{}, fmt.Errorf("media[%d] features range [%d, +%d) exceeds prompt length %d", i, item.Offset, item.Length, promptLen)
		}
		ids := make([]uint32, 1+item.Length)
		ids[0] = bosToken
		copy(ids[1:], promptTokenIDs[item.Offset:item.Offset+item.Length])
		plan.subTokenIDs[i] = ids
	}
	return plan, nil
}

// fanoutVLLMEncoder dispatches unary /vllm.Inference/Generate encoder requests.
// Preprocessed features items fan out one sub-request per item across
// encoderHostPorts. Raw media items send one full-prompt request so models
// with start and end markers see every placeholder paired with its media item.
func (s *Server) fanoutVLLMEncoder(
	ctx context.Context,
	baseReq *http.Request,
	scanned *vllmwire.Request,
	plan vllmEncoderPlan,
	encoderHostPorts []string,
	requestID string,
	prepareItemEC func(idx int) map[string]any,
	prepareBatchEC func() map[string]any,
	perItem func(idx int, outcome vllmPrefillOutcome) error,
	onBatch func(outcome vllmPrefillOutcome) error,
) error {
	if !plan.hasFeatures {
		targetHost := encoderHostPorts[0]
		encoderHandler, err := s.encoderProxyHandler(targetHost)
		if err != nil {
			return fmt.Errorf("create encoder handler for %s: %w", targetHost, err)
		}
		var ecParams map[string]any
		if prepareBatchEC != nil {
			ecParams = prepareBatchEC()
		}
		subReqID := requestID + "-mm-0"
		segs, err := scanned.FrameSegments(vllmwire.LegOptions{
			RequestID:        subReqID,
			CapSingleToken:   true,
			Media:            vllmwire.MediaAll,
			ECTransferParams: ecParams,
		})
		if err != nil {
			return fmt.Errorf("build raw media encoder frame: %w", err)
		}
		subReq := baseReq.Clone(ctx)
		setUnaryGeneratePath(subReq)
		grpcframe.SetSegmentedBody(subReq, segs)
		subReq.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		subReq.Header.Set(reqcommon.RequestIDHeaderKey, subReqID)

		pw := &bufferedResponseWriter{}
		runErr := servePrefill(encoderHandler, pw, subReq)
		outcome, extractErr := extractVLLMPrefillFinishInfo(pw, runErr)
		if extractErr != nil {
			return fmt.Errorf("encoder %s failed: %w", targetHost, extractErr)
		}
		if onBatch != nil {
			return onBatch(outcome)
		}
		return nil
	}

	grp, grpCtx := errgroup.WithContext(ctx)
	numEncoders := len(encoderHostPorts)
	one := uint64(1)

	for idx, item := range plan.items {
		grp.Go(func() error {
			targetHost := encoderHostPorts[idx%numEncoders]
			encoderHandler, err := s.encoderProxyHandler(targetHost)
			if err != nil {
				return fmt.Errorf("create encoder handler for %s: %w", targetHost, err)
			}
			var ecParams map[string]any
			if prepareItemEC != nil {
				ecParams = prepareItemEC(idx)
			}
			uuidOverride := ""
			if idx < len(plan.uuidOverrides) {
				uuidOverride = plan.uuidOverrides[idx]
			}
			subReqID := fmt.Sprintf("%s-mm-%d", requestID, idx)
			frame, err := scanned.EncoderSubFrame(vllmwire.EncoderSubFrameOptions{
				RequestID:        subReqID,
				TokenIDs:         plan.subTokenIDs[idx],
				Item:             item,
				UUIDOverride:     uuidOverride,
				FeaturesOffset:   &one,
				ECTransferParams: ecParams,
			})
			if err != nil {
				return fmt.Errorf("build encoder frame for item %d: %w", idx, err)
			}

			subReq := cloneRequestWithBody(grpCtx, baseReq, frame)
			setUnaryGeneratePath(subReq)
			subReq.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			subReq.Header.Set(reqcommon.RequestIDHeaderKey, subReqID)

			pw := &bufferedResponseWriter{}
			runErr := servePrefill(encoderHandler, pw, subReq)
			outcome, extractErr := extractVLLMPrefillFinishInfo(pw, runErr)
			if extractErr != nil {
				return fmt.Errorf("encoder %s failed for item %d: %w", targetHost, idx, extractErr)
			}
			if perItem != nil {
				return perItem(idx, outcome)
			}
			return nil
		})
	}
	return grp.Wait()
}

// handleECSharedStorageVLLMGRPC primes the encoder cache over vllm.Inference
// before forwarding to the configured P/D connector or local decoder.
func (s *Server) handleECSharedStorageVLLMGRPC(
	w http.ResponseWriter,
	r *http.Request,
	prefillEndPoint string,
	encodeEndPoints []string,
	p2p p2pRouting,
) {
	logger := log.FromContext(r.Context())
	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	requestID, ok := s.newECRequestID(w, r)
	if !ok {
		return
	}

	encodeRan := false
	if len(encodeEndPoints) > 0 && scanned.HasMedia() {
		plan, err := prepareVLLMEncoderPlan(scanned, nil)
		if err != nil {
			grpcframe.WriteError(w, grpccodes.InvalidArgument, err.Error())
			return
		}
		encodeStart := time.Now()
		if err := s.fanoutVLLMEncoder(r.Context(), r, scanned, plan, encodeEndPoints, requestID, nil, nil, nil, nil); err != nil {
			observeStageError(r.Context(), routing.StageEncode)
			logger.Error(err, "gRPC encoder primer failed", "requestID", requestID)
			_ = errorBadGateway(w, r)
			return
		}
		observeStageDuration(r.Context(), routing.StageEncode, time.Since(encodeStart))
		encodeRan = true
	}

	s.runVLLMGRPCPDPipeline(w, r, scanned, prefillEndPoint, requestID, p2p, nil, encodeRan)
}

// handleECNIXLVLLMGRPC dispatches encoder requests over vllm.Inference, merges
// FinishInfo.ec_transfer_params, and forwards to P/D or decode.
func (s *Server) handleECNIXLVLLMGRPC(
	w http.ResponseWriter,
	r *http.Request,
	prefillEndPoint string,
	encodeEndPoints []string,
	p2p p2pRouting,
) {
	logger := log.FromContext(r.Context())
	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	requestID, ok := s.newECRequestID(w, r)
	if !ok {
		return
	}

	var mergedEC map[string]any
	encodeRan := false
	if len(encodeEndPoints) > 0 && scanned.HasMedia() {
		plan, err := prepareVLLMEncoderPlan(scanned, nil)
		if err != nil {
			grpcframe.WriteError(w, grpccodes.InvalidArgument, err.Error())
			return
		}
		var (
			params      = make(map[string]any)
			paramsMu    sync.Mutex
			contributed int
		)
		collect := func(outcome vllmPrefillOutcome) error {
			if len(outcome.ecTransferParams) == 0 {
				return nil
			}
			paramsMu.Lock()
			defer paramsMu.Unlock()
			maps.Copy(params, outcome.ecTransferParams)
			contributed++
			return nil
		}
		encodeStart := time.Now()
		err = s.fanoutVLLMEncoder(
			r.Context(),
			r,
			scanned,
			plan,
			encodeEndPoints,
			requestID,
			nil,
			nil,
			func(_ int, outcome vllmPrefillOutcome) error { return collect(outcome) },
			collect,
		)
		if err != nil {
			observeStageError(r.Context(), routing.StageEncode)
			logger.Error(err, "gRPC EC-NIXL encoder failed", "requestID", requestID)
			_ = errorBadGateway(w, r)
			return
		}
		observeStageDuration(r.Context(), routing.StageEncode, time.Since(encodeStart))
		encodeRan = true
		if contributed > 0 {
			mergedEC = params
		} else {
			logger.Info("warning: no gRPC encoder response carried ec_transfer_params, forwarding without it",
				"requestID", requestID, "items", len(plan.items))
		}
	}

	s.runVLLMGRPCPDPipeline(w, r, scanned, prefillEndPoint, requestID, p2p, mergedEC, encodeRan)
}

// handleECMooncakeVLLMGRPC dispatches encoder pushes over vllm.Inference for
// ECMooncakeConnector and forwards ec_transfer_params to the EC consumer.
func (s *Server) handleECMooncakeVLLMGRPC(
	w http.ResponseWriter,
	r *http.Request,
	prefillEndPoint string,
	encodeEndPoints []string,
	p2p p2pRouting,
) {
	logger := log.FromContext(r.Context())
	scanned, ok := s.requestVLLMGRPC(r, w, p2p)
	if !ok {
		return
	}
	requestID, ok := s.newECRequestID(w, r)
	if !ok {
		return
	}

	var mergedEC map[string]any
	encodeRan := false
	if len(encodeEndPoints) > 0 && scanned.HasMedia() {
		rawItems, parseErr := scanned.MediaItems()
		if parseErr != nil {
			grpcframe.WriteError(w, grpccodes.InvalidArgument, parseErr.Error())
			return
		}
		consumerZMQ, zmqErr := s.ecMooncakeConsumerZMQ(prefillEndPoint)
		if zmqErr != nil {
			observeStageError(r.Context(), routing.StageEncode)
			logger.Error(zmqErr, "gRPC EC-Mooncake consumer address missing", "requestID", requestID)
			_ = errorBadGateway(w, r)
			return
		}

		mcItems := make([]ecMooncakeItem, len(rawItems))
		uuidOverrides := make([]string, len(rawItems))
		needsMediaRewrite := false
		for i, it := range rawItems {
			mcItems[i].transferID = newUUID()
			id := it.UUID
			if !it.HasFeatures && id == "" {
				id = it.ContentDigest()
				uuidOverrides[i] = id
				needsMediaRewrite = true
			}
			mcItems[i].uuid = id
		}

		if needsMediaRewrite {
			rewritten := make([][]byte, len(rawItems))
			for i, it := range rawItems {
				rewritten[i] = it.RewrittenBody(uuidOverrides[i], nil)
			}
			if err := scanned.SetMediaBodies(rewritten); err != nil {
				grpcframe.WriteError(w, grpccodes.Internal, err.Error())
				return
			}
		}

		plan, planErr := prepareVLLMEncoderPlan(scanned, uuidOverrides)
		if planErr != nil {
			grpcframe.WriteError(w, grpccodes.InvalidArgument, planErr.Error())
			return
		}

		matchItem := func(idx int, params map[string]any) error {
			key := mcItems[idx].uuid
			if _, ok := params[key]; !ok || key == "" {
				if len(params) != 1 {
					return fmt.Errorf("encoder metadata for item %d cannot be matched: %d entries", idx, len(params))
				}
				for k := range params {
					key = k
				}
			}
			reported, ok := params[key].(map[string]any)
			if !ok {
				return fmt.Errorf("encoder ec_transfer_params entry for item %d is %T, want map", idx, params[key])
			}
			mcItems[idx].ecMMHash = key
			mcItems[idx].reported = reported
			return nil
		}

		prepareItemEC := func(idx int) map[string]any {
			return map[string]any{
				ecMooncakeFieldConsumerZMQ: consumerZMQ,
				ecMooncakeFieldItems: []map[string]any{
					{ecMooncakeFieldTransferID: mcItems[idx].transferID},
				},
			}
		}
		prepareBatchEC := func() map[string]any {
			items := make([]map[string]any, len(mcItems))
			for i := range mcItems {
				items[i] = map[string]any{ecMooncakeFieldTransferID: mcItems[i].transferID}
			}
			return map[string]any{
				ecMooncakeFieldConsumerZMQ: consumerZMQ,
				ecMooncakeFieldItems:       items,
			}
		}
		perItem := func(idx int, outcome vllmPrefillOutcome) error {
			if len(outcome.ecTransferParams) == 0 {
				return nil
			}
			return matchItem(idx, outcome.ecTransferParams)
		}
		onBatch := func(outcome vllmPrefillOutcome) error {
			if len(outcome.ecTransferParams) == 0 {
				return nil
			}
			for i := range mcItems {
				if err := matchItem(i, outcome.ecTransferParams); err != nil {
					return err
				}
			}
			return nil
		}

		encodeStart := time.Now()
		if err := s.fanoutVLLMEncoder(r.Context(), r, scanned, plan, encodeEndPoints, requestID, prepareItemEC, prepareBatchEC, perItem, onBatch); err != nil {
			observeStageError(r.Context(), routing.StageEncode)
			logger.Error(err, "gRPC EC-Mooncake encoder failed", "requestID", requestID)
			_ = errorBadGateway(w, r)
			return
		}
		observeStageDuration(r.Context(), routing.StageEncode, time.Since(encodeStart))
		encodeRan = true

		params := make(map[string]any)
		var transferItems []map[string]any
		for i := range mcItems {
			if mcItems[i].ecMMHash != "" {
				params[mcItems[i].ecMMHash] = mcItems[i].reported
				transferItems = append(transferItems, mcItems[i].transferItem())
			}
		}
		if len(transferItems) > 0 {
			params[ecMooncakeFieldItems] = transferItems
		}
		if len(params) > 0 {
			mergedEC = params
		}
	}

	s.runVLLMGRPCPDPipeline(w, r, scanned, prefillEndPoint, requestID, p2p, mergedEC, encodeRan)
}

func (s *Server) newECRequestID(w http.ResponseWriter, r *http.Request) (string, bool) {
	reqUUID, err := uuid.NewUUID()
	if err != nil {
		log.FromContext(r.Context()).Error(err, "failed to generate request ID")
		_ = errorBadGateway(w, r)
		return "", false
	}
	return reqUUID.String(), true
}

// runVLLMGRPCPDPipeline forwards a post-encoder vllm.Inference request to the
// configured P/D connector or, when no prefiller was selected, directly to decode.
func (s *Server) runVLLMGRPCPDPipeline(
	w http.ResponseWriter,
	r *http.Request,
	scanned *vllmwire.Request,
	prefillEndPoint string,
	requestID string,
	p2p p2pRouting,
	ecParams map[string]any,
	encodeRan bool,
) {
	pdRequest := r.Clone(r.Context())
	pdRequest.Header.Add(reqcommon.RequestIDHeaderKey, requestID)

	nextP2P := p2pRouting{
		prefill:   p2p.prefill,
		source:    p2p.source,
		grpcReq:   scanned,
		ecParams:  ecParams,
		encodeRan: encodeRan,
	}

	if len(prefillEndPoint) > 0 {
		s.handlePDConnector(w, pdRequest, prefillEndPoint, nextP2P, reqcommon.APITypeVLLMGenerate)
		return
	}

	if p2p.source != nil {
		decodeStart := time.Now()
		decodeWriter, decodeStatus := captureResponseStatus(w)
		decodeReturned := false
		defer recordDecodeAbort(r.Context(), &decodeReturned, decodeStart)
		s.decodeWithP2PSourceVLLMGRPC(decodeWriter, pdRequest, nextP2P)
		decodeReturned = true
		observeStageDuration(r.Context(), routing.StageDecode, time.Since(decodeStart))
		if decodeStatus.failed() {
			observeStageError(r.Context(), routing.StageDecode)
		}
		return
	}

	dsegs, dok := s.buildLegSegmentsOrWriteError(w, pdRequest, scanned, vllmwire.LegOptions{
		CapSingleToken:   false,
		Media:            decodeOnlyMediaMode(nextP2P),
		ECTransferParams: ecParams,
	})
	if !dok {
		return
	}
	decodeStart := time.Now()
	decodeWriter, decodeStatus := captureResponseStatus(w)
	decodeReturned := false
	defer recordDecodeAbort(r.Context(), &decodeReturned, decodeStart)
	s.serveVLLMGRPCDecode(decodeWriter, pdRequest, dsegs, nil)
	decodeReturned = true
	observeStageDuration(r.Context(), routing.StageDecode, time.Since(decodeStart))
	if decodeStatus.failed() {
		observeStageError(r.Context(), routing.StageDecode)
	}
}
