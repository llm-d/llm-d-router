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

package steps

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/sglangwire"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	// maxSGLangGRPCFrameBytes bounds individual upstream SGLang gRPC stream frames.
	maxSGLangGRPCFrameBytes = 64 << 20
	// sglangPrefillWaitOnTerminal bounds how long decode waits for prefill's
	// cached_tokens frame before emitting decode's terminal frame.
	sglangPrefillWaitOnTerminal = 100 * time.Millisecond
	// sglangPrefillGrace limits how long a concurrent prefill stream may linger
	// after decode has already finished.
	sglangPrefillGrace = 30 * time.Second
)

type grpcStatusWire struct {
	status     string
	message    string
	detailsBin string
}

func (s *PrefillStep) executeSGLangGRPC(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)
	if s.kv.Name() != kv.SGLang {
		return fmt.Errorf("prefill: SGLang gRPC Generate requires %s connector, got %s", kv.SGLang, s.kv.Name())
	}

	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)
	port, ok := int64FromParam(kvParams[reqcommon.FieldBootstrapPort])
	if !ok {
		return errors.New("prefill: missing SGLang bootstrap_port")
	}
	room, ok := int64FromParam(kvParams[reqcommon.FieldBootstrapRoom])
	if !ok {
		return errors.New("prefill: missing SGLang bootstrap_room")
	}

	// Copy the client payload byte for byte and set only the bootstrap
	// fields: no decode, deep copy, or re-encode of token IDs or media.
	bodyBytes, err := sglangwire.FrameGenerateRequest(reqCtx.GRPCGeneratePayload, &sglangwire.Bootstrap{Port: port, Room: room})
	if err != nil {
		return fmt.Errorf("prefill: build GenerateRequest: %w", err)
	}

	// Cancel prefill if the client disconnects while decode is running, while
	// allowing DecodeStep to detach from r.Context() once decode finishes so
	// prefill's trailing cleanup is not aborted by request teardown.
	prefillCtx, cancelPrefill := context.WithCancel(context.WithoutCancel(ctx))
	stopAfter := context.AfterFunc(ctx, cancelPrefill)
	ps := pipeline.NewSGLangPrefillStream(cancelPrefill, stopAfter)

	path := sglangapiv1.SglangService_Generate_FullMethodName
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path)

	resp, err := streamGRPCToGateway(prefillCtx, s.gwClient, coordmetrics.UpstreamPrefill, path, bodyBytes, gatewayHeaders(reqCtx, gateway.PhasePrefill))
	if err != nil {
		ps.Close()
		ps.MarkDone()
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("prefill: %w", ctxErr)
		}
		return fmt.Errorf("prefill: %w: %w", &pipeline.UpstreamError{Step: PrefillStepName}, err)
	}

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		err := upstreamError(PrefillStepName, resp.StatusCode, readErrorBody(resp.Body))
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		return err
	}

	if st, hasStatus := extractGRPCStatus(resp.Header); hasStatus && st.status != grpcframe.StatusOK {
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		writeRawGRPCError(reqCtx.ResponseWriter, st)
		httpStatus := grpcStatusToHTTPStatus(st.status)
		return &pipeline.ResponseWrittenError{
			Step:       PrefillStepName,
			StatusCode: httpStatus,
			Cause:      &pipeline.UpstreamError{Step: PrefillStepName, StatusCode: httpStatus},
		}
	}

	// GenerateResponse has no bootstrap field, so the gateway must echo the
	// selected prefill worker's host:port on the response headers.
	prefillEndpoint := strings.TrimSpace(routing.TakeRoutingHeaderValue(resp.Header, routing.PrefillEndpointHeader))
	if prefillEndpoint == "" {
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		cause := fmt.Errorf("prefill: response missing %s header", routing.PrefillEndpointHeader)
		logger.Error(cause, "prefill response missing endpoint header", "header", routing.PrefillEndpointHeader)
		grpcframe.WriteError(reqCtx.ResponseWriter, codes.Internal, reqcommon.InternalErrorMessage(reqCtx.RequestID))
		return &pipeline.ResponseWrittenError{
			Step:       PrefillStepName,
			StatusCode: http.StatusBadGateway,
			Cause:      cause,
		}
	}

	reqCtx.KVTransferParams[reqcommon.FieldBootstrapHost] = extractHost(prefillEndpoint)
	reqCtx.KVTransferParams[reqcommon.FieldBootstrapPort] = port
	reqCtx.KVTransferParams[reqcommon.FieldBootstrapRoom] = room
	reqCtx.PeerTopology = resp.Header.Get(reqcommon.PeerTopologyHeaderKey)
	reqCtx.CaptureResponseHeaders(resp.Header)
	reqCtx.PrefillStream = ps

	// Return once response headers arrive: SGLang prefill blocks in its
	// bootstrap queue until decode connects with the matching bootstrap_room,
	// so waiting for prefill body frames before starting decode would deadlock.
	go drainSGLangPrefillStream(prefillCtx, logger, resp, ps)
	return nil
}

func drainSGLangPrefillStream(ctx context.Context, logger logr.Logger, resp *http.Response, ps *pipeline.SGLangPrefillStream) {
	defer ps.MarkDone()
	defer func() { _ = resp.Body.Close() }()

	frames := grpcframe.NewReader(resp.Body, maxSGLangGRPCFrameBytes)
	for {
		frame, err := frames.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if ctx.Err() == nil {
				logger.Error(err, "prefill gRPC stream read error")
				ps.Fail(
					err,
					strconv.FormatUint(uint64(codes.Unavailable), 10),
					grpcframe.EncodeStatusMessage("prefill stream interrupted"),
					"",
				)
			}
			return
		}
		payload := frame[grpcframe.HeaderLen:]
		// The wire scan reads index and cached_tokens without decoding
		// output, logprob, or prompt token fields.
		item, err := sglangwire.ScanStreamItem(payload)
		if err != nil {
			if ctx.Err() == nil {
				logger.Error(err, "prefill GenerateStreamItem decode error")
				ps.Fail(
					err,
					strconv.FormatUint(uint64(codes.Internal), 10),
					grpcframe.EncodeStatusMessage("invalid prefill stream frame"),
					"",
				)
			}
			return
		}
		if item.IsError && ctx.Err() == nil {
			failPrefillStreamError(logger, payload, ps)
			return
		}
		if !item.IsError && item.HasMetaInfo {
			ps.RecordCachedTokens(item.Index, item.CachedTokens)
		}
	}

	if ctx.Err() != nil {
		return
	}
	st, ok := extractGRPCStatus(resp.Trailer)
	if !ok {
		st, ok = extractGRPCStatus(resp.Header)
	}
	if !ok {
		cause := errors.New("prefill: upstream closed stream without grpc-status")
		logger.Error(cause, "prefill stream closed without grpc-status")
		ps.Fail(
			cause,
			strconv.FormatUint(uint64(codes.Internal), 10),
			grpcframe.EncodeStatusMessage("prefill stream closed without grpc-status"),
			"",
		)
		return
	}
	if st.status != grpcframe.StatusOK {
		cause := fmt.Errorf("prefill returned grpc-status %s", st.status)
		logger.Error(cause, "prefill stream failed", "grpcStatus", st.status)
		ps.Fail(cause, st.status, st.message, st.detailsBin)
	}
}

// failPrefillStreamError decodes an in-stream prefill error item, which is
// rare, and records it on ps.
func failPrefillStreamError(logger logr.Logger, payload []byte, ps *pipeline.SGLangPrefillStream) {
	var item sglangapiv1.GenerateStreamItem
	if err := proto.Unmarshal(payload, &item); err != nil {
		logger.Error(err, "prefill GenerateStreamItem decode error")
		ps.Fail(
			err,
			strconv.FormatUint(uint64(codes.Internal), 10),
			grpcframe.EncodeStatusMessage("invalid prefill stream frame"),
			"",
		)
		return
	}
	errBody := item.GetError().GetError()
	msg := errBody.GetMessage()
	code := sglangErrorBodyGRPCCode(errBody.GetCode())
	cause := fmt.Errorf("prefill stream error (code %d): %s", errBody.GetCode(), msg)
	logger.Error(cause, "prefill stream reported error", "code", errBody.GetCode())
	ps.Fail(cause, strconv.FormatUint(uint64(code), 10), grpcframe.EncodeStatusMessage(msg), "")
}

func (s *DecodeStep) executeSGLangGRPC(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	// A nil bootstrap strips any client bootstrap fields (decode-only).
	var bootstrap *sglangwire.Bootstrap
	if len(reqCtx.KVTransferParams) > 0 {
		kvParams := s.kv.PrepareDecodeKVParams(ctx, reqCtx)
		host, _ := kvParams[reqcommon.FieldBootstrapHost].(string)
		port, portOK := int64FromParam(kvParams[reqcommon.FieldBootstrapPort])
		room, roomOK := int64FromParam(kvParams[reqcommon.FieldBootstrapRoom])
		if host == "" || !portOK || !roomOK {
			return errors.New("decode: incomplete SGLang bootstrap parameters")
		}
		bootstrap = &sglangwire.Bootstrap{Host: host, Port: port, Room: room}
	}

	bodyBytes, err := sglangwire.FrameGenerateRequest(reqCtx.GRPCGeneratePayload, bootstrap)
	if err != nil {
		return fmt.Errorf("decode: build GenerateRequest: %w", err)
	}

	ps := reqCtx.PrefillStream
	reqCtx.PrefillStream = nil

	decodeCtx, cancelDecode := context.WithCancelCause(ctx)
	defer cancelDecode(nil)
	if ps != nil {
		ps.OnFailure(func(prefillErr error) {
			cancelDecode(prefillErr)
		})
	}

	decodeSucceeded := false
	if ps != nil {
		defer func() {
			if !decodeSucceeded {
				ps.Close()
				return
			}
			ps.DetachRequestContext()
			// Close is idempotent, so the timer may fire after the prefill
			// reader is done without effect.
			select {
			case <-ps.Done():
			default:
				time.AfterFunc(sglangPrefillGrace, ps.Close)
			}
		}()
	}

	headers := gatewayHeaders(reqCtx, gateway.PhaseDecode)
	if reqCtx.PeerTopology != "" {
		headers[reqcommon.PeerTopologyHeaderKey] = reqCtx.PeerTopology
	}
	path := sglangapiv1.SglangService_Generate_FullMethodName
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path, "stream", reqCtx.Stream)

	resp, err := streamGRPCToGateway(decodeCtx, s.gwClient, coordmetrics.UpstreamDecode, path, bodyBytes, headers)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("decode: %w", ctxErr)
		}
		if ps != nil {
			if pStatus, pMsg, pDetails, prefillErr := ps.Failure(); prefillErr != nil {
				writeRawGRPCError(reqCtx.ResponseWriter, grpcStatusWire{status: pStatus, message: pMsg, detailsBin: pDetails})
				httpStatus := grpcStatusToHTTPStatus(pStatus)
				return &pipeline.ResponseWrittenError{
					Step:       PrefillStepName,
					StatusCode: httpStatus,
					Cause:      &pipeline.UpstreamError{Step: PrefillStepName, StatusCode: httpStatus},
				}
			}
		}
		grpcframe.WriteError(reqCtx.ResponseWriter, codes.Unavailable, reqcommon.InternalErrorMessage(reqCtx.RequestID))
		return &pipeline.ResponseWrittenError{
			Step:       DecodeStepName,
			StatusCode: http.StatusBadGateway,
			Cause:      fmt.Errorf("decode: %w: %w", &pipeline.UpstreamError{Step: DecodeStepName}, err),
		}
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		upstreamErr := upstreamError(DecodeStepName, resp.StatusCode, readErrorBody(resp.Body))
		grpcframe.WriteError(reqCtx.ResponseWriter, pipeline.HTTPStatusToGRPCCode(resp.StatusCode), fmt.Sprintf("upstream HTTP %d (request_id: %s)", resp.StatusCode, reqCtx.RequestID))
		return &pipeline.ResponseWrittenError{
			Step:       DecodeStepName,
			StatusCode: resp.StatusCode,
			Cause:      upstreamErr,
		}
	}

	pipeline.StripInternalHeaders(resp.Header)
	w := reqCtx.ResponseWriter
	if st, hasStatus := extractGRPCStatus(resp.Header); hasStatus && st.status != grpcframe.StatusOK {
		writeRawGRPCError(w, st)
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: grpcStatusToHTTPStatus(st.status)}
	}

	rc := http.NewResponseController(w)
	committed := false
	commitHeaders := func() {
		if committed {
			return
		}
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		w.WriteHeader(http.StatusOK)
		committed = true
	}

	// frames reuses one buffer for the whole stream. Each frame keeps its
	// header, so a frame that needs no rewrite is written without a copy.
	frames := grpcframe.NewReader(resp.Body, maxSGLangGRPCFrameBytes)
	loggedRewriteErr := false
	for {
		frame, readErr := frames.Next()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return fmt.Errorf("decode: %w", ctxErr)
			}
			if ps != nil {
				if pStatus, pMsg, pDetails, prefillErr := ps.Failure(); prefillErr != nil {
					pWire := grpcStatusWire{status: pStatus, message: pMsg, detailsBin: pDetails}
					if !committed {
						writeRawGRPCError(w, pWire)
					} else {
						setPrefixedGRPCTrailer(w.Header(), pWire)
					}
					return &pipeline.UpstreamStreamedError{
						Step:       PrefillStepName,
						StatusCode: grpcStatusToHTTPStatus(pStatus),
						Cause:      prefillErr,
					}
				}
			}
			wrappedErr := fmt.Errorf("decode: %w: %w", &pipeline.UpstreamError{Step: DecodeStepName}, readErr)
			if !committed {
				grpcframe.WriteError(w, codes.Unavailable, reqcommon.InternalErrorMessage(reqCtx.RequestID))
				return &pipeline.ResponseWrittenError{Step: DecodeStepName, StatusCode: http.StatusBadGateway, Cause: wrappedErr}
			}
			setPrefixedGRPCTrailer(w.Header(), grpcStatusWire{
				status:  strconv.FormatUint(uint64(codes.Unavailable), 10),
				message: grpcframe.EncodeStatusMessage("decode stream interrupted"),
			})
			return &pipeline.UpstreamStreamedError{Step: DecodeStepName, Cause: wrappedErr}
		}

		framedOut, err := rewriteDecodeFrameCachedTokens(decodeCtx, logger, frame, ps)
		if err != nil && !loggedRewriteErr {
			// Forward the frame unchanged. Log once per stream, not per frame.
			logger.Error(err, "failed to process decode GenerateStreamItem")
			loggedRewriteErr = true
		}

		commitHeaders()
		if _, writeErr := w.Write(framedOut); writeErr != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return fmt.Errorf("decode: %w", ctxErr)
			}
			return &pipeline.UpstreamStreamedError{Step: DecodeStepName, Cause: writeErr}
		}
		_ = rc.Flush()
	}

	if ps != nil {
		if pStatus, pMsg, pDetails, prefillErr := ps.Failure(); prefillErr != nil {
			pWire := grpcStatusWire{status: pStatus, message: pMsg, detailsBin: pDetails}
			if !committed {
				writeRawGRPCError(w, pWire)
			} else {
				setPrefixedGRPCTrailer(w.Header(), pWire)
			}
			return &pipeline.UpstreamStreamedError{
				Step:       PrefillStepName,
				StatusCode: grpcStatusToHTTPStatus(pStatus),
				Cause:      prefillErr,
			}
		}
	}

	st, ok := extractGRPCStatus(resp.Trailer)
	if !ok {
		st, ok = extractGRPCStatus(resp.Header)
	}
	if !ok {
		cause := fmt.Errorf("decode: %w: upstream closed stream without grpc-status", &pipeline.UpstreamError{Step: DecodeStepName})
		logger.Error(cause, "decode stream closed without grpc-status")
		if !committed {
			grpcframe.WriteError(w, codes.Internal, reqcommon.InternalErrorMessage(reqCtx.RequestID))
			return &pipeline.ResponseWrittenError{
				Step:       DecodeStepName,
				StatusCode: http.StatusBadGateway,
				Cause:      cause,
			}
		}
		setPrefixedGRPCTrailer(w.Header(), grpcStatusWire{
			status:  strconv.FormatUint(uint64(codes.Internal), 10),
			message: grpcframe.EncodeStatusMessage("decode stream closed without grpc-status"),
		})
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, Cause: cause}
	}

	if !committed {
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		writeRawGRPCStatus(w.Header(), st)
		w.WriteHeader(http.StatusOK)
	} else {
		setPrefixedGRPCTrailer(w.Header(), st)
	}

	if st.status != grpcframe.StatusOK {
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: grpcStatusToHTTPStatus(st.status)}
	}
	decodeSucceeded = true
	return nil
}

// rewriteDecodeFrameCachedTokens substitutes prefill's prompt-cache hit count
// onto terminal decode frames because the decode worker counts KV blocks
// transferred from prefill rather than hits in prefill's prompt cache.
//
// frame is a full gRPC frame, header included. A wire scan finds the
// terminal frame without allocating. Every other frame is returned as is,
// so per-token frames are forwarded without a copy. Only a terminal frame
// whose cached_tokens changes is decoded and re-encoded. On a scan error,
// rewriteDecodeFrameCachedTokens returns frame unchanged with the error.
func rewriteDecodeFrameCachedTokens(ctx context.Context, logger logr.Logger, frame []byte, ps *pipeline.SGLangPrefillStream) ([]byte, error) {
	if ps == nil {
		return frame, nil
	}
	payload := frame[grpcframe.HeaderLen:]
	scanned, err := sglangwire.ScanStreamItem(payload)
	if err != nil {
		return frame, err
	}
	if scanned.IsError || !scanned.Finished {
		return frame, nil
	}
	index := scanned.Index
	if _, ok := ps.CachedTokens(index); !ok {
		if !ps.Wait(ctx, sglangPrefillWaitOnTerminal) {
			logger.V(logutil.DEBUG).Info("timed out waiting for prefill cached_tokens on terminal decode frame", "index", index)
		}
	}
	cached, ok := ps.CachedTokens(index)
	if !ok || scanned.CachedTokens == cached {
		return frame, nil
	}
	var item sglangapiv1.GenerateStreamItem
	if err := proto.Unmarshal(payload, &item); err != nil {
		return frame, err
	}
	item.GetFrame().GetMetaInfo().CachedTokens = cached
	return grpcframe.MarshalProto(&item)
}

func streamGRPCToGateway(ctx context.Context, gwClient *gateway.Client, upstream, path string, body []byte, headers map[string]string) (*http.Response, error) {
	call := coordmetrics.StartUpstreamCall(upstream)
	resp, err := gwClient.StreamGRPC(ctx, path, body, headers)
	call.Done()
	return resp, err
}

func extractGRPCStatus(h http.Header) (grpcStatusWire, bool) {
	statusStr, ok := grpcframe.Status(h)
	if !ok {
		return grpcStatusWire{}, false
	}
	return grpcStatusWire{
		status:     statusStr,
		message:    grpcHeaderValue(h, grpcframe.HeaderMessage),
		detailsBin: grpcHeaderValue(h, grpcframe.HeaderStatusDetailsBin),
	}, true
}

func grpcHeaderValue(h http.Header, key string) string {
	canonical := http.CanonicalHeaderKey(key)
	for _, k := range [...]string{canonical, http.TrailerPrefix + canonical, http.TrailerPrefix + key} {
		for _, v := range h[k] {
			if trimmed := strings.TrimSpace(v); trimmed != "" {
				return trimmed
			}
		}
	}
	return ""
}

func writeRawGRPCError(w http.ResponseWriter, st grpcStatusWire) {
	h := w.Header()
	h.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	writeRawGRPCStatus(h, st)
	w.WriteHeader(http.StatusOK)
}

func writeRawGRPCStatus(h http.Header, st grpcStatusWire) {
	statusStr := st.status
	if statusStr == "" {
		statusStr = strconv.FormatUint(uint64(codes.Internal), 10)
	}
	h.Set(grpcframe.HeaderStatus, statusStr)
	if st.message != "" {
		h.Set(grpcframe.HeaderMessage, st.message)
	}
	if st.detailsBin != "" {
		h.Set(grpcframe.HeaderStatusDetailsBin, st.detailsBin)
	}
}

func setPrefixedGRPCTrailer(dst http.Header, st grpcStatusWire) {
	dst.Set(http.TrailerPrefix+grpcframe.HeaderStatus, st.status)
	if st.message != "" {
		dst.Set(http.TrailerPrefix+grpcframe.HeaderMessage, st.message)
	}
	if st.detailsBin != "" {
		dst.Set(http.TrailerPrefix+grpcframe.HeaderStatusDetailsBin, st.detailsBin)
	}
}

func copyUpstreamResponseHeaders(dst, src http.Header) {
	for k, vv := range src {
		lower := strings.ToLower(k)
		if lower == "content-length" || lower == "transfer-encoding" || lower == "connection" || lower == "trailer" ||
			strings.EqualFold(k, grpcframe.HeaderStatus) ||
			strings.EqualFold(k, grpcframe.HeaderMessage) ||
			strings.EqualFold(k, grpcframe.HeaderStatusDetailsBin) {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func extractHost(hostWithPort string) string {
	hostWithPort, _ = strings.CutPrefix(hostWithPort, "http://")
	hostWithPort, _ = strings.CutPrefix(hostWithPort, "https://")
	host, _, err := net.SplitHostPort(hostWithPort)
	if err != nil {
		return hostWithPort
	}
	return host
}

func int64FromParam(v any) (int64, bool) {
	switch val := v.(type) {
	case int:
		return int64(val), true
	case int64:
		return val, true
	case float64:
		return int64(val), true
	default:
		return 0, false
	}
}

func sglangErrorBodyGRPCCode(httpCode uint32) codes.Code {
	switch int(httpCode) {
	case 0, http.StatusInternalServerError:
		return codes.Internal
	case http.StatusNotFound:
		return codes.NotFound
	default:
		return pipeline.HTTPStatusToGRPCCode(int(httpCode))
	}
}

func grpcStatusToHTTPStatus(statusStr string) int {
	codeNum, err := strconv.ParseUint(statusStr, 10, 32)
	if err != nil {
		return http.StatusBadGateway
	}
	switch codes.Code(codeNum) { //#nosec G115 -- bitSize 32 bounds codeNum to uint32
	case codes.InvalidArgument, codes.FailedPrecondition, codes.OutOfRange:
		return http.StatusBadRequest
	case codes.Unauthenticated:
		return http.StatusUnauthorized
	case codes.PermissionDenied:
		return http.StatusForbidden
	case codes.NotFound:
		return http.StatusNotFound
	case codes.ResourceExhausted:
		return http.StatusTooManyRequests
	case codes.DeadlineExceeded:
		return http.StatusGatewayTimeout
	default:
		return http.StatusBadGateway
	}
}
