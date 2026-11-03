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

	prefillReq := proto.Clone(reqCtx.GenerateRequest).(*sglangapiv1.GenerateRequest)
	prefillReq.BootstrapHost = nil
	prefillReq.BootstrapPort = optionalInt64One(port)
	prefillReq.BootstrapRoom = optionalInt64One(room)

	bodyBytes, err := grpcframe.MarshalProto(prefillReq)
	if err != nil {
		return fmt.Errorf("prefill: marshal GenerateRequest: %w", err)
	}

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
		bodyStr := readBoundedErrorBody(resp.Body)
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		return &pipeline.UpstreamError{Step: PrefillStepName, StatusCode: resp.StatusCode, Body: bodyStr}
	}

	if statusStr, hasStatus := grpcframe.Status(resp.Header); hasStatus && statusStr != grpcframe.StatusOK {
		msg := resp.Header.Get(grpcframe.HeaderMessage)
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		writeRawGRPCError(reqCtx.ResponseWriter, statusStr, msg)
		return &pipeline.ResponseWrittenError{
			Step:       PrefillStepName,
			StatusCode: grpcStatusToHTTPStatus(statusStr),
			Cause:      &pipeline.UpstreamError{Step: PrefillStepName, StatusCode: grpcStatusToHTTPStatus(statusStr)},
		}
	}

	prefillEndpoint := strings.TrimSpace(routing.TakeRoutingHeaderValue(resp.Header, routing.PrefillEndpointHeader))
	if prefillEndpoint == "" {
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		cause := fmt.Errorf("prefill: response missing %s header", routing.PrefillEndpointHeader)
		logger.Error(cause, "prefill response missing endpoint header", "header", routing.PrefillEndpointHeader)
		grpcframe.WriteError(reqCtx.ResponseWriter, codes.Internal, fmt.Sprintf("internal error (request_id: %s)", reqCtx.RequestID))
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

	go drainSGLangPrefillStream(prefillCtx, logger, resp, ps)
	return nil
}

func drainSGLangPrefillStream(ctx context.Context, logger logr.Logger, resp *http.Response, ps *pipeline.SGLangPrefillStream) {
	defer ps.MarkDone()
	defer func() { _ = resp.Body.Close() }()

	for {
		payload, err := grpcframe.ReadFrame(resp.Body, maxSGLangGRPCFrameBytes)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if ctx.Err() == nil {
				logger.Error(err, "prefill gRPC stream read error")
				ps.Fail(err, strconv.FormatUint(uint64(codes.Unavailable), 10), "prefill stream interrupted")
			}
			return
		}
		var item sglangapiv1.GenerateStreamItem
		if err := proto.Unmarshal(payload, &item); err != nil {
			if ctx.Err() == nil {
				logger.Error(err, "prefill GenerateStreamItem decode error")
				ps.Fail(err, strconv.FormatUint(uint64(codes.Internal), 10), "invalid prefill stream frame")
			}
			return
		}
		if streamErr := item.GetError(); streamErr != nil && ctx.Err() == nil {
			msg := streamErr.GetError().GetMessage()
			ps.Fail(fmt.Errorf("prefill stream error: %s", msg), strconv.FormatUint(uint64(codes.Internal), 10), msg)
			return
		}
		if frame := item.GetFrame(); frame != nil && frame.GetMetaInfo() != nil {
			ps.RecordCachedTokens(frame.GetMetaInfo().GetCachedTokens())
		}
	}

	if statusStr, ok := grpcframe.Status(resp.Trailer); ok && statusStr != grpcframe.StatusOK && ctx.Err() == nil {
		msg := resp.Trailer.Get(grpcframe.HeaderMessage)
		ps.Fail(fmt.Errorf("prefill returned grpc-status %s", statusStr), statusStr, msg)
	}
}

func (s *DecodeStep) executeSGLangGRPC(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	decodeReq := proto.Clone(reqCtx.GenerateRequest).(*sglangapiv1.GenerateRequest)
	decodeReq.BootstrapHost = nil
	decodeReq.BootstrapPort = nil
	decodeReq.BootstrapRoom = nil

	if len(reqCtx.KVTransferParams) > 0 {
		kvParams := s.kv.PrepareDecodeKVParams(ctx, reqCtx)
		host, _ := kvParams[reqcommon.FieldBootstrapHost].(string)
		port, portOK := int64FromParam(kvParams[reqcommon.FieldBootstrapPort])
		room, roomOK := int64FromParam(kvParams[reqcommon.FieldBootstrapRoom])
		if host == "" || !portOK || !roomOK {
			return errors.New("decode: incomplete SGLang bootstrap parameters")
		}
		decodeReq.BootstrapHost = optionalStringOne(host)
		decodeReq.BootstrapPort = optionalInt64One(port)
		decodeReq.BootstrapRoom = optionalInt64One(room)
	}

	bodyBytes, err := grpcframe.MarshalProto(decodeReq)
	if err != nil {
		return fmt.Errorf("decode: marshal GenerateRequest: %w", err)
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
			select {
			case <-ps.Done():
			default:
				graceTimer := time.AfterFunc(sglangPrefillGrace, ps.Close)
				go func() {
					<-ps.Done()
					graceTimer.Stop()
				}()
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
			if pStatus, pMsg, prefillErr := ps.Failure(); prefillErr != nil {
				writeRawGRPCError(reqCtx.ResponseWriter, pStatus, pMsg)
				return &pipeline.ResponseWrittenError{
					Step:       PrefillStepName,
					StatusCode: grpcStatusToHTTPStatus(pStatus),
					Cause:      &pipeline.UpstreamError{Step: PrefillStepName, StatusCode: grpcStatusToHTTPStatus(pStatus)},
				}
			}
		}
		grpcframe.WriteError(reqCtx.ResponseWriter, codes.Unavailable, fmt.Sprintf("internal error (request_id: %s)", reqCtx.RequestID))
		return &pipeline.ResponseWrittenError{
			Step:       DecodeStepName,
			StatusCode: http.StatusBadGateway,
			Cause:      err,
		}
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		bodyStr := readBoundedErrorBody(resp.Body)
		grpcframe.WriteError(reqCtx.ResponseWriter, pipeline.HTTPStatusToGRPCCode(resp.StatusCode), fmt.Sprintf("upstream HTTP %d (request_id: %s)", resp.StatusCode, reqCtx.RequestID))
		return &pipeline.ResponseWrittenError{
			Step:       DecodeStepName,
			StatusCode: resp.StatusCode,
			Cause:      &pipeline.UpstreamError{Step: DecodeStepName, StatusCode: resp.StatusCode, Body: bodyStr},
		}
	}

	pipeline.StripInternalHeaders(resp.Header)
	w := reqCtx.ResponseWriter
	if statusStr, hasStatus := grpcframe.Status(resp.Header); hasStatus && statusStr != grpcframe.StatusOK {
		writeRawGRPCError(w, statusStr, resp.Header.Get(grpcframe.HeaderMessage))
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: grpcStatusToHTTPStatus(statusStr)}
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

	for {
		payload, readErr := grpcframe.ReadFrame(resp.Body, maxSGLangGRPCFrameBytes)
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return fmt.Errorf("decode: %w", ctxErr)
			}
			if ps != nil {
				if pStatus, pMsg, prefillErr := ps.Failure(); prefillErr != nil {
					if !committed {
						writeRawGRPCError(w, pStatus, pMsg)
					} else {
						setPrefixedGRPCTrailer(w.Header(), pStatus, pMsg)
					}
					return &pipeline.UpstreamStreamedError{
						Step:       PrefillStepName,
						StatusCode: grpcStatusToHTTPStatus(pStatus),
						Cause:      prefillErr,
					}
				}
			}
			if !committed {
				grpcframe.WriteError(w, codes.Unavailable, fmt.Sprintf("internal error (request_id: %s)", reqCtx.RequestID))
				return &pipeline.ResponseWrittenError{Step: DecodeStepName, StatusCode: http.StatusBadGateway, Cause: readErr}
			}
			setPrefixedGRPCTrailer(w.Header(), strconv.FormatUint(uint64(codes.Unavailable), 10), "decode stream interrupted")
			return &pipeline.UpstreamStreamedError{Step: DecodeStepName, Cause: readErr}
		}

		framedOut, err := rewriteDecodeFrameCachedTokens(decodeCtx, payload, ps)
		if err != nil {
			logger.Error(err, "failed to process decode GenerateStreamItem")
			framedOut, err = grpcframe.FramePayload(payload)
			if err != nil {
				return err
			}
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
		if pStatus, pMsg, prefillErr := ps.Failure(); prefillErr != nil {
			if !committed {
				writeRawGRPCError(w, pStatus, pMsg)
			} else {
				setPrefixedGRPCTrailer(w.Header(), pStatus, pMsg)
			}
			return &pipeline.UpstreamStreamedError{
				Step:       PrefillStepName,
				StatusCode: grpcStatusToHTTPStatus(pStatus),
				Cause:      prefillErr,
			}
		}
	}

	if !committed {
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		if statusStr, ok := grpcframe.Status(resp.Trailer); ok {
			w.Header().Set(grpcframe.HeaderStatus, statusStr)
			if msg := resp.Trailer.Get(grpcframe.HeaderMessage); msg != "" {
				w.Header().Set(grpcframe.HeaderMessage, msg)
			}
		} else if _, hasHeaderStatus := grpcframe.Status(resp.Header); !hasHeaderStatus {
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
		}
		w.WriteHeader(http.StatusOK)
	} else {
		grpcframe.CopyTrailers(w.Header(), resp.Trailer)
		if _, ok := grpcframe.Status(resp.Trailer); !ok {
			if headerStatus, hasHeader := grpcframe.Status(resp.Header); hasHeader {
				w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, headerStatus)
			} else {
				w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
			}
		}
	}

	if statusStr, ok := grpcframe.Status(resp.Trailer); ok && statusStr != grpcframe.StatusOK {
		return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: grpcStatusToHTTPStatus(statusStr)}
	}
	decodeSucceeded = true
	return nil
}

func rewriteDecodeFrameCachedTokens(ctx context.Context, payload []byte, ps *pipeline.SGLangPrefillStream) ([]byte, error) {
	if ps == nil {
		return grpcframe.FramePayload(payload)
	}
	var item sglangapiv1.GenerateStreamItem
	if err := proto.Unmarshal(payload, &item); err != nil {
		return nil, err
	}
	frame := item.GetFrame()
	if frame == nil || frame.GetMetaInfo() == nil {
		return grpcframe.FramePayload(payload)
	}
	if frame.GetMetaInfo().GetFinishReason() != nil {
		ps.Wait(ctx, sglangPrefillWaitOnTerminal)
	}
	cached, ok := ps.CachedTokens()
	if !ok || frame.GetMetaInfo().GetCachedTokens() == cached {
		return grpcframe.FramePayload(payload)
	}
	frame.MetaInfo.CachedTokens = cached
	return grpcframe.MarshalProto(&item)
}

func streamGRPCToGateway(ctx context.Context, gwClient *gateway.Client, upstream, path string, body []byte, headers map[string]string) (*http.Response, error) {
	coordmetrics.IncUpstreamRequestTotal(upstream)
	start := time.Now()
	resp, err := gwClient.StreamGRPC(ctx, path, body, headers)
	coordmetrics.RecordUpstreamRequestDuration(upstream, time.Since(start))
	return resp, err
}

func writeRawGRPCError(w http.ResponseWriter, statusStr, rawPercentEncodedMsg string) {
	h := w.Header()
	h.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	if statusStr == "" {
		statusStr = strconv.FormatUint(uint64(codes.Internal), 10)
	}
	h.Set(grpcframe.HeaderStatus, statusStr)
	if rawPercentEncodedMsg != "" {
		h.Set(grpcframe.HeaderMessage, rawPercentEncodedMsg)
	}
	w.WriteHeader(http.StatusOK)
}

func setPrefixedGRPCTrailer(dst http.Header, statusStr, rawPercentEncodedMsg string) {
	dst.Set(http.TrailerPrefix+grpcframe.HeaderStatus, statusStr)
	if rawPercentEncodedMsg != "" {
		dst.Set(http.TrailerPrefix+grpcframe.HeaderMessage, rawPercentEncodedMsg)
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

func readBoundedErrorBody(r io.Reader) string {
	const maxErrBytes = 4096
	raw, _ := io.ReadAll(io.LimitReader(r, maxErrBytes))
	return string(raw)
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

func optionalStringOne(v string) *sglangapiv1.OptionalStringOrList {
	return &sglangapiv1.OptionalStringOrList{
		Value: &sglangapiv1.OptionalStringOrList_One{
			One: &sglangapiv1.OptionalString{Value: proto.String(v)},
		},
	}
}

func optionalInt64One(v int64) *sglangapiv1.OptionalInt64OrList {
	return &sglangapiv1.OptionalInt64OrList{
		Value: &sglangapiv1.OptionalInt64OrList_One{
			One: &sglangapiv1.OptionalInt64{Value: proto.Int64(v)},
		},
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
