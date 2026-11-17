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
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/protoscan"
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

// sglangPrefillWaitOnTerminal bounds how long decode waits for prefill's
// cached_tokens frame or trailing status before finishing the stream.
// A variable so tests can widen it under race-detector load.
var sglangPrefillWaitOnTerminal = 100 * time.Millisecond

// sglangPrefillGrace limits how long a concurrent prefill stream may linger
// after decode has already finished. A variable so tests can shorten it.
var sglangPrefillGrace = 30 * time.Second

var sglangInputIDsField = protoscan.FieldNum(&sglangapiv1.GenerateRequest{}, "input_ids")

func (s *PrefillStep) executeSGLangGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, sglangReq *pipeline.SGLangGRPCRequest) error {
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

	// The leg body is the client payload bytes, shared with decode and not
	// copied, between a frame header and the bootstrap fields.
	payload := sglangReq.Payload
	head, tail, err := sglangwire.LegEnvelope(len(payload), &sglangwire.Bootstrap{Port: port, Room: room})
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

	resp, err := streamGRPCToGateway(prefillCtx, s.gwClient, coordmetrics.UpstreamPrefill, path, gatewayHeaders(reqCtx, gateway.PhasePrefill), head, payload, tail)
	if err != nil {
		ps.Close()
		ps.MarkDone()
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("prefill: %w", ctxErr)
		}
		return fmt.Errorf("prefill: %w: %w", &pipeline.UpstreamError{Step: PrefillStepName}, err)
	}

	if err := checkGRPCLegHeaders(PrefillStepName, reqCtx, resp); err != nil {
		_ = resp.Body.Close()
		ps.Close()
		ps.MarkDone()
		return err
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
		return internalUpstreamError(PrefillStepName, reqCtx, cause)
	}

	reqCtx.KVTransferParams[reqcommon.FieldBootstrapHost] = extractHost(prefillEndpoint)
	reqCtx.KVTransferParams[reqcommon.FieldBootstrapPort] = port
	reqCtx.KVTransferParams[reqcommon.FieldBootstrapRoom] = room
	reqCtx.PeerTopology = resp.Header.Get(reqcommon.PeerTopologyHeaderKey)
	reqCtx.CaptureResponseHeaders(resp.Header)
	sglangReq.PrefillStream = ps

	// Return once response headers arrive: SGLang prefill blocks in its
	// bootstrap queue until decode connects with the matching bootstrap_room,
	// so waiting for prefill body frames before starting decode would deadlock.
	go drainSGLangPrefillStream(prefillCtx, logger, resp, ps)
	return nil
}

func drainSGLangPrefillStream(ctx context.Context, logger logr.Logger, resp *http.Response, ps *pipeline.SGLangPrefillStream) {
	defer ps.MarkDone()
	defer func() { _ = resp.Body.Close() }()

	frames := grpcframe.NewReader(resp.Body, maxGRPCFrameBytes)
	defer frames.Release()
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
		// The wire scan reads index, cached_tokens, and prompt_tokens without
		// decoding output or logprob fields.
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
			if item.PromptTokens > 0 {
				ps.RecordPromptTokens(item.Index, item.PromptTokens)
			}
			ps.RecordCachedTokens(item.Index, item.CachedTokens)
		}
	}

	if ctx.Err() != nil {
		return
	}
	st, ok := finalGRPCStatus(resp)
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

func (s *DecodeStep) executeSGLangGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, sglangReq *pipeline.SGLangGRPCRequest) error {
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

	payload := sglangReq.Payload
	head, tail, err := sglangwire.LegEnvelope(len(payload), bootstrap)
	if err != nil {
		return fmt.Errorf("decode: build GenerateRequest: %w", err)
	}

	ps := sglangReq.PrefillStream
	sglangReq.PrefillStream = nil

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

	headers := decodeLegHeaders(reqCtx)
	path := sglangapiv1.SglangService_Generate_FullMethodName
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path, "stream", reqCtx.Stream)

	resp, err := streamGRPCToGateway(decodeCtx, s.gwClient, coordmetrics.UpstreamDecode, path, headers, head, payload, tail)
	if err != nil {
		if ps != nil && ctx.Err() == nil {
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
		return legTransportError(ctx, DecodeStepName, reqCtx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if err := checkGRPCDecodeHeaders(reqCtx, resp); err != nil {
		return err
	}

	var (
		rewrite   frameRewriter
		checkPeer peerFailureCheck
	)
	if ps != nil {
		var (
			parityReported0 bool
			parityReported  map[uint32]bool
			onTerminal      func(index, promptTokens uint32)
		)
		if s.promptTokenParity != promptTokenParityOff {
			onTerminal = func(index, promptTokens uint32) {
				if index == 0 {
					if parityReported0 {
						return
					}
					parityReported0 = true
				} else {
					if parityReported[index] {
						return
					}
					if parityReported == nil {
						parityReported = make(map[uint32]bool)
					}
					parityReported[index] = true
				}
				expected, ok := ps.PromptTokens(index)
				if !ok || expected == promptTokens {
					return
				}
				recordPromptTokenMismatch(
					logger,
					s.promptTokenParity,
					DecodeStepName,
					reqCtx.RequestID,
					reqCtx.Model,
					reqcommon.APITypeSGLangGenerate,
					sglangPayloadHasInputIDs(payload),
					int(expected),
					int(promptTokens),
				)
			}
		}
		rewrite = func(frame []byte) ([]byte, error) {
			return rewriteSGLangDecodeFrame(decodeCtx, logger, frame, ps, onTerminal)
		}
		checkPeer = func(wait bool) (string, grpcStatusWire, error, bool) {
			if wait && !ps.Wait(decodeCtx, sglangPrefillWaitOnTerminal) {
				logger.V(logutil.DEBUG).Info("prefill stream still open after decode ended", "waited", sglangPrefillWaitOnTerminal)
			}
			pStatus, pMsg, pDetails, prefillErr := ps.Failure()
			if prefillErr == nil {
				return "", grpcStatusWire{}, nil, false
			}
			return PrefillStepName, grpcStatusWire{status: pStatus, message: pMsg, detailsBin: pDetails}, prefillErr, true
		}
	}

	if _, err := relayGRPCStream(ctx, logger, DecodeStepName, reqCtx, resp, false, rewrite, checkPeer); err != nil {
		return err
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
	return rewriteSGLangDecodeFrame(ctx, logger, frame, ps, nil)
}

func rewriteSGLangDecodeFrame(
	ctx context.Context,
	logger logr.Logger,
	frame []byte,
	ps *pipeline.SGLangPrefillStream,
	onTerminal func(index, promptTokens uint32),
) ([]byte, error) {
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
	if onTerminal != nil && scanned.PromptTokens > 0 {
		onTerminal(index, scanned.PromptTokens)
	}
	cached, ok := ps.CachedTokens(index)
	if !ok || scanned.CachedTokens == cached {
		return frame, nil
	}
	var item sglangapiv1.GenerateStreamItem
	if err := proto.Unmarshal(payload, &item); err != nil {
		return frame, err
	}
	meta := item.GetFrame().GetMetaInfo()
	if meta == nil {
		return frame, nil
	}
	meta.CachedTokens = cached
	rewritten, err := grpcframe.MarshalProto(&item)
	if err != nil {
		return frame, err
	}
	return rewritten, nil
}

func sglangPayloadHasInputIDs(payload []byte) bool {
	found := false
	_ = protoscan.ForEachField(payload, func(num protowire.Number, _ protowire.Type, _, _ []byte) error {
		if num == sglangInputIDsField {
			found = true
		}
		return nil
	})
	return found
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
