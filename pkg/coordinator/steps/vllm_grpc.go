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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func vllmMethodPath(stream bool) string {
	if stream {
		return vllmpb.Inference_GenerateStream_FullMethodName
	}
	return vllmpb.Inference_Generate_FullMethodName
}

// vllmUnaryResult holds the scanned fields and decoded transfer-parameter maps
// from a unary /vllm.Inference/Generate worker response.
type vllmUnaryResult struct {
	info     vllmwire.ResponseInfo
	kvParams map[string]any
	ecParams map[string]any
}

func (s *ConditionalDecodeStep) executeVLLMGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, vllmReq *pipeline.VLLMGRPCRequest) error {
	logger := log.FromContext(ctx).WithName(ConditionalDecodeStepName)

	// Conditional-decode probes decode workers directly; decode workers never
	// decode media bytes (Section 7.1).
	segs, err := vllmReq.Req.FrameSegments(vllmwire.LegOptions{
		Media: vllmwire.MediaNone,
	})
	if err != nil {
		return fmt.Errorf("%s: build GenerateRequest: %w", ConditionalDecodeStepName, err)
	}

	headers := gatewayHeaders(reqCtx, gateway.PhaseDecode)
	headers["Prefer"] = "if-available"
	path := vllmMethodPath(reqCtx.Stream)
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path, "stream", reqCtx.Stream)

	resp, err := streamGRPCToGateway(ctx, s.gwClient, coordmetrics.UpstreamConditionalDecode, path, headers, segs...)
	if err != nil {
		if ctx.Err() == nil {
			coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultTransportError)
		}
		return legTransportError(ctx, ConditionalDecodeStepName, reqCtx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusPreconditionFailed {
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultDeferred)
		logger.V(logutil.DEFAULT).Info("cache miss (HTTP 412), continuing pipeline")
		return nil
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultError)
		return uncommittedHTTPError(ConditionalDecodeStepName, reqCtx, resp)
	}

	pipeline.StripInternalHeaders(resp.Header)
	if st, hasStatus := extractGRPCStatus(resp.Header); hasStatus && st.status != grpcframe.StatusOK {
		if st.status == failedPreconditionStatus {
			coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultDeferred)
			logger.V(logutil.DEFAULT).Info("cache miss (FAILED_PRECONDITION), continuing pipeline")
			return nil
		}
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultError)
		return upstreamGRPCError(ConditionalDecodeStepName, reqCtx, st)
	}

	outcome, err := forwardVLLMDecodeResponse(ctx, logger, ConditionalDecodeStepName, reqCtx, vllmReq, resp, true, s.promptTokenParity)
	switch {
	case outcome.Deferred:
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultDeferred)
		logger.V(logutil.DEFAULT).Info("cache miss (FAILED_PRECONDITION trailer), continuing pipeline")
		return nil
	case outcome.Committed:
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultServed)
	case outcome.TransportErr:
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultTransportError)
	case err != nil && !pipeline.IsClientCancellation(ctx, err):
		coordmetrics.IncConditionalDecodeProbes(coordmetrics.ProbeResultError)
	}
	if err != nil {
		return err
	}
	logger.V(logutil.DEFAULT).Info("cache hit, response forwarded")
	return pipeline.ErrPipelineDone
}

func (s *EncodeStep) executeVLLMGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, vllmReq *pipeline.VLLMGRPCRequest) error {
	if !vllmReq.Req.HasMedia() {
		return nil
	}
	logger := log.FromContext(ctx).WithName(EncodeStepName)
	reqCtx.EncodeFanout = 1

	segs, err := vllmReq.Req.FrameSegments(vllmwire.LegOptions{
		CapSingleToken: true,
		Media:          vllmwire.MediaAll,
	})
	if err != nil {
		return fmt.Errorf("%s: build GenerateRequest: %w", EncodeStepName, err)
	}

	path := vllmpb.Inference_Generate_FullMethodName
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path)

	resp, err := streamGRPCToGateway(ctx, s.gwClient, coordmetrics.UpstreamEncode, path, gatewayHeaders(reqCtx, gateway.PhaseEncode), segs...)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("%s: %w", EncodeStepName, ctxErr)
		}
		return fmt.Errorf("%s: %w: %w", EncodeStepName, &pipeline.UpstreamError{Step: EncodeStepName}, err)
	}
	defer func() { _ = resp.Body.Close() }()

	res, err := readUnaryVLLMResponse(ctx, logger, EncodeStepName, reqCtx, resp)
	if err != nil {
		return err
	}
	vllmReq.EncodeRan = true
	s.ec.MergeEncodeResponse(ctx, reqCtx, res.ecParams)
	reqCtx.CaptureResponseHeaders(resp.Header)

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func (s *PrefillStep) executeVLLMGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, vllmReq *pipeline.VLLMGRPCRequest) error {
	logger := log.FromContext(ctx).WithName(PrefillStepName)
	if s.kv.Name() == kv.SGLang {
		return fmt.Errorf("prefill: vLLM gRPC Generate does not support %s connector", kv.SGLang)
	}

	ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	kvParams := s.kv.PreparePrefillKVParams(ctx, reqCtx)

	mediaMode := vllmwire.MediaAll
	if vllmReq.EncodeRan {
		mediaMode = vllmwire.MediaMetadataOnly
	}
	segs, err := vllmReq.Req.FrameSegments(vllmwire.LegOptions{
		CapSingleToken:   true,
		Media:            mediaMode,
		KVTransferParams: kvParams,
		ECTransferParams: ecParams,
	})
	if err != nil {
		return fmt.Errorf("prefill: build GenerateRequest: %w", err)
	}

	path := vllmpb.Inference_Generate_FullMethodName
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path)

	resp, err := streamGRPCToGateway(ctx, s.gwClient, coordmetrics.UpstreamPrefill, path, gatewayHeaders(reqCtx, gateway.PhasePrefill), segs...)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("prefill: %w", ctxErr)
		}
		return fmt.Errorf("prefill: %w: %w", &pipeline.UpstreamError{Step: PrefillStepName}, err)
	}
	defer func() { _ = resp.Body.Close() }()

	res, err := readUnaryVLLMResponse(ctx, logger, PrefillStepName, reqCtx, resp)
	if err != nil {
		return err
	}

	reqCtx.PeerTopology = resp.Header.Get(reqcommon.PeerTopologyHeaderKey)
	if reader, ok := s.kv.(kv.PrefillResponseReader); ok {
		if err := reader.ReadPrefillResponse(reqCtx, resp.Header); err != nil {
			return fmt.Errorf("prefill: %w", err)
		}
	}

	reqCtx.KVTransferParams = res.kvParams
	vllmReq.PrefillRan = true
	if res.info.HasPromptInfo {
		n := res.info.NumPromptTokens
		vllmReq.PrefillPromptTokens = &n
	}

	var cachedTokens *int
	if res.info.HasCachedTokens {
		c := int(res.info.NumCachedTokens)
		cachedTokens = &c
	}
	recordPrefillOutcome(logger, reqCtx, s.kv.Name(), path, cachedTokens)

	reqCtx.CaptureResponseHeaders(resp.Header)

	logger.V(logutil.DEFAULT).Info("complete")
	return nil
}

func (s *DecodeStep) executeVLLMGRPC(ctx context.Context, reqCtx *pipeline.RequestContext, vllmReq *pipeline.VLLMGRPCRequest) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)
	if s.kv.Name() == kv.SGLang {
		return fmt.Errorf("decode: vLLM gRPC Generate does not support %s connector", kv.SGLang)
	}

	var opts vllmwire.LegOptions
	switch {
	case vllmReq.PrefillRan:
		// EngineCore pulls KV from prefill; strip all media items and do not
		// attach ec_transfer_params because kv-offloading uses
		// remote_prefiller rather than do_remote_prefill and would otherwise
		// also run EC lookup.
		opts.Media = vllmwire.MediaNone
		opts.KVTransferParams = s.kv.PrepareDecodeKVParams(ctx, reqCtx)
	case vllmReq.EncodeRan:
		opts.Media = vllmwire.MediaMetadataOnly
		if len(reqCtx.ECTransferParams) > 0 {
			ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
			if err != nil {
				return fmt.Errorf("decode: %w", err)
			}
			opts.ECTransferParams = ecParams
		}
	default:
		opts.Media = vllmwire.MediaAll
		if len(reqCtx.ECTransferParams) > 0 {
			ecParams, err := s.ec.PreparePrefillECParams(ctx, reqCtx)
			if err != nil {
				return fmt.Errorf("decode: %w", err)
			}
			opts.ECTransferParams = ecParams
		}
	}

	segs, err := vllmReq.Req.FrameSegments(opts)
	if err != nil {
		return fmt.Errorf("decode: build GenerateRequest: %w", err)
	}

	headers := decodeLegHeaders(reqCtx)
	path := vllmMethodPath(reqCtx.Stream)
	logger.V(logutil.DEFAULT).Info("sending gRPC request", "path", path, "stream", reqCtx.Stream)

	resp, err := streamGRPCToGateway(ctx, s.gwClient, coordmetrics.UpstreamDecode, path, headers, segs...)
	if err != nil {
		return legTransportError(ctx, DecodeStepName, reqCtx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if err := checkGRPCDecodeHeaders(reqCtx, resp); err != nil {
		return err
	}

	_, err = forwardVLLMDecodeResponse(ctx, logger, DecodeStepName, reqCtx, vllmReq, resp, false, s.promptTokenParity)
	return err
}

func readUnaryVLLMResponse(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	resp *http.Response,
) (vllmUnaryResult, error) {
	if err := checkGRPCLegHeaders(step, reqCtx, resp); err != nil {
		return vllmUnaryResult{}, err
	}

	frames := grpcframe.NewReader(resp.Body, maxGRPCFrameBytes)
	defer frames.Release()

	frame, readErr := frames.Next()
	if errors.Is(readErr, io.EOF) {
		st, ok := finalGRPCStatus(resp)
		if ok && st.status != grpcframe.StatusOK {
			return vllmUnaryResult{}, upstreamGRPCError(step, reqCtx, st)
		}
		cause := fmt.Errorf("%s: %w: upstream unary Generate returned no message", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "unary gRPC response contained no message")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	}
	if readErr != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return vllmUnaryResult{}, fmt.Errorf("%s: %w", step, ctxErr)
		}
		return vllmUnaryResult{}, fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, readErr)
	}

	payload := frame[grpcframe.HeaderLen:]
	info, err := vllmwire.ScanGenerateResponse(payload)
	if err != nil {
		cause := fmt.Errorf("%s: %w: invalid GenerateResponse: %w", step, &pipeline.UpstreamError{Step: step}, err)
		logger.Error(cause, "unary GenerateResponse wire scan failed")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	}

	var res vllmUnaryResult
	res.info = info
	res.kvParams, err = info.KVTransferParams()
	if err != nil {
		cause := fmt.Errorf("%s: %w: decode kv_transfer_params: %w", step, &pipeline.UpstreamError{Step: step}, err)
		logger.Error(cause, "failed to decode kv_transfer_params struct")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	}
	res.ecParams, err = info.ECTransferParams()
	if err != nil {
		cause := fmt.Errorf("%s: %w: decode ec_transfer_params: %w", step, &pipeline.UpstreamError{Step: step}, err)
		logger.Error(cause, "failed to decode ec_transfer_params struct")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	}

	// Drain to EOF so net/http populates resp.Trailer.
	if _, extraErr := frames.Next(); extraErr == nil {
		cause := fmt.Errorf("%s: %w: upstream unary Generate returned multiple messages", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "unary gRPC response contained extra message")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	} else if !errors.Is(extraErr, io.EOF) {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return vllmUnaryResult{}, fmt.Errorf("%s: %w", step, ctxErr)
		}
		return vllmUnaryResult{}, fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, extraErr)
	}

	st, ok := finalGRPCStatus(resp)
	if !ok {
		cause := fmt.Errorf("%s: %w: upstream closed unary stream without grpc-status", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "unary gRPC stream closed without grpc-status")
		return vllmUnaryResult{}, internalUpstreamError(step, reqCtx, cause)
	}
	if st.status != grpcframe.StatusOK {
		return vllmUnaryResult{}, upstreamGRPCError(step, reqCtx, st)
	}
	return res, nil
}

func forwardVLLMDecodeResponse(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	vllmReq *pipeline.VLLMGRPCRequest,
	resp *http.Response,
	conditionalProbe bool,
	parityMode string,
) (relayOutcome, error) {
	if vllmReq.HTTP != nil {
		return forwardVLLMAsHTTPGenerate(ctx, logger, step, reqCtx, vllmReq, resp, conditionalProbe, parityMode)
	}
	return forwardVLLMNativeGRPC(ctx, logger, step, reqCtx, vllmReq, resp, conditionalProbe, parityMode)
}

func forwardVLLMNativeGRPC(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	vllmReq *pipeline.VLLMGRPCRequest,
	resp *http.Response,
	conditionalProbe bool,
	parityMode string,
) (relayOutcome, error) {
	rewriteCached, cachedTokens := resolveCachedTokensUint32(reqCtx.PrefillCachedTokens)
	checkParity := parityMode != promptTokenParityOff && vllmReq.PrefillPromptTokens != nil
	idsForwarded := vllmReq.Req.Prompt == vllmwire.PromptTokenIDs
	parityChecked := false

	var rewrite frameRewriter
	if rewriteCached || checkParity {
		rewrite = func(frame []byte) ([]byte, error) {
			if checkParity && !parityChecked {
				if payload, pErr := grpcframe.ParsePayload(frame); pErr == nil {
					if info, sErr := vllmwire.ScanGenerateResponse(payload); sErr == nil && info.HasPromptInfo {
						parityChecked = true
						recordPromptTokenMismatch(
							logger,
							parityMode,
							step,
							reqCtx.RequestID,
							reqCtx.Model,
							reqcommon.APITypeVLLMGenerate,
							idsForwarded,
							int(*vllmReq.PrefillPromptTokens),
							int(info.NumPromptTokens),
						)
					}
				}
			}
			if rewriteCached {
				return vllmwire.RewriteTerminalCachedTokens(frame, cachedTokens)
			}
			return frame, nil
		}
	}

	return relayGRPCStream(ctx, logger, step, reqCtx, resp, conditionalProbe, rewrite, nil)
}

func forwardVLLMAsHTTPGenerate(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	vllmReq *pipeline.VLLMGRPCRequest,
	resp *http.Response,
	conditionalProbe bool,
	parityMode string,
) (relayOutcome, error) {
	if !reqCtx.Stream {
		return forwardVLLMAsHTTPUnaryGenerate(ctx, logger, step, reqCtx, vllmReq, resp, conditionalProbe, parityMode)
	}
	return forwardVLLMAsHTTPStreamGenerate(ctx, logger, step, reqCtx, vllmReq, resp, conditionalProbe, parityMode)
}

func forwardVLLMAsHTTPUnaryGenerate(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	vllmReq *pipeline.VLLMGRPCRequest,
	resp *http.Response,
	conditionalProbe bool,
	parityMode string,
) (relayOutcome, error) {
	var out relayOutcome
	frames := grpcframe.NewReader(resp.Body, maxGRPCFrameBytes)
	defer frames.Release()

	frame, readErr := frames.Next()
	if errors.Is(readErr, io.EOF) {
		st, ok := finalGRPCStatus(resp)
		if ok {
			out.Status = st
			out.HasStatus = true
		}
		if ok && conditionalProbe && st.status == failedPreconditionStatus {
			out.Deferred = true
			return out, nil
		}
		if ok && st.status != grpcframe.StatusOK {
			return out, &pipeline.UpstreamError{Step: step, StatusCode: grpcStatusToHTTPStatus(st.status)}
		}
		return out, fmt.Errorf("%s: %w: upstream unary Generate returned no message", step, &pipeline.UpstreamError{Step: step})
	}
	if readErr != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return out, fmt.Errorf("%s: %w", step, ctxErr)
		}
		out.TransportErr = true
		return out, fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, readErr)
	}

	var genResp vllmpb.GenerateResponse
	if err := proto.Unmarshal(frame[grpcframe.HeaderLen:], &genResp); err != nil {
		return out, fmt.Errorf("%s: %w: invalid GenerateResponse: %w", step, &pipeline.UpstreamError{Step: step}, err)
	}

	if _, extraErr := frames.Next(); extraErr == nil {
		return out, fmt.Errorf("%s: %w: upstream unary Generate returned multiple messages", step, &pipeline.UpstreamError{Step: step})
	} else if !errors.Is(extraErr, io.EOF) {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return out, fmt.Errorf("%s: %w", step, ctxErr)
		}
		out.TransportErr = true
		return out, fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, extraErr)
	}

	st, ok := finalGRPCStatus(resp)
	if !ok {
		cause := fmt.Errorf("%s: %w: upstream closed unary stream without grpc-status", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "unary gRPC stream closed without grpc-status", "step", step)
		return out, cause
	}
	out.Status = st
	out.HasStatus = true
	if st.status != grpcframe.StatusOK {
		return out, &pipeline.UpstreamError{Step: step, StatusCode: grpcStatusToHTTPStatus(st.status)}
	}

	if parityMode != promptTokenParityOff && vllmReq.PrefillPromptTokens != nil && genResp.GetPromptInfo() != nil {
		recordPromptTokenMismatch(
			logger,
			parityMode,
			step,
			reqCtx.RequestID,
			reqCtx.Model,
			reqcommon.APITypeVLLMGenerate,
			vllmReq.Req.Prompt == vllmwire.PromptTokenIDs,
			int(*vllmReq.PrefillPromptTokens),
			int(genResp.GetPromptInfo().GetNumPromptTokens()),
		)
	}

	httpResp := buildHTTPGenerateResponse(reqCtx, vllmReq.HTTP, &genResp)
	bodyBytes, err := json.Marshal(&httpResp)
	if err != nil {
		return out, fmt.Errorf("%s: marshal HTTP generate response: %w", step, err)
	}

	w := reqCtx.ResponseWriter
	copyUpstreamResponseHeaders(w.Header(), resp.Header)
	w.Header().Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
	w.WriteHeader(http.StatusOK)
	out.Committed = true
	if _, writeErr := w.Write(bodyBytes); writeErr != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return out, fmt.Errorf("%s: %w", step, ctxErr)
		}
		return out, &pipeline.UpstreamStreamedError{Step: step, Cause: writeErr}
	}
	return out, nil
}

func forwardVLLMAsHTTPStreamGenerate(
	ctx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	vllmReq *pipeline.VLLMGRPCRequest,
	resp *http.Response,
	conditionalProbe bool,
	parityMode string,
) (relayOutcome, error) {
	w := reqCtx.ResponseWriter
	rc := http.NewResponseController(w)
	var out relayOutcome

	commitHeaders := func() {
		if out.Committed {
			return
		}
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, reqcommon.MediaTypeEventStream)
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		out.Committed = true
	}

	reqID := httpGenerateRequestID(reqCtx, vllmReq.HTTP)
	returnPromptIDs := vllmReq.HTTP != nil && vllmReq.HTTP.ReturnTokenIDs
	includeUsage := vllmReq.HTTP != nil && vllmReq.HTTP.IncludeUsage
	checkParity := parityMode != promptTokenParityOff && vllmReq.PrefillPromptTokens != nil
	idsForwarded := vllmReq.Req.Prompt == vllmwire.PromptTokenIDs
	parityChecked := false

	var (
		promptTokens          int
		completionTokens      int
		decodeCachedTokens    *int
		pendingPromptLogprobs *httpGenerateLogprobs
		pendingPromptTokenIDs []uint32
		genResp               vllmpb.GenerateResponse
		choiceBuf             [1]httpGenerateStreamChoice
		sseBuf                bytes.Buffer
	)
	enc := json.NewEncoder(&sseBuf)
	enc.SetEscapeHTML(false)

	writeSSEChunk := func(chunk *httpGenerateStreamChunk) error {
		sseBuf.Reset()
		sseBuf.WriteString(reqcommon.SSEDataPrefix)
		if err := enc.Encode(chunk); err != nil {
			return err
		}
		sseBuf.WriteByte('\n')
		_, err := w.Write(sseBuf.Bytes())
		return err
	}

	frames := grpcframe.NewReader(resp.Body, maxGRPCFrameBytes)
	defer frames.Release()

	for {
		frame, readErr := frames.Next()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return out, fmt.Errorf("%s: %w", step, ctxErr)
			}
			wrappedErr := fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, readErr)
			if !out.Committed {
				out.TransportErr = true
				return out, wrappedErr
			}
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: wrappedErr}
		}

		genResp.Reset()
		if unmarshalErr := proto.Unmarshal(frame[grpcframe.HeaderLen:], &genResp); unmarshalErr != nil {
			wrappedErr := fmt.Errorf("%s: %w: invalid GenerateResponse: %w", step, &pipeline.UpstreamError{Step: step}, unmarshalErr)
			if !out.Committed {
				return out, wrappedErr
			}
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: wrappedErr}
		}

		if pi := genResp.GetPromptInfo(); pi != nil {
			promptTokens = int(pi.GetNumPromptTokens())
			if checkParity && !parityChecked {
				parityChecked = true
				recordPromptTokenMismatch(
					logger,
					parityMode,
					step,
					reqCtx.RequestID,
					reqCtx.Model,
					reqcommon.APITypeVLLMGenerate,
					idsForwarded,
					int(*vllmReq.PrefillPromptTokens),
					promptTokens,
				)
			}
			if lp := buildHTTPPromptLogprobs(pi); lp != nil {
				pendingPromptLogprobs = lp
			}
			if returnPromptIDs && len(pi.GetTokenIds()) > 0 {
				pendingPromptTokenIDs = pi.GetTokenIds()
			}
		}

		seq := genResp.GetOutputs()
		if seq == nil {
			continue
		}

		completionTokens += int(seq.GetNumTokens())
		if fin := seq.GetFinishInfo(); fin != nil {
			completionTokens = int(fin.GetNumOutputTokens())
			if fin.NumCachedTokens != nil {
				c := int(fin.GetNumCachedTokens())
				decodeCachedTokens = &c
			}
		}

		choiceBuf[0] = buildHTTPGenerateStreamChoice(seq)
		chunk := httpGenerateStreamChunk{
			RequestID:      reqID,
			Model:          reqCtx.Model,
			Choices:        choiceBuf[:],
			PromptTokenIDs: pendingPromptTokenIDs,
			PromptLogprobs: pendingPromptLogprobs,
		}
		pendingPromptTokenIDs = nil
		pendingPromptLogprobs = nil

		commitHeaders()
		if err := writeSSEChunk(&chunk); err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return out, fmt.Errorf("%s: %w", step, ctxErr)
			}
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: err}
		}
		_ = rc.Flush()
	}

	st, ok := finalGRPCStatus(resp)
	if !ok {
		cause := fmt.Errorf("%s: %w: upstream closed stream without grpc-status", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "gRPC stream closed without grpc-status", "step", step)
		if !out.Committed {
			return out, cause
		}
		return out, &pipeline.UpstreamStreamedError{Step: step, Cause: cause}
	}
	out.Status = st
	out.HasStatus = true

	if !out.Committed && conditionalProbe && st.status == failedPreconditionStatus {
		out.Deferred = true
		return out, nil
	}
	if st.status != grpcframe.StatusOK {
		httpStatus := grpcStatusToHTTPStatus(st.status)
		if !out.Committed {
			return out, &pipeline.UpstreamError{Step: step, StatusCode: httpStatus}
		}
		return out, &pipeline.UpstreamStreamedError{Step: step, StatusCode: httpStatus}
	}

	commitHeaders()
	if includeUsage {
		cachedPtr := decodeCachedTokens
		if reqCtx.PrefillCachedTokens != nil {
			cachedPtr = reqCtx.PrefillCachedTokens
		}
		usageVal := buildHTTPGenerateUsage(promptTokens, completionTokens, cachedPtr)
		usageChunk := httpGenerateStreamChunk{
			RequestID: reqID,
			Model:     reqCtx.Model,
			Choices:   []httpGenerateStreamChoice{},
			Usage:     &usageVal,
		}
		if err := writeSSEChunk(&usageChunk); err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return out, fmt.Errorf("%s: %w", step, ctxErr)
			}
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: err}
		}
	}
	if _, err := w.Write([]byte("data: [DONE]\n\n")); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return out, fmt.Errorf("%s: %w", step, ctxErr)
		}
		return out, &pipeline.UpstreamStreamedError{Step: step, Cause: err}
	}
	_ = rc.Flush()
	return out, nil
}

type httpGenerateUnaryResponse struct {
	RequestID        string                `json:"request_id"`
	Created          int64                 `json:"created"`
	Model            string                `json:"model,omitempty"`
	Choices          [1]httpGenerateChoice `json:"choices"`
	Usage            httpGenerateUsage     `json:"usage"`
	PromptTokenIDs   []uint32              `json:"prompt_token_ids,omitempty"`
	PromptLogprobs   *httpGenerateLogprobs `json:"prompt_logprobs"`
	KVTransferParams any                   `json:"kv_transfer_params"`
}

type httpGenerateChoice struct {
	Index        uint32                `json:"index"`
	Text         string                `json:"text,omitempty"`
	TokenIDs     []uint32              `json:"token_ids"`
	FinishReason *string               `json:"finish_reason"`
	StopReason   any                   `json:"stop_reason"`
	Logprobs     *httpGenerateLogprobs `json:"logprobs"`
}

type httpGenerateStreamChunk struct {
	RequestID      string                     `json:"request_id"`
	Model          string                     `json:"model,omitempty"`
	Choices        []httpGenerateStreamChoice `json:"choices"`
	PromptTokenIDs []uint32                   `json:"prompt_token_ids,omitempty"`
	PromptLogprobs *httpGenerateLogprobs      `json:"prompt_logprobs,omitempty"`
	Usage          *httpGenerateUsage         `json:"usage,omitempty"`
}

type httpGenerateStreamChoice struct {
	Index        uint32                `json:"index"`
	Text         string                `json:"text,omitempty"`
	TokenIDs     []uint32              `json:"token_ids"`
	FinishReason *string               `json:"finish_reason"`
	StopReason   any                   `json:"stop_reason"`
	Logprobs     *httpGenerateLogprobs `json:"logprobs,omitempty"`
}

type httpGeneratePromptDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type httpGenerateUsage struct {
	PromptTokens        int                        `json:"prompt_tokens"`
	CompletionTokens    int                        `json:"completion_tokens"`
	TotalTokens         int                        `json:"total_tokens"`
	PromptTokensDetails *httpGeneratePromptDetails `json:"prompt_tokens_details"`
}

type httpGenerateLogprobs struct {
	Content []httpGenerateTokenLogprob `json:"content"`
}

type httpGenerateTokenLogprob struct {
	TokenID     uint32                         `json:"token_id"`
	Logprob     *float32                       `json:"logprob"`
	Rank        *uint32                        `json:"rank,omitempty"`
	TopLogprobs []httpGenerateCandidateLogprob `json:"top_logprobs"`
}

type httpGenerateCandidateLogprob struct {
	TokenID uint32   `json:"token_id"`
	Logprob *float32 `json:"logprob"`
	Rank    uint32   `json:"rank,omitempty"`
}

func httpGenerateRequestID(reqCtx *pipeline.RequestContext, reply *pipeline.HTTPGenerateReply) string {
	if reply != nil && reply.ClientRequestID != "" {
		return reply.ClientRequestID
	}
	return reqCtx.RequestID
}

func buildHTTPGenerateResponse(
	reqCtx *pipeline.RequestContext,
	reply *pipeline.HTTPGenerateReply,
	genResp *vllmpb.GenerateResponse,
) httpGenerateUnaryResponse {
	pi := genResp.GetPromptInfo()
	seq := genResp.GetOutputs()
	promptTokens := int(pi.GetNumPromptTokens())
	completionTokens := int(seq.GetNumTokens())
	var cachedPtr *int
	if fin := seq.GetFinishInfo(); fin != nil {
		completionTokens = int(fin.GetNumOutputTokens())
		if fin.NumCachedTokens != nil {
			c := int(fin.GetNumCachedTokens())
			cachedPtr = &c
		}
	}
	if reqCtx.PrefillCachedTokens != nil {
		cachedPtr = reqCtx.PrefillCachedTokens
	}

	sc := buildHTTPGenerateStreamChoice(seq)
	var promptTokenIDs []uint32
	if reply != nil && reply.ReturnTokenIDs && len(pi.GetTokenIds()) > 0 {
		promptTokenIDs = pi.GetTokenIds()
	}

	return httpGenerateUnaryResponse{
		RequestID: httpGenerateRequestID(reqCtx, reply),
		Created:   time.Now().Unix(),
		Model:     reqCtx.Model,
		Choices: [1]httpGenerateChoice{{
			Index:        sc.Index,
			Text:         sc.Text,
			TokenIDs:     sc.TokenIDs,
			FinishReason: sc.FinishReason,
			StopReason:   sc.StopReason,
			Logprobs:     sc.Logprobs,
		}},
		Usage:            buildHTTPGenerateUsage(promptTokens, completionTokens, cachedPtr),
		PromptTokenIDs:   promptTokenIDs,
		PromptLogprobs:   buildHTTPPromptLogprobs(pi),
		KVTransferParams: nil,
	}
}

func buildHTTPGenerateStreamChoice(seq *vllmpb.SequenceOutput) httpGenerateStreamChoice {
	tokenIDs := seq.GetTokenIds()
	if tokenIDs == nil {
		tokenIDs = []uint32{}
	}
	var finishReason *string
	var stopReason any
	if fin := seq.GetFinishInfo(); fin != nil {
		finishReason = mapVLLMFinishReason(fin.GetFinishReason())
		stopReason = mapVLLMStopReason(fin)
	}
	return httpGenerateStreamChoice{
		Index:        seq.GetIndex(),
		Text:         seq.GetText(),
		TokenIDs:     tokenIDs,
		FinishReason: finishReason,
		StopReason:   stopReason,
		Logprobs:     buildHTTPSequenceLogprobs(seq),
	}
}

func buildHTTPSequenceLogprobs(seq *vllmpb.SequenceOutput) *httpGenerateLogprobs {
	if seq == nil {
		return nil
	}
	return buildHTTPLogprobs(seq.GetTokenIds(), seq.GetLogprobs(), seq.GetRanks(), seq.GetCandidateTokens())
}

func buildHTTPPromptLogprobs(pi *vllmpb.PromptInfo) *httpGenerateLogprobs {
	if pi == nil {
		return nil
	}
	return buildHTTPLogprobs(pi.GetTokenIds(), pi.GetLogprobs(), pi.GetRanks(), pi.GetCandidateTokens())
}

func buildHTTPLogprobs(
	tokenIDs []uint32,
	logprobs []float32,
	ranks []uint32,
	candidates []*vllmpb.CandidateTokenInfo,
) *httpGenerateLogprobs {
	if len(logprobs) == 0 && len(candidates) == 0 {
		return nil
	}
	n := max(len(logprobs), len(candidates))
	content := make([]httpGenerateTokenLogprob, 0, n)
	for i := range n {
		var tokID uint32
		if i < len(tokenIDs) {
			tokID = tokenIDs[i]
		}
		var lpPtr *float32
		if i < len(logprobs) && isFiniteFloat32(logprobs[i]) {
			v := logprobs[i]
			lpPtr = &v
		}
		var rankPtr *uint32
		if i < len(ranks) {
			r := ranks[i]
			rankPtr = &r
		}
		var cands []*vllmpb.CandidateTokenInfo_TokenInfo
		if i < len(candidates) {
			cands = candidates[i].GetTokens()
		}
		top := make([]httpGenerateCandidateLogprob, 0, len(cands))
		for _, ct := range cands {
			var clp *float32
			if isFiniteFloat32(ct.GetLogprob()) {
				v := ct.GetLogprob()
				clp = &v
			}
			top = append(top, httpGenerateCandidateLogprob{
				TokenID: ct.GetId(),
				Logprob: clp,
				Rank:    ct.GetRank(),
			})
		}
		content = append(content, httpGenerateTokenLogprob{
			TokenID:     tokID,
			Logprob:     lpPtr,
			Rank:        rankPtr,
			TopLogprobs: top,
		})
	}
	return &httpGenerateLogprobs{Content: content}
}

func isFiniteFloat32(v float32) bool {
	f := float64(v)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

func buildHTTPGenerateUsage(promptTokens, completionTokens int, cachedTokens *int) httpGenerateUsage {
	var details *httpGeneratePromptDetails
	if cachedTokens != nil {
		details = &httpGeneratePromptDetails{CachedTokens: *cachedTokens}
	}
	return httpGenerateUsage{
		PromptTokens:        promptTokens,
		CompletionTokens:    completionTokens,
		TotalTokens:         promptTokens + completionTokens,
		PromptTokensDetails: details,
	}
}

var (
	finishReasonStop   = "stop"
	finishReasonLength = "length"
	finishReasonAbort  = "abort"
)

func mapVLLMFinishReason(r vllmpb.FinishInfo_FinishReason) *string {
	switch r {
	case vllmpb.FinishInfo_STOP:
		return &finishReasonStop
	case vllmpb.FinishInfo_LENGTH:
		return &finishReasonLength
	case vllmpb.FinishInfo_ABORTED:
		return &finishReasonAbort
	default:
		return nil
	}
}

func mapVLLMStopReason(fin *vllmpb.FinishInfo) any {
	switch sr := fin.GetStopReason().(type) {
	case *vllmpb.FinishInfo_StopTokenId:
		return sr.StopTokenId
	case *vllmpb.FinishInfo_StopString:
		return sr.StopString
	default:
		return nil
	}
}

func resolveCachedTokensUint32(prefillCached *int) (bool, uint32) {
	if prefillCached == nil || *prefillCached < 0 || *prefillCached > math.MaxUint32 {
		return false, 0
	}
	return true, uint32(*prefillCached) //#nosec G115 -- guarded by bounds check above
}
