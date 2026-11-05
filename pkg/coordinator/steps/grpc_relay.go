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
	"net/http"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// maxGRPCFrameBytes bounds individual upstream gRPC stream frames read by
// coordinator steps.
const maxGRPCFrameBytes = 64 << 20

var failedPreconditionStatus = strconv.FormatUint(uint64(codes.FailedPrecondition), 10)

type grpcStatusWire struct {
	status     string
	message    string
	detailsBin string
}

// relayOutcome summarizes a gRPC stream or HTTP-translated decode relay so
// ConditionalDecodeStep can record its probe outcome in one place.
type relayOutcome struct {
	Committed    bool
	Deferred     bool
	TransportErr bool
	Status       grpcStatusWire
	HasStatus    bool
}

type frameRewriter func(frame []byte) ([]byte, error)

// peerFailureCheck inspects whether a concurrent peer leg (such as SGLang's
// concurrent prefill stream) has failed. When wait is true (at stream EOF
// before reading decode trailers), it may wait briefly for the peer leg to
// report its status.
type peerFailureCheck func(wait bool) (peerStep string, st grpcStatusWire, cause error, failed bool)

// relayGRPCStream copies upstream gRPC frames to reqCtx.ResponseWriter,
// applying rewrite to each frame when non-nil, and writes the terminal gRPC
// status headers or trailers. It writes nothing to the client when conditional
// probing sees an uncommitted FAILED_PRECONDITION trailer.
func relayGRPCStream(
	clientCtx context.Context,
	logger logr.Logger,
	step string,
	reqCtx *pipeline.RequestContext,
	resp *http.Response,
	conditionalProbe bool,
	rewrite frameRewriter,
	checkPeer peerFailureCheck,
) (relayOutcome, error) {
	w := reqCtx.ResponseWriter
	rc := http.NewResponseController(w)
	var out relayOutcome

	commitHeaders := func() {
		if out.Committed {
			return
		}
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		w.WriteHeader(http.StatusOK)
		out.Committed = true
	}

	frames := grpcframe.NewReader(resp.Body, maxGRPCFrameBytes)
	defer frames.Release()
	loggedRewriteErr := false

	for {
		frame, readErr := frames.Next()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			if ctxErr := clientCtx.Err(); ctxErr != nil {
				return out, fmt.Errorf("%s: %w", step, ctxErr)
			}
			if checkPeer != nil {
				if peerStep, pWire, peerErr, failed := checkPeer(false); failed {
					if !out.Committed {
						writeRawGRPCError(w, pWire)
					} else {
						setPrefixedGRPCTrailer(w.Header(), pWire)
					}
					return out, &pipeline.UpstreamStreamedError{
						Step:       peerStep,
						StatusCode: grpcStatusToHTTPStatus(pWire.status),
						Cause:      peerErr,
					}
				}
			}
			wrappedErr := fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, readErr)
			if !out.Committed {
				out.TransportErr = true
				grpcframe.WriteError(w, codes.Unavailable, reqcommon.InternalErrorMessage(reqCtx.RequestID))
				return out, &pipeline.ResponseWrittenError{
					Step:       step,
					StatusCode: http.StatusBadGateway,
					Cause:      wrappedErr,
				}
			}
			setPrefixedGRPCTrailer(w.Header(), grpcStatusWire{
				status:  strconv.FormatUint(uint64(codes.Unavailable), 10),
				message: grpcframe.EncodeStatusMessage(step + " stream interrupted"),
			})
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: wrappedErr}
		}

		framedOut := frame
		if rewrite != nil {
			rewritten, rewriteErr := rewrite(frame)
			if rewriteErr != nil {
				if !loggedRewriteErr {
					logger.Error(rewriteErr, "failed to rewrite gRPC stream frame", "step", step)
					loggedRewriteErr = true
				}
			} else {
				framedOut = rewritten
			}
		}

		commitHeaders()
		if _, writeErr := w.Write(framedOut); writeErr != nil {
			if ctxErr := clientCtx.Err(); ctxErr != nil {
				return out, fmt.Errorf("%s: %w", step, ctxErr)
			}
			return out, &pipeline.UpstreamStreamedError{Step: step, Cause: writeErr}
		}
		_ = rc.Flush()
	}

	if checkPeer != nil {
		if peerStep, pWire, peerErr, failed := checkPeer(true); failed {
			if !out.Committed {
				writeRawGRPCError(w, pWire)
			} else {
				setPrefixedGRPCTrailer(w.Header(), pWire)
			}
			return out, &pipeline.UpstreamStreamedError{
				Step:       peerStep,
				StatusCode: grpcStatusToHTTPStatus(pWire.status),
				Cause:      peerErr,
			}
		}
	}

	st, ok := finalGRPCStatus(resp)
	if !ok {
		cause := fmt.Errorf("%s: %w: upstream closed stream without grpc-status", step, &pipeline.UpstreamError{Step: step})
		logger.Error(cause, "gRPC stream closed without grpc-status", "step", step)
		if !out.Committed {
			grpcframe.WriteError(w, codes.Internal, reqcommon.InternalErrorMessage(reqCtx.RequestID))
			return out, &pipeline.ResponseWrittenError{
				Step:       step,
				StatusCode: http.StatusBadGateway,
				Cause:      cause,
			}
		}
		setPrefixedGRPCTrailer(w.Header(), grpcStatusWire{
			status:  strconv.FormatUint(uint64(codes.Internal), 10),
			message: grpcframe.EncodeStatusMessage(step + " stream closed without grpc-status"),
		})
		return out, &pipeline.UpstreamStreamedError{Step: step, Cause: cause}
	}

	out.Status = st
	out.HasStatus = true

	if !out.Committed {
		if conditionalProbe && st.status == failedPreconditionStatus {
			out.Deferred = true
			return out, nil
		}
		if st.status != grpcframe.StatusOK {
			writeRawGRPCError(w, st)
			return out, &pipeline.UpstreamStreamedError{Step: step, StatusCode: grpcStatusToHTTPStatus(st.status)}
		}
		copyUpstreamResponseHeaders(w.Header(), resp.Header)
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		writeRawGRPCStatus(w.Header(), st)
		w.WriteHeader(http.StatusOK)
		out.Committed = true
		return out, nil
	}

	setPrefixedGRPCTrailer(w.Header(), st)
	if st.status != grpcframe.StatusOK {
		return out, &pipeline.UpstreamStreamedError{Step: step, StatusCode: grpcStatusToHTTPStatus(st.status)}
	}
	return out, nil
}

func streamGRPCToGateway(ctx context.Context, gwClient *gateway.Client, upstream, path string, headers map[string]string, body ...[]byte) (*http.Response, error) {
	call := coordmetrics.StartUpstreamCall(upstream)
	resp, err := gwClient.StreamGRPC(ctx, path, headers, body...)
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
		message:    grpcframe.Message(h),
		detailsBin: grpcframe.StatusDetailsBin(h),
	}, true
}

// finalGRPCStatus returns the gRPC status of a finished response. It reads
// the trailers first, then the headers of a trailers-only response.
func finalGRPCStatus(resp *http.Response) (grpcStatusWire, bool) {
	if st, ok := extractGRPCStatus(resp.Trailer); ok {
		return st, true
	}
	return extractGRPCStatus(resp.Header)
}

// checkGRPCLegHeaders maps a failed leg response to a pipeline error before
// the caller reads the body: a non-2xx HTTP status, or a non-OK grpc-status
// in the headers of a trailers-only response.
func checkGRPCLegHeaders(step string, reqCtx *pipeline.RequestContext, resp *http.Response) error {
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return upstreamError(step, resp.StatusCode, readErrorBody(resp.Body))
	}
	if st, ok := extractGRPCStatus(resp.Header); ok && st.status != grpcframe.StatusOK {
		return upstreamGRPCError(step, reqCtx, st)
	}
	return nil
}

// checkGRPCDecodeHeaders is checkGRPCLegHeaders for the decode leg, which
// runs before any byte reaches the client. It also strips internal headers
// so they are not relayed.
func checkGRPCDecodeHeaders(reqCtx *pipeline.RequestContext, resp *http.Response) error {
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return uncommittedHTTPError(DecodeStepName, reqCtx, resp)
	}
	pipeline.StripInternalHeaders(resp.Header)
	if st, ok := extractGRPCStatus(resp.Header); ok && st.status != grpcframe.StatusOK {
		return decodePreCommitGRPCError(DecodeStepName, reqCtx, st)
	}
	return nil
}

// decodeLegHeaders returns the gateway headers for a decode leg, with the
// peer topology that prefill reported.
func decodeLegHeaders(reqCtx *pipeline.RequestContext) map[string]string {
	headers := gatewayHeaders(reqCtx, gateway.PhaseDecode)
	if reqCtx.PeerTopology != "" {
		headers[reqcommon.PeerTopologyHeaderKey] = reqCtx.PeerTopology
	}
	return headers
}

// legTransportError maps a failed leg send to a pipeline error: the client
// context error when the client went away, or an uncommitted transport error.
func legTransportError(ctx context.Context, step string, reqCtx *pipeline.RequestContext, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return fmt.Errorf("%s: %w", step, ctxErr)
	}
	return uncommittedTransportError(step, reqCtx, err)
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
		if strings.EqualFold(k, "content-length") || strings.EqualFold(k, "transfer-encoding") ||
			strings.EqualFold(k, "connection") || strings.EqualFold(k, "trailer") ||
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

func isHTTPTranslatedGRPC(reqCtx *pipeline.RequestContext) bool {
	return reqCtx != nil && reqCtx.GRPC != nil && !reqCtx.IsNativeGRPC()
}

func upstreamGRPCError(step string, reqCtx *pipeline.RequestContext, st grpcStatusWire) error {
	httpStatus := grpcStatusToHTTPStatus(st.status)
	if isHTTPTranslatedGRPC(reqCtx) {
		return &pipeline.UpstreamError{Step: step, StatusCode: httpStatus}
	}
	writeRawGRPCError(reqCtx.ResponseWriter, st)
	return &pipeline.ResponseWrittenError{
		Step:       step,
		StatusCode: httpStatus,
		Cause:      &pipeline.UpstreamError{Step: step, StatusCode: httpStatus},
	}
}

func decodePreCommitGRPCError(step string, reqCtx *pipeline.RequestContext, st grpcStatusWire) error {
	httpStatus := grpcStatusToHTTPStatus(st.status)
	if isHTTPTranslatedGRPC(reqCtx) {
		return &pipeline.UpstreamError{Step: step, StatusCode: httpStatus}
	}
	writeRawGRPCError(reqCtx.ResponseWriter, st)
	return &pipeline.UpstreamStreamedError{Step: step, StatusCode: httpStatus}
}

func uncommittedTransportError(step string, reqCtx *pipeline.RequestContext, err error) error {
	cause := fmt.Errorf("%s: %w: %w", step, &pipeline.UpstreamError{Step: step}, err)
	if isHTTPTranslatedGRPC(reqCtx) {
		return cause
	}
	grpcframe.WriteError(reqCtx.ResponseWriter, codes.Unavailable, reqcommon.InternalErrorMessage(reqCtx.RequestID))
	return &pipeline.ResponseWrittenError{
		Step:       step,
		StatusCode: http.StatusBadGateway,
		Cause:      cause,
	}
}

func uncommittedHTTPError(step string, reqCtx *pipeline.RequestContext, resp *http.Response) error {
	upstreamErr := upstreamError(step, resp.StatusCode, readErrorBody(resp.Body))
	if isHTTPTranslatedGRPC(reqCtx) {
		return upstreamErr
	}
	grpcframe.WriteError(
		reqCtx.ResponseWriter,
		pipeline.HTTPStatusToGRPCCode(resp.StatusCode),
		fmt.Sprintf("upstream HTTP %d (request_id: %s)", resp.StatusCode, reqCtx.RequestID),
	)
	return &pipeline.ResponseWrittenError{
		Step:       step,
		StatusCode: resp.StatusCode,
		Cause:      upstreamErr,
	}
}

func internalUpstreamError(step string, reqCtx *pipeline.RequestContext, cause error) error {
	if isHTTPTranslatedGRPC(reqCtx) {
		return cause
	}
	grpcframe.WriteError(reqCtx.ResponseWriter, codes.Internal, reqcommon.InternalErrorMessage(reqCtx.RequestID))
	return &pipeline.ResponseWrittenError{
		Step:       step,
		StatusCode: http.StatusBadGateway,
		Cause:      cause,
	}
}
