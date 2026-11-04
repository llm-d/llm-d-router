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
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestGRPCAndHTTPStatusMappings(t *testing.T) {
	grpcToHTTP := []struct {
		grpcStatus string
		wantHTTP   int
	}{
		{grpcStatus: strconv.Itoa(int(codes.InvalidArgument)), wantHTTP: http.StatusBadRequest},
		{grpcStatus: strconv.Itoa(int(codes.FailedPrecondition)), wantHTTP: http.StatusBadRequest},
		{grpcStatus: strconv.Itoa(int(codes.OutOfRange)), wantHTTP: http.StatusBadRequest},
		{grpcStatus: strconv.Itoa(int(codes.Unauthenticated)), wantHTTP: http.StatusUnauthorized},
		{grpcStatus: strconv.Itoa(int(codes.PermissionDenied)), wantHTTP: http.StatusForbidden},
		{grpcStatus: strconv.Itoa(int(codes.NotFound)), wantHTTP: http.StatusNotFound},
		{grpcStatus: strconv.Itoa(int(codes.ResourceExhausted)), wantHTTP: http.StatusTooManyRequests},
		{grpcStatus: strconv.Itoa(int(codes.DeadlineExceeded)), wantHTTP: http.StatusGatewayTimeout},
		{grpcStatus: strconv.Itoa(int(codes.Internal)), wantHTTP: http.StatusBadGateway},
		{grpcStatus: strconv.Itoa(int(codes.Unavailable)), wantHTTP: http.StatusBadGateway},
		{grpcStatus: "not-a-number", wantHTTP: http.StatusBadGateway},
	}
	for _, tc := range grpcToHTTP {
		require.Equal(t, tc.wantHTTP, grpcStatusToHTTPStatus(tc.grpcStatus), "grpcStatus=%q", tc.grpcStatus)
	}

	httpToGRPC := []struct {
		httpStatus int
		wantGRPC   codes.Code
	}{
		{httpStatus: http.StatusBadRequest, wantGRPC: codes.InvalidArgument},
		{httpStatus: http.StatusUnprocessableEntity, wantGRPC: codes.InvalidArgument},
		{httpStatus: http.StatusUnauthorized, wantGRPC: codes.Unauthenticated},
		{httpStatus: http.StatusForbidden, wantGRPC: codes.PermissionDenied},
		{httpStatus: http.StatusNotFound, wantGRPC: codes.Unimplemented},
		{httpStatus: http.StatusMethodNotAllowed, wantGRPC: codes.Unimplemented},
		{httpStatus: http.StatusRequestEntityTooLarge, wantGRPC: codes.ResourceExhausted},
		{httpStatus: http.StatusTooManyRequests, wantGRPC: codes.ResourceExhausted},
		{httpStatus: http.StatusGatewayTimeout, wantGRPC: codes.DeadlineExceeded},
		{httpStatus: http.StatusBadGateway, wantGRPC: codes.Unavailable},
		{httpStatus: http.StatusServiceUnavailable, wantGRPC: codes.Unavailable},
	}
	for _, tc := range httpToGRPC {
		require.Equal(t, tc.wantGRPC, pipeline.HTTPStatusToGRPCCode(tc.httpStatus), "httpStatus=%d", tc.httpStatus)
	}

	errBodyToGRPC := []struct {
		code     uint32
		wantGRPC codes.Code
	}{
		{code: 0, wantGRPC: codes.Internal},
		{code: http.StatusInternalServerError, wantGRPC: codes.Internal},
		{code: http.StatusBadRequest, wantGRPC: codes.InvalidArgument},
		{code: http.StatusUnprocessableEntity, wantGRPC: codes.InvalidArgument},
		{code: http.StatusNotFound, wantGRPC: codes.NotFound},
		{code: http.StatusTooManyRequests, wantGRPC: codes.ResourceExhausted},
		{code: http.StatusServiceUnavailable, wantGRPC: codes.Unavailable},
	}
	for _, tc := range errBodyToGRPC {
		require.Equal(t, tc.wantGRPC, sglangErrorBodyGRPCCode(tc.code), "code=%d", tc.code)
	}
}

func TestRewriteDecodeFrameCachedTokens_OnlyTerminalFrames(t *testing.T) {
	ps := pipeline.NewSGLangPrefillStream(func() {}, nil)
	ps.RecordCachedTokens(0, 48)

	marshal := func(finish *sglangapiv1.FinishReason) []byte {
		framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
			Item: &sglangapiv1.GenerateStreamItem_Frame{
				Frame: &sglangapiv1.GenerateResponse{
					MetaInfo: &sglangapiv1.GenerateMetaInfo{CachedTokens: 0, FinishReason: finish},
				},
			},
		})
		require.NoError(t, err)
		return framed
	}
	cachedTokens := func(framed []byte) uint32 {
		payload, err := grpcframe.ParsePayload(framed)
		require.NoError(t, err)
		var item sglangapiv1.GenerateStreamItem
		require.NoError(t, proto.Unmarshal(payload, &item))
		return item.GetFrame().GetMetaInfo().GetCachedTokens()
	}

	// A non-terminal frame is forwarded as the same slice: no copy.
	nonTerminal := marshal(nil)
	out, err := rewriteDecodeFrameCachedTokens(context.Background(), logr.Discard(), nonTerminal, ps)
	require.NoError(t, err)
	require.Same(t, &nonTerminal[0], &out[0])
	require.Equal(t, uint32(0), cachedTokens(out))
	require.Zero(t, testing.AllocsPerRun(100, func() {
		_, _ = rewriteDecodeFrameCachedTokens(context.Background(), logr.Discard(), nonTerminal, ps)
	}))

	// Without a prefill stream (decode-only), frames are also forwarded as is.
	out, err = rewriteDecodeFrameCachedTokens(context.Background(), logr.Discard(), nonTerminal, nil)
	require.NoError(t, err)
	require.Same(t, &nonTerminal[0], &out[0])

	// A malformed frame is forwarded unchanged with an error.
	malformed := []byte{0, 0, 0, 0, 1, 0xff}
	out, err = rewriteDecodeFrameCachedTokens(context.Background(), logr.Discard(), malformed, ps)
	require.Error(t, err)
	require.Equal(t, malformed, out)

	// A terminal frame gets the prefill cached_tokens.
	terminal := marshal(&sglangapiv1.FinishReason{Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}}})
	out, err = rewriteDecodeFrameCachedTokens(context.Background(), logr.Discard(), terminal, ps)
	require.NoError(t, err)
	require.Equal(t, uint32(48), cachedTokens(out))
}

func TestSGLangPrefillStreamLifecycle(t *testing.T) {
	parentCtx, cancelParent := context.WithCancel(context.Background())
	var prefillCancelled atomic.Bool
	prefillCtx, cancelPrefill := context.WithCancel(context.WithoutCancel(parentCtx))
	go func() {
		<-prefillCtx.Done()
		prefillCancelled.Store(true)
	}()
	stopAfter := context.AfterFunc(parentCtx, cancelPrefill)
	ps := pipeline.NewSGLangPrefillStream(cancelPrefill, stopAfter)

	// DetachRequestContext prevents parent teardown from cancelling prefill.
	ps.DetachRequestContext()
	cancelParent()
	time.Sleep(20 * time.Millisecond)
	require.False(t, prefillCancelled.Load())

	// Close explicitly cancels the prefill context.
	ps.Close()
	require.Eventually(t, prefillCancelled.Load, time.Second, 5*time.Millisecond)
}

func TestDecodeSGLangGRPC_ClassifiesTransportErrorsAsUpstreamTransport(t *testing.T) {
	// Closed gateway address triggers round-trip transport failure.
	closedSrv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := closedSrv.URL
	closedSrv.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: closedURL})
	step, err := NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
	require.NoError(t, err)

	rec := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:           "req-transport-1",
		GRPCGeneratePayload: []byte{},
		ResponseWriter:      rec,
	}
	execErr := step.Execute(context.Background(), reqCtx)
	require.Error(t, execErr)
	require.Equal(t, coordmetrics.ErrorCodeUpstreamTransport, coordmetrics.ClassifyErrorCode(execErr, pipeline.ClassifyOpts))

	// Truncated mid-frame response before headers commit also classifies as upstream_transport.
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	truncSrv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		w.WriteHeader(http.StatusOK)
		// Write an incomplete 5-byte gRPC header (2 bytes) then end stream.
		_, _ = w.Write([]byte{0x00, 0x01})
	}))
	truncSrv.Config.Protocols = protocols
	truncSrv.Start()
	t.Cleanup(truncSrv.Close)

	truncGW := gateway.New(config.GatewayConfig{Address: truncSrv.URL})
	truncStep, err := NewDecodeStep(truncGW, map[string]any{"kv_connector": kv.SGLang})
	require.NoError(t, err)

	rec2 := httptest.NewRecorder()
	reqCtx2 := &pipeline.RequestContext{
		RequestID:           "req-transport-2",
		GRPCGeneratePayload: []byte{},
		ResponseWriter:      rec2,
	}
	truncErr := truncStep.Execute(context.Background(), reqCtx2)
	require.Error(t, truncErr)
	require.Equal(t, coordmetrics.ErrorCodeUpstreamTransport, coordmetrics.ClassifyErrorCode(truncErr, pipeline.ClassifyOpts))
}
