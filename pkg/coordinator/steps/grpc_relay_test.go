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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func nativeGRPCReqCtx() (*pipeline.RequestContext, *httptest.ResponseRecorder) {
	rec := httptest.NewRecorder()
	return &pipeline.RequestContext{RequestID: "req-1", ResponseWriter: rec}, rec
}

// hdr builds canonical headers from key and value pairs, as net/http does.
func hdr(kv ...string) http.Header {
	h := http.Header{}
	for i := 0; i+1 < len(kv); i += 2 {
		h.Set(kv[i], kv[i+1])
	}
	return h
}

func grpcResponse(code int, header http.Header) *http.Response {
	if header == nil {
		header = http.Header{}
	}
	return &http.Response{StatusCode: code, Header: header, Trailer: http.Header{}, Body: io.NopCloser(strings.NewReader("upstream body"))}
}

func TestFinalGRPCStatus(t *testing.T) {
	resp := grpcResponse(http.StatusOK, hdr(grpcframe.HeaderStatus, "14"))
	defer func() { _ = resp.Body.Close() }()
	st, ok := finalGRPCStatus(resp)
	require.True(t, ok)
	require.Equal(t, "14", st.status, "trailers-only response reads the headers")

	resp.Trailer.Set(grpcframe.HeaderStatus, "0")
	resp.Trailer.Set(grpcframe.HeaderMessage, "done")
	st, ok = finalGRPCStatus(resp)
	require.True(t, ok)
	require.Equal(t, grpcStatusWire{status: "0", message: "done"}, st, "trailers win over headers")

	empty := grpcResponse(http.StatusOK, nil)
	defer func() { _ = empty.Body.Close() }()
	_, ok = finalGRPCStatus(empty)
	require.False(t, ok)
}

func TestCheckGRPCLegHeaders(t *testing.T) {
	tests := []struct {
		name       string
		code       int
		header     http.Header
		wantErr    bool
		wantStatus int
		wantGRPC   string
	}{
		{name: "ok without status", code: http.StatusOK},
		{name: "ok status", code: http.StatusOK, header: hdr(grpcframe.HeaderStatus, "0")},
		{name: "http error", code: http.StatusServiceUnavailable, wantErr: true, wantStatus: http.StatusServiceUnavailable},
		{name: "grpc error", code: http.StatusOK, header: hdr(grpcframe.HeaderStatus, "8", grpcframe.HeaderMessage, "busy"), wantErr: true, wantGRPC: "8"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reqCtx, rec := nativeGRPCReqCtx()
			resp := grpcResponse(tc.code, tc.header)
			defer func() { _ = resp.Body.Close() }()
			err := checkGRPCLegHeaders(PrefillStepName, reqCtx, resp)
			if !tc.wantErr {
				require.NoError(t, err)
				return
			}
			if tc.wantStatus != 0 {
				upstreamErr, ok := errors.AsType[*pipeline.UpstreamError](err)
				require.True(t, ok)
				require.Equal(t, tc.wantStatus, upstreamErr.StatusCode)
				require.Equal(t, "upstream body", upstreamErr.Body)
			}
			if tc.wantGRPC != "" {
				_, ok := errors.AsType[*pipeline.ResponseWrittenError](err)
				require.True(t, ok)
				require.Equal(t, tc.wantGRPC, rec.Header().Get(grpcframe.HeaderStatus))
				require.Equal(t, "busy", rec.Header().Get(grpcframe.HeaderMessage))
			}
		})
	}
}

func TestCheckGRPCDecodeHeaders(t *testing.T) {
	reqCtx, _ := nativeGRPCReqCtx()
	resp := grpcResponse(http.StatusOK, hdr(reqcommon.PeerTopologyHeaderKey, "host=spoofed"))
	defer func() { _ = resp.Body.Close() }()
	require.NoError(t, checkGRPCDecodeHeaders(reqCtx, resp))
	require.Empty(t, resp.Header.Get(reqcommon.PeerTopologyHeaderKey), "internal headers are stripped")

	reqCtx, rec := nativeGRPCReqCtx()
	httpErr := grpcResponse(http.StatusBadGateway, nil)
	defer func() { _ = httpErr.Body.Close() }()
	err := checkGRPCDecodeHeaders(reqCtx, httpErr)
	written, ok := errors.AsType[*pipeline.ResponseWrittenError](err)
	require.True(t, ok)
	require.Equal(t, http.StatusBadGateway, written.StatusCode)
	require.NotEmpty(t, rec.Header().Get(grpcframe.HeaderStatus))

	reqCtx, rec = nativeGRPCReqCtx()
	grpcErr := grpcResponse(http.StatusOK, hdr(grpcframe.HeaderStatus, "13"))
	defer func() { _ = grpcErr.Body.Close() }()
	err = checkGRPCDecodeHeaders(reqCtx, grpcErr)
	_, ok = errors.AsType[*pipeline.UpstreamStreamedError](err)
	require.True(t, ok)
	require.Equal(t, "13", rec.Header().Get(grpcframe.HeaderStatus))
}

func TestDecodeLegHeaders(t *testing.T) {
	reqCtx := &pipeline.RequestContext{RequestID: "req-1"}
	headers := decodeLegHeaders(reqCtx)
	require.Equal(t, gateway.PhaseDecode, headers[reqcommon.EPPProfileHeaderKey])
	require.NotContains(t, headers, reqcommon.PeerTopologyHeaderKey)

	reqCtx.PeerTopology = "rack=a"
	require.Equal(t, "rack=a", decodeLegHeaders(reqCtx)[reqcommon.PeerTopologyHeaderKey])
}

func TestLegTransportError(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	reqCtx, rec := nativeGRPCReqCtx()
	err := legTransportError(canceled, DecodeStepName, reqCtx, errors.New("dial"))
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, rec.Header().Get(grpcframe.HeaderStatus), "nothing is written for a gone client")

	err = legTransportError(context.Background(), DecodeStepName, reqCtx, errors.New("dial"))
	written, ok := errors.AsType[*pipeline.ResponseWrittenError](err)
	require.True(t, ok)
	require.Equal(t, http.StatusBadGateway, written.StatusCode)
	require.Equal(t, "14", rec.Header().Get(grpcframe.HeaderStatus))
}
