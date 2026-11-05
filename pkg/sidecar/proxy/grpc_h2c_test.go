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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func newH2CTestServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(handler)
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetUnencryptedHTTP2(true)
	srv.Config.Protocols = protocols
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

func newH2CClient() *http.Client {
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	return &http.Client{
		Transport: &http.Transport{
			Protocols: protocols,
		},
	}
}

func startTestSidecar(t *testing.T, cfg Config) string {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	srv := NewProxy(cfg)
	srv.allowlistValidator = &AllowlistValidator{enabled: false}
	done := make(chan error, 1)
	go func() {
		done <- srv.Start(ctx)
	}()
	select {
	case <-srv.readyCh:
	case err := <-done:
		require.NoError(t, err, "sidecar exited before ready")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for sidecar listener")
	}
	t.Cleanup(func() {
		cancel()
		<-done
	})
	return "http://" + srv.addr.String()
}

func TestPlaintextH2CPassthroughStreamsFramesAndTrailers(t *testing.T) {
	frame1, err := grpcframe.FramePayload([]byte("chunk-1"))
	require.NoError(t, err)
	frame2, err := grpcframe.FramePayload([]byte("chunk-2"))
	require.NoError(t, err)

	var backendSawProtoMajor atomic.Int32
	decoder := newH2CTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendSawProtoMajor.Store(int32(r.ProtoMajor))
		_, _ = io.Copy(io.Discard, r.Body)
		if grpcframe.IsContentType(r.Header.Get(reqcommon.HeaderContentType)) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(frame1)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			_, _ = w.Write(frame2)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
			return
		}
		w.Header().Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	decodeURL, err := url.Parse(decoder.URL)
	require.NoError(t, err)

	baseURL := startTestSidecar(t, Config{
		Port:          "0",
		DecoderURL:    decodeURL,
		SecureServing: false,
	})

	reqPayload, err := grpcframe.FramePayload([]byte("ping"))
	require.NoError(t, err)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, baseURL+"/sglang.api.v1.SglangService/HealthCheck", bytes.NewReader(reqPayload))
	require.NoError(t, err)
	req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)

	resp, err := newH2CClient().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, 2, resp.ProtoMajor)
	assert.Equal(t, int32(2), backendSawProtoMajor.Load())
	got1, err := grpcframe.ReadFrame(resp.Body, 64)
	require.NoError(t, err)
	assert.Equal(t, []byte("chunk-1"), got1)
	got2, err := grpcframe.ReadFrame(resp.Body, 64)
	require.NoError(t, err)
	assert.Equal(t, []byte("chunk-2"), got2)
	_, err = grpcframe.ReadFrame(resp.Body, 64)
	require.ErrorIs(t, err, io.EOF)
	// The backend sent grpc-status as a trailer; it must arrive as one.
	status, ok := grpcframe.Status(resp.Trailer)
	require.True(t, ok)
	assert.Equal(t, grpcframe.StatusOK, status)

	// Verify HTTP/1.1 JSON requests on the same sidecar continue using HTTP/1.1 to the backend.
	jsonReq, err := http.NewRequestWithContext(t.Context(), http.MethodGet, baseURL+"/v1/models", nil)
	require.NoError(t, err)
	jsonResp, err := http.DefaultClient.Do(jsonReq)
	require.NoError(t, err)
	_ = jsonResp.Body.Close()
	assert.Equal(t, http.StatusOK, jsonResp.StatusCode)
	assert.Equal(t, int32(1), backendSawProtoMajor.Load())
}

func TestRunConcurrentPDRelaysGRPCPrefillError(t *testing.T) {
	tests := []struct {
		name        string
		writeError  func(w http.ResponseWriter)
		wantCode    string
		wantMessage string
		wantBody    []byte
		// wantInTrailer selects where the status must arrive: a trailers-only
		// response carries it in the HTTP/2 HEADERS frame, a mid-stream error
		// in the trailing HEADERS frame. The relay must not move it.
		wantInTrailer bool
	}{
		{
			name: "trailers-only gRPC error",
			writeError: func(w http.ResponseWriter) {
				grpcframe.WriteError(w, codes.InvalidArgument, "invalid input_ids")
			},
			wantCode:    "3",
			wantMessage: "invalid input_ids",
		},
		{
			name: "mid-stream gRPC trailer error",
			writeError: func(w http.ResponseWriter) {
				frame, _ := grpcframe.FramePayload([]byte("partial"))
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(frame)
				w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, "14")
				w.Header().Set(http.TrailerPrefix+grpcframe.HeaderMessage, "prefill unavailable")
			},
			wantCode:      "14",
			wantMessage:   "prefill unavailable",
			wantBody:      func() []byte { b, _ := grpcframe.FramePayload([]byte("partial")); return b }(),
			wantInTrailer: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), concurrentPDTestWait)
			defer cancel()

			decodeStarted := make(chan struct{})
			prefill := newH2CTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				select {
				case <-decodeStarted:
				case <-r.Context().Done():
					return
				}
				tt.writeError(w)
			}))
			decode := newH2CTestServer(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(decodeStarted)
				<-r.Context().Done()
			}))

			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)
			decodeURL, err := url.Parse(decode.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: decodeURL})
			s.allowlistValidator = &AllowlistValidator{enabled: false}
			s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)

			// Serve runConcurrentPD over a real h2c listener so HTTP/2 headers and trailers are exercised end-to-end.
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			protocols := new(http.Protocols)
			protocols.SetUnencryptedHTTP2(true)
			body, err := grpcframe.FramePayload([]byte("req"))
			require.NoError(t, err)
			front := &http.Server{
				Protocols: protocols,
				Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_, _ = io.Copy(io.Discard, r.Body)
					s.runConcurrentPD(w, r, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil, nil)
				}),
			}
			go func() { _ = front.Serve(ln) }()
			t.Cleanup(func() { _ = front.Close() })

			before := snapshotStageMetrics(t)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+ln.Addr().String()+"/sglang.api.v1.SglangService/Generate", bytes.NewReader(body))
			require.NoError(t, err)
			req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)

			resp, err := newH2CClient().Do(req)
			require.NoError(t, err)
			respBody, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			_ = resp.Body.Close()

			require.NoError(t, ctx.Err(), "decode must be cancelled before the client deadline")
			assert.Equal(t, http.StatusOK, resp.StatusCode)
			if len(tt.wantBody) > 0 {
				assert.Equal(t, tt.wantBody, respBody)
			} else {
				assert.Empty(t, respBody)
			}

			statusIn, statusNotIn := resp.Header, resp.Trailer
			if tt.wantInTrailer {
				statusIn, statusNotIn = resp.Trailer, resp.Header
			}
			gotStatus, ok := grpcframe.Status(statusIn)
			gotMsg := statusIn.Get(grpcframe.HeaderMessage)
			require.True(t, ok, "expected grpc-status where the backend sent it (trailer=%v)", tt.wantInTrailer)
			_, moved := grpcframe.Status(statusNotIn)
			assert.False(t, moved, "grpc-status must not be duplicated or moved")
			assert.Equal(t, tt.wantCode, gotStatus)
			assert.Equal(t, tt.wantMessage, gotMsg)

			require.Eventually(t, func() bool {
				delta := snapshotStageMetrics(t).delta(before)
				return delta.prefillCount == 1 && delta.prefillErrors == 1
			}, time.Second, 10*time.Millisecond)
		})
	}
}

func TestRunConcurrentPDDecodeGRPCErrorCancelsPrefill(t *testing.T) {
	prefillStarted := make(chan struct{})
	var prefillCancelled atomic.Bool
	prefill := newH2CTestServer(t, http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		close(prefillStarted)
		<-r.Context().Done()
		prefillCancelled.Store(true)
	}))
	prefillURL, err := url.Parse(prefill.URL)
	require.NoError(t, err)

	decode := newH2CTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		<-prefillStarted
		grpcframe.WriteError(w, codes.Internal, "decode engine failure")
	}))
	decodeURL, err := url.Parse(decode.URL)
	require.NoError(t, err)

	s := NewProxy(Config{Port: "0", DecoderURL: decodeURL})
	s.allowlistValidator = &AllowlistValidator{enabled: false}
	s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)

	body, err := grpcframe.FramePayload([]byte("req"))
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodPost, "/sglang.api.v1.SglangService/Generate", bytes.NewReader(body))
	req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)

	before := snapshotStageMetrics(t)
	rec := httptest.NewRecorder()
	s.runConcurrentPD(rec, req, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil, nil)

	require.Eventually(t, prefillCancelled.Load, concurrentPDTestWait, 10*time.Millisecond, "prefill was not cancelled after decode gRPC error")
	var delta stageMetrics
	require.Eventually(t, func() bool {
		delta = snapshotStageMetrics(t).delta(before)
		return delta.prefillCount == 1
	}, concurrentPDTestWait, 10*time.Millisecond)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1}, delta)
}
