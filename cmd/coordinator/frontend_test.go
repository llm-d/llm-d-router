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

package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/internal/frontendproxy"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

func TestFrontendRequestCanGenerateDuringDrain(t *testing.T) {
	publicPort, err := fwknet.GetFreePort()
	require.NoError(t, err)
	generation, err := fwknet.ReserveListener()
	require.NoError(t, err)
	public := fmt.Sprintf("127.0.0.1:%d", publicPort)
	private := generation.Addr().String()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	client := grpctest.NewH2CClient()
	client.Timeout = 3 * time.Second
	defer client.CloseIdleConnections()
	// A truncated frame exercises the real gRPC handler without needing a worker.
	checkGenerate := func(addr string) (string, error) {
		res, err := client.Post("http://"+addr+vllmpb.Inference_GenerateStream_FullMethodName, "application/grpc", strings.NewReader("x"))
		if err != nil {
			return "", err
		}
		defer res.Body.Close()
		_, err = io.Copy(io.Discard, res.Body)
		return res.Trailer.Get("Grpc-Status") + res.Header.Get("Grpc-Status"), err
	}
	release := make(chan struct{})
	frontend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(body)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		status, err := checkGenerate(private)
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = io.WriteString(w, status)
	}))
	defer frontend.Close()
	cfg := config.ServerConfig{ListenAddr: public, MaxRequestBodySize: 1, ShutdownTimeout: 3 * time.Second}
	rpcFrontend := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, 2, r.ProtoMajor)
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		assert.Equal(t, "x", string(body))
		w.Header().Set("Content-Type", "application/grpc")
		w.Header().Set("Trailer", "Grpc-Status")
		w.WriteHeader(http.StatusOK)
		w.Header().Set("Grpc-Status", "7")
	}))
	apiURL, err := url.Parse(frontend.URL)
	require.NoError(t, err)
	rpcURL, err := url.Parse(rpcFrontend.URL)
	require.NoError(t, err)
	srv := newTestServer(t, public, publicFrontend(frontendproxy.New(apiURL, 1024, false), frontendproxy.New(rpcURL, 1024, true)))
	done := make(chan error, 1)
	go func() { done <- runWithFrontend(ctx, srv, cfg, generation) }()
	waitForHealthz(t, public, 3*time.Second)
	status, err := checkGenerate(public)
	require.NoError(t, err)
	require.Equal(t, "7", status)
	res, err := client.Get("http://" + private + "/healthz")
	require.NoError(t, err)
	res.Body.Close()
	require.Equal(t, http.StatusNotFound, res.StatusCode)
	const body = "opaque\x00body"
	res, err = client.Post("http://"+public+"/v1/chat/completions", "application/octet-stream", strings.NewReader(body))
	require.NoError(t, err)
	defer res.Body.Close()
	first := make([]byte, len(body))
	_, err = io.ReadFull(res.Body, first)
	require.NoError(t, err)
	require.Equal(t, body, string(first))
	cancel()
	require.Eventually(t, func() bool {
		response, err := client.Get("http://" + public + "/healthz")
		if err != nil {
			return true
		}
		response.Body.Close()
		return false
	}, 2*time.Second, 10*time.Millisecond)
	close(release)
	tail, err := io.ReadAll(res.Body)
	require.NoError(t, err)
	require.Equal(t, "3", string(tail))
	require.NoError(t, <-done)
}
