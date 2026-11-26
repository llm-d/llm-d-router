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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

// prefillRankPath is a running proxy on one of the paths that send a prefill request.
type prefillRankPath struct {
	baseAddr        string
	prefillHostPort string
	prefill, decode *mock.ChatCompletionHandler
}

// startPrefillRankTestProxy starts a proxy with mock prefill and decode backends.
func startPrefillRankTestProxy(t *testing.T, connector string, moriWriteParallel bool, bootstrapPort int) prefillRankPath {
	t.Helper()

	decodeHandler := &mock.ChatCompletionHandler{
		Connector:       connector,
		Role:            mock.RoleDecode,
		MoRIIOWriteMode: moriWriteParallel,
	}
	decodeBackend := httptest.NewServer(decodeHandler)
	t.Cleanup(decodeBackend.Close)

	prefillHandler := &mock.ChatCompletionHandler{
		Connector:       connector,
		Role:            mock.RolePrefill,
		MoRIIOWriteMode: moriWriteParallel,
	}
	prefillBackend := httptest.NewServer(prefillHandler)
	t.Cleanup(prefillBackend.Close)

	decodeURL, err := url.Parse(decodeBackend.URL)
	require.NoError(t, err)

	cfg := Config{
		Port:                  "0",
		DecoderURL:            decodeURL,
		KVConnector:           connector,
		MooncakeBootstrapPort: bootstrapPort,
	}
	if moriWriteParallel {
		cfg.MoRIIOWriteMode = true
		cfg.MoRIIOParallelDispatch = true
		cfg.MoRIIODecodePodIP = decodeURL.Hostname()
		cfg.MoRIIODecodeNotifyPort = 61005
		cfg.MoRIIODecodeHandshakePort = 6301
		cfg.MoRIIOPrefillNotifyPort = 61006
		cfg.MoRIIOPrefillHandshakePort = 6302
		cfg.MoRIIOTPSize = 1
		cfg.MoRIIODPSize = 1
	}

	srv := NewProxy(cfg)
	srv.allowlistValidator = &AllowlistValidator{enabled: false}

	ctx, cancel := context.WithCancel(t.Context())
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Start(ctx)
	}()

	select {
	case <-srv.readyCh:
	case err := <-errCh:
		cancel()
		require.NoError(t, err)
		t.Fatal("proxy stopped before becoming ready")
	}

	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-errCh)
	})

	return prefillRankPath{
		baseAddr:        "http://" + srv.addr.String(),
		prefillHostPort: strings.TrimPrefix(prefillBackend.URL, "http://"),
		prefill:         prefillHandler,
		decode:          decodeHandler,
	}
}

func sendChatCompletionsRequest(t *testing.T, baseAddr string, headers map[string]string) *http.Response {
	t.Helper()

	req, err := http.NewRequestWithContext(
		t.Context(),
		http.MethodPost,
		baseAddr+reqcommon.PathChatCompletions,
		bytes.NewBufferString(chatCompletionsRequestBody),
	)
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	return resp
}

type mooncakeBootstrapStub struct {
	port    int
	queries atomic.Int32

	mu     sync.RWMutex
	status int
	body   string
}

func startMooncakeBootstrapStub(t *testing.T, initialBody string) *mooncakeBootstrapStub {
	t.Helper()

	stub := &mooncakeBootstrapStub{
		status: http.StatusOK,
		body:   initialBody,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		stub.queries.Add(1)
		stub.mu.RLock()
		status := stub.status
		body := stub.body
		stub.mu.RUnlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)

	u, err := url.Parse(srv.URL)
	require.NoError(t, err)
	port, err := strconv.Atoi(u.Port())
	require.NoError(t, err)
	stub.port = port
	return stub
}

func (b *mooncakeBootstrapStub) setResponse(status int, body string) {
	b.mu.Lock()
	b.status = status
	b.body = body
	b.mu.Unlock()
}

func waitForCompletionCount(t *testing.T, h *mock.ChatCompletionHandler, want int) {
	t.Helper()

	require.EventuallyWithT(t, func(c *assert.CollectT) {
		assert.Len(c, h.GetCompletionRequests(), want)
	}, 2*time.Second, 10*time.Millisecond)
}

func decodeRemoteEngineID(t *testing.T, h *mock.ChatCompletionHandler, idx int) string {
	t.Helper()

	reqs := h.GetCompletionRequests()
	require.Greater(t, len(reqs), idx)
	kvParams, ok := reqs[idx][reqcommon.FieldKVTransferParams].(map[string]any)
	require.True(t, ok)
	engineID, ok := kvParams[reqcommon.FieldRemoteEngineID].(string)
	require.True(t, ok)
	return engineID
}

// Each path builds its prefill request from the client request, which carries the
// decode rank. The NIXL v2 serial, Mooncake, SGLang, and vllm.Inference paths have
// rank tests of their own.
func TestPrefillDataParallelRank(t *testing.T) {
	cases := []struct {
		name              string
		connector         string
		moriWriteParallel bool
		decodeRank        string
		prefillRankHeader string
		wantPrefillRank   string
	}{
		{
			name:              "offloading_explicit_prefill_rank",
			connector:         constants.KVConnectorOffloading,
			decodeRank:        "0",
			prefillRankHeader: "1",
			wantPrefillRank:   "1",
		},
		{
			name:              "offloading_none_removes_decode_rank",
			connector:         constants.KVConnectorOffloading,
			decodeRank:        "1",
			prefillRankHeader: routing.NoDataParallelRank,
			wantPrefillRank:   "",
		},
		{
			name:              "shared_storage_explicit_prefill_rank",
			connector:         constants.KVConnectorSharedStorage,
			decodeRank:        "0",
			prefillRankHeader: "1",
			wantPrefillRank:   "1",
		},
		{
			name:              "shared_storage_none_removes_decode_rank",
			connector:         constants.KVConnectorSharedStorage,
			decodeRank:        "1",
			prefillRankHeader: routing.NoDataParallelRank,
			wantPrefillRank:   "",
		},
		{
			name:              "nixl_write_parallel_explicit_prefill_rank",
			connector:         constants.KVConnectorNIXLV2,
			moriWriteParallel: true,
			decodeRank:        "0",
			prefillRankHeader: "1",
			wantPrefillRank:   "1",
		},
		{
			name:              "nixl_write_parallel_none_removes_decode_rank",
			connector:         constants.KVConnectorNIXLV2,
			moriWriteParallel: true,
			decodeRank:        "1",
			prefillRankHeader: routing.NoDataParallelRank,
			wantPrefillRank:   "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := startPrefillRankTestProxy(t, tc.connector, tc.moriWriteParallel, 0)
			resp := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
				routing.PrefillEndpointHeader:         path.prefillHostPort,
				routing.KVCachePrefillHeader:          prefillP2PHostPort(path.prefillHostPort),
				routing.DataParallelRankHeader:        tc.decodeRank,
				routing.PrefillDataParallelRankHeader: tc.prefillRankHeader,
			})
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)

			waitForCompletionCount(t, path.prefill, 1)
			require.Equal(t, tc.wantPrefillRank, path.prefill.GetCompletionHeaders()[0].Get(routing.DataParallelRankHeader))
			require.Equal(t, tc.decodeRank, path.decode.GetCompletionHeaders()[0].Get(routing.DataParallelRankHeader))
		})
	}
}

func TestMooncakePrefillDPRankRefreshesStaleCache(t *testing.T) {
	const (
		rank0Body   = `{"0": {"engine_id": "engine-dp0"}}`
		rank01Body  = `{"0": {"engine_id": "engine-dp0"}, "1": {"engine_id": "engine-dp1"}}`
		engineRank0 = "engine-dp0"
		engineRank1 = "engine-dp1"
	)

	bootstrap := startMooncakeBootstrapStub(t, rank0Body)
	path := startPrefillRankTestProxy(t, constants.KVConnectorMooncake, false, bootstrap.port)

	resp0 := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
		routing.PrefillEndpointHeader:         path.prefillHostPort,
		routing.PrefillDataParallelRankHeader: "0",
		routing.DataParallelRankHeader:        "0",
	})
	defer resp0.Body.Close()
	require.Equal(t, http.StatusOK, resp0.StatusCode)
	waitForCompletionCount(t, path.prefill, 1)
	require.Equal(t, int32(1), bootstrap.queries.Load())
	require.Equal(t, engineRank0, decodeRemoteEngineID(t, path.decode, 0))

	bootstrap.setResponse(http.StatusOK, rank01Body)

	resp1 := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
		routing.PrefillEndpointHeader:         path.prefillHostPort,
		routing.PrefillDataParallelRankHeader: "1",
		routing.DataParallelRankHeader:        "0",
	})
	defer resp1.Body.Close()
	require.Equal(t, http.StatusOK, resp1.StatusCode)
	waitForCompletionCount(t, path.prefill, 2)
	require.Equal(t, int32(2), bootstrap.queries.Load())
	require.Equal(t, "1", path.prefill.GetCompletionHeaders()[1].Get(routing.DataParallelRankHeader))
	require.Equal(t, engineRank1, decodeRemoteEngineID(t, path.decode, 1))
	require.Equal(t, "0", path.decode.GetCompletionHeaders()[1].Get(routing.DataParallelRankHeader))
}

func TestMooncakePrefillDPRankMissingReturnsBadGateway(t *testing.T) {
	const rank0Body = `{"0": {"engine_id": "engine-dp0"}}`

	t.Run("cold_cache_queries_once", func(t *testing.T) {
		bootstrap := startMooncakeBootstrapStub(t, rank0Body)
		path := startPrefillRankTestProxy(t, constants.KVConnectorMooncake, false, bootstrap.port)

		resp := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "2",
			routing.DataParallelRankHeader:        "0",
		})
		defer resp.Body.Close()
		require.Equal(t, http.StatusBadGateway, resp.StatusCode)
		require.Equal(t, int32(1), bootstrap.queries.Load())
		require.Empty(t, path.prefill.GetCompletionRequests())
		require.Empty(t, path.decode.GetCompletionRequests())
	})

	t.Run("warm_cache_refreshes_once", func(t *testing.T) {
		bootstrap := startMooncakeBootstrapStub(t, rank0Body)
		path := startPrefillRankTestProxy(t, constants.KVConnectorMooncake, false, bootstrap.port)

		seed := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "0",
		})
		defer seed.Body.Close()
		require.Equal(t, http.StatusOK, seed.StatusCode)
		waitForCompletionCount(t, path.prefill, 1)

		resp := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "2",
		})
		defer resp.Body.Close()
		require.Equal(t, http.StatusBadGateway, resp.StatusCode)
		require.Equal(t, int32(2), bootstrap.queries.Load())
		require.Len(t, path.prefill.GetCompletionRequests(), 1)
		require.Len(t, path.decode.GetCompletionRequests(), 1)
	})

	t.Run("refresh_error_preserves_cached_ranks", func(t *testing.T) {
		bootstrap := startMooncakeBootstrapStub(t, rank0Body)
		path := startPrefillRankTestProxy(t, constants.KVConnectorMooncake, false, bootstrap.port)

		seed := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "0",
		})
		defer seed.Body.Close()
		require.Equal(t, http.StatusOK, seed.StatusCode)
		waitForCompletionCount(t, path.prefill, 1)

		bootstrap.setResponse(http.StatusInternalServerError, "")
		missing := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "1",
		})
		defer missing.Body.Close()
		require.Equal(t, http.StatusBadGateway, missing.StatusCode)
		require.Equal(t, int32(2), bootstrap.queries.Load())

		cached := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.PrefillDataParallelRankHeader: "0",
		})
		defer cached.Body.Close()
		require.Equal(t, http.StatusOK, cached.StatusCode)
		waitForCompletionCount(t, path.prefill, 2)
		require.Equal(t, int32(2), bootstrap.queries.Load())
		require.Equal(t, "engine-dp0", decodeRemoteEngineID(t, path.decode, 1))
	})
}

func TestMooncakePrefillDPRankNoneSelectsRandomRank(t *testing.T) {
	engines := map[string]string{"0": "engine-dp0", "1": "engine-dp1"}
	bootstrap := startMooncakeBootstrapStub(t, `{"0": {"engine_id": "engine-dp0"}, "1": {"engine_id": "engine-dp1"}}`)
	path := startPrefillRankTestProxy(t, constants.KVConnectorMooncake, false, bootstrap.port)

	resp := sendChatCompletionsRequest(t, path.baseAddr, map[string]string{
		routing.PrefillEndpointHeader:         path.prefillHostPort,
		routing.PrefillDataParallelRankHeader: routing.NoDataParallelRank,
		routing.DataParallelRankHeader:        "0",
	})
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	waitForCompletionCount(t, path.prefill, 1)

	gotRank := path.prefill.GetCompletionHeaders()[0].Get(routing.DataParallelRankHeader)
	wantEngine, ok := engines[gotRank]
	require.True(t, ok, "unexpected prefill rank %q", gotRank)
	require.Equal(t, wantEngine, decodeRemoteEngineID(t, path.decode, 0))
}
