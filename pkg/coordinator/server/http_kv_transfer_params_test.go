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

package server_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

type capturedGatewayRequest struct {
	phase  string
	prefer string
	body   map[string]any
	err    error
}

func assertNoNestedTransferParams(t *testing.T, body map[string]any, wantExtraArg string) {
	t.Helper()
	sp, ok := body[reqcommon.FieldSamplingParams].(map[string]any)
	require.True(t, ok, "sampling_params must be preserved: %v", body)
	extraArgs, ok := sp[reqcommon.FieldExtraArgs].(map[string]any)
	require.True(t, ok, "sampling_params.extra_args must be preserved: %v", body)
	require.NotContains(t, extraArgs, reqcommon.FieldKVTransferParams,
		"client-supplied sampling_params.extra_args.kv_transfer_params must not reach upstream")
	require.NotContains(t, extraArgs, reqcommon.FieldECTransferParams,
		"client-supplied sampling_params.extra_args.ec_transfer_params must not reach upstream")
	if wantExtraArg != "" {
		require.Equal(t, wantExtraArg, extraArgs["custom"])
	}
}

func TestHTTPGenerate_ConditionalDecodeDropsNestedKVTransferParams(t *testing.T) {
	const replyBody = `{"request_id":"g","choices":[{"index":0,"token_ids":[42]}]}`
	var (
		mu       sync.Mutex
		captured []capturedGatewayRequest
	)
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		var parsed map[string]any
		if err == nil {
			err = json.Unmarshal(raw, &parsed)
		}
		mu.Lock()
		captured = append(captured, capturedGatewayRequest{
			phase:  r.Header.Get(reqcommon.EPPProfileHeaderKey),
			prefer: r.Header.Get(routing.PreferHeader),
			body:   parsed,
			err:    err,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, replyBody)
	}))
	defer gw.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	cond, err := steps.NewConditionalDecodeStep(gwClient, nil)
	require.NoError(t, err)
	prefill, err := steps.NewPrefillStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)
	decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)

	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{cond, prefill, decode}), gwClient)

	reqBody := `{
		"model": "llama-3",
		"stream": false,
		"token_ids": [1, 2, 3],
		"kv_transfer_params": {"do_remote_prefill": true},
		"ec_transfer_params": {"h": {"peer_host": "10.9.9.9", "peer_port": 5501}},
		"sampling_params": {
			"max_tokens": 8,
			"extra_args": {
				"kv_transfer_params": {"do_remote_prefill": true, "remote_host": "10.9.9.9"},
				"ec_transfer_params": {"h": {"peer_host": "10.9.9.9", "peer_port": 5501}},
				"custom": "keep"
			}
		}
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(reqBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	gotBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.JSONEq(t, replyBody, string(gotBody))

	mu.Lock()
	calls := append([]capturedGatewayRequest(nil), captured...)
	mu.Unlock()

	require.Len(t, calls, 1, "served conditional-decode probe must bypass prefill and decode")
	require.NoError(t, calls[0].err)
	require.Equal(t, gateway.PhaseDecode, calls[0].phase)
	require.Equal(t, routing.PreferIfAvailable, calls[0].prefer)
	require.NotContains(t, calls[0].body, reqcommon.FieldKVTransferParams)
	require.NotContains(t, calls[0].body, reqcommon.FieldECTransferParams)
	assertNoNestedTransferParams(t, calls[0].body, "keep")
}

func TestHTTPChatCompletions_ConditionalDecodeDropsVLLMXargsTransferParams(t *testing.T) {
	const replyBody = `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	var (
		mu       sync.Mutex
		captured []capturedGatewayRequest
	)
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		var parsed map[string]any
		if err == nil {
			err = json.Unmarshal(raw, &parsed)
		}
		mu.Lock()
		captured = append(captured, capturedGatewayRequest{
			phase:  r.Header.Get(reqcommon.EPPProfileHeaderKey),
			prefer: r.Header.Get(routing.PreferHeader),
			body:   parsed,
			err:    err,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, replyBody)
	}))
	defer gw.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	cond, err := steps.NewConditionalDecodeStep(gwClient, nil)
	require.NoError(t, err)
	prefill, err := steps.NewPrefillStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)
	decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)

	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{cond, prefill, decode}), gwClient)

	reqBody := `{
		"model": "llama-3",
		"stream": false,
		"messages": [{"role": "user", "content": "hi"}],
		"vllm_xargs": {
			"kv_transfer_params": "x",
			"ec_transfer_params": "y",
			"session_id": "keep"
		}
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://"+addr+reqcommon.PathChatCompletions, strings.NewReader(reqBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	gotBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.JSONEq(t, replyBody, string(gotBody))

	mu.Lock()
	calls := append([]capturedGatewayRequest(nil), captured...)
	mu.Unlock()

	require.Len(t, calls, 1, "served conditional-decode probe must bypass prefill and decode")
	require.NoError(t, calls[0].err)
	require.Equal(t, gateway.PhaseDecode, calls[0].phase)
	require.Equal(t, routing.PreferIfAvailable, calls[0].prefer)
	require.NotContains(t, calls[0].body, reqcommon.FieldKVTransferParams)
	require.NotContains(t, calls[0].body, reqcommon.FieldECTransferParams)
	vllmXargs, ok := calls[0].body[reqcommon.FieldVLLMXargs].(map[string]any)
	require.True(t, ok, "vllm_xargs must be preserved: %v", calls[0].body)
	require.NotContains(t, vllmXargs, reqcommon.FieldKVTransferParams)
	require.NotContains(t, vllmXargs, reqcommon.FieldECTransferParams)
	require.Equal(t, "keep", vllmXargs["session_id"])
}

func TestHTTPGenerate_ForceStreamFallbackDropsNestedKVTransferParams(t *testing.T) {
	const passthroughReply = `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var (
		mu             sync.Mutex
		captured       []capturedGatewayRequest
		decodeAttempts int
	)
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		var parsed map[string]any
		if err == nil {
			err = json.Unmarshal(raw, &parsed)
		}
		phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
		mu.Lock()
		captured = append(captured, capturedGatewayRequest{
			phase: phase,
			body:  parsed,
			err:   err,
		})
		var attempt int
		if phase == gateway.PhaseDecode {
			decodeAttempts++
			attempt = decodeAttempts
		}
		mu.Unlock()

		switch {
		case phase == gateway.PhasePrefill:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				reqcommon.FieldKVTransferParams: map[string]any{
					reqcommon.FieldDoRemotePrefill: true,
					reqcommon.FieldRemoteBlockIDs:  []any{1, 2},
					reqcommon.FieldRemoteEngineID:  "prefill-pod-a",
				},
			})
		case attempt == 1:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: not-json\n\n")
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, passthroughReply)
		}
	}))
	defer gw.Close()

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	prefill, err := steps.NewPrefillStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)
	decode, err := steps.NewDecodeStep(gwClient, map[string]any{steps.ParamKVConnector: kv.NIXL})
	require.NoError(t, err)
	require.NoError(t, steps.EnableForceStream([]pipeline.Step{decode}, "1GiB", "1GiB"))

	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{prefill, decode}), gwClient)

	reqBody := `{
		"model": "llama-3",
		"stream": false,
		"token_ids": [1, 2, 3],
		"sampling_params": {
			"max_tokens": 8,
			"extra_args": {
				"kv_transfer_params": {"do_remote_prefill": true},
				"custom": "keep"
			}
		}
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://"+addr+reqcommon.PathVLLMGenerate, strings.NewReader(reqBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	gotBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.JSONEq(t, passthroughReply, string(gotBody))

	mu.Lock()
	calls := append([]capturedGatewayRequest(nil), captured...)
	mu.Unlock()

	require.Len(t, calls, 3, "expected prefill, forced decode attempt, and fallback decode recompute")
	for i, call := range calls {
		require.NoError(t, call.err, "call %d", i)
	}
	require.Equal(t, gateway.PhasePrefill, calls[0].phase)
	require.Equal(t, gateway.PhaseDecode, calls[1].phase)
	require.Equal(t, true, calls[1].body["stream"])
	require.Contains(t, calls[1].body, reqcommon.FieldKVTransferParams)
	assertNoNestedTransferParams(t, calls[1].body, "keep")
	require.Equal(t, gateway.PhaseDecode, calls[2].phase)
	require.Equal(t, false, calls[2].body["stream"])
	require.NotContains(t, calls[2].body, reqcommon.FieldKVTransferParams)
	assertNoNestedTransferParams(t, calls[2].body, "keep")
}
