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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// sseServer returns an upstream that asserts the forced request enabled
// streaming and usage, then writes the given SSE data frames followed by [DONE].
// capturedBody receives the decoded request body for the caller to inspect.
func sseServer(t *testing.T, frames []string, capturedBody *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		if capturedBody != nil {
			*capturedBody = parsed
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, f := range frames {
			fmt.Fprintf(w, "data: %s\n\n", f)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

// newForceStreamStep builds a decode step with force-streaming on and the given
// buffer size, and registers a fresh metrics registry the caller reads back.
func newForceStreamStep(t *testing.T, serverURL, bufferSize string) (*DecodeStep, *prometheus.Registry) {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()

	gwClient := gateway.New(config.GatewayConfig{Address: serverURL})
	// Pin the per-request cap to the whole budget so these tests exercise budget
	// and ceiling behavior with a single request free to reserve all of it; the
	// default cap (a fixed 256KiB, clamped to the budget only when smaller) is
	// covered separately.
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     bufferSize,
		ParamForceStreamMaxRequestSize: bufferSize,
	})
	require.NoError(t, err)
	return step.(*DecodeStep), reg
}

func forceStreamCount(t *testing.T, reg *prometheus.Registry, result string) float64 {
	t.Helper()
	return gatherLabeled(t, reg, "llm_d_coordinator_force_stream_total", "result", result)
}

// forceStreamCountModel reads force_stream_total for one (model_name, result)
// pair, so a test can assert the counter is scoped by model.
func forceStreamCountModel(t *testing.T, reg *prometheus.Registry, model, result string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_coordinator_force_stream_total" {
			continue
		}
		for _, m := range mf.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["model_name"] == model && labels["result"] == result {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

func forceStreamGauge(t *testing.T, reg *prometheus.Registry) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != "llm_d_coordinator_force_stream_buffered_bytes" {
			continue
		}
		return mf.GetMetric()[0].GetGauge().GetValue()
	}
	return 0
}

func gatherLabeled(t *testing.T, reg *prometheus.Registry, name, label, value string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == label && l.GetValue() == value {
					return m.GetCounter().GetValue()
				}
			}
		}
	}
	return 0
}

func TestForceStreamLossy(t *testing.T) {
	// chat and text requests that may carry tool/function calls or logprobs are lossy.
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"tools": []any{map[string]any{"type": "function"}}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"functions": []any{map[string]any{"name": "f"}}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"logprobs": true}))
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": 3}))
	// On the legacy Completions API logprobs is a count; its presence (even 0)
	// requests the sampled token's logprob, which the reassembler drops, so a
	// numeric value of any size is lossy. JSON decodes the count to float64.
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": 0}))
	require.True(t, forceStreamLossy(sseShapeText, map[string]any{"logprobs": float64(0)}))

	// vLLM chat reasoning (delta.reasoning) is not folded, so a chat
	// request that activates thinking via chat_template_kwargs.enable_thinking or
	// reasoning_effort is lossy and takes the pass-through.
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}}))
	require.True(t, forceStreamLossy(sseShapeChat, map[string]any{"reasoning_effort": "high"}))

	// Absent, empty, or false forms are not lossy.
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"tools": []any{}}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"logprobs": false}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}))
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"reasoning_effort": ""}))
	// Reasoning is a chat concept; the text shape has no reasoning to drop.
	require.False(t, forceStreamLossy(sseShapeText, map[string]any{"reasoning_effort": "high"}))

	// The responses shape emits the upstream object verbatim, so it is never lossy
	// even when the request declares these fields.
	require.False(t, forceStreamLossy(sseShapeResponses, map[string]any{"tools": []any{map[string]any{"type": "function"}}}))
	// The generate route reads logprobs from sampling_params, not the top level.
	require.False(t, forceStreamLossy(sseShapeGenerate, map[string]any{"logprobs": true}))

	// The generate fold drops logprobs, so a sampling_params that sets either
	// count is lossy. 0 still returns the chosen token's logprob and -1 the whole
	// vocabulary; only an absent or null value requests none.
	for _, field := range []string{"logprobs", "prompt_logprobs"} {
		for _, v := range []any{float64(0), float64(5), float64(-1)} {
			require.True(t, forceStreamLossy(sseShapeGenerate, map[string]any{"sampling_params": map[string]any{field: v}}), "%s=%v", field, v)
		}
		require.False(t, forceStreamLossy(sseShapeGenerate, map[string]any{"sampling_params": map[string]any{field: nil}}), "%s=null", field)
	}
	require.False(t, forceStreamLossy(sseShapeGenerate, map[string]any{"sampling_params": map[string]any{"max_tokens": float64(8)}}))
	require.True(t, forceStreamLossy(sseShapeGenerate, map[string]any{"sampling_params": map[string]any{"logprobs": json.RawMessage(`2`)}}),
		"a raw value is resolved before the check")

	// The messages shape folds only text blocks, so a request declaring tools is
	// lossy (tool_use blocks would be dropped); logprobs/reasoning are not Messages
	// concepts and do not gate it.
	require.True(t, forceStreamLossy(sseShapeMessages, map[string]any{"tools": []any{map[string]any{"name": "get_weather"}}}))
	require.False(t, forceStreamLossy(sseShapeMessages, map[string]any{"tools": []any{}}))
	require.False(t, forceStreamLossy(sseShapeMessages, map[string]any{}))
	require.False(t, forceStreamLossy(sseShapeMessages, map[string]any{"logprobs": true}))

	// A Messages request that turns thinking on streams thinking blocks the fold
	// drops, so it is lossy. adaptive lets the model think too; disabled does not.
	require.True(t, forceStreamLossy(sseShapeMessages, map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024}}))
	require.True(t, forceStreamLossy(sseShapeMessages, map[string]any{"thinking": map[string]any{"type": "adaptive"}}))
	require.False(t, forceStreamLossy(sseShapeMessages, map[string]any{"thinking": map[string]any{"type": "disabled"}}))
	require.False(t, forceStreamLossy(sseShapeMessages, map[string]any{"thinking": nil}))
	// thinking is a Messages field; chat turns reasoning on with its own fields.
	require.False(t, forceStreamLossy(sseShapeChat, map[string]any{"thinking": map[string]any{"type": "enabled"}}))
}

// TestForceStreamLossy_RawMessageFields guards the production body shape: the
// coordinator decodes only an allow-list of request fields, so tools, logprobs, and
// reasoning arrive as json.RawMessage, not native Go values. forceStreamLossy must
// resolve them (via reqcommon.FieldValue) rather than type-assert, or a lossy
// request would be force-streamed and silently drop the field.
func TestForceStreamLossy_RawMessageFields(t *testing.T) {
	// rawBody decodes a JSON request the way the coordinator's handler does: only
	// the inspected fields become Go values; everything else stays json.RawMessage.
	rawBody := func(t *testing.T, jsonBody string) map[string]any {
		t.Helper()
		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal([]byte(jsonBody), &fields))
		inspected := map[string]bool{reqcommon.FieldModel: true, reqcommon.FieldStream: true}
		body := make(map[string]any, len(fields))
		for k, v := range fields {
			if inspected[k] {
				var decoded any
				require.NoError(t, json.Unmarshal(v, &decoded))
				body[k] = decoded
			} else {
				body[k] = v // json.RawMessage, exactly as production leaves it
			}
		}
		return body
	}

	lossy := []struct {
		name string
		json string
	}{
		{"tools", `{"model":"m","stream":false,"tools":[{"type":"function"}]}`},
		{"functions", `{"model":"m","stream":false,"functions":[{"name":"f"}]}`},
		{"logprobs true", `{"model":"m","stream":false,"logprobs":true}`},
		{"logprobs count", `{"model":"m","stream":false,"logprobs":3}`},
		{"enable_thinking", `{"model":"m","stream":false,"chat_template_kwargs":{"enable_thinking":true}}`},
		{"reasoning_effort", `{"model":"m","stream":false,"reasoning_effort":"high"}`},
	}
	for _, tc := range lossy {
		t.Run("lossy/"+tc.name, func(t *testing.T) {
			require.True(t, forceStreamLossy(sseShapeChat, rawBody(t, tc.json)),
				"a lossy field as json.RawMessage must still be detected")
		})
	}

	// A plain request with no lossy field stays force-streamable.
	require.False(t, forceStreamLossy(sseShapeChat,
		rawBody(t, `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`)))

	// The Messages shape resolves tools through the same path: a json.RawMessage
	// tools array is detected, a plain request is not.
	require.True(t, forceStreamLossy(sseShapeMessages,
		rawBody(t, `{"model":"m","stream":false,"tools":[{"name":"get_weather"}]}`)))
	require.True(t, forceStreamLossy(sseShapeMessages,
		rawBody(t, `{"model":"m","stream":false,"thinking":{"type":"enabled","budget_tokens":1024}}`)))
	require.False(t, forceStreamLossy(sseShapeMessages,
		rawBody(t, `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`)))
}

// TestForceStreamBody_StreamOptions checks the forced body enables streaming and,
// for the chat and text shapes, requests include_usage while preserving any
// stream_options the client already sent, without mutating the source body.
func TestForceStreamBody_StreamOptions(t *testing.T) {
	t.Run("preserves existing stream_options and adds include_usage", func(t *testing.T) {
		src := map[string]any{
			"model":          "m",
			"stream":         false,
			"stream_options": map[string]any{"continuous_usage_stats": true},
		}
		body := forceStreamBody(src, sseShapeChat)

		require.Equal(t, true, body["stream"])
		opts := body["stream_options"].(map[string]any)
		require.Equal(t, true, opts["include_usage"], "include_usage is added")
		require.Equal(t, true, opts["continuous_usage_stats"], "the client's existing option is kept")

		// The shared source body is left untouched: still non-streaming, its
		// stream_options unchanged.
		require.Equal(t, false, src["stream"])
		srcOpts := src["stream_options"].(map[string]any)
		require.NotContains(t, srcOpts, "include_usage", "the source stream_options is not mutated")
	})

	t.Run("adds stream_options when the client sent none", func(t *testing.T) {
		body := forceStreamBody(map[string]any{"model": "m"}, sseShapeText)
		opts := body["stream_options"].(map[string]any)
		require.Equal(t, true, opts["include_usage"])
	})

	t.Run("no stream_options for responses, generate, or messages", func(t *testing.T) {
		for _, shape := range []sseShape{sseShapeResponses, sseShapeGenerate, sseShapeMessages} {
			body := forceStreamBody(map[string]any{"model": "m"}, shape)
			require.Equal(t, true, body["stream"])
			_, has := body["stream_options"]
			require.False(t, has, "these shapes carry no stream_options")
		}
	})
}

func TestForceStream_ParamsOptional(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})

	// Both params absent: the step builds and force-streaming stays off, so an
	// operator who configures no force_stream keys keeps the pass-through.
	off, err := NewDecodeStep(gwClient, map[string]any{})
	require.NoError(t, err)
	require.False(t, off.(*DecodeStep).forceStream, "force_stream defaults to off when the param is absent")
	require.Nil(t, off.(*DecodeStep).budget, "no budget is allocated while force_stream is off")

	// force_stream on with no buffer size: the default budget applies, so the
	// size param is optional even when the feature is enabled.
	on, err := NewDecodeStep(gwClient, map[string]any{ParamForceStream: true})
	require.NoError(t, err)
	ds := on.(*DecodeStep)
	require.True(t, ds.forceStream)
	require.NotNil(t, ds.budget, "force_stream on without a buffer size uses the default budget")
	require.Equal(t, int64(1<<30), ds.budget.max, "the default buffer budget is 1GiB")
	require.Equal(t, int64(256<<10), ds.budget.perRequestMax, "the default per-request cap is 256KiB")
}

// TestForceStream_ClientRequestedStreaming_Skipped verifies the !reqCtx.Stream
// guard: a client that already asked for streaming is never force-streamed. The
// request takes the normal pass-through, so the client receives the upstream SSE
// verbatim and no force-stream outcome is recorded.
func TestForceStream_ClientRequestedStreaming_Skipped(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-streaming",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           true,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "llama-3", "stream": true},
		ResponseWriter:   recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	httpResult := recorder.Result()
	require.Equal(t, "text/event-stream", httpResult.Header.Get("Content-Type"),
		"a streaming client gets the upstream SSE verbatim, not a reassembled JSON reply")
	body, _ := io.ReadAll(httpResult.Body)
	require.Contains(t, string(body), "[DONE]", "the raw SSE stream is passed through")

	// None of the force-stream outcomes fire: the request never entered the branch.
	for _, rv := range []string{
		coordmetrics.ForceStreamResultForced,
		coordmetrics.ForceStreamResultFallbackBudget,
		coordmetrics.ForceStreamResultFallbackUnbounded,
		coordmetrics.ForceStreamResultFallbackUnsupported,
		coordmetrics.ForceStreamResultFallbackUnreadable,
	} {
		require.InDelta(t, 0.0, forceStreamCount(t, reg, rv), 1e-9, "no force-stream outcome for a streaming client: %s", rv)
	}
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

// TestForceStream_BudgetConfigErrors verifies that a malformed buffer-size or
// per-request-size value fails config load rather than running a silent default.
func TestForceStream_BudgetConfigErrors(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})
	cases := []struct {
		name   string
		params map[string]any
	}{
		{"zero buffer", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "0"}},
		{"negative buffer", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "-1GiB"}},
		{"non-numeric buffer", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "abc"}},
		{"zero per-request", map[string]any{ParamForceStream: true, ParamForceStreamMaxRequestSize: "0"}},
		{"non-numeric per-request", map[string]any{ParamForceStream: true, ParamForceStreamMaxRequestSize: "nope"}},
		{"per-request exceeds buffer", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "1GiB", ParamForceStreamMaxRequestSize: "2GiB"}},
		{"buffer below envelope overhead", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "1KiB"}},
		{"per-request below envelope overhead", map[string]any{ParamForceStream: true, ParamForceStreamBufferSize: "1GiB", ParamForceStreamMaxRequestSize: "1KiB"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewDecodeStep(gwClient, tc.params)
			require.Error(t, err)
		})
	}
}

// TestForceStream_PerRequestCap_FallsBack verifies a request whose estimate
// exceeds the per-request cap takes the pass-through even when the shared budget
// is far larger, so one large request cannot claim the whole budget.
func TestForceStream_PerRequestCap_FallsBack(t *testing.T) {
	var upstreamBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &upstreamBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}}})
	}))
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "1GiB",
		ParamForceStreamMaxRequestSize: "4KiB",
	})
	require.NoError(t, err)

	// A generate request whose max_tokens would reserve far more than the 4KiB
	// per-request cap: it is bounded but too large, so it falls back rather than
	// claiming a large share of the 1GiB budget.
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-big",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1, 2, 3},
			"sampling_params": map[string]any{"max_tokens": 100000},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.(*DecodeStep).Execute(context.Background(), reqCtx))

	require.Equal(t, false, upstreamBody["stream"], "a request over the per-request cap must pass through unchanged")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnbounded), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

// TestForceStream_OversizedFrame_CeilingAbort verifies that a single SSE frame
// larger than the request's reservation trips the scanner's buffer limit, is
// counted error_ceiling, and falls back to the pass-through (not a generic read
// error, and not a 5xx): nothing reached the client, so the request is re-sent.
func TestForceStream_OversizedFrame_CeilingAbort(t *testing.T) {
	var big strings.Builder
	big.WriteString(`{"request_id":"g","choices":[{"index":0,"token_ids":[`)
	for i := 0; i < 2000; i++ {
		if i > 0 {
			big.WriteByte(',')
		}
		big.WriteString("12345")
	}
	big.WriteString(`]}]}`)
	passthroughReply := `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			fmt.Fprintf(w, "data: %s\n\n", big.String())
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	// A 4KiB budget reserves roughly 2.3KiB for a max_tokens=1 generate request, so
	// the single ~12KB frame overflows the scanner buffer sized to that reservation.
	step, reg := newForceStreamStep(t, server.URL, "4KiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-bigframe",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 1},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "an oversized frame must fall back, not 5xx")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt then a pass-through")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_TotalCarriesModelLabel verifies force_stream_total is scoped by
// model, so budget-exhaustion fallbacks can be attributed to a model's traffic.
func TestForceStream_TotalCarriesModelLabel(t *testing.T) {
	frames := []string{
		`{"id":"c-1","model":"llama-3","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-model",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.InDelta(t, 1.0, forceStreamCountModel(t, reg, "llama-3", coordmetrics.ForceStreamResultForced), 1e-9)
}

func TestForceStream_Chat_ReassemblesToJSON(t *testing.T) {
	frames := []string{
		`{"id":"c-1","model":"llama-3","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"c-1","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"c-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c-1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-1",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":      "llama-3",
			"stream":     false,
			"max_tokens": 16,
			"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	// The upstream must have been asked to stream with a trailing usage frame.
	require.Equal(t, true, upstreamBody["stream"], "force-stream must send stream:true upstream")
	opts, ok := upstreamBody["stream_options"].(map[string]any)
	require.True(t, ok, "force-stream must set stream_options")
	require.Equal(t, true, opts["include_usage"], "force-stream must request include_usage")

	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	require.Equal(t, "application/json", result.Header.Get("Content-Type"),
		"client of a non-streaming request must receive JSON, not an event stream")

	respBody, _ := io.ReadAll(result.Body)
	require.NotEmpty(t, result.Header.Get("Content-Length"),
		"a buffered reply is written whole, so it carries a Content-Length")

	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got), "reassembled reply must be valid JSON")
	require.Equal(t, "chat.completion", got["object"])
	choices := got["choices"].([]any)
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	require.Equal(t, "assistant", msg["role"])
	require.Equal(t, "Hello world", msg["content"])
	require.Equal(t, "stop", choice["finish_reason"])
	usage := got["usage"].(map[string]any)
	require.Equal(t, float64(7), usage["total_tokens"])

	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation must be released after the reply")
}

func TestForceStream_Responses_ReassemblesToJSON(t *testing.T) {
	completed := `{"id":"resp-1","object":"response","model":"llama-3","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`
	frames := []string{
		`{"type":"response.created","response":{"id":"resp-1","model":"llama-3","status":"in_progress"}}`,
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":" world"}`,
		`{"type":"response.completed","response":` + completed + `}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-resp",
		OriginalPath:     reqcommon.PathResponses,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":             "llama-3",
			"stream":            false,
			"max_output_tokens": 16,
			"input":             "hi",
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	// The Responses API has no stream_options; force-stream must enable streaming
	// without adding it (usage arrives in the response.completed event).
	require.Equal(t, true, upstreamBody["stream"], "force-stream must send stream:true upstream")
	_, hasOpts := upstreamBody["stream_options"]
	require.False(t, hasOpts, "force-stream must not add stream_options for the Responses API")

	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	require.Equal(t, "application/json", result.Header.Get("Content-Type"),
		"client of a non-streaming request must receive JSON, not an event stream")

	respBody, _ := io.ReadAll(result.Body)
	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got), "reassembled reply must be valid JSON")

	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(completed), &want))
	require.Equal(t, want, got, "the forced reply must equal the pod's non-streaming response object")

	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation must be released after the reply")
}

func TestForceStream_Messages_ReassemblesToJSON(t *testing.T) {
	// The Anthropic Messages stream: typed events, no stream_options, no [DONE]
	// (ends on message_stop). sseServer appends a trailing [DONE] which the scanner
	// harmlessly breaks on after message_stop already marked the stream terminal.
	frames := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude","role":"assistant","usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":5,"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-msg",
		OriginalPath:     reqcommon.PathMessages,
		Model:            "claude",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":      "claude",
			"stream":     false,
			"max_tokens": 16,
			"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	// Messages has no stream_options; force-stream enables streaming without it.
	require.Equal(t, true, upstreamBody["stream"], "force-stream must send stream:true upstream")
	_, hasOpts := upstreamBody["stream_options"]
	require.False(t, hasOpts, "force-stream must not add stream_options for the Messages API")

	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	require.Equal(t, "application/json", result.Header.Get("Content-Type"))

	var got map[string]any
	respBody, _ := io.ReadAll(result.Body)
	require.NoError(t, json.Unmarshal(respBody, &got), "reassembled reply must be valid JSON")
	require.Equal(t, "msg_1", got["id"])
	require.Equal(t, "message", got["type"])
	require.Equal(t, "assistant", got["role"])
	require.Equal(t, "end_turn", got["stop_reason"])
	block := got["content"].([]any)[0].(map[string]any)
	require.Equal(t, "text", block["type"])
	require.Equal(t, "Hello world", block["text"])
	usage := got["usage"].(map[string]any)
	require.EqualValues(t, 5, usage["input_tokens"])
	require.EqualValues(t, 2, usage["output_tokens"])

	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation must be released after the reply")
}

// A Messages stream carrying a non-text block (server-side tool_use the request did
// not declare) is detected mid-stream and falls back to the pass-through, which
// preserves the block. Counted fallback_unsupported.
func TestForceStream_Messages_NonTextFallsBack(t *testing.T) {
	passthroughReply := `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"tool_use","id":"t1","name":"get_weather","input":{}}],"stop_reason":"tool_use"}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"get_weather\"}}\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-msg-tool",
		OriginalPath:     reqcommon.PathMessages,
		Model:            "claude",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "claude", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "weather?"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "a non-text block must fall back, not drop silently")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body), "the pass-through reply keeps the tool_use block")
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

func TestForceStream_Text_ReassemblesToJSON(t *testing.T) {
	frames := []string{
		`{"id":"t-1","object":"text_completion","model":"llama-3","choices":[{"index":0,"text":"He"}]}`,
		`{"id":"t-1","choices":[{"index":0,"text":"llo","finish_reason":"length"}]}`,
		`{"id":"t-1","choices":[],"usage":{"total_tokens":3}}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-text",
		OriginalPath:     reqcommon.PathCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "llama-3", "stream": false, "max_tokens": 8, "prompt": "Hello"},
		ResponseWriter:   recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	result := recorder.Result()
	require.Equal(t, "application/json", result.Header.Get("Content-Type"))
	respBody, _ := io.ReadAll(result.Body)
	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got))
	require.Equal(t, "text_completion", got["object"])
	choice := got["choices"].([]any)[0].(map[string]any)
	require.Equal(t, "Hello", choice["text"])
	require.Equal(t, "length", choice["finish_reason"])
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

func TestForceStream_Generate_ReassemblesToJSON(t *testing.T) {
	// vLLM's tokens-in generate API streams raw token ids: each chunk carries one
	// choices[].token_ids entry under a request_id envelope with no object/id/model,
	// a trailing choices:[] usage frame closes the stream, and the non-streaming
	// reply folds the tokens into a request_id envelope with prompt_logprobs and
	// kv_transfer_params null, no object, and no usage block. The request body
	// places max_tokens under sampling_params, where OutputTokenLimit reads it for
	// this API; a top-level cap would be invisible and the request would fall back
	// to the pass-through instead of force-streaming.
	frames := []string{
		`{"request_id":"generate-tokens-g1","choices":[{"index":0,"logprobs":null,"finish_reason":null,"token_ids":[28715]}],"usage":null}`,
		`{"request_id":"generate-tokens-g1","choices":[{"index":0,"logprobs":null,"finish_reason":null,"token_ids":[314]}],"usage":null}`,
		`{"request_id":"generate-tokens-g1","choices":[{"index":0,"logprobs":null,"finish_reason":null,"token_ids":[678]}],"usage":null}`,
		`{"request_id":"generate-tokens-g1","choices":[{"index":0,"logprobs":null,"finish_reason":"length","token_ids":[921]}],"usage":null}`,
		`{"request_id":"generate-tokens-g1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":           "llama-3",
			"stream":          false,
			"token_ids":       []any{1, 2, 3},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	// The sampling_params cap is read, so the request is reservable and forced
	// rather than passed through unchanged.
	require.Equal(t, true, upstreamBody["stream"], "generate with a sampling_params cap must be force-streamed")

	result := recorder.Result()
	require.Equal(t, "application/json", result.Header.Get("Content-Type"))
	respBody, _ := io.ReadAll(result.Body)
	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got))

	// The generate reply is token-level: a request_id envelope with prompt_logprobs
	// and kv_transfer_params null, token_ids folded per choice, no object, and no
	// usage block (unlike chat and text, the non-streaming generate reply omits it).
	require.Equal(t, "generate-tokens-g1", got["request_id"])
	require.NotContains(t, got, "object")
	require.Contains(t, got, "prompt_logprobs")
	require.Nil(t, got["prompt_logprobs"])
	require.Contains(t, got, "kv_transfer_params")
	require.Nil(t, got["kv_transfer_params"])
	require.NotContains(t, got, "usage")
	choice := got["choices"].([]any)[0].(map[string]any)
	require.NotContains(t, choice, "routed_experts")
	require.Equal(t, []any{float64(28715), float64(314), float64(678), float64(921)}, choice["token_ids"])
	require.Equal(t, "length", choice["finish_reason"])
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

func TestForceStream_MultipleChoices(t *testing.T) {
	frames := []string{
		`{"choices":[{"index":0,"delta":{"content":"A1"}},{"index":1,"delta":{"content":"B1"}}]}`,
		`{"choices":[{"index":1,"delta":{"content":"B2"},"finish_reason":"stop"},{"index":0,"delta":{"content":"A2"},"finish_reason":"stop"}]}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, _ := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-n",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false, "max_tokens": 16, "n": 2,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	respBody, _ := io.ReadAll(recorder.Result().Body)
	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got))
	choices := got["choices"].([]any)
	require.Len(t, choices, 2)
	// Choices come out ordered by index regardless of frame interleaving.
	first := choices[0].(map[string]any)
	second := choices[1].(map[string]any)
	require.Equal(t, float64(0), first["index"])
	require.Equal(t, "A1A2", first["message"].(map[string]any)["content"])
	require.Equal(t, float64(1), second["index"])
	require.Equal(t, "B1B2", second["message"].(map[string]any)["content"])
}

// TestForceStream_ChatBothTokenLimits_ReservesForCompletionTokens reproduces the
// reviewer's measured 5xx end to end: a chat n:2 request carrying both max_tokens
// (16) and max_completion_tokens (4000). The engine honors max_completion_tokens,
// so the reply can run well past the 16-token reservation the old precedence would
// have sized. The streamed content below exceeds that wrong reservation (16 tokens
// ~= 3 KiB for n:2) but fits the correct one (4000 tokens ~= 128 KiB), so the
// request force-streams successfully instead of tripping the ceiling into a 5xx.
func TestForceStream_ChatBothTokenLimits_ReservesForCompletionTokens(t *testing.T) {
	// ~2 KiB of content per choice, more than the wrong (16-token) reservation
	// allows for the two choices combined, less than the correct (4000-token) one.
	chunk := strings.Repeat("x", 2048)
	frames := []string{
		fmt.Sprintf(`{"id":"c","model":"llama-3","choices":[{"index":0,"delta":{"role":"assistant","content":%q}},{"index":1,"delta":{"role":"assistant","content":%q}}]}`, chunk, chunk),
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}]}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-both-limits",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false, "n": 2,
			"max_tokens":            16,
			"max_completion_tokens": 4000,
			"messages":              []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx),
		"reserving for max_completion_tokens must let the reply through, not 5xx at the ceiling")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	require.Equal(t, "application/json", result.Header.Get("Content-Type"))

	var got map[string]any
	respBody, _ := io.ReadAll(result.Body)
	require.NoError(t, json.Unmarshal(respBody, &got), "the full reply is returned as one JSON")
	require.Len(t, got["choices"].([]any), 2)
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9,
		"the conforming request must not hit the ceiling")
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the reply")
}

func TestForceStream_FallbackBudget(t *testing.T) {
	var upstreamBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &upstreamBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}}})
	}))
	defer server.Close()

	// Size the budget to exactly one request's reservation, then pre-reserve all
	// of it so the request under test cannot fit and must fall back.
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-fb",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1, 2, 3},
			"sampling_params": map[string]any{"max_tokens": 10},
		},
		ResponseWriter: httptest.NewRecorder(),
	}

	sizing, _ := newForceStreamStep(t, server.URL, "1GiB")
	reserved, ok := sizing.estimateReservation(reqCtx)
	require.True(t, ok)
	// Build the real step with the budget pinned to one reservation, then claim
	// all of it so the request under test cannot fit.
	step, reg := newForceStreamStep(t, server.URL, strconv.FormatInt(reserved, 10))
	require.True(t, step.budget.tryReserve(reserved), "pre-reserve the whole budget")
	require.InDelta(t, float64(reserved), forceStreamGauge(t, reg), 1e-9, "the gauge reports the live reservation")

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	require.Equal(t, false, upstreamBody["stream"], "fallback must send the client's non-streaming body unchanged")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackBudget), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)

	step.budget.release(reserved)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "the gauge drops by the released reservation")
}

// TestForceStream_NoTokenLimit_Forced verifies a request with no declared output
// token limit is still force-streamed: it reserves the whole per-request cap (most
// chat clients send no max_tokens, so refusing them would leave the common case
// unforced) and the per-request ceiling bounds the reply.
func TestForceStream_NoTokenLimit_Forced(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"llama-3","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}`,
		`{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	var upstreamBody map[string]any
	server := sseServer(t, frames, &upstreamBody)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-nolimit",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	require.Equal(t, true, upstreamBody["stream"], "a request without a token limit is still force-streamed")
	require.Equal(t, "application/json", recorder.Result().Header.Get("Content-Type"))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnbounded), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the reply")
}

// TestForceStream_LossyRequest_PassThrough verifies that a chat or text request
// whose reply could carry fields the reassembler drops (tool/function calls or
// logprobs) is not force-streamed: it takes the pass-through unchanged and is
// counted as fallback_unsupported.
func TestForceStream_LossyRequest_PassThrough(t *testing.T) {
	cases := []struct {
		name  string
		path  string
		extra map[string]any
	}{
		{"chat tools", reqcommon.PathChatCompletions, map[string]any{"tools": []any{map[string]any{"type": "function"}}}},
		{"chat functions", reqcommon.PathChatCompletions, map[string]any{"functions": []any{map[string]any{"name": "f"}}}},
		{"chat logprobs", reqcommon.PathChatCompletions, map[string]any{"logprobs": true}},
		{"completions logprobs", reqcommon.PathCompletions, map[string]any{"logprobs": 5}},
		{"chat enable_thinking", reqcommon.PathChatCompletions, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}}},
		{"chat reasoning_effort", reqcommon.PathChatCompletions, map[string]any{"reasoning_effort": "high"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var upstreamBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, &upstreamBody)
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}}})
			}))
			defer server.Close()

			step, reg := newForceStreamStep(t, server.URL, "1GiB")
			body := map[string]any{"model": "llama-3", "stream": false, "max_tokens": 16}
			for k, v := range tc.extra {
				body[k] = v
			}
			if tc.path == reqcommon.PathChatCompletions {
				body["messages"] = []any{map[string]any{"role": "user", "content": "hi"}}
			} else {
				body["prompt"] = "hi"
			}
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-lossy",
				OriginalPath:     tc.path,
				Model:            "llama-3",
				Stream:           false,
				KVTransferParams: map[string]any{},
				Body:             body,
				ResponseWriter:   httptest.NewRecorder(),
			}
			require.NoError(t, step.Execute(context.Background(), reqCtx))

			require.Equal(t, false, upstreamBody["stream"], "a lossy request must pass through with its non-streaming body unchanged")
			require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
			require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
		})
	}
}

// TestForceStream_ServerDefaultReasoning_FallsBack covers the frame-level lossy
// gate: a plain chat request (no tools, no reasoning signal) whose pod streams
// reasoning by default. forceStreamLossy cannot see this in the request, so
// the forced attempt starts, detects the unfoldable delta mid-stream, and falls back
// to the pass-through, which preserves reasoning. Counted fallback_unsupported.
func TestForceStream_ServerDefaultReasoning_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Paris","reasoning":"let me think"}}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			// vLLM emits reasoning before content; the first reasoning delta flags the
			// reply lossy before anything is written to the client.
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"let me think\"}}]}\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-reasoning",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "a server-default lossy field must fall back, not drop silently")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body), "the pass-through reply keeps reasoning")
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

func TestForceStream_UpstreamError_ForwardsVerbatim(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusServiceUnavailable} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":"boom"}`))
			}))
			defer server.Close()

			step, reg := newForceStreamStep(t, server.URL, "1GiB")

			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-err",
				OriginalPath:     reqcommon.PathChatCompletions,
				Model:            "llama-3",
				Stream:           false,
				KVTransferParams: map[string]any{},
				Body: map[string]any{
					"model": "llama-3", "stream": false, "max_tokens": 16,
					"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				},
				ResponseWriter: recorder,
			}

			err := step.Execute(context.Background(), reqCtx)
			var streamed *pipeline.UpstreamStreamedError
			require.True(t, errors.As(err, &streamed), "an upstream error with a written body must report UpstreamStreamedError")
			require.Equal(t, status, streamed.StatusCode)

			result := recorder.Result()
			require.Equal(t, status, result.StatusCode)
			respBody, _ := io.ReadAll(result.Body)
			require.Equal(t, `{"error":"boom"}`, string(respBody), "upstream error body must forward verbatim")
			// A written error must not be double-counted as a forced success; it is
			// attributed to the upstream-error outcome instead.
			require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
			require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
			require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
		})
	}
}

func TestForceStream_TransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	serverURL := server.URL
	server.Close()

	step, reg := newForceStreamStep(t, serverURL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-transport",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	require.Error(t, err)
	var streamed *pipeline.UpstreamStreamedError
	require.False(t, errors.As(err, &streamed),
		"a transport failure writes nothing, so it must return a plain error for a clean 502")

	require.Equal(t, 0, recorder.Body.Len(), "nothing may be written to the client on a transport failure")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorTransport), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

// TestForceStream_Buffered_UpstreamError covers the buffered path's upstream-error
// accounting: a generate request (n=1 generate takes the buffered path) whose
// upstream returns a 5xx is forwarded verbatim and counted as error_upstream.
func TestForceStream_Buffered_UpstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen-err",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	var streamed *pipeline.UpstreamStreamedError
	require.True(t, errors.As(err, &streamed), "an upstream error with a written body reports UpstreamStreamedError")
	require.Equal(t, http.StatusServiceUnavailable, recorder.Result().StatusCode)
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after an upstream error")
}

// TestForceStream_Buffered_TransportError covers the buffered path's transport
// accounting: a generate request whose upstream never responds is counted as
// error_transport, with nothing written and the reservation released.
func TestForceStream_Buffered_TransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	serverURL := server.URL
	server.Close()

	step, reg := newForceStreamStep(t, serverURL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-gen-transport",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	err := step.Execute(context.Background(), reqCtx)
	require.Error(t, err)
	var streamed *pipeline.UpstreamStreamedError
	require.False(t, errors.As(err, &streamed), "a transport failure writes nothing, so it is a plain error")
	require.Equal(t, 0, recorder.Body.Len(), "nothing may be written on a transport failure")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorTransport), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after a transport error")
}

// TestForceStream_CeilingAbort_FallsBack verifies that a buffered reply exceeding
// its reservation (usually a short per-token byte estimate, not a pod ignoring the
// limit) falls back to the pass-through rather than failing a conforming reply.
// Nothing reached the client, so the request is re-sent non-forced; the outcome
// keeps the error_ceiling label for alerting on the estimate shortfall.
func TestForceStream_CeilingAbort_FallsBack(t *testing.T) {
	passthroughReply := `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			// max_tokens=1 reserves a small ceiling; stream token_ids well past it to
			// trip the abort before anything reaches the client.
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			for i := 0; i < 100; i++ {
				fmt.Fprint(w, "data: {\"request_id\":\"g\",\"choices\":[{\"index\":0,\"token_ids\":[1,2,3,4,5,6,7,8,9,10]}],\"usage\":null}\n\n")
			}
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "4KiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-ceiling",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 1},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "a ceiling abort must fall back, not 5xx")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_CeilingSpentHandoff_Recomputes verifies a ceiling abort on a
// NIXL path whose block handoff the forced attempt already consumed falls back
// without replaying it: the pass-through drops the handoff and the decode pod
// recomputes the prompt.
func TestForceStream_CeilingSpentHandoff_Recomputes(t *testing.T) {
	const passthroughReply = `{"request_id":"g","choices":[{"index":0,"token_ids":[1]}]}`
	// Stream token_ids well past the small reservation to trip the ceiling.
	var forced strings.Builder
	for i := 0; i < 100; i++ {
		forced.WriteString("data: {\"request_id\":\"g\",\"choices\":[{\"index\":0,\"token_ids\":[1,2,3,4,5,6,7,8,9,10]}],\"usage\":null}\n\n")
	}
	var calls []spentHandoffCall
	server := spentHandoffServer(t, forced.String(), passthroughReply, &calls)
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "4KiB",
		ParamForceStreamMaxRequestSize: "4KiB",
		ParamKVConnector:               kv.NIXL,
	})
	require.NoError(t, err)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-ceiling-spent",
		OriginalPath: reqcommon.PathVLLMGenerate,
		Model:        "llama-3",
		Stream:       false,
		KVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{1, 2},
			reqcommon.FieldRemoteEngineID: "prefill-pod-a",
		},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 1},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.(*DecodeStep).Execute(context.Background(), reqCtx))
	requireRecomputed(t, calls)
	body, _ := io.ReadAll(recorder.Result().Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_MalformedFrame_BufferedPath verifies that an unreadable 200
// stream on the buffered path falls back to the pass-through rather than failing:
// the forced (stream:true) attempt gets unparsable SSE, so nothing reaches the
// client and the request is re-sent non-forced, which the pod serves normally.
// Generate takes the buffered path. The outcome is counted fallback_unreadable.
func TestForceStream_MalformedFrame_BufferedPath(t *testing.T) {
	passthroughReply := `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			fmt.Fprint(w, "data: not-json\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-badframe",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "an unreadable stream must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// spentHandoffCall records one upstream call of a spent-handoff test: whether it
// asked to stream, and the kv_transfer_params it carried.
type spentHandoffCall struct {
	stream   any
	kvParams any
	hasKV    bool
}

// spentHandoffServer fails the forced (stream:true) attempt with forcedStream and
// answers the pass-through with passthroughReply, recording every call.
func spentHandoffServer(t *testing.T, forcedStream, passthroughReply string, calls *[]spentHandoffCall) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		kvParams, hasKV := parsed[reqcommon.FieldKVTransferParams]
		*calls = append(*calls, spentHandoffCall{stream: parsed["stream"], kvParams: kvParams, hasKV: hasKV})
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, forcedStream)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
}

// requireRecomputed asserts a spent-handoff request was served by a pass-through
// that dropped the handoff: the forced attempt carried the block ids, the second
// call carried no kv_transfer_params, so the decode pod recomputed the prompt.
func requireRecomputed(t *testing.T, calls []spentHandoffCall) {
	t.Helper()
	require.Len(t, calls, 2, "a forced attempt, then one recompute pass-through")
	require.Equal(t, true, calls[0].stream)
	require.True(t, calls[0].hasKV, "the forced attempt carries the handoff")
	require.Equal(t, false, calls[1].stream)
	require.False(t, calls[1].hasKV, "the recompute pass-through must not replay the spent handoff, got %v", calls[1].kvParams)
}

// TestForceStream_SpentHandoff_Recomputes verifies that when the forced attempt
// fails after the upstream 200 and the request carries a one-shot P2P block
// handoff (remote_block_ids, the NIXL path), the fallback does not replay it: the
// pass-through drops kv_transfer_params, so the decode pod recomputes the prompt,
// and the client gets a correct reply. A shared-storage request (do_remote_prefill
// but no block ids) keeps its params on the fallback, covered by
// TestForceStream_MalformedFrame_BufferedPath.
func TestForceStream_SpentHandoff_Recomputes(t *testing.T) {
	const passthroughReply = `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var calls []spentHandoffCall
	server := spentHandoffServer(t, "data: not-json\n\n", passthroughReply, &calls)
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "1GiB",
		ParamForceStreamMaxRequestSize: "1GiB",
		ParamKVConnector:               kv.NIXL,
	})
	require.NoError(t, err)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-spent-handoff",
		OriginalPath: reqcommon.PathVLLMGenerate,
		Model:        "llama-3",
		Stream:       false,
		// The prefiller streamed these blocks to this decode pod; NIXL frees them
		// once pulled, so a replay is unsafe.
		KVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{1, 2},
			reqcommon.FieldRemoteEngineID: "prefill-pod-a",
		},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.(*DecodeStep).Execute(context.Background(), reqCtx))
	requireRecomputed(t, calls)
	require.Equal(t, http.StatusOK, recorder.Result().StatusCode)
	body, _ := io.ReadAll(recorder.Result().Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_LossyFrameSpentHandoff_Recomputes covers the frame-level lossy
// gate on a spent handoff: a plain chat request whose pod streams reasoning by
// default, on a NIXL path whose block handoff the forced attempt already consumed.
// The lossy field is detected mid-stream; the fallback drops the handoff so the
// decode pod recomputes the prompt, and the reply keeps its reasoning.
func TestForceStream_LossyFrameSpentHandoff_Recomputes(t *testing.T) {
	const passthroughReply = `{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Paris","reasoning":"let me think"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}`
	var calls []spentHandoffCall
	server := spentHandoffServer(t, "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"let me think\"}}]}\n\n", passthroughReply, &calls)
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "1GiB",
		ParamForceStreamMaxRequestSize: "1GiB",
		ParamKVConnector:               kv.NIXL,
	})
	require.NoError(t, err)

	prefillCached := 32
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-lossy-spent",
		OriginalPath: reqcommon.PathChatCompletions,
		Model:        "llama-3",
		Stream:       false,
		// The prefill cache-hit count describes the handoff the recompute drops.
		PrefillCachedTokens: &prefillCached,
		KVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{1, 2},
			reqcommon.FieldRemoteEngineID: "prefill-pod-a",
		},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.(*DecodeStep).Execute(context.Background(), reqCtx))
	requireRecomputed(t, calls)
	body, _ := io.ReadAll(recorder.Result().Body)
	require.JSONEq(t, passthroughReply, string(body),
		"the reply keeps the reasoning the fold would drop, and the recompute's own cached count, not prefill's")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_MessagesThinkingSpentHandoff_PassesThrough covers a Messages
// request that turns thinking on, on a NIXL path carrying a one-shot block
// handoff. Its thinking blocks would make a forced attempt lossy, and the attempt
// would spend the handoff, so the fallback would have to recompute the prompt. The
// request is declined before any forced attempt and served by the pass-through
// with its handoff intact.
func TestForceStream_MessagesThinkingSpentHandoff_PassesThrough(t *testing.T) {
	const passthroughReply = `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"hi"}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "1GiB",
		ParamForceStreamMaxRequestSize: "1GiB",
		ParamKVConnector:               kv.NIXL,
	})
	require.NoError(t, err)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-thinking-spent",
		OriginalPath: reqcommon.PathMessages,
		Model:        "m",
		Stream:       false,
		KVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{1, 2},
			reqcommon.FieldRemoteEngineID: "prefill-pod-a",
		},
		Body: map[string]any{
			"model": "m", "stream": false, "max_tokens": 2048,
			"thinking": json.RawMessage(`{"type":"enabled","budget_tokens":1024}`),
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.(*DecodeStep).Execute(context.Background(), reqCtx))
	require.Equal(t, []any{false}, streams, "no forced attempt; the pass-through serves the request once")
	respBody, _ := io.ReadAll(recorder.Result().Body)
	require.JSONEq(t, passthroughReply, string(respBody), "the thinking block reaches the client")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "nothing was reserved")
}

// TestForceStream_ForwardsPeerTopology covers the peer-topology header on the
// forced decode request. The EPP's topology-affinity filter and scorer read it on
// the decode leg, and the pass-through sets it, so the forced request carries it
// too and both legs are scheduled the same way.
func TestForceStream_ForwardsPeerTopology(t *testing.T) {
	for _, tc := range []struct {
		name     string
		topology string
	}{
		{"set", "zone=a,rack=r1"},
		{"unset", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r.Header.Values(reqcommon.PeerTopologyHeaderKey)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer server.Close()

			step, reg := newForceStreamStep(t, server.URL, "1GiB")
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-topology",
				OriginalPath:     reqcommon.PathChatCompletions,
				Model:            "m",
				Stream:           false,
				KVTransferParams: map[string]any{},
				PeerTopology:     tc.topology,
				Body: map[string]any{
					"model": "m", "stream": false, "max_tokens": 16,
					"messages": []any{map[string]any{"role": "user", "content": "hi"}},
				},
				ResponseWriter: httptest.NewRecorder(),
			}

			require.NoError(t, step.Execute(context.Background(), reqCtx))
			require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
			if tc.topology == "" {
				require.Empty(t, got, "no topology header without a prefill topology")
				return
			}
			require.Equal(t, []string{tc.topology}, got, "the forced decode request carries the prefill topology")
		})
	}
}

// TestSpendsRemoteBlocks covers the one-shot P2P handoff detector that gates the
// fallback's recompute: only a non-empty remote_block_ids list qualifies, so a
// shared-storage body (do_remote_prefill but no block ids) and bodies with no or a
// malformed kv_transfer_params all report false and fall back with their params.
func TestSpendsRemoteBlocks(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
		want bool
	}{
		{"no kv_transfer_params", map[string]any{"model": "m"}, false},
		{"kv_transfer_params not a map", map[string]any{reqcommon.FieldKVTransferParams: "nope"}, false},
		{"shared storage: do_remote_prefill, no ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{reqcommon.FieldDoRemotePrefill: true}}, false},
		{"empty remote_block_ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{reqcommon.FieldRemoteBlockIDs: []any{}}}, false},
		{"remote_block_ids wrong type", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{reqcommon.FieldRemoteBlockIDs: "1,2"}}, false},
		{"populated remote_block_ids", map[string]any{reqcommon.FieldKVTransferParams: map[string]any{reqcommon.FieldRemoteBlockIDs: []any{1, 2}}}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, spendsRemoteBlocks(tc.body))
		})
	}
}

// TestScanForcedResponse_CRLFFraming verifies the scan handles a CRLF-framed
// stream: ssePayload trims the trailing carriage return, so a \r\n-delimited chat
// stream reassembles the same as an \n-delimited one.
func TestScanForcedResponse_CRLFFraming(t *testing.T) {
	stream := "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\r\n\r\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n" +
		"data: [DONE]\r\n\r\n"
	r := newSSEReassembler(sseShapeChat, 1)
	require.NoError(t, scanForcedResponse(strings.NewReader(stream), r, 1<<20, 0))

	choices := r.result()["choices"].([]any)
	require.Len(t, choices, 1)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	require.Equal(t, "hi", msg["content"], "CRLF framing must reassemble like LF framing")
	require.Equal(t, "stop", choices[0].(map[string]any)["finish_reason"])
}

// TestScanForcedResponse_TerminalMarker checks the buffered path's completeness
// gate: a stream is complete only if it carried the [DONE] marker or a shape
// terminal (a Responses terminal event, or a finish_reason on another shape). A
// clean EOF with neither is a truncation that must fall back, not a 200. The
// Responses case guards the stream that closes on its terminal event with no
// [DONE] line.
func TestScanForcedResponse_TerminalMarker(t *testing.T) {
	tests := []struct {
		name      string
		shape     sseShape
		stream    string
		truncated bool
	}{
		{
			name:   "chat with done",
			shape:  sseShapeChat,
			stream: `data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n" + "data: [DONE]\n\n",
		},
		{
			name:   "chat finish without done",
			shape:  sseShapeChat,
			stream: `data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}` + "\n\n",
		},
		{
			name:      "chat truncated",
			shape:     sseShapeChat,
			stream:    `data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n",
			truncated: true,
		},
		{
			name:      "empty stream",
			shape:     sseShapeChat,
			stream:    "",
			truncated: true,
		},
		{
			name:   "responses completed without done",
			shape:  sseShapeResponses,
			stream: `data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n" + `data: {"type":"response.completed","response":{"id":"r","status":"completed"}}` + "\n\n",
		},
		{
			// [DONE] is not enough for Responses: without a terminal event the reply
			// has no status or usage, so it is a truncation and must fall back.
			name:      "responses done without terminal event",
			shape:     sseShapeResponses,
			stream:    `data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n" + "data: [DONE]\n\n",
			truncated: true,
		},
		{
			name:      "responses truncated",
			shape:     sseShapeResponses,
			stream:    `data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n",
			truncated: true,
		},
		{
			name:   "messages terminal (message_stop)",
			shape:  sseShapeMessages,
			stream: `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}` + "\n\n" + `data: {"type":"message_stop"}` + "\n\n",
		},
		{
			// A Messages stream cut before message_delta/message_stop has no terminal
			// signal, so it is a truncation and must fall back even with text buffered.
			name:      "messages truncated",
			shape:     sseShapeMessages,
			stream:    `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}` + "\n\n",
			truncated: true,
		},
		{
			// [DONE] is not enough for Messages either: without message_delta/
			// message_stop the reply has no stop_reason or final usage, so a stream
			// that ends on [DONE] alone is still a truncation.
			name:      "messages done without terminal event",
			shape:     sseShapeMessages,
			stream:    `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}` + "\n\n" + "data: [DONE]\n\n",
			truncated: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := newSSEReassembler(tc.shape, 1)
			err := scanForcedResponse(strings.NewReader(tc.stream), r, 1<<20, 0)
			if tc.truncated {
				require.ErrorIs(t, err, errForceStreamTruncated)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

// TestForceStream_Truncated_BufferedPath_FallsBack covers the buffered path's
// completeness gate end to end: a forced stream cut off (clean EOF) before [DONE]
// and before a finish_reason reassembles into a complete-looking reply, so it
// falls back to the pass-through instead of returning a truncated 200. Generate
// takes the buffered path. The outcome is counted fallback_unreadable.
func TestForceStream_Truncated_BufferedPath_FallsBack(t *testing.T) {
	passthroughReply := `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			// One content frame with no finish_reason, then the stream closes with no
			// [DONE] line.
			fmt.Fprint(w, "data: {\"request_id\":\"g\",\"choices\":[{\"index\":0,\"token_ids\":[1]}]}\n\n")
			flusher.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-truncated",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "a truncated stream must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_SeveredConnection_BufferedPath_FallsBack covers the read-error
// branch, distinct from a JSON parse error: the forced upstream sends a valid frame
// then severs the connection (a chunked body with no terminating chunk and no
// [DONE]), so the scanner returns a read-upstream error. Nothing reached the
// client, so the request falls back to the pass-through. Counted fallback_unreadable.
func TestForceStream_SeveredConnection_BufferedPath_FallsBack(t *testing.T) {
	passthroughReply := `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			hj, ok := w.(http.Hijacker)
			require.True(t, ok, "the test server must support hijacking")
			conn, buf, err := hj.Hijack()
			require.NoError(t, err)
			defer conn.Close()
			// One complete chunk, then close with no terminating chunk, so the
			// client's chunked read fails mid-stream rather than ending cleanly.
			_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
			f := "data: {\"request_id\":\"g\",\"choices\":[{\"index\":0,\"token_ids\":[1]}]}\n\n"
			fmt.Fprintf(buf, "%x\r\n%s\r\n", len(f), f)
			_ = buf.Flush()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-severed",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids":       []any{1},
			"sampling_params": map[string]any{"max_tokens": 8},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "a severed connection must fall back, not error")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client gets the pass-through reply, not a 5xx")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body))
	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the fallback")
}

// TestForceStream_Responses_OversizedFrame_FallsBack covers the backstop: a
// Responses completed frame whose bytes exceed even the reservation plus the request
// body size. Here the pod emits a 16KB instructions field the request never sent, so
// it is accounted for by neither the reservation (max_output_tokens=16) nor the tiny
// request body. Rather than failing a reply the pod returned intact, the request
// falls back to the pass-through: the client gets 200, counted fallback_oversize.
func TestForceStream_Responses_OversizedFrame_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"resp-1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			// The completed frame carries a 16KB field the request never sent, so it
			// exceeds the reservation (~2.5KB) plus the tiny request body and trips the
			// scanner's frame cap.
			echo := strings.Repeat("x", 16<<10)
			completed := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp-1","model":"llama-3","status":"completed","instructions":%q,"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}}`, echo)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher := w.(http.Flusher)
			for _, f := range []string{
				`{"type":"response.created","response":{"id":"resp-1","model":"llama-3","status":"in_progress"}}`,
				`{"type":"response.output_text.delta","delta":"hi"}`,
				completed,
			} {
				fmt.Fprintf(w, "data: %s\n\n", f)
				flusher.Flush()
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		// Fallback pass-through: the pod returns the reply non-streaming.
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(passthroughReply))
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-resp-oversize",
		OriginalPath:     reqcommon.PathResponses,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":             "llama-3",
			"stream":            false,
			"max_output_tokens": 16,
			"input":             "hi",
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "an oversized completed frame must fall back, not error")

	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode, "the client must get the pod's reply, not a 500")
	body, _ := io.ReadAll(result.Body)
	require.JSONEq(t, passthroughReply, string(body), "the client receives the pass-through reply")

	require.Equal(t, []any{true, false}, streams, "a forced attempt (stream:true) then a pass-through (stream:false)")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackOversize), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "the reservation is released after the fallback")
}

// TestForceStream_Responses_EchoedToolsRelayed covers the oversize-relay: a request
// that declares a large tools menu (far bigger than its max_output_tokens
// reservation) whose completed frame echoes that menu back. The frame fits within
// reservation + request body size, so it is read and relayed verbatim as the reply
// (forced) rather than falling back and regenerating.
func TestForceStream_Responses_EchoedToolsRelayed(t *testing.T) {
	// A ~16KB tools menu the client declares in the request.
	bigTools := strings.Repeat("x", 16<<10)
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		// The completed frame echoes the request's tools back, so it is large even
		// though generation was short. It must be read and relayed, not fall back.
		completed := fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp-1","model":"llama-3","status":"completed","instructions":%q,"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}}`, bigTools)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, f := range []string{
			`{"type":"response.created","response":{"id":"resp-1","model":"llama-3","status":"in_progress"}}`,
			`{"type":"response.output_text.delta","delta":"hi"}`,
			completed,
		} {
			fmt.Fprintf(w, "data: %s\n\n", f)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-resp-echo",
		OriginalPath:     reqcommon.PathResponses,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model":             "llama-3",
			"stream":            false,
			"max_output_tokens": 16,
			"input":             "hi",
			// The request carries the large tools menu the frame echoes back, so the
			// frame fits within reservation + request body size.
			"instructions": bigTools,
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx), "the echoed frame must be relayed, not fall back")
	result := recorder.Result()
	require.Equal(t, http.StatusOK, result.StatusCode)
	require.Equal(t, "application/json", result.Header.Get("Content-Type"))
	require.Equal(t, []any{true}, streams, "only the forced attempt runs; no fallback re-POST")

	var got map[string]any
	body, _ := io.ReadAll(result.Body)
	require.NoError(t, json.Unmarshal(body, &got))
	require.Equal(t, "completed", got["status"], "the pod's completed object is relayed verbatim")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackOversize), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the reply")
}

func TestForceStream_Concurrency(t *testing.T) {
	frames := []string{
		`{"request_id":"c","choices":[{"index":0,"token_ids":[1,2]}],"usage":null}`,
		`{"request_id":"c","choices":[{"index":0,"finish_reason":"length","token_ids":[3]}],"usage":{"total_tokens":3}}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	// Generate takes the buffered path, where the budget gates each request. One
	// request's reservation is ~2464 bytes (max_tokens=10); a budget of 5000 admits
	// only a couple at a time, so concurrent requests produce a mix of forced and
	// budget-fallback outcomes.
	step, reg := newForceStreamStep(t, server.URL, "5000")

	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-c",
				OriginalPath:     reqcommon.PathVLLMGenerate,
				Model:            "llama-3",
				Stream:           false,
				KVTransferParams: map[string]any{},
				Body: map[string]any{
					"model": "llama-3", "stream": false,
					"token_ids":       []any{1},
					"sampling_params": map[string]any{"max_tokens": 10},
				},
				ResponseWriter: httptest.NewRecorder(),
			}
			require.NoError(t, step.Execute(context.Background(), reqCtx))
		}()
	}
	wg.Wait()

	forced := forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced)
	fallback := forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackBudget)
	require.InDelta(t, float64(n), forced+fallback, 1e-9, "every request must be accounted as forced or fallback")
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "no reservation may leak after all requests complete")
}

// TestForceStream_MidStreamError_ForwardsEnvelope covers an error the pod reports
// after its 200 headers, as an SSE error frame. vLLM's chat, text, and generate
// streams send {"error":{...}} and then [DONE]; its Responses and Messages streams
// send a typed error event and end. The pod's error envelope reaches the client
// with the status the frame names and no fallback re-send, since the pod already
// answered (and a retry would re-send to a pod reporting itself overloaded).
// Counted error_upstream.
func TestForceStream_MidStreamError_ForwardsEnvelope(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		body       map[string]any
		frames     []string
		wantStatus int
		wantBody   string
	}{
		{
			name: "chat error then done",
			path: reqcommon.PathChatCompletions,
			body: map[string]any{"model": "m", "stream": false, "max_tokens": 16, "messages": []any{map[string]any{"role": "user", "content": "hi"}}},
			frames: []string{
				`data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}`,
				`data: {"error":{"message":"Internal server error","type":"InternalServerError","code":500}}`,
				`data: [DONE]`,
			},
			wantStatus: http.StatusInternalServerError,
			wantBody:   `{"error":{"message":"Internal server error","type":"InternalServerError","code":500}}`,
		},
		{
			name: "text error then done",
			path: reqcommon.PathCompletions,
			body: map[string]any{"model": "m", "stream": false, "max_tokens": 16, "prompt": "hi"},
			frames: []string{
				`data: {"id":"c","choices":[{"index":0,"text":"partial"}]}`,
				`data: {"error":{"message":"Service unavailable","type":"ServiceUnavailableError","code":503}}`,
				`data: [DONE]`,
			},
			wantStatus: http.StatusServiceUnavailable,
			wantBody:   `{"error":{"message":"Service unavailable","type":"ServiceUnavailableError","code":503}}`,
		},
		{
			name: "generate error then done",
			path: reqcommon.PathVLLMGenerate,
			body: map[string]any{"model": "m", "stream": false, "token_ids": []any{1}, "sampling_params": map[string]any{"max_tokens": 8}},
			frames: []string{
				`data: {"request_id":"g","choices":[{"index":0,"token_ids":[1]}]}`,
				`data: {"error":{"message":"Internal server error","type":"InternalServerError","code":500}}`,
				`data: [DONE]`,
			},
			wantStatus: http.StatusInternalServerError,
			wantBody:   `{"error":{"message":"Internal server error","type":"InternalServerError","code":500}}`,
		},
		{
			name: "responses error event",
			path: reqcommon.PathResponses,
			body: map[string]any{"model": "m", "stream": false, "max_output_tokens": 16, "input": "hi"},
			frames: []string{
				`data: {"type":"response.output_text.delta","delta":"partial"}`,
				`data: {"type":"error","code":"server_error","message":"boom","param":null}`,
			},
			wantStatus: http.StatusInternalServerError,
			wantBody:   `{"type":"error","code":"server_error","message":"boom","param":null}`,
		},
		{
			name: "messages overloaded error",
			path: reqcommon.PathMessages,
			body: map[string]any{"model": "m", "stream": false, "max_tokens": 16, "messages": []any{map[string]any{"role": "user", "content": "hi"}}},
			frames: []string{
				"event: message_start\n" + `data: {"type":"message_start","message":{"id":"msg_1","role":"assistant"}}`,
				"event: content_block_delta\n" + `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}`,
				"event: error\n" + `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
			},
			wantStatus: http.StatusServiceUnavailable,
			wantBody:   `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var streams []any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				var parsed map[string]any
				_ = json.Unmarshal(raw, &parsed)
				streams = append(streams, parsed["stream"])
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				for _, f := range tc.frames {
					fmt.Fprintf(w, "%s\n\n", f)
				}
			}))
			defer server.Close()

			step, reg := newForceStreamStep(t, server.URL, "1GiB")
			recorder := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-midstream-error",
				OriginalPath:     tc.path,
				Model:            "m",
				Stream:           false,
				KVTransferParams: map[string]any{},
				Body:             tc.body,
				ResponseWriter:   recorder,
			}

			err := step.Execute(context.Background(), reqCtx)
			var streamed *pipeline.UpstreamStreamedError
			require.True(t, errors.As(err, &streamed), "a forwarded error must report UpstreamStreamedError, got %v", err)
			require.Equal(t, tc.wantStatus, streamed.StatusCode)

			result := recorder.Result()
			require.Equal(t, tc.wantStatus, result.StatusCode)
			require.Equal(t, "application/json", result.Header.Get("Content-Type"))
			respBody, _ := io.ReadAll(result.Body)
			require.JSONEq(t, tc.wantBody, string(respBody), "the pod's error envelope reaches the client")
			require.Equal(t, []any{true}, streams, "the pod answered; there is no fallback re-send")
			require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
			require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
			require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnreadable), 1e-9)
			require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation released after the error")
		})
	}
}

// TestForceStream_MidStreamErrorSpentHandoff_ForwardsEnvelope covers a mid-stream
// error on a NIXL path whose block handoff the forced attempt already spent. The
// error is forwarded rather than a fallback, so there is no recompute and the
// client gets the pod's error.
func TestForceStream_MidStreamErrorSpentHandoff_ForwardsEnvelope(t *testing.T) {
	const envelope = `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "event: error\ndata: "+envelope+"\n\n")
	}))
	defer server.Close()

	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	step, err := NewDecodeStep(gwClient, map[string]any{
		ParamForceStream:               true,
		ParamForceStreamBufferSize:     "1GiB",
		ParamForceStreamMaxRequestSize: "1GiB",
		ParamKVConnector:               kv.NIXL,
	})
	require.NoError(t, err)

	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:    "req-midstream-error-spent",
		OriginalPath: reqcommon.PathMessages,
		Model:        "m",
		Stream:       false,
		KVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{1, 2},
			reqcommon.FieldRemoteEngineID: "prefill-pod-a",
		},
		Body: map[string]any{
			"model": "m", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	err = step.(*DecodeStep).Execute(context.Background(), reqCtx)
	var streamed *pipeline.UpstreamStreamedError
	require.True(t, errors.As(err, &streamed), "the pod's error is forwarded, got %v", err)
	require.Equal(t, http.StatusServiceUnavailable, recorder.Result().StatusCode)
	respBody, _ := io.ReadAll(recorder.Result().Body)
	require.JSONEq(t, envelope, string(respBody))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
}

// TestUpstreamErrorStatus maps a mid-stream error frame to the HTTP status the
// client gets: the OpenAI envelope's numeric code, the Responses error event's
// string code, or the Anthropic error type. A frame that names no usable status
// is a server error.
func TestUpstreamErrorStatus(t *testing.T) {
	tests := []struct {
		name  string
		frame string
		want  int
	}{
		{"openai numeric code", `{"error":{"message":"m","type":"BadRequestError","code":400}}`, http.StatusBadRequest},
		{"openai 5xx code", `{"error":{"message":"m","type":"InternalServerError","code":500}}`, http.StatusInternalServerError},
		{"openai code out of range", `{"error":{"message":"m","code":200}}`, http.StatusInternalServerError},
		{"openai no code", `{"error":{"message":"m"}}`, http.StatusInternalServerError},
		{"responses string code", `{"type":"error","code":"server_error","message":"m"}`, http.StatusInternalServerError},
		{"anthropic overloaded", `{"type":"error","error":{"type":"overloaded_error","message":"m"}}`, http.StatusServiceUnavailable},
		{"anthropic rate limit", `{"type":"error","error":{"type":"rate_limit_error","message":"m"}}`, http.StatusTooManyRequests},
		{"anthropic invalid request", `{"type":"error","error":{"type":"invalid_request_error","message":"m"}}`, http.StatusBadRequest},
		{"anthropic not found", `{"type":"error","error":{"type":"not_found_error","message":"m"}}`, http.StatusNotFound},
		{"anthropic api error", `{"type":"error","error":{"type":"api_error","message":"m"}}`, http.StatusInternalServerError},
		{"anthropic unknown type", `{"type":"error","error":{"type":"some_new_error","message":"m"}}`, http.StatusInternalServerError},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var f map[string]any
			require.NoError(t, json.Unmarshal([]byte(tc.frame), &f))
			require.Equal(t, tc.want, upstreamErrorStatus(f))
		})
	}
}

// An error frame is recognised on every shape, in both forms vLLM streams: the
// OpenAI envelope with a top-level error object (chat, text, generate) and a typed
// error event (Responses, Messages). A data frame that is neither is not an error.
func TestIsErrorFrame(t *testing.T) {
	tests := []struct {
		name  string
		frame string
		want  bool
	}{
		{"openai envelope", `{"error":{"message":"m","type":"InternalServerError","code":500}}`, true},
		{"responses error event", `{"type":"error","code":"server_error","message":"m"}`, true},
		{"anthropic error event", `{"type":"error","error":{"type":"overloaded_error","message":"m"}}`, true},
		{"chat chunk", `{"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}`, false},
		{"responses failed event", `{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error"}}}`, false},
		{"null error field", `{"id":"c","choices":[],"error":null}`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, isErrorFrame(frame(t, tc.frame)))
		})
	}
}

// TestForceStream_PrefillCachedTokens covers the prefill cache-hit count on a
// forced reply. A decode pod counts the KV blocks transferred from prefill as
// prompt-cache hits, so the coordinator reports prefill's count instead, on the
// forced reply as on the pass-through.
func TestForceStream_PrefillCachedTokens(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c","model":"m","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101,"prompt_tokens_details":{"cached_tokens":96}}}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	prefillCached := 32
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:           "req-cached",
		OriginalPath:        reqcommon.PathChatCompletions,
		Model:               "m",
		Stream:              false,
		KVTransferParams:    map[string]any{},
		PrefillCachedTokens: &prefillCached,
		Body: map[string]any{
			"model": "m", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))
	result := recorder.Result()
	respBody, _ := io.ReadAll(result.Body)
	// The rewrite can change the body size, so the stale Content-Length is dropped.
	require.Empty(t, result.Header.Get("Content-Length"))
	var got map[string]any
	require.NoError(t, json.Unmarshal(respBody, &got))
	details := got["usage"].(map[string]any)["prompt_tokens_details"].(map[string]any)
	require.EqualValues(t, prefillCached, details["cached_tokens"], "the forced reply reports the prefill cache-hit count")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

// TestForceStream_PromptTokenParity covers the prompt token parity check on a
// forced reply. The forced reply is written through the decode response writer,
// so a decode prompt count that differs from the rendered token count is counted
// the same as on the pass-through.
func TestForceStream_PromptTokenParity(t *testing.T) {
	frames := []string{
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c","model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
	}
	server := sseServer(t, frames, nil)
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-forced-parity",
		OriginalPath:     reqcommon.PathChatCompletions,
		Model:            "m",
		Stream:           false,
		TokenIDs:         []int{1, 2, 3},
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "m", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 1.0, counterValue(t, reg, parityMetric, map[string]string{"api_type": "chat_completions"}), 1e-9,
		"the forced reply's 5 prompt tokens differ from the 3 rendered")
	var got map[string]any
	respBody, _ := io.ReadAll(recorder.Result().Body)
	require.NoError(t, json.Unmarshal(respBody, &got))
	require.EqualValues(t, 5, got["usage"].(map[string]any)["prompt_tokens"], "the check does not change the reply")
}
