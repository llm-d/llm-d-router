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
// larger than the request's reservation trips the scanner's buffer limit and is
// reported as a ceiling abort (counted and clean), not a generic read error.
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
	server := sseServer(t, []string{big.String()}, nil)
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

	err := step.Execute(context.Background(), reqCtx)
	require.ErrorIs(t, err, errForceStreamCeiling)
	require.Equal(t, 0, recorder.Body.Len(), "nothing may be written to the client when the frame overflows")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation must be released after an abort")
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
	require.Empty(t, result.Header.Get("Content-Length"),
		"a chat single-choice reply is written incrementally (chunked), so it carries no Content-Length")
	require.True(t, recorder.Flushed, "the incremental reply must be flushed to the client")

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

func TestForceStream_FallbackBudget(t *testing.T) {
	var upstreamBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &upstreamBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}}})
	}))
	defer server.Close()

	// Size the budget to exactly one request's reservation, then pre-reserve all
	// of it so the request under test cannot fit and must fall back. A generate
	// request takes the buffered path (only chat/text single-choice stream
	// incrementally), so the budget still gates it.
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

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	require.Equal(t, false, upstreamBody["stream"], "fallback must send the client's non-streaming body unchanged")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackBudget), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
}

func TestForceStream_NoTokenLimit_PassThrough(t *testing.T) {
	var upstreamBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &upstreamBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}}})
	}))
	defer server.Close()

	step, reg := newForceStreamStep(t, server.URL, "1GiB")

	// A generate request with no token cap takes the buffered path and is not
	// reservable, so it falls through to the pass-through unchanged. Chat and text
	// single-choice requests are instead streamed incrementally even without a cap
	// (see TestForceStreamIncremental_NoTokenLimit_StillForced).
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-nolimit",
		OriginalPath:     reqcommon.PathVLLMGenerate,
		Model:            "llama-3",
		Stream:           false,
		KVTransferParams: map[string]any{},
		Body: map[string]any{
			"model": "llama-3", "stream": false,
			"token_ids": []any{1, 2, 3},
		},
		ResponseWriter: recorder,
	}

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	// An unbounded buffered request is not force-streamed (no limit to reserve); it
	// takes the pass-through with its body (stream:false) unchanged and is counted
	// as fallback_unbounded.
	require.Equal(t, false, upstreamBody["stream"])
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnbounded), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 0.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackBudget), 1e-9)
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

func TestForceStream_CeilingAbort(t *testing.T) {
	// sampling_params.max_tokens=1 reserves a small per-request ceiling; stream
	// token_ids well past it to trip the abort before anything reaches the client.
	// Generate takes the buffered path, where the ceiling applies.
	frames := make([]string, 0, 100)
	for i := 0; i < 100; i++ {
		frames = append(frames, `{"request_id":"g","choices":[{"index":0,"token_ids":[1,2,3,4,5,6,7,8,9,10]}],"usage":null}`)
	}
	server := sseServer(t, frames, nil)
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

	err := step.Execute(context.Background(), reqCtx)
	require.ErrorIs(t, err, errForceStreamCeiling)
	var streamed *pipeline.UpstreamStreamedError
	require.False(t, errors.As(err, &streamed), "a ceiling abort writes nothing, so it must return a plain error")

	require.Equal(t, 0, recorder.Body.Len(), "nothing may be written to the client when the ceiling trips")
	require.InDelta(t, 1.0, forceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorCeiling), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9, "reservation must be released after an abort")
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
			name:   "responses done without terminal event",
			shape:  sseShapeResponses,
			stream: `data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n" + "data: [DONE]\n\n",
		},
		{
			name:      "responses truncated",
			shape:     sseShapeResponses,
			stream:    `data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n",
			truncated: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := newSSEReassembler(tc.shape, 1)
			err := scanForcedResponse(strings.NewReader(tc.stream), r, 1<<20)
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

// TestForceStream_Responses_OversizedFrame_FallsBack covers the regression where
// a Responses response.completed frame echoes request fields the reservation does
// not price (tools, instructions, schema), so the frame can exceed the reservation
// that is also its scanner cap. Rather than failing a reply the pod returned
// intact, the request falls back to the pass-through: the client gets 200, and the
// outcome is counted as fallback_oversize.
func TestForceStream_Responses_OversizedFrame_FallsBack(t *testing.T) {
	passthroughReply := `{"id":"resp-1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`
	var streams []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		streams = append(streams, parsed["stream"])
		if parsed["stream"] == true {
			// The completed frame's echoed request field dwarfs the ~2.5KB reserved
			// for max_output_tokens=16, so it trips the scanner's frame cap.
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
