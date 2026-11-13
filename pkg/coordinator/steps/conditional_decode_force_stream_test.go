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
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// condForcedFrames is a complete streamed chat reply.
var condForcedFrames = []string{
	`{"id":"c","model":"llama-3","choices":[{"index":0,"delta":{"role":"assistant","content":"cached"}}]}`,
	`{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
}

// condCall records one request the conditional-decode upstream received.
type condCall struct {
	body   map[string]any
	prefer string
	phase  string
}

// condUpstream answers each request with the next handler in order and records
// what it received. A call past the last handler fails the test.
func condUpstream(t *testing.T, handlers ...http.HandlerFunc) (*httptest.Server, *[]condCall) {
	t.Helper()
	var (
		mu    sync.Mutex
		calls []condCall
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		i := len(calls)
		calls = append(calls, condCall{body: body, prefer: r.Header.Get(routing.PreferHeader), phase: r.Header.Get(reqcommon.EPPProfileHeaderKey)})
		mu.Unlock()
		if i >= len(handlers) {
			t.Errorf("unexpected upstream call %d", i+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		handlers[i](w, r)
	}))
	t.Cleanup(server.Close)
	return server, &calls
}

// sseReply streams frames followed by [DONE].
func sseReply(frames ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, f := range frames {
			fmt.Fprintf(w, "data: %s\n\n", f)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}
}

// jsonReply answers a non-streaming chat completion.
func jsonReply(content string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": content}}},
		})
	}
}

func deferProbe(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusPreconditionFailed)
}

// newForcedConditionalStep builds a conditional-decode step with force-streaming
// enabled from a pipeline-level budget, and a fresh metrics registry.
func newForcedConditionalStep(t *testing.T, serverURL string, params map[string]any) (*ConditionalDecodeStep, *prometheus.Registry) {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	step, err := NewConditionalDecodeStep(gateway.New(config.GatewayConfig{Address: serverURL}), params)
	require.NoError(t, err)
	require.NoError(t, EnableForceStream([]pipeline.Step{step}, "1GiB", "1GiB"))
	return step.(*ConditionalDecodeStep), reg
}

func condForcedRequest(recorder http.ResponseWriter) *pipeline.RequestContext {
	return &pipeline.RequestContext{
		RequestID:    "req-cond-forced",
		OriginalPath: reqcommon.PathChatCompletions,
		Model:        "llama-3",
		Stream:       false,
		Body: map[string]any{
			"model": "llama-3", "stream": false, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		},
		ResponseWriter: recorder,
	}
}

func condForceStreamCount(t *testing.T, reg *prometheus.Registry, result string) float64 {
	t.Helper()
	return counterValue(t, reg, forceStreamTotalMetric, map[string]string{"step": ConditionalDecodeStepName, "result": result})
}

// condForceStreamTotal sums force_stream_total over every result recorded by
// conditional-decode.
func condForceStreamTotal(t *testing.T, reg *prometheus.Registry) float64 {
	t.Helper()
	return counterValue(t, reg, forceStreamTotalMetric, map[string]string{"step": ConditionalDecodeStepName})
}

// A non-streaming request the worker serves is force-streamed: the probe goes out
// streaming with the cache-probe headers and the client gets one JSON reply.
func TestConditionalDecodeForceStream_ServedIsForced(t *testing.T) {
	server, calls := condUpstream(t, sseReply(condForcedFrames...))
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), condForcedRequest(recorder))
	require.ErrorIs(t, err, pipeline.ErrPipelineDone)

	require.Len(t, *calls, 1)
	got := (*calls)[0]
	require.Equal(t, true, got.body["stream"])
	require.Equal(t, map[string]any{"include_usage": true}, got.body["stream_options"])
	require.Equal(t, routing.PreferIfAvailable, got.prefer)
	require.Equal(t, gateway.PhaseDecode, got.phase)

	require.Equal(t, reqcommon.ContentTypeJSON, recorder.Header().Get("Content-Type"))
	var reply map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &reply))
	msg := reply["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	require.Equal(t, "cached", msg["content"])

	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultServed), 1e-9)
	require.InDelta(t, 1.0, counterValue(t, reg, "llm_d_coordinator_upstream_request_total", map[string]string{"upstream": coordmetrics.UpstreamConditionalDecode}), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

// A 412 to the forced probe writes nothing and continues the pipeline. The
// request's force-stream outcome is left to decode, so none is recorded here.
func TestConditionalDecodeForceStream_DeferredProbe(t *testing.T) {
	server, calls := condUpstream(t, deferProbe)
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()
	reqCtx := condForcedRequest(recorder)

	require.NoError(t, step.Execute(context.Background(), reqCtx))

	require.Len(t, *calls, 1, "a miss is not re-sent")
	require.Equal(t, true, (*calls)[0].body["stream"])
	require.Zero(t, recorder.Body.Len())
	require.Equal(t, false, reqCtx.Body["stream"], "the forced body must not leak into reqCtx.Body")
	require.NotContains(t, reqCtx.Body, "stream_options")
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultDeferred), 1e-9)
	require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

// A stream carrying a field the reassembler drops falls back to the probe
// pass-through, which re-sends the client's non-streaming body.
func TestConditionalDecodeForceStream_LossyFrameFallsBackToProbe(t *testing.T) {
	lossy := `{"id":"c","choices":[{"index":0,"delta":{"reasoning":"think"}}]}`
	server, calls := condUpstream(t, sseReply(lossy), jsonReply("passthrough"))
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), condForcedRequest(recorder))
	require.ErrorIs(t, err, pipeline.ErrPipelineDone)

	require.Len(t, *calls, 2)
	require.Equal(t, true, (*calls)[0].body["stream"])
	require.Equal(t, false, (*calls)[1].body["stream"], "the fallback re-sends the client's body")
	require.Equal(t, routing.PreferIfAvailable, (*calls)[1].prefer, "the fallback is still a conditional-decode probe")
	require.Contains(t, recorder.Body.String(), "passthrough")
	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackLossyFrame), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultServed), 1e-9)
	require.InDelta(t, 0.0, forceStreamGauge(t, reg), 1e-9)
}

// A fallback whose re-sent probe is deferred leaves the request to decode, so
// conditional-decode records no force-stream outcome for it.
func TestConditionalDecodeForceStream_FallbackThenDeferredRecordsNoOutcome(t *testing.T) {
	lossy := `{"id":"c","choices":[{"index":0,"delta":{"reasoning":"think"}}]}`
	server, calls := condUpstream(t, sseReply(lossy), deferProbe)
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()

	require.NoError(t, step.Execute(context.Background(), condForcedRequest(recorder)))

	require.Len(t, *calls, 2)
	require.Zero(t, recorder.Body.Len())
	require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultDeferred), 1e-9)
	require.InDelta(t, 0.0, probeCount(t, reg, coordmetrics.ProbeResultServed), 1e-9)
}

// A client that leaves before the worker answers a fallback's re-sent probe
// records no force-stream outcome.
func TestConditionalDecodeForceStream_FallbackThenCancelRecordsNoOutcome(t *testing.T) {
	lossy := `{"id":"c","choices":[{"index":0,"delta":{"reasoning":"think"}}]}`
	ctx, cancel := context.WithCancel(context.Background())
	release := make(chan struct{})
	server, calls := condUpstream(t, sseReply(lossy), func(http.ResponseWriter, *http.Request) {
		cancel()
		<-release
	})
	defer close(release)
	step, reg := newForcedConditionalStep(t, server.URL, nil)

	err := step.Execute(ctx, condForcedRequest(httptest.NewRecorder()))
	require.ErrorIs(t, err, context.Canceled)
	require.Len(t, *calls, 2)
	require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
}

// The forced probe sends the step's prepared body: a completions request carries
// the rendered token IDs as its prompt, and a chat request forwarding them
// carries them in kv_transfer_params.
func TestConditionalDecodeForceStream_SendsPreparedBody(t *testing.T) {
	t.Run("completions", func(t *testing.T) {
		server, calls := condUpstream(t, sseReply(
			`{"id":"t","object":"text_completion","model":"llama-3","choices":[{"index":0,"text":"cached","finish_reason":"stop"}]}`))
		step, reg := newForcedConditionalStep(t, server.URL, nil)
		reqCtx := condForcedRequest(httptest.NewRecorder())
		reqCtx.OriginalPath = reqcommon.PathCompletions
		reqCtx.Body = map[string]any{"model": "llama-3", "stream": false, "max_tokens": 16, "prompt": "hi"}
		reqCtx.TokenIDs = []int{1, 2, 3}

		require.ErrorIs(t, step.Execute(context.Background(), reqCtx), pipeline.ErrPipelineDone)
		require.Len(t, *calls, 1)
		require.Equal(t, true, (*calls)[0].body["stream"])
		require.Equal(t, []any{1.0, 2.0, 3.0}, (*calls)[0].body[reqcommon.FieldPrompt])
		require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	})
	t.Run("chat forwarding token IDs", func(t *testing.T) {
		server, calls := condUpstream(t, sseReply(condForcedFrames...))
		step, reg := newForcedConditionalStep(t, server.URL, map[string]any{ParamForwardPromptTokenIDs: true})
		reqCtx := condForcedRequest(httptest.NewRecorder())
		reqCtx.TokenIDs = []int{1, 2, 3}

		require.ErrorIs(t, step.Execute(context.Background(), reqCtx), pipeline.ErrPipelineDone)
		require.Len(t, *calls, 1)
		require.Equal(t, true, (*calls)[0].body["stream"])
		require.Equal(t, map[string]any{reqcommon.FieldPromptTokenIDs: []any{1.0, 2.0, 3.0}}, (*calls)[0].body[reqcommon.FieldKVTransferParams])
		require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	})
}

// A request declaring a field the reassembler drops takes the probe
// pass-through without a forced attempt. Its decline is recorded only when the
// probe serves it.
func TestConditionalDecodeForceStream_LossyRequestDeclines(t *testing.T) {
	withTools := func(reqCtx *pipeline.RequestContext) *pipeline.RequestContext {
		reqCtx.Body["tools"] = json.RawMessage(`[{"type":"function","function":{"name":"f"}}]`)
		return reqCtx
	}

	t.Run("served", func(t *testing.T) {
		server, calls := condUpstream(t, jsonReply("tools"))
		step, reg := newForcedConditionalStep(t, server.URL, nil)
		err := step.Execute(context.Background(), withTools(condForcedRequest(httptest.NewRecorder())))
		require.ErrorIs(t, err, pipeline.ErrPipelineDone)
		require.Len(t, *calls, 1)
		require.Equal(t, false, (*calls)[0].body["stream"])
		require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultFallbackUnsupported), 1e-9)
	})
	t.Run("deferred", func(t *testing.T) {
		server, _ := condUpstream(t, deferProbe)
		step, reg := newForcedConditionalStep(t, server.URL, nil)
		require.NoError(t, step.Execute(context.Background(), withTools(condForcedRequest(httptest.NewRecorder()))))
		require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
	})
}

// A worker error on the forced probe is forwarded and attributed to
// conditional-decode, both on the error and on the probe counter.
func TestConditionalDecodeForceStream_UpstreamError(t *testing.T) {
	server, _ := condUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":{"message":"boom","code":500}}`)
	})
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), condForcedRequest(recorder))
	var streamed *pipeline.UpstreamStreamedError
	require.True(t, errors.As(err, &streamed), "got %T (%v)", err, err)
	require.Equal(t, ConditionalDecodeStepName, streamed.Step)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	require.Contains(t, recorder.Body.String(), "boom")
	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultError), 1e-9)
}

// An error frame streamed after the 200 is a worker error too.
func TestConditionalDecodeForceStream_MidStreamErrorFrame(t *testing.T) {
	server, _ := condUpstream(t, sseReply(`{"error":{"message":"overloaded","code":503}}`))
	step, reg := newForcedConditionalStep(t, server.URL, nil)
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), condForcedRequest(recorder))
	var streamed *pipeline.UpstreamStreamedError
	require.True(t, errors.As(err, &streamed), "got %T (%v)", err, err)
	require.Equal(t, ConditionalDecodeStepName, streamed.Step)
	require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorUpstream), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultError), 1e-9)
}

func TestConditionalDecodeForceStream_TransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	serverURL := server.URL
	server.Close()
	step, reg := newForcedConditionalStep(t, serverURL, nil)
	recorder := httptest.NewRecorder()

	err := step.Execute(context.Background(), condForcedRequest(recorder))
	require.Error(t, err)
	require.False(t, errors.Is(err, pipeline.ErrPipelineDone))
	require.Equal(t, coordmetrics.ErrorCodeUpstreamTransport, coordmetrics.ClassifyErrorCode(err, pipeline.ClassifyOpts))
	require.Zero(t, recorder.Body.Len(), "nothing is written, so the server answers a clean 502")
	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultErrorTransport), 1e-9)
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultTransportError), 1e-9)
}

func TestConditionalDecodeForceStream_ClientCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	release := make(chan struct{})
	server, _ := condUpstream(t, func(http.ResponseWriter, *http.Request) {
		cancel()
		<-release
	})
	defer close(release)
	step, reg := newForcedConditionalStep(t, server.URL, nil)

	err := step.Execute(ctx, condForcedRequest(httptest.NewRecorder()))
	require.ErrorIs(t, err, context.Canceled)
	require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
	for _, result := range []string{coordmetrics.ProbeResultServed, coordmetrics.ProbeResultDeferred, coordmetrics.ProbeResultError, coordmetrics.ProbeResultTransportError} {
		require.InDelta(t, 0.0, probeCount(t, reg, result), 1e-9, "no probe result for %q", result)
	}
}

// A client that asked for a stream keeps the probe pass-through.
func TestConditionalDecodeForceStream_StreamingClientNotForced(t *testing.T) {
	server, calls := condUpstream(t, sseReply(condForcedFrames...))
	step, reg := newForcedConditionalStep(t, server.URL, map[string]any{ParamPromptTokenParity: "off"})
	recorder := httptest.NewRecorder()
	reqCtx := condForcedRequest(recorder)
	reqCtx.Stream = true
	reqCtx.Body["stream"] = true

	require.ErrorIs(t, step.Execute(context.Background(), reqCtx), pipeline.ErrPipelineDone)
	require.Len(t, *calls, 1)
	require.NotContains(t, (*calls)[0].body, "stream_options")
	require.Contains(t, recorder.Body.String(), "data: ")
	require.InDelta(t, 0.0, condForceStreamTotal(t, reg), 1e-9)
}

// A forced reply is checked for prompt-token parity under the conditional-decode
// step, as a served pass-through reply is.
func TestConditionalDecodeForceStream_PromptTokenParity(t *testing.T) {
	frames := append(append([]string{}, condForcedFrames...),
		`{"id":"c","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}`)
	server, _ := condUpstream(t, sseReply(frames...))
	step, reg := newForcedConditionalStep(t, server.URL, map[string]any{ParamPromptTokenParity: "error"})
	reqCtx := condForcedRequest(httptest.NewRecorder())
	reqCtx.TokenIDs = []int{1, 2, 3}

	require.ErrorIs(t, step.Execute(context.Background(), reqCtx), pipeline.ErrPipelineDone)
	require.InDelta(t, 1.0, condForceStreamCount(t, reg, coordmetrics.ForceStreamResultForced), 1e-9)
	require.InDelta(t, 1.0, counterValue(t, reg, parityMetric, map[string]string{"api_type": "chat_completions"}), 1e-9)
	require.InDelta(t, 1.0, counterValue(t, reg, stepErrorMetric, map[string]string{"step": ConditionalDecodeStepName, "error_code": coordmetrics.ErrorCodePromptTokenMismatch}), 1e-9)
}

// One client request deferred by conditional-decode and forced by decode
// records exactly one force-stream outcome, under decode.
func TestConditionalDecodeForceStream_PipelineCountsOnce(t *testing.T) {
	server, calls := condUpstream(t, deferProbe, sseReply(condForcedFrames...))
	gwClient := gateway.New(config.GatewayConfig{Address: server.URL})
	reg := prometheus.NewRegistry()
	require.NoError(t, coordmetrics.Register(reg))
	coordmetrics.Reset()
	cond, err := NewConditionalDecodeStep(gwClient, nil)
	require.NoError(t, err)
	decode, err := NewDecodeStep(gwClient, map[string]any{})
	require.NoError(t, err)
	stepList := []pipeline.Step{cond, decode}
	require.NoError(t, EnableForceStream(stepList, "1GiB", "1GiB"))

	recorder := httptest.NewRecorder()
	reqCtx := condForcedRequest(recorder)
	reqCtx.KVTransferParams = map[string]any{}
	require.NoError(t, pipeline.New(stepList).Execute(context.Background(), reqCtx))

	require.Len(t, *calls, 2)
	require.Equal(t, routing.PreferIfAvailable, (*calls)[0].prefer)
	require.Empty(t, (*calls)[1].prefer)
	require.Contains(t, recorder.Body.String(), "cached")
	require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric, map[string]string{"step": DecodeStepName, "result": coordmetrics.ForceStreamResultForced}), 1e-9)
	require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric, nil), 1e-9, "one client request, one outcome")
	require.InDelta(t, 1.0, probeCount(t, reg, coordmetrics.ProbeResultDeferred), 1e-9)
}

// Each step reserves a forced request under its own step label, releases it
// under the same label, and records its outcome under that label.
func TestForceStream_MetricsCarryStepLabel(t *testing.T) {
	for _, tc := range []struct {
		step    string
		build   func(t *testing.T, serverURL string) (pipeline.Step, *prometheus.Registry)
		wantErr error
	}{
		{
			step: DecodeStepName,
			build: func(t *testing.T, serverURL string) (pipeline.Step, *prometheus.Registry) {
				return newForceStreamStep(t, serverURL, "1GiB")
			},
		},
		{
			step: ConditionalDecodeStepName,
			build: func(t *testing.T, serverURL string) (pipeline.Step, *prometheus.Registry) {
				return newForcedConditionalStep(t, serverURL, nil)
			},
			wantErr: pipeline.ErrPipelineDone,
		},
	} {
		t.Run(tc.step, func(t *testing.T) {
			var reg *prometheus.Registry
			var during map[string]float64
			var gatherErr error
			server, _ := condUpstream(t, func(w http.ResponseWriter, r *http.Request) {
				during, gatherErr = forceStreamGaugeByStep(reg)
				sseReply(condForcedFrames...)(w, r)
			})
			var step pipeline.Step
			step, reg = tc.build(t, server.URL)
			reqCtx := condForcedRequest(httptest.NewRecorder())
			reqCtx.KVTransferParams = map[string]any{}

			err := step.Execute(context.Background(), reqCtx)
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			require.NoError(t, gatherErr)
			require.Len(t, during, 1, "only %s holds a reservation", tc.step)
			require.Positive(t, during[tc.step])
			after, err := forceStreamGaugeByStep(reg)
			require.NoError(t, err)
			require.Equal(t, map[string]float64{tc.step: 0}, after, "released under the label it reserved under")
			require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric, map[string]string{"step": tc.step, "result": coordmetrics.ForceStreamResultForced}), 1e-9)
			require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric, nil), 1e-9)
		})
	}
}

// EnableForceStream hands one budget to every step that force-streams, and
// rejects a size that fails validation.
func TestEnableForceStream(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})
	cond, err := NewConditionalDecodeStep(gwClient, nil)
	require.NoError(t, err)
	decode, err := NewDecodeStep(gwClient, map[string]any{})
	require.NoError(t, err)
	prefill, err := NewPrefillStep(gwClient, map[string]any{"use_openai_format": true})
	require.NoError(t, err)

	require.NoError(t, EnableForceStream([]pipeline.Step{prefill, cond, decode}, "", ""))
	shared := decode.(*DecodeStep).budget
	require.NotNil(t, shared)
	require.Same(t, shared, cond.(*ConditionalDecodeStep).budget, "both steps reserve from one budget")
	require.Equal(t, int64(1<<30), shared.max)
	require.Equal(t, int64(256<<10), shared.perRequestMax)

	require.Error(t, EnableForceStream([]pipeline.Step{decode}, "1KiB", ""))
}

// The force-stream keys are pipeline settings; a step that sets them fails
// config load rather than silently not force-streaming.
func TestForceStream_StepParamsRejected(t *testing.T) {
	gwClient := gateway.New(config.GatewayConfig{})
	for _, key := range []string{ParamForceStream, ParamForceStreamBufferSize, ParamForceStreamMaxRequestSize} {
		_, err := NewDecodeStep(gwClient, map[string]any{key: true})
		require.ErrorContains(t, err, "pipeline."+key, "decode %s", key)
		_, err = NewConditionalDecodeStep(gwClient, map[string]any{key: true})
		require.ErrorContains(t, err, "pipeline."+key, "conditional-decode %s", key)
	}
}
