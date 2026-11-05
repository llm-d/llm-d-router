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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	parityMetric    = "llm_d_coordinator_prompt_token_mismatch_total"
	stepErrorMetric = "llm_d_coordinator_step_errors_total"
)

// counterValue sums a counter family's samples that carry every label in want.
func counterValue(t *testing.T, reg *prometheus.Registry, name string, want map[string]string) float64 {
	t.Helper()
	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, m := range family.GetMetric() {
			if labelsContain(m.GetLabel(), want) {
				total += m.GetCounter().GetValue()
			}
		}
	}
	return total
}

func labelsContain(actual []*dto.LabelPair, want map[string]string) bool {
	for k, v := range want {
		found := false
		for _, pair := range actual {
			if pair.GetName() == k && pair.GetValue() == v {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// parityServer answers a non-streaming chat completion whose usage reports
// promptTokens.
func parityServer(t *testing.T, promptTokens int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": promptTokens, "completion_tokens": 1},
		})
	}))
}

func parityRequest(server *httptest.Server, params map[string]any, tokenIDs []int, stream bool) (*DecodeStep, *pipeline.RequestContext, *httptest.ResponseRecorder, error) {
	step, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), params)
	if err != nil {
		return nil, nil, nil, err
	}
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-parity",
		OriginalPath:     testChatCompletionsPath,
		Model:            "llama-3",
		Stream:           stream,
		TokenIDs:         tokenIDs,
		KVTransferParams: map[string]any{},
		Body:             map[string]any{"model": "llama-3", "stream": stream, "messages": []any{map[string]any{"role": "user", "content": "hi"}}},
		ResponseWriter:   recorder,
	}
	return step.(*DecodeStep), reqCtx, recorder, nil
}

func TestDecodeStep_PromptTokenParity_MismatchCountsAndKeepsResponse(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := parityServer(t, 7)
	defer server.Close()

	step, reqCtx, recorder, err := parityRequest(server, map[string]any{}, []int{1, 2, 3, 4, 5}, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := counterValue(t, reg, parityMetric, map[string]string{"model_name": "llama-3", "api_type": "chat_completions", "ids_forwarded": "false"}); got != 1 {
		t.Fatalf("mismatch counter = %v, want 1", got)
	}
	if got := counterValue(t, reg, stepErrorMetric, map[string]string{"step": DecodeStepName}); got != 0 {
		t.Fatalf("warn mode must not record a step error, got %v", got)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"content":"ok"`) || !strings.Contains(body, `"prompt_tokens":7`) {
		t.Fatalf("response must reach the client unchanged, got %s", body)
	}
}

func TestDecodeStep_PromptTokenParity_MatchIsSilent(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := parityServer(t, 5)
	defer server.Close()

	step, reqCtx, _, err := parityRequest(server, map[string]any{}, []int{1, 2, 3, 4, 5}, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 0 {
		t.Fatalf("mismatch counter = %v, want 0", got)
	}
}

func TestDecodeStep_PromptTokenParity_ErrorModeRecordsStepError(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := parityServer(t, 7)
	defer server.Close()

	step, reqCtx, recorder, err := parityRequest(server, map[string]any{ParamPromptTokenParity: "error"}, []int{1, 2, 3, 4, 5}, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("error mode must not fail the request: %v", err)
	}
	if got := counterValue(t, reg, stepErrorMetric, map[string]string{"step": DecodeStepName, "error_code": coordmetrics.ErrorCodePromptTokenMismatch}); got != 1 {
		t.Fatalf("step error counter = %v, want 1", got)
	}
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

func TestDecodeStep_PromptTokenParity_OffAndNoTokensSkip(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := parityServer(t, 7)
	defer server.Close()

	for name, tc := range map[string]struct {
		params   map[string]any
		tokenIDs []int
	}{
		"off":          {params: map[string]any{ParamPromptTokenParity: "off"}, tokenIDs: []int{1, 2, 3}},
		"not rendered": {params: map[string]any{}, tokenIDs: nil},
	} {
		t.Run(name, func(t *testing.T) {
			step, reqCtx, _, err := parityRequest(server, tc.params, tc.tokenIDs, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := counterValue(t, reg, parityMetric, nil); got != 0 {
				t.Fatalf("mismatch counter = %v, want 0", got)
			}
		})
	}
}

func TestDecodeStep_PromptTokenParity_RejectsUnknownMode(t *testing.T) {
	server := parityServer(t, 7)
	defer server.Close()
	if _, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{ParamPromptTokenParity: "loud"}); err == nil {
		t.Fatal("expected an error for an unknown prompt_token_parity value")
	}
}

// A streamed chat leg is asked for include_usage so the trailing usage frame
// exists to compare against; the client did not ask for it, so it is dropped.
func TestDecodeStep_PromptTokenParity_StreamingInjectsAndDropsUsageFrame(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &received)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, event := range []string{
			`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1}}`,
			`data: [DONE]`,
		} {
			fmt.Fprintf(w, "%s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	step, reqCtx, recorder, err := parityRequest(server, map[string]any{}, []int{1, 2, 3}, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	options, _ := received["stream_options"].(map[string]any)
	if options["include_usage"] != true {
		t.Fatalf("upstream stream_options = %v, want include_usage=true", received["stream_options"])
	}
	body := recorder.Body.String()
	if strings.Contains(body, `"usage"`) {
		t.Fatalf("injected usage frame reached the client: %s", body)
	}
	if !strings.Contains(body, `"content":"Hello"`) || !strings.Contains(body, "[DONE]") {
		t.Fatalf("content or [DONE] lost: %s", body)
	}
	if got := counterValue(t, reg, parityMetric, map[string]string{"api_type": "chat_completions"}); got != 1 {
		t.Fatalf("mismatch counter = %v, want 1 (rendered 3, decode reported 9)", got)
	}
}

// When the client asked for include_usage itself, the frame is theirs and
// stays. The server leaves stream_options as raw JSON (it is not one of the
// fields it decodes), so both representations are covered.
func TestDecodeStep_PromptTokenParity_StreamingKeepsClientRequestedUsage(t *testing.T) {
	for name, options := range map[string]any{
		"decoded map": map[string]any{"include_usage": true},
		"raw json":    json.RawMessage(`{"include_usage": true}`),
	} {
		t.Run(name, func(t *testing.T) {
			metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}\n\ndata: [DONE]\n\n")
			}))
			defer server.Close()

			step, reqCtx, recorder, err := parityRequest(server, map[string]any{}, []int{1, 2, 3}, true)
			if err != nil {
				t.Fatal(err)
			}
			reqCtx.Body["stream_options"] = options
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !strings.Contains(recorder.Body.String(), `"usage"`) {
				t.Fatalf("client-requested usage frame was dropped: %s", recorder.Body.String())
			}
		})
	}
}

// A Messages response reports the cached part of the prompt beside
// input_tokens. With prefill's KV reused, most of the prompt is
// cache_read_input_tokens; the sum must compare as a match.
func TestDecodeStep_PromptTokenParity_MessagesCacheSplitMatches(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"type": "message", "role": "assistant",
			"content": []map[string]any{{"type": "text", "text": "ok"}},
			"usage":   map[string]any{"input_tokens": 1, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 1, "output_tokens": 1},
		})
	}))
	defer server.Close()

	step, reqCtx, _, err := parityRequest(server, map[string]any{}, []int{1, 2, 3, 4, 5}, false)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx.OriginalPath = "/v1/messages"
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 0 {
		t.Fatalf("mismatch counter = %v, want 0: 1 + 3 + 1 cached equals the 5 rendered tokens", got)
	}
}

// A client that set continuous_usage_stats without include_usage gets no usage
// today; adding include_usage would put a usage block in every content chunk,
// so the step leaves the request alone and skips the check.
func TestDecodeStep_PromptTokenParity_SkipsInjectionForContinuousUsageStats(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &received)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	step, reqCtx, recorder, err := parityRequest(server, map[string]any{}, []int{1, 2, 3}, true)
	if err != nil {
		t.Fatal(err)
	}
	reqCtx.Body["stream_options"] = json.RawMessage(`{"continuous_usage_stats": true}`)
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	options, _ := received["stream_options"].(map[string]any)
	if _, injected := options["include_usage"]; injected {
		t.Fatalf("include_usage must not be injected beside continuous_usage_stats, got %v", options)
	}
	if !strings.Contains(recorder.Body.String(), `"content":"Hello"`) {
		t.Fatalf("stream changed: %s", recorder.Body.String())
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 0 {
		t.Fatalf("mismatch counter = %v, want 0 (no usage to compare)", got)
	}
}
