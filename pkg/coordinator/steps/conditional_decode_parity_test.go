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
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func conditionalParityRequest(t *testing.T, server *httptest.Server, params map[string]any, stream bool) (pipeline.Step, *pipeline.RequestContext, *httptest.ResponseRecorder) {
	t.Helper()
	step, err := NewConditionalDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), params)
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	reqCtx := &pipeline.RequestContext{
		RequestID:      "req-cond",
		OriginalPath:   testChatCompletionsPath,
		Model:          "llama-3",
		Stream:         stream,
		Body:           map[string]any{"model": "llama-3", "stream": stream, "messages": []any{}},
		TokenIDs:       []int{1, 2, 3, 4, 5},
		ResponseWriter: recorder,
	}
	return step, reqCtx, recorder
}

// A served cache hit is checked like a decode response: the worker's count is
// compared with the rendered one and the hit is still returned to the client.
func TestConditionalDecodeStep_PromptTokenParity_HitMismatchCounts(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "cached"}}},
			"usage":   map[string]any{"prompt_tokens": 7, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	step, reqCtx, recorder := conditionalParityRequest(t, server, nil, false)
	if err := step.Execute(context.Background(), reqCtx); !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("expected ErrPipelineDone, got %v", err)
	}
	if got := counterValue(t, reg, parityMetric, map[string]string{"api_type": "chat_completions", "ids_forwarded": "false"}); got != 1 {
		t.Fatalf("mismatch counter = %v, want 1", got)
	}
	if !strings.Contains(recorder.Body.String(), "cached") {
		t.Fatalf("hit must reach the client, got %s", recorder.Body.String())
	}
}

// A 412 writes nothing, so there is no usage to compare and the pipeline goes
// on; the decode step then sees the client's body without any injection.
func TestConditionalDecodeStep_PromptTokenParity_MissIsSilent(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPreconditionFailed)
	}))
	defer server.Close()

	step, reqCtx, recorder := conditionalParityRequest(t, server, nil, true)
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("a miss must continue the pipeline, got %v", err)
	}
	if recorder.Body.Len() != 0 {
		t.Fatalf("a miss must write nothing, got %q", recorder.Body.String())
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 0 {
		t.Fatalf("mismatch counter = %v, want 0", got)
	}
	if _, injected := reqCtx.Body["stream_options"]; injected {
		t.Fatal("the probe's include_usage injection must not leak into reqCtx.Body")
	}
}

// A streamed hit is asked for the usage frame, which is dropped again before
// it reaches the client.
func TestConditionalDecodeStep_PromptTokenParity_StreamingHitInjectsAndDrops(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &received)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":9}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	step, reqCtx, recorder := conditionalParityRequest(t, server, nil, true)
	if err := step.Execute(context.Background(), reqCtx); !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("expected ErrPipelineDone, got %v", err)
	}
	options, _ := received["stream_options"].(map[string]any)
	if options["include_usage"] != true {
		t.Fatalf("probe must ask for include_usage, got %v", received["stream_options"])
	}
	body := recorder.Body.String()
	if strings.Contains(body, `"usage"`) || !strings.Contains(body, `"content":"Hi"`) {
		t.Fatalf("usage frame must be dropped and content kept, got %s", body)
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 1 {
		t.Fatalf("mismatch counter = %v, want 1 (rendered 5, worker 9)", got)
	}
}

func TestConditionalDecodeStep_PromptTokenParity_OffAndErrorModes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{}, "usage": map[string]any{"prompt_tokens": 7}})
	}))
	defer server.Close()

	t.Run("off", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		step, reqCtx, _ := conditionalParityRequest(t, server, map[string]any{ParamPromptTokenParity: "off"}, false)
		if err := step.Execute(context.Background(), reqCtx); !errors.Is(err, pipeline.ErrPipelineDone) {
			t.Fatalf("expected ErrPipelineDone, got %v", err)
		}
		if got := counterValue(t, reg, parityMetric, nil); got != 0 {
			t.Fatalf("mismatch counter = %v, want 0", got)
		}
	})
	t.Run("error", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		step, reqCtx, _ := conditionalParityRequest(t, server, map[string]any{ParamPromptTokenParity: "error"}, false)
		if err := step.Execute(context.Background(), reqCtx); !errors.Is(err, pipeline.ErrPipelineDone) {
			t.Fatalf("expected ErrPipelineDone, got %v", err)
		}
		if got := counterValue(t, reg, stepErrorMetric, map[string]string{"step": ConditionalDecodeStepName, "error_code": coordmetrics.ErrorCodePromptTokenMismatch}); got != 1 {
			t.Fatalf("step error counter = %v, want 1", got)
		}
	})
	t.Run("unknown mode rejected", func(t *testing.T) {
		if _, err := NewConditionalDecodeStep(gateway.New(config.GatewayConfig{Address: server.URL}), map[string]any{ParamPromptTokenParity: "loud"}); err == nil {
			t.Fatal("expected an error for an unknown prompt_token_parity value")
		}
	})
}

// The probe's include_usage must not reach the decode step through the
// client's own stream_options object, which the step's shallow clone shares.
func TestConditionalDecodeStep_PromptTokenParity_MissKeepsClientStreamOptions(t *testing.T) {
	metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPreconditionFailed)
	}))
	defer server.Close()

	step, reqCtx, _ := conditionalParityRequest(t, server, nil, true)
	client := map[string]any{"include_obfuscation": false}
	reqCtx.Body["stream_options"] = client
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("a miss must continue the pipeline, got %v", err)
	}
	if _, leaked := client["include_usage"]; leaked {
		t.Fatal("the probe's include_usage leaked into the client's stream_options")
	}
	if got, _ := reqCtx.Body["stream_options"].(map[string]any); got["include_usage"] != nil {
		t.Fatalf("reqCtx.Body stream_options = %v, want the client's object unchanged", got)
	}
}

// A worker error on a streamed probe with the check on is held by the usage
// writer until finalize flushes it: the client must still get the status and
// the body, and nothing is counted as a mismatch.
func TestConditionalDecodeStep_PromptTokenParity_StreamedWorkerErrorReachesClient(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":{"message":"boom","type":"server_error"}}`)
	}))
	defer server.Close()

	step, reqCtx, recorder := conditionalParityRequest(t, server, nil, true)
	err := step.Execute(context.Background(), reqCtx)
	var streamed *pipeline.UpstreamStreamedError
	if !errors.As(err, &streamed) || streamed.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected an UpstreamStreamedError with status 500, got %T (%v)", err, err)
	}
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("client status = %d, want 500", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), `"boom"`) {
		t.Fatalf("client must receive the worker's error body, got %q", recorder.Body.String())
	}
	if got := counterValue(t, reg, parityMetric, nil); got != 0 {
		t.Fatalf("mismatch counter = %v, want 0", got)
	}
}
