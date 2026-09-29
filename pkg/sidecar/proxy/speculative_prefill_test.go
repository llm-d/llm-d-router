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
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestSpeculativePrefillWarmsSelectedDataParallelDecoder(t *testing.T) {
	s := NewProxy(Config{Port: "0", EnableSpeculativePrefill: true})
	selectedHostPort := "127.0.0.1:9001"

	var rank0Requests atomic.Int32
	s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		rank0Requests.Add(1)
	})

	var warmupBody map[string]any
	var selectedRankRequests atomic.Int32
	s.dataParallelProxies = map[string]http.Handler{
		selectedHostPort: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			selectedRankRequests.Add(1)
			if err := json.NewDecoder(r.Body).Decode(&warmupBody); err != nil {
				t.Errorf("failed to decode warmup body: %v", err)
			}
			w.WriteHeader(http.StatusNoContent)
		}),
	}

	originalBody := map[string]any{
		requestFieldMessages:  json.RawMessage(`[{"role":"user","content":"Hi"}]`),
		requestFieldMaxTokens: float64(128),
		requestFieldStream:    true,
	}
	var released atomic.Bool
	s.triggerSpeculativePrefill(context.Background(), originalBody, "Hello from decode", nil, selectedHostPort, func() { released.Store(true) }, nil)

	if !released.Load() {
		t.Fatal("expected speculative prefill slot to be released")
	}
	if got := rank0Requests.Load(); got != 0 {
		t.Fatalf("expected rank 0 decoder not to be warmed, got %d requests", got)
	}
	if got := selectedRankRequests.Load(); got != 1 {
		t.Fatalf("expected selected data-parallel decoder to be warmed once, got %d requests", got)
	}
	if warmupBody[requestFieldMaxTokens] != float64(1) {
		t.Fatalf("expected max_tokens to be capped to 1, got %#v", warmupBody[requestFieldMaxTokens])
	}
	if warmupBody[requestFieldStream] != false {
		t.Fatalf("expected stream to be disabled, got %#v", warmupBody[requestFieldStream])
	}
	messages, ok := warmupBody[requestFieldMessages].([]any)
	if !ok {
		t.Fatalf("expected messages array, got %T", warmupBody[requestFieldMessages])
	}
	if len(messages) != 3 {
		t.Fatalf("expected original, assistant, and placeholder messages, got %d", len(messages))
	}
	assistantMessage, ok := messages[1].(map[string]any)
	if !ok {
		t.Fatalf("expected assistant message object, got %T", messages[1])
	}
	if assistantMessage[requestFieldRole] != roleAssistant || assistantMessage[requestFieldContent] != "Hello from decode" {
		t.Fatalf("unexpected assistant message: %#v", assistantMessage)
	}
}

func TestSpeculativePrefillSkipsMissingDataParallelDecoder(t *testing.T) {
	s := NewProxy(Config{Port: "0", EnableSpeculativePrefill: true})
	s.dataParallelProxies = map[string]http.Handler{}

	var rank0Requests atomic.Int32
	s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		rank0Requests.Add(1)
	})

	originalBody := map[string]any{
		requestFieldMessages: json.RawMessage(`[{"role":"user","content":"Hi"}]`),
	}
	var released atomic.Bool
	s.triggerSpeculativePrefill(context.Background(), originalBody, "Hello from decode", nil, "127.0.0.1:9002", func() { released.Store(true) }, nil)

	if !released.Load() {
		t.Fatal("expected speculative prefill slot to be released")
	}
	if got := rank0Requests.Load(); got != 0 {
		t.Fatalf("expected missing data-parallel target not to fall back to rank 0, got %d requests", got)
	}
}

func TestSpeculativePrefillDataParallelWarmupUsesChatCompletionsPath(t *testing.T) {
	s := NewProxy(Config{Port: "0", EnableSpeculativePrefill: true})
	selectedHostPort := "127.0.0.1:9001"

	var path string
	s.dataParallelProxies = map[string]http.Handler{
		selectedHostPort: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			path = r.URL.Path
			w.WriteHeader(http.StatusOK)
		}),
	}

	originalBody := map[string]any{
		requestFieldMessages: json.RawMessage(`[{"role":"user","content":"Hi"}]`),
	}
	s.triggerSpeculativePrefill(context.Background(), originalBody, "Hello from decode", nil, selectedHostPort, func() {}, nil)

	if path != reqcommon.PathChatCompletions {
		t.Fatalf("expected warmup path %q, got %q", reqcommon.PathChatCompletions, path)
	}
}
