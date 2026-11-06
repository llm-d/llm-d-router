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

package usage

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPromptTokens(t *testing.T) {
	tests := []struct {
		name  string
		block any
		want  int
		ok    bool
	}{
		{name: "chat prompt_tokens", block: map[string]any{"prompt_tokens": 12.0, "completion_tokens": 3.0}, want: 12, ok: true},
		{name: "responses input_tokens", block: map[string]any{"input_tokens": 40.0, "output_tokens": 5.0}, want: 40, ok: true},
		{name: "messages input_tokens int", block: map[string]any{"input_tokens": 9}, want: 9, ok: true},
		// vLLM reports the cached part of a Messages prompt beside input_tokens, not inside it.
		{name: "messages with cache split", block: map[string]any{"input_tokens": 3.0, "cache_read_input_tokens": 40.0, "cache_creation_input_tokens": 5.0}, want: 48, ok: true},
		{name: "chat prompt_tokens ignores messages cache fields", block: map[string]any{"prompt_tokens": 12.0, "cache_read_input_tokens": 40.0}, want: 12, ok: true},
		{name: "output only (messages delta)", block: map[string]any{"output_tokens": 5.0}, ok: false},
		{name: "not an object", block: nil, ok: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := PromptTokens(tc.block)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("PromptTokens = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestUsageBlock(t *testing.T) {
	tests := []struct {
		name string
		body string
		want map[string]any
	}{
		{name: "top level", body: `{"usage":{"prompt_tokens":1}}`, want: map[string]any{"prompt_tokens": 1.0}},
		{name: "responses stream event", body: `{"type":"response.completed","response":{"usage":{"input_tokens":2}}}`, want: map[string]any{"input_tokens": 2.0}},
		{name: "messages message_start", body: `{"type":"message_start","message":{"usage":{"input_tokens":3}}}`, want: map[string]any{"input_tokens": 3.0}},
		{name: "null usage on an early responses event", body: `{"type":"response.created","response":{"usage":null}}`, want: nil},
		{name: "no usage", body: `{"choices":[]}`, want: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var response map[string]any
			if err := json.Unmarshal([]byte(tc.body), &response); err != nil {
				t.Fatal(err)
			}
			got, ok := UsageBlock(response)
			if (tc.want == nil) == ok {
				t.Fatalf("UsageBlock ok = %v, want %v", ok, tc.want != nil)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Fatalf("UsageBlock[%q] = %v, want %v", k, got[k], v)
				}
			}
		})
	}
}

// observedPrompts runs body through a writer with only an observer set and
// returns every prompt count the observer saw, plus what reached the client.
func observedPrompts(t *testing.T, body string, streaming, dropUsageOnly bool) ([]int, string) {
	t.Helper()
	var seen []int
	recorder := httptest.NewRecorder()
	writer, finalize := NewResponseWriter(recorder, Options{
		OnUsage: func(block map[string]any) {
			if n, ok := PromptTokens(block); ok {
				seen = append(seen, n)
			}
		},
		DropUsageOnlyFrame: dropUsageOnly,
	}, streaming)
	if _, err := writer.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}
	return seen, recorder.Body.String()
}

func TestResponseWriter_ObservesUsage_JSON(t *testing.T) {
	body := `{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":17,"completion_tokens":1}}`
	seen, out := observedPrompts(t, body, false, false)
	if len(seen) != 1 || seen[0] != 17 {
		t.Fatalf("observed %v, want [17]", seen)
	}
	if out != body {
		t.Fatalf("body changed without a rewrite: %s", out)
	}
}

func TestResponseWriter_ObservesUsage_ChatStreamFinalFrame(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n" +
		"data: [DONE]\n\n"
	seen, out := observedPrompts(t, body, true, false)
	if len(seen) != 1 || seen[0] != 5 {
		t.Fatalf("observed %v, want [5]", seen)
	}
	if out != body {
		t.Fatalf("stream changed without a rewrite: %s", out)
	}
}

func TestResponseWriter_ObservesUsage_ResponsesAndMessagesEvents(t *testing.T) {
	responses := "data: {\"type\":\"response.created\",\"response\":{\"usage\":null}}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":21,\"output_tokens\":2}}}\n\n"
	if seen, _ := observedPrompts(t, responses, true, false); len(seen) != 1 || seen[0] != 21 {
		t.Fatalf("responses: observed %v, want [21]", seen)
	}
	messages := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":8,\"output_tokens\":1}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4}}\n\n"
	if seen, _ := observedPrompts(t, messages, true, false); len(seen) != 1 || seen[0] != 8 {
		t.Fatalf("messages: observed %v, want [8]", seen)
	}
}

func TestResponseWriter_DropsInjectedUsageFrame(t *testing.T) {
	content := "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"
	usageFrame := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n"
	done := "data: [DONE]\n\n"

	seen, out := observedPrompts(t, content+usageFrame+done, true, true)
	if len(seen) != 1 || seen[0] != 5 {
		t.Fatalf("observed %v, want [5]: the frame is read before it is dropped", seen)
	}
	if strings.Contains(out, `"usage"`) {
		t.Fatalf("usage-only frame reached the client: %s", out)
	}
	if !strings.Contains(out, `"content":"a"`) || !strings.Contains(out, "[DONE]") {
		t.Fatalf("content or [DONE] lost: %s", out)
	}

	// A frame that carries usage beside choices is the client's own and stays.
	withChoices := "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5}}\n\n"
	if _, out := observedPrompts(t, withChoices, true, true); !strings.Contains(out, `"usage"`) {
		t.Fatalf("frame with choices was dropped: %s", out)
	}
}

func TestResponseWriter_ObserverAndCachedTokensTogether(t *testing.T) {
	cached := 3
	var seen []int
	recorder := httptest.NewRecorder()
	writer, finalize := NewResponseWriter(recorder, Options{
		CachedTokens: &cached,
		OnUsage: func(block map[string]any) {
			n, _ := PromptTokens(block)
			seen = append(seen, n)
		},
	}, false)
	if _, err := writer.Write([]byte(`{"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}`)); err != nil {
		t.Fatal(err)
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != 10 {
		t.Fatalf("observed %v, want [10]", seen)
	}
	if got := cachedTokensOf(t, recorder.Body.Bytes()); got != 3 {
		t.Fatalf("cached_tokens = %d, want 3", got)
	}
}

// recordingWriter records writes as they happen, so a test can tell whether a
// body was forwarded before or after finalize.
type recordingWriter struct {
	*httptest.ResponseRecorder
	writes int
}

func (w *recordingWriter) Write(b []byte) (int, error) {
	w.writes++
	return w.ResponseRecorder.Write(b)
}

func TestResponseWriter_PassthroughKeepsContentLengthAndForwardsEagerly(t *testing.T) {
	for name, streaming := range map[string]bool{"json": false, "sse": true} {
		t.Run(name, func(t *testing.T) {
			rec := &recordingWriter{ResponseRecorder: httptest.NewRecorder()}
			var seen int
			writer, finalize := NewResponseWriter(rec, Options{OnUsage: func(block map[string]any) {
				seen, _ = PromptTokens(block)
			}}, streaming)
			writer.Header().Set("Content-Length", "123")
			writer.WriteHeader(200)
			if rec.Header().Get("Content-Length") != "123" {
				t.Fatalf("passthrough must keep Content-Length, got %q", rec.Header().Get("Content-Length"))
			}
			first, second := `{"choices":[],`, `"usage":{"prompt_tokens":4}}`
			if streaming {
				first, second = "data: {\"choices\":[{\"delta\":{}}]}\n\n", "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4}}\n\ndata: [DONE]\n\n"
			}
			if _, err := writer.Write([]byte(first)); err != nil {
				t.Fatal(err)
			}
			if rec.writes != 1 || rec.Body.String() != first {
				t.Fatalf("first write must be forwarded immediately, got %d writes, body %q", rec.writes, rec.Body.String())
			}
			if _, err := writer.Write([]byte(second)); err != nil {
				t.Fatal(err)
			}
			if err := finalize(); err != nil {
				t.Fatal(err)
			}
			if rec.Body.String() != first+second {
				t.Fatalf("body changed in passthrough: %q", rec.Body.String())
			}
			if seen != 4 {
				t.Fatalf("observer saw %d, want 4", seen)
			}
		})
	}
}

func TestResponseWriter_RewriteModeStillDropsContentLength(t *testing.T) {
	rec := httptest.NewRecorder()
	cached := 1
	writer, _ := NewResponseWriter(rec, Options{CachedTokens: &cached}, false)
	writer.Header().Set("Content-Length", "123")
	writer.WriteHeader(200)
	if rec.Header().Get("Content-Length") != "" {
		t.Fatalf("a rewriting writer must drop Content-Length, got %q", rec.Header().Get("Content-Length"))
	}
}

func TestResponseWriter_PassthroughSkipsObservationPastCap(t *testing.T) {
	old := maxObserveBytes
	maxObserveBytes = 16
	t.Cleanup(func() { maxObserveBytes = old })
	rec := httptest.NewRecorder()
	called := false
	writer, finalize := NewResponseWriter(rec, Options{OnUsage: func(map[string]any) { called = true }}, false)
	body := `{"padding":"xxxxxxxxxxxxxxxxxxxx","usage":{"prompt_tokens":4}}`
	if _, err := writer.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}
	if rec.Body.String() != body {
		t.Fatalf("body must be forwarded even past the cap: %q", rec.Body.String())
	}
	if called {
		t.Fatal("observer must not run on a truncated copy")
	}
}

func TestResponseWriter_PassthroughReadFrom(t *testing.T) {
	rec := &readerFromResponseWriter{}
	var seen int
	writer, finalize := NewResponseWriter(rec, Options{OnUsage: func(block map[string]any) {
		seen, _ = PromptTokens(block)
	}}, false)
	body := `{"choices":[],"usage":{"prompt_tokens":6}}`
	if _, err := writer.(io.ReaderFrom).ReadFrom(strings.NewReader(body)); err != nil {
		t.Fatal(err)
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}
	if rec.body.String() != body || seen != 6 {
		t.Fatalf("ReadFrom passthrough: body %q seen %d", rec.body.String(), seen)
	}
}
