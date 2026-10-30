/*
Copyright 2025 The llm-d Authors.

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
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

type readerFromResponseWriter struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (w *readerFromResponseWriter) Header() http.Header {
	if w.header == nil {
		w.header = http.Header{}
	}
	return w.header
}

func (w *readerFromResponseWriter) Write(body []byte) (int, error) {
	return w.body.Write(body)
}

func (w *readerFromResponseWriter) WriteHeader(statusCode int) {
	w.status = statusCode
}

func (w *readerFromResponseWriter) ReadFrom(src io.Reader) (int64, error) {
	return io.Copy(&w.body, src)
}

// cachedTokensOf reads usage.prompt_tokens_details.cached_tokens out of a JSON
// document, failing the test if any level is missing.
func cachedTokensOf(t *testing.T, body []byte) int {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal(body, &document); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	cachedTokens, ok := CachedTokens(document[reqcommon.FieldUsage])
	if !ok {
		t.Fatalf("no cached tokens in %s", body)
	}
	return cachedTokens
}

func TestReplaceCachedTokens(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "json usage",
			body: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`,
			want: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`,
		},
		{
			name: "json usage details without cached tokens",
			body: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{}}}`,
			want: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`,
		},
		{
			name: "json usage without details",
			body: `{"usage":{"prompt_tokens":64}}`,
			want: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`,
		},
		{
			// Only usage is decoded: key order and number formatting elsewhere
			// stay as the worker wrote them.
			name: "json values outside usage",
			body: `{"choices":[{"message":{"role":"assistant","content":"hi"},"index":0,"logprobs":{"content":[{"token":"hi","logprob":0.0}]}}],"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`,
			want: `{"choices":[{"message":{"role":"assistant","content":"hi"},"index":0,"logprobs":{"content":[{"token":"hi","logprob":0.0}]}}],"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`,
		},
		{
			// Responses reports input_tokens_details; the prompt_tokens_details
			// the other routes use must not be added alongside it.
			name: "responses usage",
			body: `{"usage":{"input_tokens":64,"input_tokens_details":{"cached_tokens":500},"total_tokens":68}}`,
			want: `{"usage":{"input_tokens":64,"input_tokens_details":{"cached_tokens":7},"total_tokens":68}}`,
		},
		{
			// The Messages API carries its own cache counts and defines neither
			// details object, so the block is left as it came.
			name: "messages usage",
			body: `{"usage":{"cache_read_input_tokens":500,"input_tokens":64,"output_tokens":4}}`,
			want: `{"usage":{"cache_read_input_tokens":500,"input_tokens":64,"output_tokens":4}}`,
		},
		{
			name: "streamed usage frame",
			body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n",
			want: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n\ndata: [DONE]\n",
		},
		{
			name: "streamed usage frame without cached tokens",
			body: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{}}}\n\ndata: [DONE]\n",
			want: "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n\ndata: [DONE]\n",
		},
		{
			name: "non-JSON streamed data lines",
			body: "event: ping\ndata: not-json\n\ndata: [DONE]\n",
			want: "event: ping\ndata: not-json\n\ndata: [DONE]\n",
		},
		{
			// The common streamed frame: the guard skips it before any unmarshalling.
			name: "streamed content frame without usage",
			body: `data: {"choices":[{"delta":{"content":" the"}}]}` + "\n\ndata: [DONE]\n",
			want: `data: {"choices":[{"delta":{"content":" the"}}]}` + "\n\ndata: [DONE]\n",
		},
		{
			// JSON escapes any quote inside a string, so free text only produces
			// the `"usage"` byte sequence when a whole string value is the word
			// itself, which is what a model streaming that word one token at a
			// time sends. The guard matches and the parse does run; the frame must
			// still come back byte-for-byte unchanged.
			name: "streamed content frame past the usage guard",
			body: `data: {"choices":[{"delta":{"content":"usage"}}]}` + "\n",
			want: `data: {"choices":[{"delta":{"content":"usage"}}]}` + "\n",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := string(replaceCachedTokens([]byte(test.body), 7)); got != test.want {
				t.Errorf("replaceCachedTokens() = %s, want %s", got, test.want)
			}
		})
	}
}

// The guard is what keeps the per-frame cost off content frames, so assert that
// the two frames above sit on the sides of it the comments claim.
func TestUsageKeyGuard(t *testing.T) {
	contentFrame := []byte(`data: {"choices":[{"delta":{"content":" the"}}]}` + "\n")
	if bytes.Contains(contentFrame, usageKey) {
		t.Errorf("content frame must not match the usage guard: %s", contentFrame)
	}

	wordFrame := []byte(`data: {"choices":[{"delta":{"content":"usage"}}]}` + "\n")
	if !bytes.Contains(wordFrame, usageKey) {
		t.Errorf("frame streaming the word itself must match the usage guard: %s", wordFrame)
	}
	payload := bytes.TrimPrefix(bytes.TrimRight(wordFrame, "\n"), []byte(reqcommon.SSEDataPrefix))
	if !json.Valid(payload) {
		t.Errorf("frame payload must be valid JSON for the parse to run: %s", payload)
	}
}

func TestCachedTokens(t *testing.T) {
	tests := []struct {
		name   string
		block  any
		want   int
		wantOK bool
	}{
		{
			name:  "absent usage block",
			block: nil,
		},
		{
			name:  "usage without prompt token details",
			block: map[string]any{"prompt_tokens": float64(64)},
		},
		{
			name:   "explicit zero",
			block:  map[string]any{"prompt_tokens_details": map[string]any{"cached_tokens": float64(0)}},
			wantOK: true,
		},
		{
			name:   "cached tokens",
			block:  map[string]any{"prompt_tokens_details": map[string]any{"cached_tokens": float64(49)}},
			want:   49,
			wantOK: true,
		},
		{
			name:   "responses cached tokens",
			block:  map[string]any{"input_tokens_details": map[string]any{"cached_tokens": float64(49)}},
			want:   49,
			wantOK: true,
		},
		{
			name:   "both details objects",
			block:  map[string]any{"input_tokens_details": map[string]any{"cached_tokens": float64(49)}, "prompt_tokens_details": map[string]any{"cached_tokens": float64(7)}},
			want:   49,
			wantOK: true,
		},
		{
			name:  "messages cache counts",
			block: map[string]any{"cache_read_input_tokens": float64(49), "input_tokens": float64(64)},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, ok := CachedTokens(test.block)
			if ok != test.wantOK || got != test.want {
				t.Errorf("CachedTokens() = (%d, %t), want (%d, %t)", got, ok, test.want, test.wantOK)
			}
		})
	}
}

// A prefiller response carries cached tokens under its usage field; the sidecar
// and the coordinator both read it from there.
func TestCachedTokens_PrefillerResponseWithoutUsage(t *testing.T) {
	response := map[string]any{
		reqcommon.FieldKVTransferParams: map[string]any{
			reqcommon.FieldRemoteBlockIDs: []any{float64(1), float64(2), float64(3)},
		},
	}
	if _, ok := CachedTokens(response[reqcommon.FieldUsage]); ok {
		t.Error("CachedTokens() reported a count for a response without usage")
	}
}

func TestResponseWriter_BuffersJSONSplitAcrossWrites(t *testing.T) {
	recorder := httptest.NewRecorder()
	recorder.Header().Set("Content-Type", reqcommon.ContentTypeJSON)
	writer, finalize := NewCachedTokensResponseWriter(recorder, 7, false)

	writeChunk(t, writer, `{"usage":{"prompt_tokens":64,`)
	if recorder.Body.Len() != 0 {
		t.Fatalf("partial JSON was forwarded: %s", recorder.Body.String())
	}

	writeChunk(t, writer, `"prompt_tokens_details":{"cached_tokens":49}}}`)
	if err := finalize(); err != nil {
		t.Fatal(err)
	}

	want := `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`
	if got := recorder.Body.String(); got != want {
		t.Errorf("body = %s, want %s", got, want)
	}
}

func TestResponseWriter_FlushesIncompleteJSONUnchanged(t *testing.T) {
	recorder := httptest.NewRecorder()
	writer, finalize := NewCachedTokensResponseWriter(recorder, 7, false)

	const body = `{"usage":`
	writeChunk(t, writer, body)
	if recorder.Body.Len() != 0 {
		t.Fatalf("partial JSON was forwarded: %s", recorder.Body.String())
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}

	if got := recorder.Body.String(); got != body {
		t.Errorf("body = %s, want %s", got, body)
	}
}

// The request's own stream flag decides the rewrite mode: the upstream content
// type is not known when the writer is built.
func TestResponseWriter_UsesTheRequestModeNotTheContentType(t *testing.T) {
	recorder := httptest.NewRecorder()
	recorder.Header().Set("Content-Type", "text/event-stream")
	writer, finalize := NewCachedTokensResponseWriter(recorder, 7, false)

	writeChunk(t, writer, `{"usage":{"prompt_tokens_details":{"cached_tokens":49}}}`)
	if err := finalize(); err != nil {
		t.Fatal(err)
	}

	want := `{"usage":{"prompt_tokens_details":{"cached_tokens":7}}}`
	if got := recorder.Body.String(); got != want {
		t.Errorf("body = %s, want %s", got, want)
	}
}

func TestResponseWriter_BuffersStreamedFramesSplitAcrossWrites(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
	}{
		{
			name:   "split before the data prefix",
			chunks: []string{"da", "ta: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n"},
		},
		{
			name:   "split inside the JSON payload",
			chunks: []string{`data: {"choices":[],"usage":{"prompt_tokens":64,`, `"prompt_tokens_details":{"cached_tokens":49}}}` + "\n\n"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			recorder.Header().Set("Content-Type", "text/event-stream")
			writer, finalize := NewCachedTokensResponseWriter(recorder, 7, true)

			writeChunk(t, writer, test.chunks[0])
			if recorder.Body.Len() != 0 {
				t.Fatalf("partial frame was forwarded: %s", recorder.Body.String())
			}
			for _, chunk := range test.chunks[1:] {
				writeChunk(t, writer, chunk)
			}
			if err := finalize(); err != nil {
				t.Fatal(err)
			}

			if got := cachedTokensOf(t, framePayload(t, recorder.Body.String())); got != 7 {
				t.Errorf("cached_tokens = %d, want 7", got)
			}
		})
	}
}

// EOF without a final newline is malformed SSE; the frame is still rewritten
// rather than dropped.
func TestResponseWriter_FinalizesStreamedFrameWithoutTrailingNewline(t *testing.T) {
	recorder := httptest.NewRecorder()
	recorder.Header().Set("Content-Type", "text/event-stream")
	writer, finalize := NewCachedTokensResponseWriter(recorder, 7, true)

	writeChunk(t, writer, `data: {"choices":[],"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`)
	if recorder.Body.Len() != 0 {
		t.Fatalf("frame without a newline was forwarded before the flush: %s", recorder.Body.String())
	}
	if err := finalize(); err != nil {
		t.Fatal(err)
	}

	if got := cachedTokensOf(t, framePayload(t, recorder.Body.String())); got != 7 {
		t.Errorf("cached_tokens = %d, want 7", got)
	}
}

// A ResponseWriter that implements io.ReaderFrom keeps it through the wrap, and
// the rewrite still applies: ReadFrom would otherwise bypass Write.
func TestResponseWriter_PreservesReaderFrom(t *testing.T) {
	tests := []struct {
		name      string
		streaming bool
		body      string
	}{
		{
			name: "non-streaming JSON",
			body: `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`,
		},
		{
			name:      "streamed frames",
			streaming: true,
			body:      "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n",
		},
		{
			name:      "streamed frame without a trailing newline",
			streaming: true,
			body:      "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := &readerFromResponseWriter{}
			writer, _ := NewCachedTokensResponseWriter(base, 7, test.streaming)
			readerFrom, ok := writer.(io.ReaderFrom)
			if !ok {
				t.Fatal("wrapped writer dropped io.ReaderFrom")
			}

			n, err := readerFrom.ReadFrom(strings.NewReader(test.body))
			if err != nil {
				t.Fatal(err)
			}
			if n != int64(len(test.body)) {
				t.Errorf("ReadFrom() = %d, want %d", n, len(test.body))
			}

			written := base.body.String()
			payload := []byte(written)
			if test.streaming {
				if strings.Contains(test.body, reqcommon.SSEDoneMarker) && !strings.Contains(written, reqcommon.SSEDoneMarker) {
					t.Errorf("body dropped the done marker: %s", written)
				}
				payload = framePayload(t, written)
			}
			if got := cachedTokensOf(t, payload); got != 7 {
				t.Errorf("cached_tokens = %d, want 7", got)
			}
		})
	}
}

func writeChunk(t *testing.T, w http.ResponseWriter, chunk string) {
	t.Helper()
	n, err := w.Write([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}
	// The caller counts the bytes it handed over, not the rewritten length.
	if n != len(chunk) {
		t.Fatalf("Write() = %d, want %d", n, len(chunk))
	}
}

// framePayload returns the JSON of the first SSE data frame in body.
func framePayload(t *testing.T, body string) []byte {
	t.Helper()
	for line := range strings.SplitSeq(body, "\n") {
		data, ok := strings.CutPrefix(line, reqcommon.SSEDataPrefix)
		if ok && strings.TrimSpace(data) != reqcommon.SSEDoneMarker {
			return []byte(data)
		}
	}
	t.Fatalf("no data frame in %s", body)
	return nil
}

// Streamed responses send one SSE frame per token and only the final frame carries
// usage, so these two benchmarks bracket the per-frame cost of the rewrite.
var (
	benchContentFrame = []byte(`data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730000000,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":" the"},"logprobs":null,"finish_reason":null}]}` + "\n")
	benchUsageFrame   = []byte(`data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1730000000,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[],"usage":{"prompt_tokens":1024,"completion_tokens":256,"total_tokens":1280,"prompt_tokens_details":{"cached_tokens":1024}}}` + "\n")
)

func BenchmarkReplaceCachedTokensSSELineContentFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		replaceCachedTokensSSELine(benchContentFrame, 512)
	}
}

func BenchmarkReplaceCachedTokensSSELineUsageFrame(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		replaceCachedTokensSSELine(benchUsageFrame, 512)
	}
}

// A large non-streaming response, written in the 32 KiB chunks
// httputil.ReverseProxy writes. The two sizes show how the cost scales.
func BenchmarkResponseWriterChunkedJSON(b *testing.B) {
	const entry = `{"token":"the","logprob":-0.1234567,"bytes":[116,104,101],"top_logprobs":[{"token":"the","logprob":-0.1234567,"bytes":[116,104,101]},{"token":"a","logprob":-2.5,"bytes":[97]}]}`
	for _, size := range []struct {
		name  string
		bytes int
	}{{"1MiB", 1 << 20}, {"8MiB", 8 << 20}} {
		var document bytes.Buffer
		document.WriteString(`{"choices":[{"index":0,"message":{"role":"assistant","content":"x"},"logprobs":{"content":[`)
		for document.Len() < size.bytes {
			document.WriteString(entry + ",")
		}
		document.WriteString(entry + `]}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}`)
		body := document.Bytes()

		b.Run(size.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(body)))
			for i := 0; i < b.N; i++ {
				writer, finalize := NewCachedTokensResponseWriter(httptest.NewRecorder(), 7, false)
				for chunk := range slices.Chunk(body, 32<<10) {
					_, _ = writer.Write(chunk)
				}
				if err := finalize(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
