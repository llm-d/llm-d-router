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

package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
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

var _ = Describe("Cached token usage rewriter", func() {
	It("should replace cached tokens in JSON responses", func() {
		body := []byte(`{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`)
		Expect(replaceCachedTokens(body, 7)).To(Equal([]byte(`{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`)))
	})

	It("should buffer non-streaming JSON split across writes", func() {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "application/json")
		writer, finalize := newCachedTokensResponseWriterWithFinalize(recorder, 7, false)

		firstChunk := []byte(`{"usage":{"prompt_tokens":64,`)
		n, err := writer.Write(firstChunk)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(firstChunk)))
		Expect(recorder.Body.Len()).To(BeZero())

		secondChunk := []byte(`"prompt_tokens_details":{"cached_tokens":49}}}`)
		n, err = writer.Write(secondChunk)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(secondChunk)))
		Expect(finalize()).To(Succeed())

		Expect(recorder.Body.String()).To(Equal(`{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`))
	})

	It("should flush an incomplete non-streaming body unchanged", func() {
		recorder := httptest.NewRecorder()
		writer, finalize := newCachedTokensResponseWriterWithFinalize(recorder, 7, false)

		body := []byte(`{"usage":`)
		n, err := writer.Write(body)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(body)))
		Expect(recorder.Body.Len()).To(BeZero())
		Expect(finalize()).To(Succeed())

		Expect(recorder.Body.Bytes()).To(Equal(body))
	})

	It("should use the request mode instead of the response content type", func() {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "text/event-stream")
		writer, finalize := newCachedTokensResponseWriterWithFinalize(recorder, 7, false)

		body := []byte(`{"usage":{"prompt_tokens_details":{"cached_tokens":49}}}`)
		n, err := writer.Write(body)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(body)))
		Expect(finalize()).To(Succeed())

		Expect(recorder.Body.String()).To(Equal(`{"usage":{"prompt_tokens_details":{"cached_tokens":7}}}`))
	})

	It("should add cached tokens when JSON usage details omit them", func() {
		body := []byte(`{"usage":{"prompt_tokens":64,"prompt_tokens_details":{}}}`)
		updated := replaceCachedTokens(body, 7)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		details := usage["prompt_tokens_details"].(map[string]any)
		Expect(details["cached_tokens"]).To(BeNumerically("==", 7))
	})

	It("should add usage details when JSON usage omits them", func() {
		body := []byte(`{"usage":{"prompt_tokens":64}}`)
		updated := replaceCachedTokens(body, 7)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		details := usage["prompt_tokens_details"].(map[string]any)
		Expect(details["cached_tokens"]).To(BeNumerically("==", 7))
	})

	It("should not extract cached tokens when prefill response has none", func() {
		prefillResponse := map[string]any{
			requestFieldKVTransferParams: map[string]any{
				requestFieldRemoteBlockIDs: []any{float64(1), float64(2), float64(3)},
			},
		}
		_, ok := extractCachedTokens(prefillResponse)
		Expect(ok).To(BeFalse())
	})

	It("should extract zero cached tokens when prefill explicitly reports zero", func() {
		prefillResponse := map[string]any{
			"usage": map[string]any{
				"prompt_tokens_details": map[string]any{
					"cached_tokens": float64(0),
				},
			},
		}
		cachedTokens, ok := extractCachedTokens(prefillResponse)
		Expect(ok).To(BeTrue())
		Expect(cachedTokens).To(Equal(0))
	})

	It("should replace cached tokens in streamed usage chunks", func() {
		body := []byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n")
		updated := replaceCachedTokens(body, 7)
		Expect(string(updated)).To(ContainSubstring(`"cached_tokens":7`))
		Expect(string(updated)).To(ContainSubstring("data: [DONE]"))
	})

	It("should buffer streamed usage chunks split before the data prefix", func() {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "text/event-stream")
		writer := newCachedTokensResponseWriter(recorder, 8, true)

		n, err := writer.Write([]byte("da"))
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(2))
		Expect(recorder.Body.String()).To(BeEmpty())

		chunk := []byte("ta: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n")
		n, err = writer.Write(chunk)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(chunk)))
		Expect(recorder.Body.String()).To(ContainSubstring(`"cached_tokens":8`))
		Expect(recorder.Body.String()).To(ContainSubstring("data: [DONE]"))
	})

	It("should buffer streamed usage chunks split inside the JSON payload", func() {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "text/event-stream")
		writer := newCachedTokensResponseWriter(recorder, 7, true)

		firstChunk := []byte(`data: {"choices":[],"usage":{"prompt_tokens":64,`)
		n, err := writer.Write(firstChunk)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(firstChunk)))
		Expect(recorder.Body.String()).To(BeEmpty())

		secondChunk := []byte(`"prompt_tokens_details":{"cached_tokens":49}}}` + "\n\n")
		n, err = writer.Write(secondChunk)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(secondChunk)))
		Expect(recorder.Body.String()).To(ContainSubstring(`"cached_tokens":7`))
	})

	It("should add cached tokens in streamed usage chunks that omit them", func() {
		body := []byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{}}}\n\ndata: [DONE]\n")
		updated := replaceCachedTokens(body, 7)
		Expect(string(updated)).To(ContainSubstring(`"cached_tokens":7`))
		Expect(string(updated)).To(ContainSubstring("data: [DONE]"))
	})

	It("should preserve non-JSON streamed data lines", func() {
		body := []byte("event: ping\ndata: not-json\n\ndata: [DONE]\n")
		updated := replaceCachedTokens(body, 7)
		Expect(updated).To(Equal(body))
	})

	It("should preserve ReaderFrom while rewriting cached tokens", func() {
		base := &readerFromResponseWriter{header: http.Header{}}
		writer := newCachedTokensResponseWriter(base, 7, false)
		readerFrom, ok := writer.(io.ReaderFrom)
		Expect(ok).To(BeTrue())

		body := `{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`
		n, err := readerFrom.ReadFrom(strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(int64(len(body))))
		Expect(base.body.String()).To(Equal(`{"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":7}}}`))
	})

	It("should preserve ReaderFrom for streamed responses while rewriting complete lines", func() {
		base := &readerFromResponseWriter{header: http.Header{"Content-Type": []string{"text/event-stream"}}}
		writer := newCachedTokensResponseWriter(base, 7, true)
		readerFrom, ok := writer.(io.ReaderFrom)
		Expect(ok).To(BeTrue())

		body := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n"
		n, err := readerFrom.ReadFrom(strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(int64(len(body))))
		Expect(base.body.String()).To(ContainSubstring(`"cached_tokens":7`))
		Expect(base.body.String()).To(ContainSubstring("data: [DONE]"))
	})

	It("should flush a trailing streamed data line without a final newline", func() {
		base := &readerFromResponseWriter{header: http.Header{"Content-Type": []string{"text/event-stream"}}}
		writer := newCachedTokensResponseWriter(base, 7, true)
		readerFrom, ok := writer.(io.ReaderFrom)
		Expect(ok).To(BeTrue())

		body := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"prompt_tokens_details\":{\"cached_tokens\":49}}}"
		n, err := readerFrom.ReadFrom(strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(int64(len(body))))
		Expect(base.body.String()).To(ContainSubstring(`"cached_tokens":7`))
	})

	It("should finalize a trailing streamed data line written without a final newline", func() {
		recorder := httptest.NewRecorder()
		recorder.Header().Set("Content-Type", "text/event-stream")
		writer, finalize := newCachedTokensResponseWriterWithFinalize(recorder, 7, true)

		body := []byte(`data: {"choices":[],"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`)
		n, err := writer.Write(body)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(body)))
		Expect(recorder.Body.String()).To(BeEmpty())

		Expect(finalize()).To(Succeed())
		Expect(recorder.Body.String()).To(ContainSubstring(`"cached_tokens":7`))
	})

	It("should preserve streamed content chunks without usage", func() {
		// The common streamed frame: the guard skips it before any unmarshalling.
		body := []byte(`data: {"choices":[{"delta":{"content":" the"}}]}` + "\n\ndata: [DONE]\n")
		Expect(bytes.Contains(body, usageKey)).To(BeFalse())
		Expect(replaceCachedTokens(body, 7)).To(Equal(body))
	})

	It("should preserve a streamed content chunk that gets past the guard", func() {
		// JSON escapes any quote inside a string, so free text only produces the
		// `"usage"` byte sequence when a whole string value is the word itself,
		// which is what a model streaming that word one token at a time sends.
		// This frame is valid JSON, so the guard matches and the parse does run;
		// it must still come back byte-for-byte unchanged.
		body := []byte(`data: {"choices":[{"delta":{"content":"usage"}}]}` + "\n")
		Expect(bytes.Contains(body, usageKey)).To(BeTrue())
		Expect(json.Valid(bytes.TrimPrefix(bytes.TrimRight(body, "\n"), []byte("data: ")))).To(BeTrue())
		Expect(replaceCachedTokens(body, 7)).To(Equal(body))
	})
})

var _ = Describe("Cached token usage rewriter with Anthropic usage", func() {
	// A decode worker reports the whole prompt as a cache read when the KV arrived
	// from a prefiller, collapsing input_tokens to 0. The prefiller's real count has
	// to land in cache_read_input_tokens with input_tokens absorbing the remainder.
	It("should rewrite cache_read_input_tokens and re-derive input_tokens", func() {
		body := []byte(`{"usage":{"input_tokens":0,"output_tokens":16,"cache_creation_input_tokens":0,"cache_read_input_tokens":1710}}`)
		updated := replaceCachedTokens(body, 448)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(448))
		Expect(usage["input_tokens"]).To(BeEquivalentTo(1262))
		Expect(usage["output_tokens"]).To(BeEquivalentTo(16))
	})

	// Fabricating the OpenAI details object on an Anthropic response is what hid the
	// bug: the response looked patched while the Anthropic fields still carried the
	// decode worker's numbers.
	It("should not fabricate prompt_tokens_details on an Anthropic response", func() {
		body := []byte(`{"usage":{"input_tokens":0,"output_tokens":16,"cache_read_input_tokens":1710}}`)
		updated := replaceCachedTokens(body, 0)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		Expect(usage).NotTo(HaveKey("prompt_tokens_details"))
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(0))
		Expect(usage["input_tokens"]).To(BeEquivalentTo(1710))
	})

	It("should preserve cache_creation_input_tokens", func() {
		body := []byte(`{"usage":{"input_tokens":6,"output_tokens":16,"cache_creation_input_tokens":768,"cache_read_input_tokens":448}}`)
		updated := replaceCachedTokens(body, 64)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		Expect(usage["cache_creation_input_tokens"]).To(BeEquivalentTo(768))
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(64))
		// The prompt total (6+768+448=1222) is invariant: 1222-64-768=390.
		Expect(usage["input_tokens"]).To(BeEquivalentTo(390))
	})

	It("should rewrite usage nested under message on the message_start event", func() {
		body := []byte(`data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":1214,"output_tokens":0}}}` + "\n")
		updated := replaceCachedTokens(body, 448)

		var event map[string]any
		payload := bytes.TrimPrefix(bytes.TrimRight(updated, "\n"), []byte("data: "))
		Expect(json.Unmarshal(payload, &event)).To(Succeed())
		usage := event["message"].(map[string]any)["usage"].(map[string]any)
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(448))
		Expect(usage["input_tokens"]).To(BeEquivalentTo(766))
	})

	It("should rewrite the message_delta usage frame", func() {
		body := []byte(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":0,"output_tokens":24,"cache_creation_input_tokens":0,"cache_read_input_tokens":1713}}` + "\n")
		updated := replaceCachedTokens(body, 0)

		var event map[string]any
		payload := bytes.TrimPrefix(bytes.TrimRight(updated, "\n"), []byte("data: "))
		Expect(json.Unmarshal(payload, &event)).To(Succeed())
		usage := event["usage"].(map[string]any)
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(0))
		Expect(usage["input_tokens"]).To(BeEquivalentTo(1713))
		Expect(usage).NotTo(HaveKey("prompt_tokens_details"))
	})

	It("should clamp input_tokens at zero when cached tokens exceed the prompt", func() {
		body := []byte(`{"usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":0}}`)
		updated := replaceCachedTokens(body, 99)

		var response map[string]any
		Expect(json.Unmarshal(updated, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		Expect(usage["cache_read_input_tokens"]).To(BeEquivalentTo(99))
		Expect(usage["input_tokens"]).To(BeEquivalentTo(0))
	})

	It("should leave an already-correct Anthropic response untouched", func() {
		body := []byte(`{"usage":{"input_tokens":1710,"output_tokens":16,"cache_read_input_tokens":0}}`)
		Expect(replaceCachedTokens(body, 0)).To(Equal(body))
	})

	// The OpenAI Responses API also carries a top-level input_tokens, so it must not
	// be mistaken for the Anthropic shape.
	It("should not treat OpenAI Responses usage as Anthropic", func() {
		usage := map[string]any{
			"input_tokens":         float64(100),
			"input_tokens_details": map[string]any{"cached_tokens": float64(20)},
			"output_tokens":        float64(8),
		}
		Expect(isAnthropicUsage(usage)).To(BeFalse())
	})

	It("should not treat OpenAI chat usage as Anthropic", func() {
		usage := map[string]any{
			"prompt_tokens":         float64(64),
			"prompt_tokens_details": map[string]any{"cached_tokens": float64(49)},
		}
		Expect(isAnthropicUsage(usage)).To(BeFalse())
	})

	// Without this the sidecar cannot learn the prefiller's hit count for a
	// /v1/messages request, so every such request looks like a cold cache.
	It("should read cached tokens from an Anthropic prefiller response", func() {
		var response map[string]any
		Expect(json.Unmarshal([]byte(`{"usage":{"input_tokens":6,"output_tokens":1,"cache_read_input_tokens":448}}`), &response)).To(Succeed())

		cachedTokens, ok := extractCachedTokens(response)
		Expect(ok).To(BeTrue())
		Expect(cachedTokens).To(Equal(448))
	})
})

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
