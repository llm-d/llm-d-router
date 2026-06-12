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
		writer := newCachedTokensResponseWriter(recorder, 8)

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
		writer := newCachedTokensResponseWriter(recorder, 7)

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
		writer := newCachedTokensResponseWriter(base, 7)
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
		writer := newCachedTokensResponseWriter(base, 7)
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
		writer := newCachedTokensResponseWriter(base, 7)
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
		writer, finalize := newCachedTokensResponseWriterWithFinalize(recorder, 7)

		body := []byte(`data: {"choices":[],"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":49}}}`)
		n, err := writer.Write(body)
		Expect(err).ToNot(HaveOccurred())
		Expect(n).To(Equal(len(body)))
		Expect(recorder.Body.String()).To(BeEmpty())

		Expect(finalize()).To(Succeed())
		Expect(recorder.Body.String()).To(ContainSubstring(`"cached_tokens":7`))
	})
})
