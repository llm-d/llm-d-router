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
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"

	"github.com/go-logr/logr/funcr"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
)

func postBody(body string) *http.Request {
	return httptest.NewRequest(http.MethodPost, ChatCompletionsPath, bytes.NewReader([]byte(body)))
}

var _ = Describe("bodyAsJSON", func() {
	It("returns the raw bytes and the parsed object", func() {
		raw, parsed, err := bodyAsJSON(postBody(`{"model":"m","max_tokens":5}`))

		Expect(err).ToNot(HaveOccurred())
		Expect(string(raw)).To(Equal(`{"model":"m","max_tokens":5}`))
		Expect(parsed).To(HaveKeyWithValue("model", json.RawMessage(`"m"`)))
		Expect(parsed).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 5)))
	})

	It("accepts an empty JSON object", func() {
		_, parsed, err := bodyAsJSON(postBody(`{}`))

		Expect(err).ToNot(HaveOccurred())
		Expect(parsed).ToNot(BeNil())
		Expect(parsed).To(BeEmpty())
	})

	// A parsed body is copied and written into by every connector. A nil map
	// accepts no writes, so the parser must never hand one back.
	DescribeTable("rejects a body that is not a JSON object",
		func(body string) {
			_, parsed, err := bodyAsJSON(postBody(body))

			Expect(err).To(MatchError(errInvalidJSON))
			Expect(parsed).To(BeNil())
		},
		Entry("null", `null`),
		Entry("array", `[]`),
		Entry("string", `"text"`),
		Entry("number", `7`),
		Entry("boolean", `true`),
		Entry("empty body", ``),
		Entry("malformed", `{"model":`),
	)

	It("wraps a read failure without marking it invalid JSON", func() {
		r := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, errReader{})

		_, parsed, err := bodyAsJSON(r)

		Expect(err).To(MatchError(ContainSubstring("failed to read request body")))
		Expect(err).ToNot(MatchError(errInvalidJSON))
		Expect(parsed).To(BeNil())
	})

	It("returns a map that callers can clone and write into", func() {
		_, parsed, err := bodyAsJSON(postBody(`{"model":"m"}`))
		Expect(err).ToNot(HaveOccurred())

		clone := maps.Clone(parsed)
		Expect(func() { clone[requestFieldKVTransferParams] = map[string]any{} }).ToNot(Panic())
		Expect(parsed).ToNot(HaveKey(requestFieldKVTransferParams))
	})
})

var _ = Describe("readJSONBody", func() {
	var proxy *Server

	BeforeEach(func() {
		proxy = NewProxy(Config{Port: "0", KVConnector: KVConnectorNIXLV2})
	})

	It("reports success and returns the parsed body", func() {
		w := httptest.NewRecorder()

		raw, parsed, ok := proxy.readJSONBody(postBody(`{"model":"m"}`), w)

		Expect(ok).To(BeTrue())
		Expect(string(raw)).To(Equal(`{"model":"m"}`))
		Expect(parsed).To(HaveKeyWithValue("model", json.RawMessage(`"m"`)))
		Expect(w.Code).To(Equal(http.StatusOK))
	})

	It("answers a null body with a vLLM-shaped 400", func() {
		w := httptest.NewRecorder()

		_, parsed, ok := proxy.readJSONBody(postBody(`null`), w)

		Expect(ok).To(BeFalse())
		Expect(parsed).To(BeNil())
		Expect(w.Code).To(Equal(http.StatusBadRequest))
		Expect(w.Body.String()).To(ContainSubstring("BadRequestError"))
		Expect(w.Body.String()).To(ContainSubstring("is not a JSON object"))
	})

	It("answers a read failure with a vLLM-shaped 400", func() {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, errReader{})

		_, _, ok := proxy.readJSONBody(r, w)

		Expect(ok).To(BeFalse())
		Expect(w.Code).To(Equal(http.StatusBadRequest))
		Expect(w.Body.String()).To(ContainSubstring("BadRequestError"))
		Expect(w.Body.String()).To(ContainSubstring("failed to read request body"))
		Expect(w.Body.String()).To(ContainSubstring("read failed"))
	})

	// A gateway unmarshals the sidecar 400 as a vLLM error. Both refusal paths
	// must therefore answer with that envelope, not with a bare Go string.
	DescribeTable("answers every refusal with a parseable vLLM error envelope",
		func(newRequest func() *http.Request) {
			w := httptest.NewRecorder()

			_, _, ok := proxy.readJSONBody(newRequest(), w)
			Expect(ok).To(BeFalse())

			expectErrorEnvelope(w.Body.Bytes())
		},
		Entry("null body", func() *http.Request { return postBody(`null`) }),
		Entry("malformed body", func() *http.Request { return postBody(`{"model":`) }),
		Entry("read failure", func() *http.Request {
			return httptest.NewRequest(http.MethodPost, ChatCompletionsPath, errReader{})
		}),
	)

	// A client that hangs up before reading the refusal leaves nowhere to send
	// it, so the error goes to the log instead of the wire.
	It("logs the refusal when the response cannot be written", func() {
		var logged []string
		proxy.logger = funcr.New(func(prefix, args string) {
			logged = append(logged, prefix+" "+args)
		}, funcr.Options{})

		_, _, ok := proxy.readJSONBody(postBody(`null`), errWriter{})

		Expect(ok).To(BeFalse())
		Expect(logged).To(ContainElement(ContainSubstring("failed to send error response to client")))
	})
})

// expectErrorEnvelope asserts the vLLM error envelope a gateway unmarshals and
// returns its message, so a caller can additionally check the refusal reason.
func expectErrorEnvelope(body []byte) string {
	GinkgoHelper()

	var got errorResponse
	Expect(json.Unmarshal(body, &got)).To(Succeed())
	Expect(got.Object).To(Equal("error"))
	Expect(got.Type).To(Equal("BadRequestError"))
	Expect(got.Code).To(Equal(http.StatusBadRequest))
	Expect(got.Message).ToNot(BeEmpty())

	return got.Message
}

// errWriter accepts a status code and then fails the body write, standing in
// for a client that hangs up before it reads the response.
type errWriter struct{}

func (errWriter) Header() http.Header       { return http.Header{} }
func (errWriter) WriteHeader(int)           {}
func (errWriter) Write([]byte) (int, error) { return 0, errors.New("write failed") }

var _ http.ResponseWriter = errWriter{}

// errReader fails every read, standing in for a client that drops the
// connection mid-body.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

var _ io.Reader = errReader{}

var _ = Describe("decodeRequestBody", func() {
	It("decodes inspected fields and keeps the rest as raw bytes", func() {
		tools := `[{"type":"function","function":{"parameters":{"properties":{"b":{},"a":{}}}}}]`
		parsed, err := decodeRequestBody([]byte(`{"stream":true,"max_tokens":5,"tools":` + tools + `}`))
		Expect(err).ToNot(HaveOccurred())

		Expect(parsed[requestFieldStream]).To(BeTrue())
		Expect(parsed[requestFieldMaxTokens]).To(BeNumerically("==", 5))
		Expect(parsed["tools"]).To(Equal(json.RawMessage(tools)))

		out, err := json.Marshal(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(string(out)).To(ContainSubstring(`"tools":` + tools))
	})

	It("rejects non-object bodies", func() {
		_, err := decodeRequestBody([]byte(`[1,2]`))
		Expect(err).To(HaveOccurred())

		_, err = decodeRequestBody([]byte(`null`))
		Expect(err).To(HaveOccurred())
	})

	It("keeps messages raw and decodes them on use", func() {
		messages := `[{"role":"user","content":"Hi"}]`
		parsed, err := decodeRequestBody([]byte(`{"messages":` + messages + `}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(parsed[requestFieldMessages]).To(Equal(json.RawMessage(messages)))

		decoded, err := requestMessages(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(decoded).To(HaveLen(1))
	})

	It("treats a null messages field as absent", func() {
		parsed, err := decodeRequestBody([]byte(`{"messages":null}`))
		Expect(err).ToNot(HaveOccurred())

		decoded, err := requestMessages(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(decoded).To(BeNil())
	})

	It("reports a messages field that is not an array", func() {
		_, err := requestMessages(map[string]any{requestFieldMessages: json.RawMessage(`{}`)})
		Expect(err).To(HaveOccurred())

		_, err = requestMessages(map[string]any{requestFieldMessages: 5})
		Expect(err).To(HaveOccurred())
	})
})
