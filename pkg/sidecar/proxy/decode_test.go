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
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// chunkedTestInfo holds a running proxy backed by a controlled decode backend.
type chunkedTestInfo struct {
	proxy     *Server
	backend   *httptest.Server
	addr      string // "http://host:port" of the proxy
	cancelFn  context.CancelFunc
	stoppedCh chan struct{}
}

// newChunkedTestSetup starts a proxy with chunked decode enabled. The backend
// serves decodeResponses in order; any extra request gets a 500.
func newChunkedTestSetup(chunkSize int, decodeResponses []string) *chunkedTestInfo {
	var reqIdx int
	return newChunkedTestSetupWithHandler(chunkSize, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if reqIdx >= len(decodeResponses) {
			http.Error(w, "unexpected request", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, decodeResponses[reqIdx]) //nolint:errcheck
		reqIdx++
	}))
}

// newChunkedTestSetupWithHandler is like newChunkedTestSetup but accepts a
// custom backend handler for tests that need to inspect or alter requests.
func newChunkedTestSetupWithHandler(chunkSize int, handler http.Handler) *chunkedTestInfo {
	backend := httptest.NewServer(handler)
	DeferCleanup(backend.Close)

	decoderURL, _ := url.Parse(backend.URL)
	cfg := Config{
		Port:            "0",
		DecoderURL:      decoderURL,
		KVConnector:     constants.KVConnectorNIXLV2,
		DecodeChunkSize: chunkSize,
	}
	proxy := NewProxy(cfg)

	ctx := newTestContext()
	ctx, cancelFn := context.WithCancel(ctx)
	stoppedCh := make(chan struct{})

	go func() {
		defer GinkgoRecover()
		_ = proxy.Start(ctx)
		stoppedCh <- struct{}{}
	}()
	<-proxy.readyCh

	return &chunkedTestInfo{
		proxy:     proxy,
		backend:   backend,
		addr:      "http://" + proxy.addr.String(),
		cancelFn:  cancelFn,
		stoppedCh: stoppedCh,
	}
}

func (ti *chunkedTestInfo) stop() {
	ti.cancelFn()
	<-ti.stoppedCh
}

// chatResponse builds a minimal non-streaming chat completion JSON response.
func chatResponse(content, finishReason string, promptTokens, completionTokens int) string {
	resp := map[string]any{
		"id":      "test-id",
		"object":  "chat.completion",
		"model":   "test-model",
		"created": 1234567890,
		"choices": []any{
			map[string]any{
				"index":         0,
				"finish_reason": finishReason,
				"message":       map[string]any{"role": "assistant", "content": content},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// doPost sends a POST request to the proxy and returns the response.
func doPost(addr, body string) *http.Response {
	req, err := http.NewRequest(http.MethodPost, addr+reqcommon.PathChatCompletions, strings.NewReader(body))
	Expect(err).ToNot(HaveOccurred())
	resp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	return resp
}

var _ = Describe("Chunked Decode", func() {
	DescribeTable("keeps one accumulated assistant message across chunks",
		func(streaming, continueFinal bool, finalReason string) {
			// Unsorted keys and the extra name field expose any re-encoding of client messages.
			userMessage := `{"role":"user","content":[{"type":"text","text":"Hi","z":1,"a":2}]}`
			history := `{"role":"assistant","content":"Earlier answer"}`
			prefixMessage := `{"role":"assistant","content":"Prefix: ","name":"writer"}`
			messages := []json.RawMessage{json.RawMessage(history), json.RawMessage(userMessage), json.RawMessage(prefixMessage)}
			requestBody := map[string]any{
				reqcommon.FieldMessages:             messages,
				reqcommon.FieldMaxTokens:            20,
				reqcommon.FieldStream:               streaming,
				reqcommon.FieldContinueFinalMessage: continueFinal,
				reqcommon.FieldAddGenerationPrompt:  !continueFinal,
				reqcommon.FieldKVTransferParams:     map[string]any{"test": true},
			}
			raw, err := json.Marshal(requestBody)
			Expect(err).ToNot(HaveOccurred())
			var requests []map[string]json.RawMessage
			texts := []string{"one", "two", "", "three"}
			server := NewProxy(Config{DecodeChunkSize: 5})
			server.logger = logr.Discard()
			server.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				Expect(json.NewDecoder(r.Body).Decode(&body)).To(Succeed())
				requests = append(requests, body)
				Expect(len(requests)).To(BeNumerically("<=", len(texts)))
				reason := finishReasonLength
				if len(requests) == len(texts) {
					reason = finalReason
				}
				_, err := io.WriteString(w, chatResponse(texts[len(requests)-1], reason, 8, 5))
				Expect(err).ToNot(HaveOccurred())
			})
			response := httptest.NewRecorder()
			server.runChunkedDecode(response, httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(string(raw))))
			Expect(response.Code).To(Equal(http.StatusOK))
			Expect(requests).To(HaveLen(4))
			for index, request := range requests {
				var gotMessages []json.RawMessage
				Expect(json.Unmarshal(request[reqcommon.FieldMessages], &gotMessages)).To(Succeed())
				Expect(gotMessages[:2]).To(Equal(messages[:2]))
				if index == 0 {
					Expect(gotMessages).To(Equal(messages))
					continue
				}
				Expect(string(request[reqcommon.FieldContinueFinalMessage])).To(Equal("true"))
				Expect(string(request[reqcommon.FieldAddGenerationPrompt])).To(Equal("false"))
				Expect(request).ToNot(HaveKey(reqcommon.FieldKVTransferParams))
				wantText := strings.Join(texts[:index], "")
				if continueFinal {
					Expect(gotMessages).To(HaveLen(3))
					wantText = "Prefix: " + wantText
				} else {
					Expect(gotMessages).To(HaveLen(4))
					Expect(gotMessages[2]).To(Equal(messages[2]))
				}
				var last map[string]any
				Expect(json.Unmarshal(gotMessages[len(gotMessages)-1], &last)).To(Succeed())
				Expect(last).To(HaveKeyWithValue(reqcommon.FieldRole, roleAssistant))
				Expect(last).To(HaveKeyWithValue(reqcommon.FieldContent, wantText))
				if continueFinal {
					Expect(last).To(HaveKeyWithValue("name", "writer"))
				}
			}
			if streaming {
				var emitted []string
				scanner := bufio.NewScanner(response.Body)
				for scanner.Scan() {
					data, ok := strings.CutPrefix(scanner.Text(), reqcommon.SSEDataPrefix)
					if !ok || scanner.Text() == reqcommon.SSEDone {
						continue
					}
					var event map[string]any
					Expect(json.Unmarshal([]byte(data), &event)).To(Succeed())
					if choice := firstChoice(event); choice != nil {
						emitted = append(emitted, choice[responseFieldDelta].(map[string]any)[reqcommon.FieldContent].(string))
					}
				}
				Expect(scanner.Err()).ToNot(HaveOccurred())
				Expect(emitted).To(Equal(texts))
			} else {
				var result map[string]any
				Expect(json.Unmarshal(response.Body.Bytes(), &result)).To(Succeed())
				Expect(extractChoiceText(firstChoice(result))).To(Equal(strings.Join(texts, "")))
			}
		},
		Entry("non-streaming new answer", false, false, "stop"),
		Entry("streaming new answer", true, false, "stop"),
		Entry("non-streaming existing prefix", false, true, "stop"),
		Entry("streaming existing prefix", true, true, "stop"),
		Entry("non-streaming exhausted budget", false, false, finishReasonLength),
	)

	DescribeTable("forwards unsupported continuation inputs unchanged",
		func(messages string) {
			raw := `{"messages":` + messages + `,"continue_final_message":true,"max_tokens":20,"stream":true}`
			server := NewProxy(Config{DecodeChunkSize: 5})
			server.logger = logr.Discard()
			calls := 0
			server.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				body, err := io.ReadAll(r.Body)
				Expect(err).ToNot(HaveOccurred())
				Expect(string(body)).To(Equal(raw))
				Expect(w.Header().Get("Content-Type")).To(BeEmpty())
				w.WriteHeader(http.StatusBadRequest)
			})
			response := httptest.NewRecorder()
			server.runChunkedDecode(response, httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(raw)))
			Expect(calls).To(Equal(1))
			Expect(response.Code).To(Equal(http.StatusBadRequest))
		},
		Entry("empty messages", `[]`),
		Entry("malformed messages", `{}`),
		Entry("missing role", `[{"content":"prefix"}]`),
		Entry("user message", `[{"role":"user","content":"prefix"}]`),
		Entry("structured content", `[{"role":"assistant","content":[{"type":"text","text":"prefix"}]}]`),
		Entry("null content", `[{"role":"assistant","content":null}]`),
	)

	DescribeTable("handles an empty first chunk",
		func(tokens int, wantCalls int) {
			server := NewProxy(Config{DecodeChunkSize: 5})
			server.logger = logr.Discard()
			calls := 0
			server.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if calls == 1 {
					_, err := io.WriteString(w, chatResponse("", finishReasonLength, 8, tokens))
					Expect(err).ToNot(HaveOccurred())
					return
				}
				var request map[string]any
				Expect(json.NewDecoder(r.Body).Decode(&request)).To(Succeed())
				Expect(request[reqcommon.FieldMessages]).To(Equal([]any{
					map[string]any{"role": "user", "content": "Hi"},
					map[string]any{"role": "assistant", "content": ""},
				}))
				_, err := io.WriteString(w, chatResponse("done", "stop", 8, 1))
				Expect(err).ToNot(HaveOccurred())
			})
			body, err := decodeRequestBody([]byte(`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":20}`))
			Expect(err).ToNot(HaveOccurred())
			original, err := json.Marshal(body)
			Expect(err).ToNot(HaveOccurred())
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(string(original)))
			server.runChunkedDecodeFromMap(response, request, body)
			Expect(calls).To(Equal(wantCalls))
			Expect(response.Code).To(Equal(http.StatusOK))
			after, err := json.Marshal(body)
			Expect(err).ToNot(HaveOccurred())
			Expect(after).To(Equal(original))
		},
		Entry("continues an empty assistant message when tokens were consumed", 5, 2),
		Entry("stops when neither tokens nor text were produced", 0, 1),
	)

	Describe("non-streaming", func() {

		It("falls back to regular decode when budget fits in one chunk", func() {
			// chunk size 512, max_tokens 10 → single pass, no chunking
			ti := newChunkedTestSetup(512, []string{
				chatResponse("hello world", "stop", 5, 10),
			})
			defer ti.stop()

			resp := doPost(ti.addr,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			var body map[string]any
			Expect(json.NewDecoder(resp.Body).Decode(&body)).To(Succeed())
			content := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
			Expect(content).To(Equal("hello world"))
		})

		It("reassembles two chunks into a single response with correct usage", func() {
			// chunk size 5, max_tokens 10 → two chunks
			ti := newChunkedTestSetup(5, []string{
				chatResponse("hello ", "length", 8, 5),
				chatResponse("world", "stop", 9, 5),
			})
			defer ti.stop()

			resp := doPost(ti.addr,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			var body map[string]any
			Expect(json.NewDecoder(resp.Body).Decode(&body)).To(Succeed())

			content := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
			Expect(content).To(Equal("hello world"))

			usage := body["usage"].(map[string]any)
			promptTokens, _ := toInt(usage["prompt_tokens"])
			completionTokens, _ := toInt(usage["completion_tokens"])
			totalTokens, _ := toInt(usage["total_tokens"])
			Expect(promptTokens).To(Equal(8))
			Expect(completionTokens).To(Equal(10))
			Expect(totalTokens).To(Equal(18))
		})

		It("stops early on terminal finish reason before budget is exhausted", func() {
			// chunk size 5, max_tokens 20 → first chunk returns "stop"
			ti := newChunkedTestSetup(5, []string{
				chatResponse("done", "stop", 5, 3),
			})
			defer ti.stop()

			resp := doPost(ti.addr,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":20}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			var body map[string]any
			Expect(json.NewDecoder(resp.Body).Decode(&body)).To(Succeed())
			content := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
			Expect(content).To(Equal("done"))
		})

		It("propagates decode backend error to client", func() {
			ti := newChunkedTestSetupWithHandler(5, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
				fmt.Fprint(w, `{"error":"backend down"}`) //nolint:errcheck
			}))
			defer ti.stop()

			resp := doPost(ti.addr,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}`)
			Expect(resp.StatusCode).To(Equal(http.StatusBadGateway))
		})
	})

	Describe("streaming", func() {

		It("emits one SSE event per chunk and terminates with [DONE]", func() {
			ti := newChunkedTestSetup(5, []string{
				chatResponse("hello ", "length", 5, 5),
				chatResponse("world", "stop", 9, 5),
			})
			defer ti.stop()

			resp := doPost(ti.addr,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"stream":true}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			Expect(resp.Header.Get("Content-Type")).To(Equal("text/event-stream"))

			var events []string
			scanner := bufio.NewScanner(resp.Body)
			for scanner.Scan() {
				if line := scanner.Text(); strings.HasPrefix(line, "data: ") {
					events = append(events, line)
				}
			}

			// Two chunk data events + usage event + [DONE]
			Expect(events).To(HaveLen(4))
			Expect(events[3]).To(Equal(reqcommon.SSEDone))

			var first map[string]any
			Expect(json.Unmarshal([]byte(strings.TrimPrefix(events[0], reqcommon.SSEDataPrefix)), &first)).To(Succeed())
			delta := first["choices"].([]any)[0].(map[string]any)[responseFieldDelta].(map[string]any)
			Expect(delta[reqcommon.FieldContent]).To(Equal("hello "))

			// Verify cumulative usage in the final usage event.
			var usageEvent map[string]any
			Expect(json.Unmarshal([]byte(strings.TrimPrefix(events[2], reqcommon.SSEDataPrefix)), &usageEvent)).To(Succeed())
			usage := usageEvent["usage"].(map[string]any)
			promptTokens, _ := toInt(usage["prompt_tokens"])
			completionTokens, _ := toInt(usage["completion_tokens"])
			totalTokens, _ := toInt(usage["total_tokens"])
			Expect(promptTokens).To(Equal(5))
			Expect(completionTokens).To(Equal(10))
			Expect(totalTokens).To(Equal(15))
		})
	})

	Describe("helper functions", func() {

		It("resolveMaxTokens prefers max_completion_tokens over max_tokens", func() {
			req := map[string]any{reqcommon.FieldMaxTokens: float64(50), reqcommon.FieldMaxCompletionTokens: float64(100)}
			Expect(resolveMaxTokens(req)).To(Equal(100))
		})

		It("resolveMaxTokens returns -1 when neither field is set", func() {
			Expect(resolveMaxTokens(map[string]any{})).To(Equal(-1))
		})

		It("remainingTokens returns -1 for unlimited budget", func() {
			Expect(remainingTokens(-1, 100)).To(Equal(-1))
		})

		It("remainingTokens returns 0 when budget is exhausted", func() {
			Expect(remainingTokens(10, 10)).To(Equal(0))
		})
	})
})
