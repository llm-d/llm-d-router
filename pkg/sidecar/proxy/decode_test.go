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

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
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
		KVConnector:     KVConnectorNIXLV2,
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
func doPost(addr, path, body string) *http.Response {
	req, err := http.NewRequest(http.MethodPost, addr+path, strings.NewReader(body))
	Expect(err).ToNot(HaveOccurred())
	resp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	return resp
}

var _ = Describe("Chunked Decode", func() {

	Describe("non-streaming", func() {

		It("falls back to regular decode when budget fits in one chunk", func() {
			// chunk size 512, max_tokens 10 → single pass, no chunking
			ti := newChunkedTestSetup(512, []string{
				chatResponse("hello world", "stop", 5, 10),
			})
			defer ti.stop()

			resp := doPost(ti.addr, ChatCompletionsPath,
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

			resp := doPost(ti.addr, ChatCompletionsPath,
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

			resp := doPost(ti.addr, ChatCompletionsPath,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":20}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			var body map[string]any
			Expect(json.NewDecoder(resp.Body).Decode(&body)).To(Succeed())
			content := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"]
			Expect(content).To(Equal("done"))
		})

		It("appends assistant message to messages on second chunk", func() {
			var secondReqBody map[string]any
			var reqIdx int
			responses := []string{
				chatResponse("hello ", "length", 5, 5),
				chatResponse("world", "stop", 9, 5),
			}

			ti := newChunkedTestSetupWithHandler(5, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				if reqIdx == 1 {
					json.Unmarshal(b, &secondReqBody) //nolint:errcheck
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, responses[reqIdx]) //nolint:errcheck
				reqIdx++
			}))
			defer ti.stop()

			resp := doPost(ti.addr, ChatCompletionsPath,
				`{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}`)
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			Expect(secondReqBody).ToNot(BeNil())
			msgs := secondReqBody[requestFieldMessages].([]any)
			Expect(msgs).To(HaveLen(2))
			lastMsg := msgs[1].(map[string]any)
			Expect(lastMsg[requestFieldRole]).To(Equal("assistant"))
			Expect(lastMsg[requestFieldContent]).To(Equal("hello "))
		})

		It("propagates decode backend error to client", func() {
			ti := newChunkedTestSetupWithHandler(5, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
				fmt.Fprint(w, `{"error":"backend down"}`) //nolint:errcheck
			}))
			defer ti.stop()

			resp := doPost(ti.addr, ChatCompletionsPath,
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

			resp := doPost(ti.addr, ChatCompletionsPath,
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

			// Two chunk data events + [DONE]
			Expect(events).To(HaveLen(3))
			Expect(events[2]).To(Equal(sseDone))

			var first map[string]any
			Expect(json.Unmarshal([]byte(strings.TrimPrefix(events[0], sseDataPrefix)), &first)).To(Succeed())
			delta := first["choices"].([]any)[0].(map[string]any)[responseFieldDelta].(map[string]any)
			Expect(delta[requestFieldContent]).To(Equal("hello "))
		})
	})

	Describe("helper functions", func() {

		It("resolveMaxTokens prefers max_completion_tokens over max_tokens", func() {
			req := map[string]any{requestFieldMaxTokens: float64(50), requestFieldMaxCompletionTokens: float64(100)}
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

		It("appendChunkToRequest appends assistant message to chat messages", func() {
			req := map[string]any{
				requestFieldMessages: []any{map[string]any{requestFieldRole: "user", requestFieldContent: "Hi"}},
			}
			appendChunkToRequest(req, "hello")
			msgs := req[requestFieldMessages].([]any)
			Expect(msgs).To(HaveLen(2))
			last := msgs[1].(map[string]any)
			Expect(last[requestFieldRole]).To(Equal("assistant"))
			Expect(last[requestFieldContent]).To(Equal("hello"))
		})

		It("appendChunkToRequest appends to legacy prompt with prefix", func() {
			req := map[string]any{requestFieldPrompt: "Tell me a story"}
			appendChunkToRequest(req, "Once upon a time")
			Expect(req[requestFieldPrompt]).To(Equal("Tell me a story" + legacyChunkPrefix + "Once upon a time"))
		})

		It("appendChunkToRequest is a no-op for empty text", func() {
			req := map[string]any{requestFieldMessages: []any{}}
			appendChunkToRequest(req, "")
			Expect(req[requestFieldMessages].([]any)).To(BeEmpty())
		})
	})
})
