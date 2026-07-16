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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/test/sidecar/mock"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const eventStreamContentType = "text/event-stream"

var _ = Describe("NIXL Connector (v2)", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorNIXLV2)
	})

	startProxy := func() string {
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})

		return "http://" + testInfo.proxy.addr.String()
	}

	sendChatCompletionsRequest := func(proxyBaseAddr string) map[string]any {
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))

		var response map[string]any
		Expect(json.Unmarshal(responseBody, &response)).To(Succeed())
		return response
	}

	sendStreamingChatCompletionsRequest := func(proxyBaseAddr string) string {
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"stream": true,
				"stream_options": {"include_usage": true}
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))
		Expect(rp.Header.Get("Content-Type")).To(ContainSubstring(eventStreamContentType))
		return string(responseBody)
	}

	cachedTokensFromResponse := func(response map[string]any) float64 {
		usage, ok := response["usage"].(map[string]any)
		Expect(ok).To(BeTrue())
		details, ok := usage["prompt_tokens_details"].(map[string]any)
		Expect(ok).To(BeTrue())
		cachedTokens, ok := details["cached_tokens"].(float64)
		Expect(ok).To(BeTrue())
		return cachedTokens
	}

	It("should successfully send request to 1. prefill 2. decode with the correct fields", func() {
		proxyBaseAddr := startProxy()

		By("sending a /v1/chat/completions request with prefill header")
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prq1 := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prq1).To(HaveKey(requestFieldKVTransferParams))
		kvTransferParams, ok := prq1[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteBlockIDs, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteEngineID, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteHost, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemotePort, BeNil()))

		Expect(prq1).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
		Expect(prq1).To(HaveKeyWithValue("stream", false))
		Expect(prq1).ToNot(HaveKey("stream_options"))

		Expect(testInfo.prefillHandler.CompletionResponses).To(HaveLen(1))
		prp1 := testInfo.prefillHandler.CompletionResponses[0]
		Expect(prp1).To(HaveKey(requestFieldKVTransferParams))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		var response map[string]any
		Expect(json.Unmarshal(responseBody, &response)).To(Succeed())
		usage := response["usage"].(map[string]any)
		details := usage["prompt_tokens_details"].(map[string]any)
		Expect(details["cached_tokens"]).To(BeNumerically("==", 7))

	})

	It("should add prefiller cached tokens when decoder usage details omit cached_tokens", func() {
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{}}}`
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 7))
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
	})

	It("should create prompt token details when decoder usage omits them", func() {
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65}}`
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 7))
	})

	It("should return zero cached tokens when prefiller does not report cached tokens", func() {
		testInfo.prefillHandler.RawResponse = `{"kv_transfer_params":{"remote_block_ids":[1,2,3],"remote_engine_id":"5b5fb28f-3f30-4bdd-9a36-958d52459200","remote_host":"ahost","remote_port":4032},"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{}}}`
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":49}}}`
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 0))
	})

	It("should overwrite decoder cached tokens when prefiller reports zero cached tokens", func() {
		testInfo.prefillHandler.RawResponse = `{"kv_transfer_params":{"remote_block_ids":[1,2,3],"remote_engine_id":"5b5fb28f-3f30-4bdd-9a36-958d52459200","remote_host":"ahost","remote_port":4032},"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":0}}}`
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{"cached_tokens":49}}}`
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 0))
	})

	It("should replace cached tokens in streamed usage chunks", func() {
		testInfo.decodeHandler.RawResponseType = eventStreamContentType
		testInfo.decodeHandler.RawResponse = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":1,\"total_tokens\":65,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n"
		proxyBaseAddr := startProxy()

		responseBody := sendStreamingChatCompletionsRequest(proxyBaseAddr)

		Expect(responseBody).To(ContainSubstring(`"content":"hello"`))
		Expect(responseBody).To(ContainSubstring(`"cached_tokens":7`))
		Expect(responseBody).ToNot(ContainSubstring(`"cached_tokens":49`))
		Expect(responseBody).To(ContainSubstring("data: [DONE]"))
	})

	It("should create cached token details in streamed usage chunks that omit them", func() {
		testInfo.decodeHandler.RawResponseType = eventStreamContentType
		testInfo.decodeHandler.RawResponse = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":1,\"total_tokens\":65}}\n\ndata: [DONE]\n"
		proxyBaseAddr := startProxy()

		responseBody := sendStreamingChatCompletionsRequest(proxyBaseAddr)

		Expect(responseBody).To(ContainSubstring(`"prompt_tokens_details":{"cached_tokens":7}`))
		Expect(responseBody).To(ContainSubstring("data: [DONE]"))
	})

	It("should return zero cached tokens in streamed usage when prefiller does not report cached tokens", func() {
		testInfo.prefillHandler.RawResponse = `{"kv_transfer_params":{"remote_block_ids":[1,2,3],"remote_engine_id":"5b5fb28f-3f30-4bdd-9a36-958d52459200","remote_host":"ahost","remote_port":4032},"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{}}}`
		testInfo.decodeHandler.RawResponseType = eventStreamContentType
		testInfo.decodeHandler.RawResponse = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":1,\"total_tokens\":65,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n"
		proxyBaseAddr := startProxy()

		responseBody := sendStreamingChatCompletionsRequest(proxyBaseAddr)

		Expect(responseBody).To(ContainSubstring(`"cached_tokens":0`))
		Expect(responseBody).ToNot(ContainSubstring(`"cached_tokens":49`))
		Expect(responseBody).To(ContainSubstring("data: [DONE]"))
	})

	sendChatCompletionsRequestWithBody := func(proxyBaseAddr, body string) {
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))
	}

	It("should cap token limits in prefill and restore originals in decode", func() {
		proxyBaseAddr := startProxy()

		sendChatCompletionsRequestWithBody(proxyBaseAddr, `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 100,
				"min_tokens": 5
			}`)

		prefillReq := testInfo.prefillHandler.CompletionRequests[0]
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 100)))
		Expect(decodeReq).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 5)))
	})

	It("should cap prefill and drop the caps in decode when the request omits them", func() {
		proxyBaseAddr := startProxy()

		sendChatCompletionsRequestWithBody(proxyBaseAddr, `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				]
			}`)

		prefillReq := testInfo.prefillHandler.CompletionRequests[0]
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).ToNot(HaveKey(requestFieldMaxTokens))
		Expect(decodeReq).ToNot(HaveKey(requestFieldMinTokens))
	})

	// Messages API tests — verify /v1/messages routes through the disaggregation
	// handler with the same token-limit fields as chat completions.

	It("should successfully send messages API request to 1. prefill 2. decode with the correct fields", func() {
		proxyBaseAddr := startProxy()

		By("sending a /v1/messages request with prefill header")
		body := `{
				"model": "claude-3-5-sonnet-20241022",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+MessagesPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prq1 := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prq1).To(HaveKey(requestFieldKVTransferParams))
		kvTransferParams, ok := prq1[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))

		Expect(prq1).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
		Expect(prq1).To(HaveKeyWithValue("stream", false))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
	})

	It("should pass through messages API request when no prefill header is set", func() {
		proxyBaseAddr := startProxy()

		By("sending a /v1/messages request without prefill header")
		body := `{
				"model": "claude-3-5-sonnet-20241022",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+MessagesPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 0))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
	})

	// Responses API tests — exercise the same NIXL v2 connector with
	// /v1/responses and the max_output_tokens field instead of max_tokens.

	It("should successfully send responses API request to 1. prefill 2. decode with the correct fields", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/responses request with prefill header")
		body := `{
				"model": "gpt-4o",
				"input": "Hello, how are you?",
				"max_output_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prq1 := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prq1).To(HaveKey(requestFieldKVTransferParams))
		kvTransferParams, ok := prq1[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteBlockIDs, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteEngineID, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemoteHost, BeNil()))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldRemotePort, BeNil()))

		Expect(prq1).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 1)))
		Expect(prq1).To(HaveKeyWithValue("stream", false))
		Expect(prq1).ToNot(HaveKey("stream_options"))

		Expect(testInfo.prefillHandler.CompletionResponses).To(HaveLen(1))
		prp1 := testInfo.prefillHandler.CompletionResponses[0]
		Expect(prp1).To(HaveKey(requestFieldKVTransferParams))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should set max_output_tokens=1 in prefill and restore original value in decode", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/responses request with max_output_tokens set")
		body := `{
				"model": "gpt-4o",
				"input": "Tell me a story",
				"max_output_tokens": 100
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		By("verifying prefill request has max_output_tokens=1")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prefillReq).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 1)))

		By("verifying decode request has original max_output_tokens=100")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]

		Expect(decodeReq).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should handle responses API request without max_output_tokens", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/responses request without max_output_tokens")
		body := `{
				"model": "gpt-4o",
				"input": "Hello!"
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		By("verifying prefill request has max_output_tokens=1")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prefillReq).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 1)))

		By("verifying decode request does not have max_output_tokens since it wasn't in original request")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]

		Expect(decodeReq).ToNot(HaveKey("max_output_tokens"))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should pass through responses API request when no prefill header is set", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/responses request without prefill header")
		body := `{
				"model": "gpt-4o",
				"input": "Hello, how are you?",
				"max_output_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 0))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	DescribeTable("should retry prefill on retryable status and succeed",
		func(statusCode int) {
			testInfo.prefillHandler.FailForFirstN = 1
			testInfo.prefillHandler.FailStatusCode = statusCode
			testInfo.proxy.config.PrefillMaxRetries = 2
			testInfo.proxy.config.PrefillRetryBackoff = time.Millisecond

			proxyBaseAddr := startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()
			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			By("verifying prefill was called twice (1 fail + 1 success)")
			Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 2))

			By("verifying decode received kv_transfer_params from the successful prefill")
			Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
			decodeReq := testInfo.decodeHandler.CompletionRequests[0]
			Expect(decodeReq).To(HaveKey(requestFieldKVTransferParams))
		},
		Entry("502 Bad Gateway", http.StatusBadGateway),
		Entry("503 Service Unavailable", http.StatusServiceUnavailable),
		Entry("504 Gateway Timeout", http.StatusGatewayTimeout),
	)

	It("should return error to client when retries are disabled and prefill fails", func() {
		testInfo.prefillHandler.FailForFirstN = 1
		testInfo.prefillHandler.FailStatusCode = http.StatusBadGateway
		testInfo.proxy.config.PrefillMaxRetries = 0

		proxyBaseAddr := startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		By("verifying the error is returned to the client")
		Expect(rp.StatusCode).To(Equal(http.StatusBadGateway))

		By("verifying prefill was called only once (no retry)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		By("verifying decode was NOT called")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 0))
	})

	It("should return error to client after exhausting all retries", func() {
		testInfo.prefillHandler.FailForFirstN = 100
		testInfo.prefillHandler.FailStatusCode = http.StatusBadGateway
		testInfo.proxy.config.PrefillMaxRetries = 2
		testInfo.proxy.config.PrefillRetryBackoff = time.Millisecond

		proxyBaseAddr := startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		By("verifying the error is returned to the client")
		Expect(rp.StatusCode).To(Equal(http.StatusBadGateway))

		By("verifying prefill was called 3 times (1 initial + 2 retries)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 3))

		By("verifying decode was NOT called")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 0))
	})

	It("should not retry on non-retryable 500 and return error to client", func() {
		testInfo.prefillHandler.FailForFirstN = 1
		testInfo.prefillHandler.FailStatusCode = http.StatusInternalServerError
		testInfo.proxy.config.PrefillMaxRetries = 2
		testInfo.proxy.config.PrefillRetryBackoff = time.Millisecond

		proxyBaseAddr := startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		By("verifying the error is returned to the client")
		Expect(rp.StatusCode).To(Equal(http.StatusInternalServerError))

		By("verifying prefill was called only once (no retry despite PrefillMaxRetries=2)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		By("verifying decode was NOT called")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 0))
	})

	It("should preserve stream settings in responses API request", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/responses request with streaming enabled")
		body := `{
				"model": "gpt-4o",
				"input": "Hello!",
				"max_output_tokens": 50,
				"stream": true
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		By("verifying prefill request has stream=false")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]
		Expect(prefillReq).To(HaveKeyWithValue("stream", false))

		By("verifying decode request has stream=true restored")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).To(HaveKeyWithValue("stream", true))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	// Generate API tests — exercise the same NIXL v2 connector with
	// /inference/v1/generate, whose token limits live under sampling_params.

	startProxyAndSendGenerate := func(body string, withPrefillHeader bool) {
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})

		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+GeneratePath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		if withPrefillHeader {
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])
		}

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()
		if rp.StatusCode != 200 {
			bp, readErr := io.ReadAll(rp.Body)
			Expect(readErr).ToNot(HaveOccurred())
			Fail(string(bp))
		}
	}

	samplingParamsOf := func(req map[string]any) map[string]any {
		sp, ok := req[requestFieldSamplingParams].(map[string]any)
		Expect(ok).To(BeTrue())
		return sp
	}

	It("should successfully send generate API request to 1. prefill 2. decode with the correct fields", func() {
		startProxyAndSendGenerate(`{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 50}
			}`, true)

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prq1 := testInfo.prefillHandler.CompletionRequests[0]

		kvTransferParams, ok := prq1[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))

		Expect(samplingParamsOf(prq1)).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(samplingParamsOf(prq1)).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))
		Expect(prq1).To(HaveKeyWithValue("stream", false))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
	})

	It("should cap sampling_params token limits in prefill and restore originals in decode", func() {
		startProxyAndSendGenerate(`{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 100, "min_tokens": 5}
			}`, true)

		prefillSP := samplingParamsOf(testInfo.prefillHandler.CompletionRequests[0])
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeSP := samplingParamsOf(testInfo.decodeHandler.CompletionRequests[0])
		Expect(decodeSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 100)))
		Expect(decodeSP).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 5)))
	})

	It("should cap prefill and drop the caps in decode when sampling_params omits them", func() {
		startProxyAndSendGenerate(`{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"temperature": 0.7}
			}`, true)

		prefillSP := samplingParamsOf(testInfo.prefillHandler.CompletionRequests[0])
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeSP := samplingParamsOf(testInfo.decodeHandler.CompletionRequests[0])
		Expect(decodeSP).ToNot(HaveKey(requestFieldMaxTokens))
		Expect(decodeSP).ToNot(HaveKey(requestFieldMinTokens))
		Expect(decodeSP).To(HaveKeyWithValue("temperature", BeNumerically("==", 0.7)))
	})

	It("should cap prefill and drop synthesized sampling_params in decode when the request omits it", func() {
		startProxyAndSendGenerate(`{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4]
			}`, true)

		prefillSP := samplingParamsOf(testInfo.prefillHandler.CompletionRequests[0])
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillSP).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).ToNot(HaveKey(requestFieldSamplingParams))
	})

	It("should pass through generate API request when no prefill header is set", func() {
		startProxyAndSendGenerate(`{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 50}
			}`, false)

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 0))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
	})

	// MoRI-IO WRITE-mode regression test.
	// When --moriio-write-mode is enabled, the sidecar must populate
	// remote_host / remote_notify_port / transfer_id on the prefill leg
	// (rather than leaving them nil as the standard NIXLv2 contract does) so
	// the prefill engine's MoRIIOConnector can issue RDMA Write to decode.
	// The same transfer_id must also be carried forward into the decode request
	// so the consumer side can bind notifications to the right transfer.
	It("populates MoRI-IO WRITE-mode kv_transfer_params when MoRIIOWriteMode is enabled", func() {
		// Manual setup because sidecarConnectionTestSetup does not accept
		// MoRI-IO config knobs.  Mirrors the helper exactly otherwise.
		ctx := newTestContext()
		ctx, cancelFn := context.WithCancel(ctx)
		stoppedCh := make(chan struct{})

		decodeHandler := &mock.ChatCompletionHandler{
			Connector:       KVConnectorNIXLV2,
			Role:            mock.RoleDecode,
			MoRIIOWriteMode: true,
		}
		decodeBackend := httptest.NewServer(decodeHandler)
		DeferCleanup(decodeBackend.Close)

		prefillHandler := &mock.ChatCompletionHandler{
			Connector:       KVConnectorNIXLV2,
			Role:            mock.RolePrefill,
			MoRIIOWriteMode: true,
		}
		prefillBackend := httptest.NewServer(prefillHandler)
		DeferCleanup(prefillBackend.Close)

		decodeURL, err := url.Parse(decodeBackend.URL)
		Expect(err).ToNot(HaveOccurred())

		cfg := Config{
			Port:                   "0",
			DecoderURL:             decodeURL,
			KVConnector:            KVConnectorNIXLV2,
			MoRIIOWriteMode:        true,
			MoRIIODecodeNotifyPort: 61005,
			// r6: kv_transfer_params["remote_host"] is sourced from this
			// field (the decode pod's routable IP) instead of
			// DecoderURL.Hostname().  Set it so the assertion at
			// line 195 (remote_host == decodeURL.Hostname()) holds.
			MoRIIODecodePodIP: decodeURL.Hostname(),
		}
		proxy := NewProxy(cfg)

		By("starting the proxy")
		go func() {
			defer GinkgoRecover()
			proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := proxy.Start(ctx)
			Expect(err).ToNot(HaveOccurred())
			stoppedCh <- struct{}{}
		}()

		<-proxy.readyCh
		proxyBaseAddr := "http://" + proxy.addr.String()

		By("sending a /v1/chat/completions request")
		body := `{
			"model": "Qwen/Qwen2-0.5B",
			"messages": [{"role": "user", "content": "Hello"}],
			"max_tokens": 50
		}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		By("verifying prefill request has WRITE-mode kv_transfer_params populated")
		Expect(prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(prefillHandler.CompletionRequests).To(HaveLen(1))
		prq := prefillHandler.CompletionRequests[0]

		Expect(prq).To(HaveKey(requestFieldKVTransferParams))
		kv, ok := prq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		// New WRITE-mode fields must be non-nil and match config / request UUID.
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteHost, decodeURL.Hostname()))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteNotifyPort, BeNumerically("==", 61005)))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteDPRank, BeNumerically("==", 0)))
		Expect(kv).To(HaveKey(requestFieldTransferID))
		Expect(kv[requestFieldTransferID]).ToNot(BeEmpty())

		// Pre-existing nil fields are still nil because they are populated by
		// the prefill engine's request_finished, not the sidecar.
		Expect(kv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kv).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteEngineID, BeNil()))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteBlockIDs, BeNil()))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemotePort, BeNil()))

		transferID := kv[requestFieldTransferID]

		By("verifying decode request carries the same transfer_id")
		Expect(decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(decodeHandler.CompletionRequests).To(HaveLen(1))
		drq := decodeHandler.CompletionRequests[0]

		Expect(drq).To(HaveKey(requestFieldKVTransferParams))
		dkv, ok := drq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(dkv).To(HaveKey(requestFieldTransferID))
		Expect(dkv[requestFieldTransferID]).To(Equal(transferID))

		cancelFn()
		<-stoppedCh
	})

	// MoRI-IO Wide-EP DP-rank pinning and multi-pod fan-out coverage for the
	// 1P1D DP=8 and 2P2D DP=16 topologies, plus the flags-off legacy path.
	//
	// These tests use mocks and build Config directly - they don't need the
	// MoRIIOFeatureEnabled gate since they bypass Options.Complete().

	// 1P1D DP=8, concurrent dispatch: both legs pinned to one DP rank, decode
	// flips do_remote_prefill, remote_dp_size carries the DP world size.
	It("parallel-dispatch 1P1D DP=8 pins both legs to one DP rank and emits remote_dp_size", func() {
		env := startMoRIProxy(func(c *Config) {
			c.MoRIIOParallelDispatch = true
			c.MoRIIODPSize = 8
		})
		env.send()

		Expect(env.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(env.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		By("prefill leg carries WRITE-mode + Wide-EP fields")
		pkv := kvParams(env.prefillHandler, 0)
		Expect(pkv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(pkv).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		Expect(pkv).To(HaveKeyWithValue(requestFieldRemoteHost, env.decodePodIP))
		Expect(pkv).To(HaveKeyWithValue("remote_dp_size", BeNumerically("==", 8)))
		Expect(pkv).To(HaveKeyWithValue(requestFieldRemoteDPRankOverride, true))
		// Single-pod: the multi-pod fan-out keys must be omitted entirely.
		Expect(pkv).ToNot(HaveKey("remote_hosts"))
		Expect(pkv).ToNot(HaveKey("remote_dp_size_local"))
		pRank, ok := pkv[requestFieldRemoteDPRank].(float64)
		Expect(ok).To(BeTrue())
		Expect(pRank).To(And(BeNumerically(">=", 0), BeNumerically("<", 8)))

		By("decode leg flips do_remote_prefill and reuses the same rank + transfer_id")
		dkv := kvParams(env.decodeHandler, 0)
		Expect(dkv).To(HaveKeyWithValue(requestFieldDoRemotePrefill, true))
		Expect(dkv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, false))
		Expect(dkv).To(HaveKeyWithValue("remote_dp_size", BeNumerically("==", 8)))
		Expect(dkv[requestFieldRemoteDPRank]).To(Equal(pRank))
		Expect(dkv[requestFieldTransferID]).To(Equal(pkv[requestFieldTransferID]))
		Expect(dkv[requestFieldTransferID]).ToNot(BeEmpty())

		By("both HTTP legs share the same X-Data-Parallel-Rank header")
		ph := dpRankHeader(env.prefillHandler, 0)
		Expect(ph).To(Equal(strconv.Itoa(int(pRank))))
		Expect(dpRankHeader(env.decodeHandler, 0)).To(Equal(ph))
	})

	// 2P2D DP=16 multi-pod fan-out: each leg's remote_hosts is the opposite
	// side's pod IPs (prefill leg -> decode IPs, decode leg -> prefill IPs).
	It("parallel-dispatch 2P2D DP=EP=16 fans out remote_hosts with opposite host lists per leg", func() {
		prefillHosts := []string{"10.0.0.1", "10.0.0.2"}
		decodeHosts := []string{"10.0.1.1", "10.0.1.2"}
		env := startMoRIProxy(func(c *Config) {
			c.MoRIIOParallelDispatch = true
			c.MoRIIODPSize = 16
			c.MoRIIODPSizeLocal = 8
			c.MoRIIORemoteHosts = prefillHosts
			c.MoRIIODecodeHosts = decodeHosts
		})
		env.send()

		Expect(env.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(env.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		By("prefill leg fans out to the DECODE-side host list")
		pkv := kvParams(env.prefillHandler, 0)
		Expect(pkv["remote_hosts"]).To(Equal([]any{"10.0.1.1", "10.0.1.2"}))
		Expect(pkv).To(HaveKeyWithValue("remote_dp_size_local", BeNumerically("==", 8)))
		Expect(pkv).To(HaveKeyWithValue("remote_dp_size", BeNumerically("==", 16)))

		By("decode leg fans out to the PREFILL-side host list")
		dkv := kvParams(env.decodeHandler, 0)
		Expect(dkv["remote_hosts"]).To(Equal([]any{"10.0.0.1", "10.0.0.2"}))
		Expect(dkv).To(HaveKeyWithValue("remote_dp_size_local", BeNumerically("==", 8)))
		Expect(dkv).To(HaveKeyWithValue(requestFieldDoRemotePrefill, true))

		By("both legs share one pinned DP rank in [0,16)")
		Expect(dpRankHeader(env.prefillHandler, 0)).To(Equal(dpRankHeader(env.decodeHandler, 0)))
		pRank, ok := pkv[requestFieldRemoteDPRank].(float64)
		Expect(ok).To(BeTrue())
		Expect(pRank).To(And(BeNumerically(">=", 0), BeNumerically("<", 16)))
	})

	// The concurrent-dispatch path stages its own prefill body rather than going
	// through the serial path's token-limit handling, so it caps min_tokens too.
	It("concurrent WRITE-mode dispatch caps min_tokens in prefill and restores it in decode", func() {
		env := startMoRIProxy(func(c *Config) {
			c.MoRIIOParallelDispatch = true
		})
		env.sendBody(`{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 100,
				"min_tokens": 5
			}`)

		prefillReq := env.prefillHandler.GetCompletionRequests()[0]
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(prefillReq).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 1)))

		decodeReq := env.decodeHandler.GetCompletionRequests()[0]
		Expect(decodeReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 100)))
		Expect(decodeReq).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 5)))
	})

	// 1P1D DP=8, serial dispatch: the prefill leg sets the DP-rank header and
	// the decode leg's kv_transfer_params are backfilled with the same rank.
	It("serial WRITE-mode DP=8 pins prefill and decode HTTP legs to the same DP rank", func() {
		env := startMoRIProxy(func(c *Config) {
			c.MoRIIODPSize = 8 // ParallelDispatch stays false -> strictly-serial path
		})
		env.send()

		pkv := kvParams(env.prefillHandler, 0)
		Expect(pkv).To(HaveKeyWithValue(requestFieldRemoteDPRankOverride, true))
		pRank, ok := pkv[requestFieldRemoteDPRank].(float64)
		Expect(ok).To(BeTrue())
		Expect(pRank).To(And(BeNumerically(">=", 0), BeNumerically("<", 8)))

		ph := dpRankHeader(env.prefillHandler, 0)
		Expect(ph).To(Equal(strconv.Itoa(int(pRank))))
		Expect(dpRankHeader(env.decodeHandler, 0)).To(Equal(ph))

		By("decode kv_transfer_params (from prefill response) is backfilled with the same rank")
		dkv := kvParams(env.decodeHandler, 0)
		Expect(dkv[requestFieldRemoteDPRank]).To(Equal(pRank))
		Expect(dkv).To(HaveKeyWithValue(requestFieldRemoteDPRankOverride, true))
	})

	// Flags-off path: the sidecar must produce the legacy NIXLv2 wire shape
	// (remote_host nil, no transfer_id / remote_dp_size) with no DP-rank header.
	It("keeps the legacy NIXLv2 wire shape and omits the DP-rank header when MoRI flags are off", func() {
		proxyBaseAddr := startProxy()
		sendChatCompletionsRequest(proxyBaseAddr)

		pkv, ok := testInfo.prefillHandler.CompletionRequests[0][requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(pkv).To(HaveKeyWithValue(requestFieldRemoteHost, BeNil()))
		Expect(pkv).ToNot(HaveKey(requestFieldTransferID))
		Expect(pkv).ToNot(HaveKey("remote_dp_size"))
		Expect(pkv).ToNot(HaveKey("remote_hosts"))

		Expect(testInfo.prefillHandler.GetCompletionHeaders()[0].Get(requestHeaderDataParallelRank)).To(BeEmpty())
		Expect(testInfo.decodeHandler.GetCompletionHeaders()[0].Get(requestHeaderDataParallelRank)).To(BeEmpty())
	})
})

// moriProxyEnv bundles a running MoRI-IO proxy with its mock prefill/decode
// backends for the Wide-EP integration tests.
type moriProxyEnv struct {
	proxy          *Server
	prefillHandler *mock.ChatCompletionHandler
	decodeHandler  *mock.ChatCompletionHandler
	prefillBackend *httptest.Server
	decodeBackend  *httptest.Server
	baseAddr       string
	decodePodIP    string
}

// startMoRIProxy spins up prefill + decode mock backends and a proxy whose
// Config is seeded with MoRI-IO WRITE-mode defaults, then customised by mutate.
// Both mock backends run with MoRIIOWriteMode so the prefill-side validation
// tolerates the populated remote_host / transfer_id fields.  Cleanup is
// registered via DeferCleanup, so callers just invoke this from within an It.
func startMoRIProxy(mutate func(cfg *Config)) *moriProxyEnv {
	ctx, cancelFn := context.WithCancel(newTestContext())
	stoppedCh := make(chan struct{})
	env := &moriProxyEnv{}

	env.decodeHandler = &mock.ChatCompletionHandler{
		Connector:       KVConnectorNIXLV2,
		Role:            mock.RoleDecode,
		MoRIIOWriteMode: true,
	}
	env.decodeBackend = httptest.NewServer(env.decodeHandler)
	DeferCleanup(env.decodeBackend.Close)

	env.prefillHandler = &mock.ChatCompletionHandler{
		Connector:       KVConnectorNIXLV2,
		Role:            mock.RolePrefill,
		MoRIIOWriteMode: true,
	}
	env.prefillBackend = httptest.NewServer(env.prefillHandler)
	DeferCleanup(env.prefillBackend.Close)

	decodeURL, err := url.Parse(env.decodeBackend.URL)
	Expect(err).ToNot(HaveOccurred())
	env.decodePodIP = decodeURL.Hostname()

	cfg := Config{
		Port:                       "0",
		DecoderURL:                 decodeURL,
		KVConnector:                KVConnectorNIXLV2,
		MoRIIOWriteMode:            true,
		MoRIIODecodePodIP:          env.decodePodIP,
		MoRIIODecodeNotifyPort:     61005,
		MoRIIODecodeHandshakePort:  6301,
		MoRIIOPrefillNotifyPort:    61006,
		MoRIIOPrefillHandshakePort: 6302,
		MoRIIOTPSize:               1,
		MoRIIODPSize:               1,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	env.proxy = NewProxy(cfg)

	go func() {
		defer GinkgoRecover()
		env.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
		err := env.proxy.Start(ctx)
		Expect(err).ToNot(HaveOccurred())
		stoppedCh <- struct{}{}
	}()

	<-env.proxy.readyCh
	env.baseAddr = "http://" + env.proxy.addr.String()
	DeferCleanup(func() {
		cancelFn()
		<-stoppedCh
	})
	return env
}

// send issues a /v1/chat/completions request with the prefill header and
// asserts a 200.  Both legs have completed (wg.Wait in the concurrent path,
// sequential in the serial path) by the time this returns, so the captured
// requests / headers are safe to read afterwards.
func (env *moriProxyEnv) send() {
	env.sendBody(chatCompletionsRequestBody)
}

// sendBody is send with a caller-supplied request body.
func (env *moriProxyEnv) sendBody(body string) {
	req, err := http.NewRequest(http.MethodPost, env.baseAddr+ChatCompletionsPath, strings.NewReader(body))
	Expect(err).ToNot(HaveOccurred())
	req.Header.Add(routing.PrefillEndpointHeader, env.prefillBackend.URL[len("http://"):])

	rp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	defer rp.Body.Close()
	responseBody, _ := io.ReadAll(rp.Body) //nolint:errcheck
	Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))
}

// kvParams returns the kv_transfer_params map of the i-th request captured by h.
func kvParams(h *mock.ChatCompletionHandler, i int) map[string]any { //nolint:unparam // i kept for future multi-request tests
	reqs := h.GetCompletionRequests()
	ExpectWithOffset(1, len(reqs)).To(BeNumerically(">", i))
	kv, ok := reqs[i][requestFieldKVTransferParams].(map[string]any)
	ExpectWithOffset(1, ok).To(BeTrue())
	return kv
}

// dpRankHeader returns the X-Data-Parallel-Rank header of the i-th request
// captured by h (empty string when unset).
func dpRankHeader(h *mock.ChatCompletionHandler, i int) string { //nolint:unparam // i kept for future multi-request tests
	hdrs := h.GetCompletionHeaders()
	ExpectWithOffset(1, len(hdrs)).To(BeNumerically(">", i))
	return hdrs[i].Get(requestHeaderDataParallelRank)
}
