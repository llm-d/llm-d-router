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
	"strings"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	chatCompletionsRequestBody = `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`
	eventStreamContentType = "text/event-stream"
)

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
})
