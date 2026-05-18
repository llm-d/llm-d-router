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
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
)

var _ = Describe("NIXL Connector (v2)", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorNIXLV2)
	})

	It("should successfully send request to 1. prefill 2. decode with the correct fields", func() {
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

		By("sending a /v1/chat/completions request with prefill header")
		//nolint:goconst
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
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

		testInfo.cancelFn()
		<-testInfo.stoppedCh
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

	It("should retry prefill on 502 and succeed when retries are enabled", func() {
		By("setting up proxy with PrefillMaxRetries=2 and a prefill that fails once")
		testInfo.prefillHandler.FailForFirstN = 1
		testInfo.prefillHandler.FailStatusCode = http.StatusBadGateway
		testInfo.proxy.config.PrefillMaxRetries = 2
		testInfo.proxy.config.PrefillRetryBackoff = time.Millisecond

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		//nolint:goconst
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [{"role": "user", "content": "Hello"}],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		By("verifying prefill was called twice (1 fail + 1 success)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 2))

		By("verifying decode received kv_transfer_params from the successful prefill")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).To(HaveKey(requestFieldKVTransferParams))
		kvParams, ok := decodeReq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(kvParams).To(HaveKey("remote_block_ids"))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should fallback to decode without kv_transfer_params when retries are disabled", func() {
		By("setting up proxy with PrefillMaxRetries=0 and a prefill that fails")
		testInfo.prefillHandler.FailForFirstN = 1
		testInfo.prefillHandler.FailStatusCode = http.StatusBadGateway
		testInfo.proxy.config.PrefillMaxRetries = 0

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [{"role": "user", "content": "Hello"}],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		By("verifying prefill was called only once (no retry)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		By("verifying decode was called via fallback (without kv_transfer_params)")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).ToNot(HaveKey(requestFieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should fallback to decode after exhausting all retries", func() {
		By("setting up proxy with PrefillMaxRetries=2 and a prefill that always fails")
		testInfo.prefillHandler.FailForFirstN = 100
		testInfo.prefillHandler.FailStatusCode = http.StatusBadGateway
		testInfo.proxy.config.PrefillMaxRetries = 2
		testInfo.proxy.config.PrefillRetryBackoff = time.Millisecond

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [{"role": "user", "content": "Hello"}],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		By("verifying prefill was called 3 times (1 initial + 2 retries)")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 3))

		By("verifying decode was called via fallback")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).ToNot(HaveKey(requestFieldKVTransferParams))

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
