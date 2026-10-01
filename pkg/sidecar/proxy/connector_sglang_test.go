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
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("SGLang Connector", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		// Mock testing setup using the SGLang connector mode
		testInfo = sidecarConnectionTestSetup(KVConnectorSGLang)
	})

	It("should successfully send concurrent requests to prefill and decode with bootstrap info", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := localProxyBaseAddr(testInfo)

		By("sending a /v1/chat/completions request with prefill header")
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Because SGLang connector sends requests concurrently (prefill in goroutine),
		// wait until the prefill handler has finished processing before reading its
		// state. The wait polls the recorded requests rather than RequestCount: the
		// mock counts a request on entry and records it after reading the body, so
		// the counter reaches 1 while the slice is still empty.
		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))

		// Validate prefill request
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		prq1 := prefillReqs[0]

		// Validate decode request
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		drq1 := decodeReqs[0]

		// Bootstrap validations for prefill
		Expect(prq1).To(HaveKey(requestFieldBootstrapHost))
		Expect(prq1).To(HaveKey(requestFieldBootstrapPort))
		Expect(prq1).To(HaveKey(requestFieldBootstrapRoom))

		expectedHost := extractHost(prefillHostPort)
		Expect(prq1[requestFieldBootstrapHost]).To(Equal(expectedHost))
		Expect(prq1[requestFieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(prq1[requestFieldBootstrapRoom]).ToNot(BeNil())

		// Bootstrap validations for decode
		Expect(drq1).To(HaveKey(requestFieldBootstrapHost))
		Expect(drq1).To(HaveKey(requestFieldBootstrapPort))
		Expect(drq1).To(HaveKey(requestFieldBootstrapRoom))

		Expect(drq1[requestFieldBootstrapHost]).To(Equal(expectedHost))
		Expect(drq1[requestFieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(drq1[requestFieldBootstrapRoom]).To(Equal(prq1[requestFieldBootstrapRoom])) // Room ID must match

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not panic when prefill response is slower than decode response", func() {
		// Stop previously injected servers
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		var prefillFinished atomic.Bool

		slowPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.prefillHandler.ServeHTTP(w, r)
			time.Sleep(300 * time.Millisecond) // Simulated load delay on KV Cache
			prefillFinished.Store(true)
		})
		testInfo.prefillBackend = httptest.NewServer(slowPrefill)

		fastDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.decodeHandler.ServeHTTP(w, r)
		})
		testInfo.decodeBackend = httptest.NewServer(fastDecode)
		testInfo.decodeURL, _ = url.Parse(testInfo.decodeBackend.URL)

		// Re-initialize proxy to fetch the new mock addresses
		cfg := Config{
			Port:        "0",
			DecoderURL:  testInfo.decodeURL,
			KVConnector: KVConnectorSGLang,
		}
		testInfo.proxy = NewProxy(cfg)

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := localProxyBaseAddr(testInfo)

		body := `{"model": "Qwen", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}`
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		// Submit request. This will complete as soon as fastDecode completes.
		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		// The original panicking goroutine takes 300ms total. Give it time to attempt finishing up!
		time.Sleep(500 * time.Millisecond)

		Expect(prefillFinished.Load()).To(BeTrue())
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should add bootstrap info to paired speculative prefill warmups", func() {
		testInfo.proxy.config.EnableSpeculativePrefill = true
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"Hello from decode"}}]}`

		testInfo.startProxy()
		proxyBaseAddr := localProxyBaseAddr(testInfo)

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)
		req.Header.Set(routing.SpeculativePrefillHeader, "true")

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck
		if resp.StatusCode != http.StatusOK {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(2))
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		warmupReq := prefillReqs[1]

		Expect(warmupReq).To(HaveKeyWithValue(requestFieldBootstrapHost, extractHost(prefillHostPort)))
		Expect(warmupReq).To(HaveKeyWithValue(requestFieldBootstrapPort, BeNumerically("==", sglangBootstrapPort)))
		Expect(warmupReq).To(HaveKey(requestFieldBootstrapRoom))
		Expect(warmupReq).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
		Expect(warmupReq).To(HaveKeyWithValue(requestFieldMaxCompletionTokens, BeNumerically("==", 1)))
		Expect(warmupReq).To(HaveKeyWithValue(requestFieldStream, false))

		messages, ok := warmupReq[requestFieldMessages].([]any)
		Expect(ok).To(BeTrue())
		Expect(messages).To(HaveLen(3))
		assistantMsg, ok := messages[1].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(assistantMsg).To(HaveKeyWithValue(requestFieldRole, roleAssistant))
		Expect(assistantMsg).To(HaveKeyWithValue(requestFieldContent, "Hello from decode"))
		placeholderMsg, ok := messages[2].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(placeholderMsg).To(HaveKeyWithValue(requestFieldRole, "user"))
		Expect(placeholderMsg).To(HaveKeyWithValue(requestFieldContent, " "))

		Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(2))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		warmupDecodeReq := decodeReqs[1]
		Expect(warmupDecodeReq).To(HaveKeyWithValue(requestFieldBootstrapHost, extractHost(prefillHostPort)))
		Expect(warmupDecodeReq).To(HaveKeyWithValue(requestFieldBootstrapRoom, warmupReq[requestFieldBootstrapRoom]))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should skip speculative prefill warmups when concurrency limit is occupied", func() {
		testInfo.proxy.config.EnableSpeculativePrefill = true
		specPrefillAdmission <- struct{}{}
		defer func() { <-specPrefillAdmission }()
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"Hello from decode"}}]}`

		testInfo.startProxy()
		proxyBaseAddr := localProxyBaseAddr(testInfo)

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)
		req.Header.Set(routing.SpeculativePrefillHeader, "true")

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close() //nolint:errcheck
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))
		Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(1))
		Consistently(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }, 100*time.Millisecond, 20*time.Millisecond).Should(Equal(1))
		Consistently(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }, 100*time.Millisecond, 20*time.Millisecond).Should(Equal(1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})

func localProxyBaseAddr(testInfo *sidecarTestInfo) string {
	return "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(testInfo.proxy.addr.(*net.TCPAddr).Port))
}
