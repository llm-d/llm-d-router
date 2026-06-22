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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("Mooncake Connector", func() {

	var testInfo *sidecarTestInfo
	var bootstrapServer *httptest.Server
	var bootstrapResponse string

	BeforeEach(func() {
		// single dp rank bootstrap response
		bootstrapResponse = `{"0": {"engine_id": "test-engine-abc123", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}}}`

		// start a mock bootstrap server that returns map
		bootstrapServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(bootstrapResponse)) //nolint:all
		}))
		DeferCleanup(bootstrapServer.Close)

		// Extract the bootstrap server port
		bootstrapURL, err := url.Parse(bootstrapServer.URL)
		Expect(err).ToNot(HaveOccurred())
		var bootstrapPort int
		_, err = fmt.Sscanf(bootstrapURL.Port(), "%d", &bootstrapPort)
		Expect(err).ToNot(HaveOccurred())

		testInfo = sidecarConnectionTestSetup(KVConnectorMooncake)
		testInfo.proxy.config.MooncakeBootstrapPort = bootstrapPort
	})

	It("should send concurrent requests with correct mooncake kv_transfer_params", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"max_completion_tokens": 100
			}`
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		// Use the bootstrap server's host as the prefill host so engine_id discovery works
		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Wait for async prefill request to be fully recorded
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Validate prefill request
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		preq := prefillReqs[0]

		// Prefill should have kv_transfer_params with do_remote_decode=true
		Expect(preq).To(HaveKey(requestFieldKVTransferParams))
		prefillKVParams, ok := preq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams[requestFieldDoRemoteDecode]).To(BeTrue())
		Expect(prefillKVParams[requestFieldDoRemotePrefill]).To(BeFalse())
		Expect(prefillKVParams[requestFieldTransferID]).ToNot(BeEmpty())

		// Prefill should have max_tokens=1, max_completion_tokens=1 and stream=false
		Expect(preq[requestFieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).To(HaveKeyWithValue(requestFieldMaxCompletionTokens, BeNumerically("==", 1)))
		Expect(preq[requestFieldStream]).To(BeFalse())

		// Validate decode request
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		dreq := decodeReqs[0]

		// Decode should have kv_transfer_params with do_remote_prefill=true
		Expect(dreq).To(HaveKey(requestFieldKVTransferParams))
		decodeKVParams, ok := dreq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams[requestFieldDoRemotePrefill]).To(BeTrue())
		Expect(decodeKVParams[requestFieldDoRemoteDecode]).To(BeFalse())
		Expect(decodeKVParams[requestFieldTransferID]).To(HavePrefix("xfer-"))
		Expect(decodeKVParams[requestFieldRemoteEngineID]).To(Equal("test-engine-abc123"))
		Expect(decodeKVParams[requestFieldRemoteBootstrapAddr]).To(ContainSubstring(fmt.Sprintf(":%d", testInfo.proxy.config.MooncakeBootstrapPort)))

		// Transfer IDs must match between prefill and decode
		Expect(decodeKVParams[requestFieldTransferID]).To(Equal(prefillKVParams[requestFieldTransferID]))

		// Prefill must be pinned to the dp rank whose engine_id is sent to decode
		prefillHeaders := testInfo.prefillHandler.GetCompletionHeaders()
		Expect(prefillHeaders).To(HaveLen(1))
		Expect(prefillHeaders[0].Get(mooncakeDataParallelRankHeader)).To(Equal("0"))

		// Decode should preserve original max_tokens and max_completion_tokens from request
		Expect(dreq[requestFieldMaxTokens]).To(BeNumerically("==", 50))
		Expect(dreq).To(HaveKeyWithValue(requestFieldMaxCompletionTokens, BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not panic when prefill response is slower than decode response", func() {
		// Stop previously injected servers
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		var prefillFinished atomic.Bool

		// create a delay on prefill
		slowPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.prefillHandler.ServeHTTP(w, r)
			time.Sleep(300 * time.Millisecond)
			prefillFinished.Store(true)
		})
		testInfo.prefillBackend = httptest.NewServer(slowPrefill)

		fastDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.decodeHandler.ServeHTTP(w, r)
		})
		testInfo.decodeBackend = httptest.NewServer(fastDecode)
		testInfo.decodeURL, _ = url.Parse(testInfo.decodeBackend.URL)

		var bootstrapPort int
		bURL, _ := url.Parse(bootstrapServer.URL)
		_, _ = fmt.Sscanf(bURL.Port(), "%d", &bootstrapPort)

		cfg := Config{
			Port:                  "0",
			DecoderURL:            testInfo.decodeURL,
			KVConnector:           KVConnectorMooncake,
			MooncakeBootstrapPort: bootstrapPort,
		}
		testInfo.proxy = NewProxy(cfg)

		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Eventually(prefillFinished.Load).Should(BeTrue()) // use default timeout from gomega
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should pin prefill to the selected dp rank and send its engine_id to decode", func() {
		engineByRank := map[string]string{"0": "engine-dp0", "1": "engine-dp1"}
		bootstrapResponse = `{
			"0": {"engine_id": "engine-dp0", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}},
			"1": {"engine_id": "engine-dp1", "worker_addr": {"0": {"0": "10.0.0.2:5000"}}}
		}`

		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Wait for async prefill request to be fully recorded
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// header includes rank_id
		prefillHeaders := testInfo.prefillHandler.GetCompletionHeaders()
		Expect(prefillHeaders).To(HaveLen(1))
		pinnedRank := prefillHeaders[0].Get(mooncakeDataParallelRankHeader)
		Expect(engineByRank).To(HaveKey(pinnedRank))

		// decode payload body has rank_id's engine_id
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		decodeKVParams, ok := decodeReqs[0][requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams[requestFieldRemoteEngineID]).To(Equal(engineByRank[pinnedRank]))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})
