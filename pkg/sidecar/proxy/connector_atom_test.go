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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("ATOM Connector", func() {
	var (
		testInfo *atomTestInfo
	)

	BeforeEach(func() {
		testInfo = atomTestSetup()
	})

	AfterEach(func() {
		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	Describe("handleATOM", func() {
		It("should inject data_parallel_rank into request body for prefill", func() {
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))
			Expect(testInfo.prefillHandler.requestCount).To(BeNumerically("==", 1))

			prefillReq := testInfo.prefillHandler.lastRequest
			Expect(prefillReq).To(HaveKey(atomFieldDataParallelRank))
			dpRank := prefillReq[atomFieldDataParallelRank]
			Expect(dpRank).To(BeNumerically(">=", 0))
		})

		It("should inject same data_parallel_rank for decode as prefill", func() {
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))
			Expect(testInfo.decodeHandler.requestCount).To(BeNumerically("==", 1))

			prefillReq := testInfo.prefillHandler.lastRequest
			decodeReq := testInfo.decodeHandler.lastRequest

			prefillDPRank := prefillReq[atomFieldDataParallelRank]
			decodeDPRank := decodeReq[atomFieldDataParallelRank]

			Expect(prefillDPRank).To(Equal(decodeDPRank), "prefill and decode should have same DP rank")
		})

		It("should relay kv_transfer_params from prefill to decode", func() {
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			decodeReq := testInfo.decodeHandler.lastRequest
			kvParams, ok := decodeReq[requestFieldKVTransferParams].(map[string]any)
			Expect(ok).To(BeTrue())

			Expect(kvParams).To(HaveKey(requestFieldDoRemotePrefill))
			Expect(kvParams[requestFieldDoRemotePrefill]).To(BeTrue())
			Expect(kvParams).To(HaveKey("remote_block_ids"))
			Expect(kvParams).To(HaveKey("remote_engine_id"))
		})

		It("should add remote_dp_rank alias from dp_rank in kv_transfer_params", func() {
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			decodeReq := testInfo.decodeHandler.lastRequest
			kvParams, ok := decodeReq[requestFieldKVTransferParams].(map[string]any)
			Expect(ok).To(BeTrue())

			Expect(kvParams).To(HaveKey(atomFieldRemoteDPRank))
			Expect(kvParams).To(HaveKey(atomFieldDPRank))
			Expect(kvParams[atomFieldRemoteDPRank]).To(Equal(kvParams[atomFieldDPRank]))
		})

		It("should set max_tokens=1 for prefill and restore original for decode", func() {
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			prefillReq := testInfo.prefillHandler.lastRequest
			Expect(prefillReq[requestFieldMaxTokens]).To(BeNumerically("==", 1))

			decodeReq := testInfo.decodeHandler.lastRequest
			Expect(decodeReq[requestFieldMaxTokens]).To(BeNumerically("==", 50))
		})

		It("should disable streaming for prefill and preserve original for decode", func() {
			proxyBaseAddr := testInfo.startProxy()

			streamingBody := `{
				"model": "meta-llama/Llama-3-8B",
				"messages": [{"role": "user", "content": "Hello"}],
				"max_tokens": 50,
				"stream": true
			}`

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(streamingBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			prefillReq := testInfo.prefillHandler.lastRequest
			Expect(prefillReq[requestFieldStream]).To(BeFalse())

			decodeReq := testInfo.decodeHandler.lastRequest
			Expect(decodeReq[requestFieldStream]).To(BeTrue())
		})

		It("should compute DP rank correctly with ATOMDPSize > 1", func() {
			testInfo.proxy.config.ATOMDPSize = 8
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
				bytes.NewReader([]byte(chatCompletionsRequestBody)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			rp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer rp.Body.Close()

			Expect(rp.StatusCode).To(Equal(http.StatusOK))

			prefillReq := testInfo.prefillHandler.lastRequest
			dpRank := int(prefillReq[atomFieldDataParallelRank].(float64))
			Expect(dpRank).To(BeNumerically(">=", 0))
			Expect(dpRank).To(BeNumerically("<", 8))
		})
	})
})

type atomTestInfo struct {
	ctx            context.Context
	cancelFn       context.CancelFunc
	stoppedCh      chan struct{}
	decodeBackend  *httptest.Server
	decodeHandler  *atomMockHandler
	prefillBackend *httptest.Server
	prefillHandler *atomMockHandler
	proxy          *Server
}

func (testInfo *atomTestInfo) startProxy() string {
	go func() {
		defer GinkgoRecover()

		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
		err := testInfo.proxy.Start(testInfo.ctx)
		Expect(err).ToNot(HaveOccurred())

		testInfo.stoppedCh <- struct{}{}
	}()

	<-testInfo.proxy.readyCh
	return "http://" + testInfo.proxy.addr.String()
}

func atomTestSetup() *atomTestInfo {
	testInfo := &atomTestInfo{}

	testInfo.ctx = newTestContext()
	testInfo.ctx, testInfo.cancelFn = context.WithCancel(testInfo.ctx)
	testInfo.stoppedCh = make(chan struct{})

	testInfo.decodeHandler = &atomMockHandler{role: "decode"}
	testInfo.decodeBackend = httptest.NewServer(testInfo.decodeHandler)

	testInfo.prefillHandler = &atomMockHandler{role: "prefill"}
	testInfo.prefillBackend = httptest.NewServer(testInfo.prefillHandler)

	decodeURL, err := url.Parse(testInfo.decodeBackend.URL)
	Expect(err).ToNot(HaveOccurred())

	cfg := Config{
		Port:       "0",
		DecoderURL: decodeURL,
		KVConnector: KVConnectorATOM,
		ATOMDPSize:  1,
	}
	testInfo.proxy = NewProxy(cfg)

	return testInfo
}

type atomMockHandler struct {
	role         string
	requestCount int
	lastRequest  map[string]any
}

func (h *atomMockHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.requestCount++

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusInternalServerError)
		return
	}
	defer r.Body.Close()

	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	h.lastRequest = req

	var resp map[string]any
	if h.role == "prefill" {
		resp = map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"model":   "meta-llama/Llama-3-8B",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "",
					},
					"finish_reason": "length",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 1,
				"total_tokens":      11,
			},
			requestFieldKVTransferParams: map[string]any{
				requestFieldDoRemotePrefill:  true,
				requestFieldDoRemoteDecode:   false,
				"remote_block_ids":           []int{0, 1, 2, 3},
				"remote_engine_id":           "10.0.0.1:6301",
				"remote_host":                "10.0.0.1",
				"remote_port":                6301,
				"remote_handshake_port":      6301,
				"tp_size":                    8,
				atomFieldDPRank:              0,
				"transfer_id":                12345,
				"first_token_id":             128000,
				"draft_token_ids":            []int{},
				"prefix_cache_hit_tokens":    0,
			},
		}
	} else {
		resp = map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"model":   "meta-llama/Llama-3-8B",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello! How can I assist you today?",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 10,
				"total_tokens":      20,
			},
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}
