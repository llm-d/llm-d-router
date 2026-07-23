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
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("P2P KV cache source header", func() {

	var testInfo *sidecarTestInfo

	const p2pConnectorPort = 7777
	// A private pod-range peer address (the kind an InferencePool would allow).
	const kvCacheSource = "10.9.9.9:8000"

	// A client that supplies its own kv_transfer_params must not have those
	// keys reach vLLM: the sidecar owns the field and rebuilds it.
	const bodyWithClientKVParams = `{
					"model": "Qwen/Qwen2-0.5B",
					"messages": [{"role": "user", "content": "Hello"}],
					"max_tokens": 50,
					"kv_transfer_params": {"do_remote_prefill": true, "remote_host": "evil.example.com"}
				}`

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorOffloading)
		testInfo.proxy.config.P2PConnectorPort = p2pConnectorPort
		// SSRF allowlist disabled in-test (an enabled one requires a live
		// InferencePool), so a well-formed source passes; the malformed and
		// empty-host cases below exercise the header validation.
		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
	})

	sendBody := func(proxyBaseAddr, body string, headers map[string]string) *http.Response {
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
			bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		for k, v := range headers {
			req.Header.Add(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}
		return resp
	}

	sendRequest := func(proxyBaseAddr string, headers map[string]string) *http.Response {
		return sendBody(proxyBaseAddr, chatCompletionsRequestBodyWithMaxCompletionTokens, headers)
	}

	It("should inject remote_kv_source params on the local request without disaggregation", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: kvCacheSource})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		dreq := decodeReqs[0]

		kvParams, ok := dreq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(kvParams).ToNot(HaveKey(requestFieldRemoteDecoder))
		Expect(kvParams).ToNot(HaveKey(requestFieldRemotePrefiller))
		p2p, ok := kvParams[requestFieldRemoteKVSource].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[requestFieldKVRequestID]).ToNot(BeEmpty())
		Expect(p2p[requestFieldRemoteHost]).To(Equal("10.9.9.9"))
		Expect(p2p[requestFieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))

		// The caller's token limits are untouched.
		Expect(dreq[requestFieldMaxTokens]).To(BeNumerically("==", 50))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should add remote_kv_source params to the prefill leg under disaggregation", func() {
		proxyBaseAddr := testInfo.startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		sendRequest(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCacheSourceHeader:   kvCacheSource,
		})

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Prefill leg: remote_decoder + remote_kv_source, each with its own
		// kv_request_id.
		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		prefillKVParams, ok := preq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		decodeParams, ok := prefillKVParams[requestFieldRemoteDecoder].(map[string]any)
		Expect(ok).To(BeTrue())
		p2p, ok := prefillKVParams[requestFieldRemoteKVSource].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[requestFieldKVRequestID]).ToNot(BeEmpty())
		Expect(p2p[requestFieldKVRequestID]).ToNot(Equal(decodeParams[requestFieldKVRequestID]))
		Expect(p2p[requestFieldRemoteHost]).To(Equal("10.9.9.9"))
		Expect(p2p[requestFieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))

		// Decode leg: remote_prefiller only, never remote_kv_source.
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		decodeKVParams, ok := decodeReqs[0][requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams).To(HaveKey(requestFieldRemotePrefiller))
		Expect(decodeKVParams).ToNot(HaveKey(requestFieldRemoteKVSource))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not add remote_kv_source params to the prefill leg when the source is the prefiller itself", func() {
		proxyBaseAddr := testInfo.startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		// The source resolves to the selected prefiller - there is nothing to
		// pull from itself, so the prefill leg carries remote_decoder only.
		sendRequest(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCacheSourceHeader:   prefillHostPort,
		})

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		prefillKVParams, ok := preq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams).To(HaveKey(requestFieldRemoteDecoder))
		Expect(prefillKVParams).ToNot(HaveKey(requestFieldRemoteKVSource))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not inject remote_kv_source params without the header", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, nil)

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		Expect(decodeReqs[0]).ToNot(HaveKey(requestFieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should ignore a malformed header value", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: "not-a-host-port"})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		Expect(decodeReqs[0]).ToNot(HaveKey(requestFieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should replace client-supplied kv_transfer_params", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendBody(proxyBaseAddr, bodyWithClientKVParams,
			map[string]string{routing.KVCacheSourceHeader: kvCacheSource})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		kvParams, ok := decodeReqs[0][requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		// Only the sidecar-owned remote_kv_source key survives; the client's keys are gone.
		Expect(kvParams).To(HaveLen(1))
		p2p, ok := kvParams[requestFieldRemoteKVSource].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[requestFieldRemoteHost]).To(Equal("10.9.9.9"))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should ignore a header with an empty host", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: ":8000"})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		Expect(decodeReqs[0]).ToNot(HaveKey(requestFieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not inject remote_kv_source params when the source is the local pod", func() {
		GinkgoT().Setenv("POD_IP", "10.9.9.9")
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: kvCacheSource})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		Expect(decodeReqs[0]).ToNot(HaveKey(requestFieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should generate a distinct kv_request_id per request", func() {
		proxyBaseAddr := testInfo.startProxy()

		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: kvCacheSource})
		sendRequest(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: kvCacheSource})

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(2))
		ids := make([]any, 0, 2)
		for _, dreq := range decodeReqs {
			kvParams, ok := dreq[requestFieldKVTransferParams].(map[string]any)
			Expect(ok).To(BeTrue())
			p2p, ok := kvParams[requestFieldRemoteKVSource].(map[string]any)
			Expect(ok).To(BeTrue())
			ids = append(ids, p2p[requestFieldKVRequestID])
		}
		Expect(ids[0]).ToNot(Equal(ids[1]))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})
