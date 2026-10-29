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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// promptTokenIDsBody is a chat request carrying the EPP's forwarded prompt_token_ids.
const promptTokenIDsBody = `{"model": "Qwen/Qwen2-0.5B", "messages": [{"role": "user", "content": "Hello"}],
	"max_tokens": 50, "kv_transfer_params": {"prompt_token_ids": [5, 6, 7]}}`

var wantPromptTokenIDs = []any{float64(5), float64(6), float64(7)}

func promptTokenIDsOf(req map[string]any) any {
	params, ok := req[reqcommon.FieldKVTransferParams].(map[string]any)
	Expect(ok).To(BeTrue(), "kv_transfer_params missing or not an object")
	return params[requestFieldPromptTokenIDs]
}

func sendPromptTokenIDs(proxyBaseAddr string, headers map[string]string) {
	req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions,
		bytes.NewReader([]byte(promptTokenIDsBody)))
	Expect(err).ToNot(HaveOccurred())
	for k, v := range headers {
		req.Header.Add(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		Fail(string(b))
	}
}

// Every P/D connector rebuilds kv_transfer_params on both hops; each must keep the
// client's prompt_token_ids so neither engine re-tokenizes the prompt.
var _ = Describe("prompt_token_ids carry-over", func() {
	expectBothHops := func(testInfo *sidecarTestInfo) {
		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))
		Expect(promptTokenIDsOf(testInfo.prefillHandler.GetCompletionRequests()[0])).To(Equal(wantPromptTokenIDs))
		Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(1))
		Expect(promptTokenIDsOf(testInfo.decodeHandler.GetCompletionRequests()[0])).To(Equal(wantPromptTokenIDs))
	}

	It("is kept by the offloading (P2P) connector on prefill and decode", func() {
		testInfo := sidecarConnectionTestSetup(constants.KVConnectorOffloading)
		testInfo.proxy.config.P2PConnectorPort = 7777
		proxyBaseAddr := testInfo.startProxy()
		sendPromptTokenIDs(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: testInfo.prefillBackend.URL[len("http://"):],
		})
		expectBothHops(testInfo)
	})

	It("is kept by the offloading connector's decode-only cache-source pull", func() {
		testInfo := sidecarConnectionTestSetup(constants.KVConnectorOffloading)
		testInfo.proxy.config.P2PConnectorPort = 7777
		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
		proxyBaseAddr := testInfo.startProxy()
		sendPromptTokenIDs(proxyBaseAddr, map[string]string{routing.KVCacheSourceHeader: "10.9.9.9:8000"})

		Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(1))
		dreq := testInfo.decodeHandler.GetCompletionRequests()[0]
		Expect(promptTokenIDsOf(dreq)).To(Equal(wantPromptTokenIDs))
		Expect(dreq[reqcommon.FieldKVTransferParams]).To(HaveKey(requestFieldRemoteKVSource))
	})

	It("is kept by the mooncake connector on prefill and decode", func() {
		bootstrap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"0": {"engine_id": "e1", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}}}`))
		}))
		DeferCleanup(bootstrap.Close)
		u, err := url.Parse(bootstrap.URL)
		Expect(err).ToNot(HaveOccurred())
		var port int
		_, err = fmt.Sscanf(u.Port(), "%d", &port)
		Expect(err).ToNot(HaveOccurred())

		testInfo := sidecarConnectionTestSetup(constants.KVConnectorMooncake)
		testInfo.proxy.config.MooncakeBootstrapPort = port
		proxyBaseAddr := testInfo.startProxy()
		sendPromptTokenIDs(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: testInfo.prefillBackend.URL[len("http://"):],
		})
		expectBothHops(testInfo)
	})

	It("is kept by the NIXL MoRI-IO parallel WRITE dispatch on prefill and decode", func() {
		var mu sync.Mutex
		seen := map[string]map[string]any{}
		record := func(role string, reply string) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				Expect(json.NewDecoder(r.Body).Decode(&body)).To(Succeed())
				mu.Lock()
				seen[role] = body
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(reply))
			}
		}
		env := startParallelCommitProxy(
			record("prefill", `{"id":"p","choices":[],"kv_transfer_params":{}}`),
			record("decode", `{"id":"d","choices":[]}`), nil)

		req, err := http.NewRequest(http.MethodPost, env.baseAddr+reqcommon.PathChatCompletions,
			strings.NewReader(promptTokenIDsBody))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, env.prefillHost)
		resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
		Expect(err).ToNot(HaveOccurred())
		_ = resp.Body.Close()

		Eventually(func() int { mu.Lock(); defer mu.Unlock(); return len(seen) }).Should(Equal(2))
		mu.Lock()
		defer mu.Unlock()
		Expect(promptTokenIDsOf(seen["prefill"])).To(Equal(wantPromptTokenIDs))
		Expect(promptTokenIDsOf(seen["decode"])).To(Equal(wantPromptTokenIDs))
	})
})

// shared-storage and sglang forward the client's kv_transfer_params as sent.
var _ = Describe("prompt_token_ids passthrough", func() {
	for _, connector := range []string{constants.KVConnectorSharedStorage, constants.KVConnectorSGLang} {
		It("reaches decode unchanged with the "+connector+" connector", func() {
			testInfo := sidecarConnectionTestSetup(connector)
			proxyBaseAddr := testInfo.startProxy()
			sendPromptTokenIDs(proxyBaseAddr, map[string]string{
				routing.PrefillEndpointHeader: testInfo.prefillBackend.URL[len("http://"):],
			})
			Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(BeNumerically(">=", 1))
			for _, dreq := range testInfo.decodeHandler.GetCompletionRequests() {
				Expect(promptTokenIDsOf(dreq)).To(Equal(wantPromptTokenIDs))
			}
		})
	}
})

// The EPP replaces a /v1/completions text prompt with its token IDs; both P/D hops must send that
// prompt as received so neither engine tokenizes it.
var _ = Describe("token-ID completions prompt", func() {
	It("reaches prefill and decode unchanged with the nixlv2 connector", func() {
		testInfo := sidecarConnectionTestSetup(constants.KVConnectorNIXLV2)
		proxyBaseAddr := testInfo.startProxy()
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathCompletions,
			bytes.NewReader([]byte(`{"model": "Qwen/Qwen2-0.5B", "prompt": [5, 6, 7], "max_tokens": 50}`)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])
		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))
		Expect(testInfo.prefillHandler.GetCompletionRequests()[0]["prompt"]).To(Equal(wantPromptTokenIDs))
		Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(1))
		Expect(testInfo.decodeHandler.GetCompletionRequests()[0]["prompt"]).To(Equal(wantPromptTokenIDs))
	})
})
