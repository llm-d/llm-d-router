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

// NIXL PD composed with the OffloadingConnector P2P tier via vLLM MultiConnector:
// the sidecar orchestrates PD over NIXL and additionally injects the p2p pull
// block on the prefill leg, gated by --enable-p2p-pull.
var _ = Describe("NIXL Connector with P2P pull", func() {

	var testInfo *sidecarTestInfo

	const p2pConnectorPort = 7777
	// A private pod-range peer address (the kind an InferencePool would allow).
	const kvCacheSource = "10.9.9.9:8000"

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorNIXLV2)
		testInfo.proxy.config.P2PConnectorPort = p2pConnectorPort
		// SSRF allowlist disabled in-test (an enabled one requires a live
		// InferencePool), so a well-formed source passes.
		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
	})

	startProxy := func() string {
		go func() {
			defer GinkgoRecover()
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

	sendRequest := func(proxyBaseAddr string, headers map[string]string) {
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath,
			bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		for k, v := range headers {
			req.Header.Add(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}
	}

	// prefillKV returns the kv_transfer_params of the single captured prefill
	// request. The serial NIXL path completes prefill before the response
	// returns, so no polling is needed.
	prefillKV := func() map[string]any {
		reqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(reqs).To(HaveLen(1))
		kv, ok := reqs[0][requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		return kv
	}

	It("composes the p2p pull onto the NIXL prefill leg when --enable-p2p-pull is set", func() {
		testInfo.proxy.config.EnableP2PPull = true
		proxyBaseAddr := startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		sendRequest(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCacheSourceHeader:   kvCacheSource,
		})

		kv := prefillKV()
		// NIXL fields still drive the NixlConnector under MultiConnector.
		Expect(kv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kv).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		// The remote_kv_source block drives the OffloadingConnector's cached-prefix pull.
		p2p, ok := kv[requestFieldP2PParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[requestFieldKVRequestID]).ToNot(BeEmpty())
		Expect(p2p[requestFieldRemoteHost]).To(Equal("10.9.9.9"))
		Expect(p2p[requestFieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))
	})

	It("ignores the source header on the NIXL path without --enable-p2p-pull", func() {
		proxyBaseAddr := startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		sendRequest(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCacheSourceHeader:   kvCacheSource,
		})

		Expect(prefillKV()).ToNot(HaveKey(requestFieldP2PParams))
	})

	It("does not compose a p2p pull when the source is the prefiller itself", func() {
		testInfo.proxy.config.EnableP2PPull = true
		proxyBaseAddr := startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		sendRequest(proxyBaseAddr, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCacheSourceHeader:   prefillHostPort,
		})

		Expect(prefillKV()).ToNot(HaveKey(requestFieldP2PParams))
	})

	// The parallel-dispatch (MoRI-IO WRITE) path builds the prefill leg in a
	// separate function, so it has its own p2p injection site.
	It("composes the p2p pull onto the NIXL prefill leg in parallel-dispatch mode", func() {
		env := startMoRIProxy(func(c *Config) {
			c.MoRIIOParallelDispatch = true
			c.EnableP2PPull = true
			c.P2PConnectorPort = p2pConnectorPort
		})

		req, err := http.NewRequest(http.MethodPost, env.baseAddr+ChatCompletionsPath,
			bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, env.prefillBackend.URL[len("http://"):])
		req.Header.Add(routing.KVCacheSourceHeader, kvCacheSource)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

		// Prefill leg keeps the NIXL WRITE fields and gains the composed remote_kv_source block.
		pkv := kvParams(env.prefillHandler, 0)
		Expect(pkv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		p2p, ok := pkv[requestFieldP2PParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[requestFieldRemoteHost]).To(Equal("10.9.9.9"))
		Expect(p2p[requestFieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))
	})
})
