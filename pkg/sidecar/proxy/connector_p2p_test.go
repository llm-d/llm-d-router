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

var _ = Describe("P2P Connector", func() {

	var testInfo *sidecarTestInfo

	const p2pConnectorPort = 7777

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorOffloading)
		testInfo.proxy.config.P2PConnectorPort = p2pConnectorPort
	})

	It("should send concurrent requests with correct p2p kv_transfer_params", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := chatCompletionsRequestBodyWithMaxCompletionTokens
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Wait for the async prefill request to be recorded.
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Prefill leg: kv_transfer_params.decode carries only kv_request_id,
		// with no peer address.
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		preq := prefillReqs[0]

		Expect(preq).To(HaveKey(requestFieldKVTransferParams))
		prefillKVParams, ok := preq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams).ToNot(HaveKey(requestFieldP2PPrefillParams))
		prefillDecode, ok := prefillKVParams[requestFieldP2PDecodeParams].(map[string]any)
		Expect(ok).To(BeTrue())
		prefillKVRequestID := prefillDecode[requestFieldKVRequestID]
		Expect(prefillKVRequestID).ToNot(BeEmpty())
		Expect(prefillDecode).ToNot(HaveKey(requestFieldRemoteHost))
		Expect(prefillDecode).ToNot(HaveKey(requestFieldRemotePort))

		// Prefill is capped to a single output token and non-streaming.
		Expect(preq[requestFieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).To(HaveKeyWithValue(requestFieldMaxCompletionTokens, BeNumerically("==", 1)))
		Expect(preq[requestFieldStream]).To(BeFalse())

		// Decode leg: kv_transfer_params.prefill carries the prefiller's
		// OffloadingConnector P2P tier address plus the matching kv_request_id.
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		dreq := decodeReqs[0]

		Expect(dreq).To(HaveKey(requestFieldKVTransferParams))
		decodeKVParams, ok := dreq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams).ToNot(HaveKey(requestFieldP2PDecodeParams))
		decodePrefill, ok := decodeKVParams[requestFieldP2PPrefillParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodePrefill[requestFieldKVRequestID]).To(Equal(prefillKVRequestID))
		Expect(decodePrefill[requestFieldRemoteHost]).To(Equal(extractHost(prefillHostPort)))
		Expect(decodePrefill[requestFieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))

		// Decode preserves the caller's original token limits.
		Expect(dreq[requestFieldMaxTokens]).To(BeNumerically("==", 50))
		Expect(dreq).To(HaveKeyWithValue(requestFieldMaxCompletionTokens, BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})

var _ = DescribeTable("p2pPullAvailable",
	func(connector string, enableP2PPull, want bool) {
		s := &Server{config: Config{KVConnector: connector, EnableP2PPull: enableP2PPull}}
		Expect(s.p2pPullAvailable()).To(Equal(want))
	},
	Entry("offloading is always available", KVConnectorOffloading, false, true),
	Entry("nixlv2 with the flag is available", KVConnectorNIXLV2, true, true),
	Entry("nixlv2 without the flag is unavailable", KVConnectorNIXLV2, false, false),
	Entry("the flag has no effect on sglang", KVConnectorSGLang, true, false),
	Entry("the flag has no effect on shared-storage", KVConnectorSharedStorage, true, false),
)

var _ = DescribeTable("p2pPortFor",
	func(dpSize, dpBasePort int, target string, want int) {
		s := &Server{
			dpBasePort: dpBasePort,
			config:     Config{P2PConnectorPort: 7777, DataParallelSize: dpSize},
		}
		Expect(s.p2pPortFor(target)).To(Equal(want))
	},
	Entry("single DP uses the base port regardless of target", 1, 8000, "10.0.0.5:8003", 7777),
	Entry("rank 0 target uses the base port", 4, 8000, "10.0.0.5:8000", 7777),
	Entry("rank 2 target offsets by 2", 4, 8000, "10.0.0.5:8002", 7779),
	Entry("last rank target offsets by dpSize-1", 4, 8000, "10.0.0.5:8003", 7780),
	Entry("port below the base falls back to the base port", 4, 8000, "10.0.0.5:7999", 7777),
	Entry("port beyond the rank range falls back to the base port", 4, 8000, "10.0.0.5:8004", 7777),
	Entry("target without a port falls back to the base port", 4, 8000, "10.0.0.5", 7777),
	Entry("unparsable port falls back to the base port", 4, 8000, "10.0.0.5:http", 7777),
	Entry("zero base port disables derivation", 4, 0, "10.0.0.5:8002", 7777),
)

var _ = Describe("p2pSourceParams", func() {
	It("offsets remote_port by the source endpoint's DP rank", func() {
		s := &Server{
			dpBasePort: 8000,
			config:     Config{P2PConnectorPort: 7777, DataParallelSize: 4},
		}
		params := s.p2pSourceParams("10.0.0.9:8002")
		Expect(params[requestFieldRemoteHost]).To(Equal("10.0.0.9"))
		Expect(params[requestFieldRemotePort]).To(Equal(7779))
		Expect(params[requestFieldKVRequestID]).ToNot(BeEmpty())
	})

	It("keeps rank derivation on Clone, which rank servers rely on", func() {
		s := &Server{
			dpBasePort: 8000,
			config:     Config{P2PConnectorPort: 7777, DataParallelSize: 4},
		}
		params := s.Clone().p2pSourceParams("10.0.0.9:8002")
		Expect(params[requestFieldRemotePort]).To(Equal(7779))
	})
})
