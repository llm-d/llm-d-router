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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// p2pConnectorPort is the OffloadingConnector P2P tier port of data parallel
// rank 0, the value EPP's p2p-source-producer computes for an endpoint on
// rank 0.
const p2pConnectorPort = 7777

// prefillP2PHostPort returns the P2P tier address EPP sets for the prefill pod
// at prefillHostPort: the same host on the rank-0 tier port.
func prefillP2PHostPort(prefillHostPort string) string {
	return net.JoinHostPort(extractHost(prefillHostPort), strconv.Itoa(p2pConnectorPort))
}

// addPrefillRouting routes req to prefillHostPort with the P2P tier address the
// offloading connector requires on a prefill request. The other connectors
// strip the address header unread.
func addPrefillRouting(req *http.Request, prefillHostPort string) {
	req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)
	req.Header.Add(routing.KVCachePrefillHeader, prefillP2PHostPort(prefillHostPort))
}

var _ = Describe("P2P Connector", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorOffloading)
	})

	It("should send both requests with correct PD Multi Tier kv_transfer_params", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := chatCompletionsRequestBodyWithMaxCompletionCap
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		addPrefillRouting(req, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// The prefill request completes before the response is returned.
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Prefill request: kv_transfer_params.remote_decoder carries only kv_request_id,
		// with no peer address.
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		preq := prefillReqs[0]

		Expect(preq).To(HaveKey(reqcommon.FieldKVTransferParams))
		prefillKVParams, ok := preq[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams).ToNot(HaveKey(reqcommon.FieldRemotePrefiller))
		prefillDecode, ok := prefillKVParams[reqcommon.FieldRemoteDecoder].(map[string]any)
		Expect(ok).To(BeTrue())
		prefillKVRequestID := prefillDecode[reqcommon.FieldKVRequestID]
		Expect(prefillKVRequestID).ToNot(BeEmpty())
		Expect(prefillDecode).ToNot(HaveKey(reqcommon.FieldRemoteHost))
		Expect(prefillDecode).ToNot(HaveKey(reqcommon.FieldRemotePort))

		// Prefill is capped to a single output token and non-streaming.
		Expect(preq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).To(HaveKeyWithValue(reqcommon.FieldMaxCompletionTokens, BeNumerically("==", 1)))
		Expect(preq[reqcommon.FieldStream]).To(BeFalse())

		// Decode request: kv_transfer_params.remote_prefiller carries the prefiller's
		// OffloadingConnector P2P tier address plus the matching kv_request_id.
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		dreq := decodeReqs[0]

		Expect(dreq).To(HaveKey(reqcommon.FieldKVTransferParams))
		decodeKVParams, ok := dreq[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams).ToNot(HaveKey(reqcommon.FieldRemoteDecoder))
		decodePrefill, ok := decodeKVParams[reqcommon.FieldRemotePrefiller].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodePrefill[reqcommon.FieldKVRequestID]).To(Equal(prefillKVRequestID))
		Expect(decodePrefill[reqcommon.FieldRemoteHost]).To(Equal(extractHost(prefillHostPort)))
		Expect(decodePrefill[reqcommon.FieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))

		// Decode preserves the caller's original token limits.
		Expect(dreq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 50))
		Expect(dreq).To(HaveKeyWithValue(reqcommon.FieldMaxCompletionTokens, BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	// Encoder disaggregation runs the P/D stage after the encoder stage; the
	// offloading connector needs the prefill P2P port there too.
	It("should carry the prefill P2P port through the encoder stage", func() {
		encoderBackend := httptest.NewServer(statusHandler(http.StatusOK, `{"choices":[{"message":{"content":""}}],"ec_transfer_params":{"hash-1":{"peer_host":"10.0.0.1","peer_port":5500}}}`))
		DeferCleanup(encoderBackend.Close)
		testInfo.proxy = NewProxy(Config{Port: "0", DecoderURL: testInfo.decodeURL, KVConnector: constants.KVConnectorOffloading, ECConnector: constants.ECConnectorNIXL})
		proxyBaseAddr := testInfo.startProxy()

		// One image, so the encoder stage runs.
		body, err := json.Marshal(userMessageRequest(imageURLItem("https://example.com/img.jpg")))
		Expect(err).ToNot(HaveOccurred())
		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		postChatCompletions(proxyBaseAddr, string(body), map[string]string{
			routing.PrefillEndpointHeader:  prefillHostPort,
			routing.KVCachePrefillHeader:   prefillP2PHostPort(prefillHostPort),
			routing.EncoderEndpointsHeader: encoderBackend.URL[len("http://"):],
		})

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))
		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		Expect(preq).To(HaveKey(reqcommon.FieldECTransferParams))

		decodePrefill := decodeRemotePrefiller(testInfo)
		Expect(decodePrefill[reqcommon.FieldRemoteHost]).To(Equal(extractHost(prefillHostPort)))
		Expect(decodePrefill[reqcommon.FieldRemotePort]).To(BeNumerically("==", p2pConnectorPort))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should strip min_tokens from the prefill request and restore it in decode", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := chatCompletionsRequestBodyWithMinCap
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		addPrefillRouting(req, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		Expect(preq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).ToNot(HaveKey(reqcommon.FieldMinTokens))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		dreq := testInfo.decodeHandler.GetCompletionRequests()[0]
		Expect(dreq).To(HaveKeyWithValue(reqcommon.FieldMinTokens, BeNumerically("==", 5)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not dispatch the decode request until the prefill request has returned", func() {
		// The decode request pulls KV from the prefiller's secondary tier. If it is
		// dispatched first, its fetch arrives before any blocks are stored and
		// burns the connector's load deadline waiting for KV that does not exist.
		proxyBaseAddr := testInfo.startProxy()

		release := make(chan struct{})
		var releaseOnce sync.Once
		releasePrefill := func() { releaseOnce.Do(func() { close(release) }) }
		// Always unblock, so a failed assertion cannot deadlock cleanup.
		defer releasePrefill()

		var prefillHits atomic.Int32
		blockingPrefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			prefillHits.Add(1)
			select {
			case <-release:
			case <-time.After(10 * time.Second):
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"choices":[]}`))
		}))
		defer blockingPrefill.Close()

		body := chatCompletionsRequestBodyWithMaxCompletionCap
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		addPrefillRouting(req, blockingPrefill.URL[len("http://"):])

		done := make(chan *http.Response, 1)
		go func() {
			defer GinkgoRecover()
			resp, doErr := http.DefaultClient.Do(req)
			Expect(doErr).ToNot(HaveOccurred())
			done <- resp
		}()

		// Prefill is in flight and blocked; decode must not have been touched.
		Eventually(prefillHits.Load).Should(Equal(int32(1)))
		Consistently(func() int32 {
			return testInfo.decodeHandler.RequestCount.Load()
		}, 200*time.Millisecond, 20*time.Millisecond).Should(BeZero())

		releasePrefill()

		var resp *http.Response
		Eventually(done).Should(Receive(&resp))
		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should return the prefill error and never dispatch decode when prefill fails", func() {
		proxyBaseAddr := testInfo.startProxy()

		failingPrefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInsufficientStorage)
			_, _ = w.Write([]byte(`{"error":"no room for kv"}`))
		}))
		defer failingPrefill.Close()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions,
			bytes.NewReader([]byte(chatCompletionsRequestBodyWithMaxCompletionCap)))
		Expect(err).ToNot(HaveOccurred())
		addPrefillRouting(req, failingPrefill.URL[len("http://"):])

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInsufficientStorage))
		respBody, err := io.ReadAll(resp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(string(respBody)).To(ContainSubstring("no room for kv"))

		Consistently(func() int32 {
			return testInfo.decodeHandler.RequestCount.Load()
		}, 200*time.Millisecond, 20*time.Millisecond).Should(BeZero())

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should add max_completion_tokens=1 to the prefill request even when absent from the original request", func() {
		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		addPrefillRouting(req, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		Expect(preq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).To(HaveKeyWithValue(reqcommon.FieldMaxCompletionTokens, BeNumerically("==", 1)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should cap sampling_params in the prefill request and restore originals in decode", func() {
		expectGenerateRequestTokenLimits(testInfo)

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})

var _ = DescribeTable("p2pPullAvailable",
	func(connector string, enableP2PPull, want bool) {
		s := &Server{config: Config{KVConnector: connector, EnableP2PPull: enableP2PPull}}
		Expect(s.p2pPullAvailable()).To(Equal(want))
	},
	Entry("offloading is always available", constants.KVConnectorOffloading, false, true),
	Entry("nixlv2 with the flag is available", constants.KVConnectorNIXLV2, true, true),
	Entry("nixlv2 without the flag is unavailable", constants.KVConnectorNIXLV2, false, false),
	Entry("the flag has no effect on sglang", constants.KVConnectorSGLang, true, false),
	Entry("the flag has no effect on shared-storage", constants.KVConnectorSharedStorage, true, false),
)

var _ = Describe("addP2PPullToPrefill", func() {
	It("copies the source block EPP wrote", func() {
		params := map[string]any{reqcommon.FieldRemoteDecoder: map[string]any{reqcommon.FieldKVRequestID: "k"}}
		source := &kvSource{kvRequestID: "k2", p2pPeer: p2pPeer{host: "10.0.6.107", port: 7780}}

		addP2PPullToPrefill(params, source)

		Expect(params).To(HaveLen(2))
		Expect(params[reqcommon.FieldRemoteKVSource]).To(Equal(map[string]any{
			reqcommon.FieldKVRequestID: "k2",
			reqcommon.FieldRemoteHost:  "10.0.6.107",
			reqcommon.FieldRemotePort:  7780,
		}))
	})

	It("leaves the params untouched without a source", func() {
		params := map[string]any{reqcommon.FieldRemoteDecoder: map[string]any{reqcommon.FieldKVRequestID: "k"}}

		addP2PPullToPrefill(params, nil)

		Expect(params).To(HaveLen(1))
		Expect(params).NotTo(HaveKey(reqcommon.FieldRemoteKVSource))
	})
})

var _ = DescribeTable("parsePrefillP2P",
	func(value string, want *p2pPeer) {
		Expect(parsePrefillP2P(logr.Discard(), value)).To(Equal(want))
	},
	Entry("absent header yields no address, which fails the request", "", nil),
	Entry("rank-0 port", "10.0.0.1:7777", &p2pPeer{host: "10.0.0.1", port: 7777}),
	Entry("offset port", "10.0.0.1:7780", &p2pPeer{host: "10.0.0.1", port: 7780}),
	Entry("IPv6", "[fd00::1]:7777", &p2pPeer{host: "fd00::1", port: 7777}),
	Entry("no port", "10.0.0.1", nil),
	Entry("no host", ":7777", nil),
	Entry("port not a number", "10.0.0.1:http", nil),
	Entry("port zero", "10.0.0.1:0", nil),
	Entry("port above the TCP range", "10.0.0.1:70000", nil),
)

// decodeRemotePrefiller returns the remote_prefiller block of the one decode
// request the decode backend received.
func decodeRemotePrefiller(testInfo *sidecarTestInfo) map[string]any {
	GinkgoHelper()
	decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
	Expect(decodeReqs).To(HaveLen(1))
	kvParams, ok := decodeReqs[0][reqcommon.FieldKVTransferParams].(map[string]any)
	Expect(ok).To(BeTrue())
	prefiller, ok := kvParams[reqcommon.FieldRemotePrefiller].(map[string]any)
	Expect(ok).To(BeTrue())
	return prefiller
}

var _ = Describe("P2P prefill address header", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorOffloading)
	})

	// remote_prefiller is the header's host and port, not the serving address
	// the sidecar connected to.
	It("copies the address from the header into remote_prefiller and strips the header", func() {
		proxyBaseAddr := testInfo.startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		postChatCompletions(proxyBaseAddr, chatCompletionsRequestBodyWithMaxCompletionCap, map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCachePrefillHeader:  "10.0.0.7:7779",
		})

		prefiller := decodeRemotePrefiller(testInfo)
		Expect(prefiller[reqcommon.FieldRemoteHost]).To(Equal("10.0.0.7"))
		Expect(prefiller[reqcommon.FieldRemotePort]).To(BeNumerically("==", 7779))
		for _, headers := range [][]http.Header{
			testInfo.prefillHandler.GetCompletionHeaders(),
			testInfo.decodeHandler.GetCompletionHeaders(),
		} {
			Expect(headers).To(HaveLen(1))
			Expect(headers[0]).ToNot(HaveKey(http.CanonicalHeaderKey(routing.KVCachePrefillHeader)))
			Expect(headers[0]).ToNot(HaveKey(http.CanonicalHeaderKey(routing.PrefillEndpointHeader)))
		}

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	// expectRejected sends a prefill-routed request with the given address
	// header value ("" for none) and expects the sidecar to fail it before any
	// pod request is sent.
	expectRejected := func(addressHeader string) {
		GinkgoHelper()
		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions,
			bytes.NewReader([]byte(chatCompletionsRequestBodyWithMaxCompletionCap)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])
		if addressHeader != "" {
			req.Header.Add(routing.KVCachePrefillHeader, addressHeader)
		}

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
		respBody, err := io.ReadAll(resp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(string(respBody)).To(ContainSubstring(routing.KVCachePrefillHeader))
		Expect(testInfo.prefillHandler.GetCompletionRequests()).To(BeEmpty())
		Expect(testInfo.decodeHandler.GetCompletionRequests()).To(BeEmpty())

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	}

	It("fails the request without the header", func() {
		expectRejected("")
	})

	It("fails the request when the header is malformed", func() {
		expectRejected("not-a-host-port")
	})

	It("fails the request when the header has no port", func() {
		expectRejected("10.0.0.7")
	})
})
