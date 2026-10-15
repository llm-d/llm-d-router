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
	"encoding/json"
	"maps"
	"net"
	"strconv"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"k8s.io/utils/set"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// sourceBody returns a chat completions body carrying source as the
// remote_kv_source block EPP's p2p-source-producer writes, next to any other
// kv_transfer_params keys in extra.
func sourceBody(source any, extra map[string]any) string {
	kvParams := map[string]any{}
	maps.Copy(kvParams, extra)
	if source != nil {
		kvParams[reqcommon.FieldRemoteKVSource] = source
	}
	body := map[string]any{
		"model":                         "Qwen/Qwen2-0.5B",
		"messages":                      []map[string]any{{"role": "user", "content": "Hello"}},
		reqcommon.FieldMaxTokens:        50,
		reqcommon.FieldKVTransferParams: kvParams,
	}
	raw, err := json.Marshal(body)
	Expect(err).ToNot(HaveOccurred())
	return string(raw)
}

var _ = Describe("P2P KV cache source in the request body", func() {

	var testInfo *sidecarTestInfo

	// A private pod-range peer (the kind an InferencePool would allow), on
	// the P2P tier port of its data parallel rank 1.
	const sourceHost = "10.9.9.9"
	const sourcePort = 7778
	const sourceKVRequestID = "k2-from-epp"

	validSource := map[string]any{
		reqcommon.FieldKVRequestID: sourceKVRequestID,
		reqcommon.FieldRemoteHost:  sourceHost,
		reqcommon.FieldRemotePort:  sourcePort,
	}
	// sourceHeaders returns the source header EPP sets next to the body
	// block, plus extra.
	sourceHeaders := func(extra map[string]string) map[string]string {
		headers := map[string]string{routing.KVCacheSourceHeader: net.JoinHostPort(sourceHost, strconv.Itoa(sourcePort))}
		maps.Copy(headers, extra)
		return headers
	}

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorOffloading)
		// SSRF allowlist disabled in-test (an enabled one requires a live
		// InferencePool), so a well-formed source passes; the allowlist cases
		// exercise kvSourceFromBody with a validator built by hand.
		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
	})

	decodeKVParams := func() (map[string]any, map[string]any) {
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		kvParams, _ := decodeReqs[0][reqcommon.FieldKVTransferParams].(map[string]any)
		return decodeReqs[0], kvParams
	}

	expectSource := func(kvParams map[string]any) {
		p2p, ok := kvParams[reqcommon.FieldRemoteKVSource].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(p2p[reqcommon.FieldKVRequestID]).To(Equal(sourceKVRequestID))
		Expect(p2p[reqcommon.FieldRemoteHost]).To(Equal(sourceHost))
		Expect(p2p[reqcommon.FieldRemotePort]).To(BeNumerically("==", sourcePort))
	}

	It("should forward remote_kv_source on the local request without disaggregation", func() {
		proxyBaseAddr := testInfo.startProxy()

		postChatCompletions(proxyBaseAddr, sourceBody(validSource, nil), sourceHeaders(nil))

		dreq, kvParams := decodeKVParams()
		Expect(kvParams).To(HaveLen(1))
		expectSource(kvParams)
		// The caller's token limits are untouched.
		Expect(dreq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 50))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	// Without the header the body is not read: the request streams to the
	// decoder as the client sent it, a client-supplied source included.
	It("should forward the body unread without the source header", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := sourceBody(validSource, map[string]any{"other": true})
		postChatCompletions(proxyBaseAddr, body, nil)

		Expect(testInfo.decodeHandler.GetCompletionRawBodies()).To(HaveLen(1))
		Expect(string(testInfo.decodeHandler.GetCompletionRawBodies()[0])).To(Equal(body))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should carry remote_kv_source into the prefill request under disaggregation", func() {
		// Start takes the proxy logger from the context.
		var logged []string
		testInfo.ctx = log.IntoContext(testInfo.ctx, funcr.New(func(_, args string) { logged = append(logged, args) }, funcr.Options{}))
		proxyBaseAddr := testInfo.startProxy()

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		postChatCompletions(proxyBaseAddr, sourceBody(validSource, nil), sourceHeaders(map[string]string{
			routing.PrefillEndpointHeader: prefillHostPort,
			routing.KVCachePrefillHeader:  prefillP2PHostPort(prefillHostPort),
		}))

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Prefill request: remote_decoder with the sidecar's kv_request_id, and
		// remote_kv_source copied from EPP with its own.
		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		prefillKVParams, ok := preq[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		decodeParams, ok := prefillKVParams[reqcommon.FieldRemoteDecoder].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeParams[reqcommon.FieldKVRequestID]).ToNot(BeEmpty())
		Expect(decodeParams[reqcommon.FieldKVRequestID]).ToNot(Equal(sourceKVRequestID))
		expectSource(prefillKVParams)

		// Decode request: remote_prefiller only, never remote_kv_source.
		_, decodeKV := decodeKVParams()
		Expect(decodeKV).To(HaveKey(reqcommon.FieldRemotePrefiller))
		Expect(decodeKV).ToNot(HaveKey(reqcommon.FieldRemoteKVSource))

		// The pull is visible at the default level, with the source peer.
		Expect(logged).To(ContainElement(And(
			ContainSubstring("running P2P protocol"),
			ContainSubstring(net.JoinHostPort(sourceHost, strconv.Itoa(sourcePort))),
		)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not inject remote_kv_source without the body field", func() {
		proxyBaseAddr := testInfo.startProxy()

		postChatCompletions(proxyBaseAddr, chatCompletionsRequestBodyWithMaxCompletionCap, sourceHeaders(nil))

		dreq, _ := decodeKVParams()
		Expect(dreq).ToNot(HaveKey(reqcommon.FieldKVTransferParams))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	DescribeTable("should drop a malformed remote_kv_source and forward the rest of kv_transfer_params",
		func(source any) {
			proxyBaseAddr := testInfo.startProxy()

			postChatCompletions(proxyBaseAddr, sourceBody(source, map[string]any{"other": true}), sourceHeaders(nil))

			_, kvParams := decodeKVParams()
			Expect(kvParams).ToNot(HaveKey(reqcommon.FieldRemoteKVSource))
			Expect(kvParams).To(HaveKeyWithValue("other", true))

			testInfo.cancelFn()
			<-testInfo.stoppedCh
		},
		Entry("not an object", "10.9.9.9:7777"),
		Entry("missing kv_request_id", map[string]any{reqcommon.FieldRemoteHost: sourceHost, reqcommon.FieldRemotePort: sourcePort}),
		Entry("empty host", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: "", reqcommon.FieldRemotePort: sourcePort}),
		Entry("host not a string", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: 10, reqcommon.FieldRemotePort: sourcePort}),
		Entry("missing port", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: sourceHost}),
		Entry("port zero", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: sourceHost, reqcommon.FieldRemotePort: 0}),
		Entry("port above the TCP range", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: sourceHost, reqcommon.FieldRemotePort: 70000}),
		Entry("port not an integer", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: sourceHost, reqcommon.FieldRemotePort: 7777.5}),
		Entry("port a string", map[string]any{reqcommon.FieldKVRequestID: "k", reqcommon.FieldRemoteHost: sourceHost, reqcommon.FieldRemotePort: "7777"}),
	)

	It("should drop client-supplied kv_transfer_params keys next to the source", func() {
		proxyBaseAddr := testInfo.startProxy()

		postChatCompletions(proxyBaseAddr, sourceBody(validSource, map[string]any{
			reqcommon.FieldDoRemotePrefill: true,
			reqcommon.FieldRemoteHost:      "evil.example.com",
		}), sourceHeaders(nil))

		_, kvParams := decodeKVParams()
		// Only the sidecar-owned remote_kv_source key survives; the client's keys are gone.
		Expect(kvParams).To(HaveLen(1))
		expectSource(kvParams)

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should copy extra keys out of the source block", func() {
		proxyBaseAddr := testInfo.startProxy()

		source := map[string]any{
			reqcommon.FieldKVRequestID: sourceKVRequestID,
			reqcommon.FieldRemoteHost:  sourceHost,
			reqcommon.FieldRemotePort:  sourcePort,
			"extra":                    "dropped",
		}
		postChatCompletions(proxyBaseAddr, sourceBody(source, nil), sourceHeaders(nil))

		_, kvParams := decodeKVParams()
		expectSource(kvParams)
		Expect(kvParams[reqcommon.FieldRemoteKVSource]).To(HaveLen(3))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})

var _ = Describe("kvSourceFromBody", func() {
	const sourceHost = "10.9.9.9"

	// parsedBody returns a body carrying source next to one other
	// kv_transfer_params key, through the sidecar's own body decoder so the
	// numbers have the types a request has.
	parsedBody := func(source map[string]any) map[string]any {
		parsed, err := decodeRequestBody([]byte(sourceBody(source, map[string]any{"other": true})))
		Expect(err).ToNot(HaveOccurred())
		return parsed
	}

	validSource := map[string]any{
		reqcommon.FieldKVRequestID: "k2",
		reqcommon.FieldRemoteHost:  sourceHost,
		reqcommon.FieldRemotePort:  7778,
	}

	const clientIP = "10.1.1.1:5555"

	It("returns a copy of an allowed source and removes it from the body", func() {
		s := &Server{allowlistValidator: &AllowlistValidator{enabled: true, allowedHosts: set.New(sourceHost)}}
		body := parsedBody(validSource)

		source, present := s.kvSourceFromBody(logr.Discard(), clientIP, body)

		Expect(present).To(BeTrue())
		Expect(source).To(Equal(&kvSource{kvRequestID: "k2", p2pPeer: p2pPeer{host: sourceHost, port: 7778}}))
		Expect(body[reqcommon.FieldKVTransferParams]).To(Equal(map[string]any{"other": true}))
	})

	It("drops a source whose host is not in the allowlist", func() {
		s := &Server{allowlistValidator: &AllowlistValidator{enabled: true, allowedHosts: set.New("10.0.0.1")}}
		body := parsedBody(validSource)

		source, present := s.kvSourceFromBody(logr.Discard(), clientIP, body)

		Expect(present).To(BeTrue())
		Expect(source).To(BeNil())
		Expect(body[reqcommon.FieldKVTransferParams]).To(Equal(map[string]any{"other": true}))
	})

	It("removes kv_transfer_params when the source was its only key", func() {
		s := &Server{allowlistValidator: &AllowlistValidator{enabled: true, allowedHosts: set.New(sourceHost)}}
		body := map[string]any{reqcommon.FieldKVTransferParams: map[string]any{reqcommon.FieldRemoteKVSource: validSource}}

		source, present := s.kvSourceFromBody(logr.Discard(), clientIP, body)

		Expect(present).To(BeTrue())
		Expect(source).ToNot(BeNil())
		Expect(body).ToNot(HaveKey(reqcommon.FieldKVTransferParams))
	})

	// The header and the block come from one EPP step, so a header alone is
	// worth a line to the operator; the request proceeds without a pull.
	It("reports no source, with a warning, when the body carries none", func() {
		s := &Server{allowlistValidator: &AllowlistValidator{enabled: true}}
		logged := captureLogs(s, 0)

		source, present := s.kvSourceFromBody(s.logger, clientIP, map[string]any{})

		Expect(present).To(BeFalse())
		Expect(source).To(BeNil())
		Expect(*logged).To(HaveLen(1))
		Expect((*logged)[0]).To(ContainSubstring("without a remote_kv_source block"))
		Expect((*logged)[0]).To(ContainSubstring(clientIP))
	})
})
