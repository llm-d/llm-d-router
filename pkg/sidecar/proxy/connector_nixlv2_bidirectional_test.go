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
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive
	"github.com/onsi/gomega/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

var _ = Describe("NIXL Connector (v2) bidirectional KV transfer", func() {
	const (
		reply = "Hi there"
		// The prefix-cache peer EPP names in the request body and header.
		sourceHost        = "10.9.9.9"
		sourcePort        = 7778
		sourceKVRequestID = "k2-from-epp"
	)

	var (
		testInfo      *sidecarTestInfo
		proxyBaseAddr string
		// sendSource makes chat carry the P2P source EPP's p2p-source-producer writes.
		sendSource bool
	)

	// The decode engine's response to every turn: the generated reply plus the
	// location of the blocks it keeps for the next turn.
	decodeJSON := `{"id":"chatcmpl-1","object":"chat.completion",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"` + reply + `"},"finish_reason":"stop"}],` +
		`"kv_transfer_params":{"do_remote_prefill":false,"do_remote_decode":true,` +
		`"remote_block_ids":[[1,2,3]],"remote_engine_id":"decode-engine","remote_request_id":"chatcmpl-1",` +
		`"remote_host":"10.0.0.7","remote_port":5600,"remote_num_tokens":192,"tp_size":1,"not_part_of_the_contract":"dropped"}}`

	decodeSSE := func() string {
		return "data: " + `{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi "}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{"content":"there"}}]}` + "\n\n" +
			"data: " + `{"choices":[],"kv_transfer_params":{"do_remote_prefill":false,"do_remote_decode":true,` +
			`"remote_block_ids":[[1,2,3]],"remote_engine_id":"decode-engine","remote_request_id":"chatcmpl-1",` +
			`"remote_host":"10.0.0.7","remote_port":5600,"remote_num_tokens":192}}` + "\n\n" +
			"data: [DONE]\n\n"
	}

	BeforeEach(func() {
		sendSource = true
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorNIXLV2)
		testInfo.prefillHandler.BidirectionalKVMode = true
		testInfo.decodeHandler.RawResponse = decodeJSON
		testInfo.proxy = NewProxy(Config{
			Port:                       "0",
			DecoderURL:                 testInfo.decodeURL,
			KVConnector:                constants.KVConnectorNIXLV2,
			EnableP2PPull:              true,
			PrefillMaxRetries:          1,
			PrefillRetryBackoff:        time.Millisecond,
			BidirectionalKVXfer:        true,
			BidirectionalSessionHeader: sessionutil.DefaultHeader,
			BidirectionalCacheSize:     16,
			BidirectionalCacheTTL:      time.Minute,
			PodName:                    testPodName,
			PodNamespace:               testPodNamespace,
			DataParallelSize:           1,
		})
	})

	// start launches the proxy once the test has finished configuring the mock
	// backends.
	start := func() {
		proxyBaseAddr = testInfo.startProxy()
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})
	}

	// chatVia posts a chat completion with the given history, routed to the
	// prefiller at prefillURL, carrying the session token the EPP issued for rank
	// 0 of pod. An empty pod sends no token, as the first turn of a conversation
	// does.
	chatVia := func(prefillURL, pod string, stream bool, messages ...any) {
		GinkgoHelper()
		request := map[string]any{
			"model":      "Qwen/Qwen2-0.5B",
			"messages":   messages,
			"max_tokens": 50,
			"stream":     stream,
		}
		if sendSource {
			request[reqcommon.FieldKVTransferParams] = map[string]any{
				reqcommon.FieldRemoteKVSource: map[string]any{
					reqcommon.FieldKVRequestID: sourceKVRequestID,
					reqcommon.FieldRemoteHost:  sourceHost,
					reqcommon.FieldRemotePort:  sourcePort,
				},
			}
		}
		body, err := json.Marshal(request)
		Expect(err).ToNot(HaveOccurred())
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, prefillURL[len("http://"):])
		if sendSource {
			req.Header.Add(routing.KVCacheSourceHeader, net.JoinHostPort(sourceHost, strconv.Itoa(sourcePort)))
		}
		if pod != "" {
			req.Header.Add(sessionutil.DefaultHeader, eppSessionToken(testPodNamespace, pod, 0))
		}
		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()
		b, err := io.ReadAll(resp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK), string(b))
	}

	chat := func(pod string, stream bool, messages ...any) {
		GinkgoHelper()
		chatVia(testInfo.prefillBackend.URL, pod, stream, messages...)
	}

	// kvOf returns the kv_transfer_params of a recorded prefill request.
	kvOf := func(req map[string]any) map[string]any {
		GinkgoHelper()
		kv, ok := req[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		return kv
	}

	// prefillKV returns the kv_transfer_params of the i-th prefill request, which
	// must be the most recent one.
	prefillKV := func(i int) map[string]any {
		GinkgoHelper()
		reqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(reqs).To(HaveLen(i + 1))
		return kvOf(reqs[i])
	}

	// theSource matches the remote_kv_source block the prefill request must carry.
	theSource := func() types.GomegaMatcher {
		return And(
			HaveKeyWithValue(reqcommon.FieldKVRequestID, sourceKVRequestID),
			HaveKeyWithValue(reqcommon.FieldRemoteHost, sourceHost),
			HaveKeyWithValue(reqcommon.FieldRemotePort, BeNumerically("==", sourcePort)),
		)
	}

	turn1 := func() []any { return []any{chatMessage("user", "Hello")} }
	turn2 := func() []any {
		return []any{chatMessage("user", "Hello"), chatMessage("assistant", reply), chatMessage("user", "And now?")}
	}

	It("replays the previous turn's decode-side blocks next to the P2P source", func() {
		start()
		chat("", false, turn1()...)
		cold := prefillKV(0)
		Expect(cold).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()), "a cold conversation has nothing to replay")
		Expect(cold).To(HaveKeyWithValue(reqcommon.FieldRemoteKVSource, theSource()), "the P2P pull does not depend on a replay")

		chat(testPodName, false, turn2()...)
		kv := prefillKV(1)
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldDoRemoteDecode, true))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldDoRemotePrefill, false))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, "decode-engine"))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteRequestID, "chatcmpl-1"))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteHost, "10.0.0.7"))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemotePort, BeNumerically("==", 5600)))
		Expect(kv).To(HaveKeyWithValue(requestFieldRemoteNumTokens, BeNumerically("==", 192)))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteBlockIDs, ConsistOf(ConsistOf(BeNumerically("==", 1), BeNumerically("==", 2), BeNumerically("==", 3)))))
		Expect(kv).ToNot(HaveKey("not_part_of_the_contract"))
		// Both sources stay on the request, unchanged: the engine's MultiConnector
		// takes the first connector that reports a hit and applies
		// kv_recompute_threshold to the NIXL read itself.
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteKVSource, theSource()))
	})

	It("replays without a P2P source when the request names none", func() {
		sendSource = false
		start()
		chat("", false, turn1()...)
		chat(testPodName, false, turn2()...)
		kv := prefillKV(1)
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, "decode-engine"))
		Expect(kv).ToNot(HaveKey(reqcommon.FieldRemoteKVSource))
	})

	It("replays an entry once", func() {
		start()
		chat("", false, turn1()...)
		chat(testPodName, false, turn2()...)
		Expect(prefillKV(1)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, "decode-engine"))

		// The same follow-up again: the blocks it named were released by the
		// first read.
		chat(testPodName, false, turn2()...)
		Expect(prefillKV(2)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()))
	})

	It("retries a failed prefill without the replayed blocks and keeps the P2P source", func() {
		// The first attempt fails before the engine records it, so the recorded
		// request is the retry. The failed attempt may already have read and
		// released the decode-side blocks, so the retry recomputes the prefill.
		flaky := &mock.ChatCompletionHandler{
			Connector:           constants.KVConnectorNIXLV2,
			Role:                mock.RolePrefill,
			BidirectionalKVMode: true,
			FailForFirstN:       1,
			FailStatusCode:      http.StatusServiceUnavailable,
		}
		flakyBackend := httptest.NewServer(flaky)
		DeferCleanup(flakyBackend.Close)

		start()
		chat("", false, turn1()...)
		chatVia(flakyBackend.URL, testPodName, false, turn2()...)

		Expect(flaky.RequestCount.Load()).To(BeEquivalentTo(2), "one failed attempt and one retry")
		reqs := flaky.GetCompletionRequests()
		Expect(reqs).To(HaveLen(1))
		kv := kvOf(reqs[0])
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()))
		Expect(kv).To(HaveKeyWithValue(reqcommon.FieldRemoteKVSource, theSource()))

		// The entry was consumed by the first attempt and is not offered again.
		chat(testPodName, false, turn2()...)
		Expect(prefillKV(1)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()))
	})

	It("does not replay into a conversation whose history differs", func() {
		start()
		chat("", false, turn1()...)
		chat(testPodName, false, chatMessage("user", "Hello"), chatMessage("assistant", "Something else"), chatMessage("user", "And now?"))
		Expect(prefillKV(1)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()))
	})

	It("ignores a session token for another pod", func() {
		start()
		chat("", false, turn1()...)
		chat("decode-1", false, turn2()...)
		Expect(prefillKV(1)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()))
	})

	It("captures a streamed decode response without delaying it", func() {
		testInfo.decodeHandler.RawResponse = decodeSSE()
		testInfo.decodeHandler.RawResponseType = eventStreamContentType
		start()

		chat("", true, turn1()...)
		chat(testPodName, true, turn2()...)
		Expect(prefillKV(1)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, "decode-engine"))
	})
})
