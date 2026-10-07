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
	"net/http"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

var _ = Describe("NIXL Connector (v2) bidirectional KV transfer", func() {
	const (
		kvCacheSource = "10.9.9.9:8000"
		reply         = "Hi there"
	)

	var (
		testInfo      *sidecarTestInfo
		proxyBaseAddr string
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
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorNIXLV2)
		testInfo.prefillHandler.BidirectionalKVMode = true
		testInfo.decodeHandler.RawResponse = decodeJSON
		testInfo.proxy = NewProxy(Config{
			Port:                       "0",
			DecoderURL:                 testInfo.decodeURL,
			KVConnector:                constants.KVConnectorNIXLV2,
			EnableP2PPull:              true,
			P2PConnectorPort:           7777,
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

	// chat posts a chat completion with the given history, routed to the test
	// prefiller, carrying the session token the EPP issued for rank 0 of pod. An
	// empty pod sends no token, as the first turn of a conversation does.
	chat := func(pod string, stream bool, messages ...any) {
		GinkgoHelper()
		body, err := json.Marshal(map[string]any{
			"model":      "Qwen/Qwen2-0.5B",
			"messages":   messages,
			"max_tokens": 50,
			"stream":     stream,
		})
		Expect(err).ToNot(HaveOccurred())
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])
		req.Header.Add(routing.KVCacheSourceHeader, kvCacheSource)
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

	// prefillKV returns the kv_transfer_params of the i-th prefill request, which
	// must be the most recent one.
	prefillKV := func(i int) map[string]any {
		GinkgoHelper()
		reqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(reqs).To(HaveLen(i + 1))
		kv, ok := reqs[i][reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		return kv
	}

	turn1 := func() []any { return []any{chatMessage("user", "Hello")} }
	turn2 := func() []any {
		return []any{chatMessage("user", "Hello"), chatMessage("assistant", reply), chatMessage("user", "And now?")}
	}

	It("replays the previous turn's decode-side blocks next to the P2P source", func() {
		start()
		chat("", false, turn1()...)
		Expect(prefillKV(0)).To(HaveKeyWithValue(reqcommon.FieldRemoteEngineID, BeNil()), "a cold conversation has nothing to replay")

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
		// Both sources stay on the request; the engine's MultiConnector picks
		// the first one that reports a hit.
		Expect(kv).To(HaveKey(requestFieldRemoteKVSource))
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
