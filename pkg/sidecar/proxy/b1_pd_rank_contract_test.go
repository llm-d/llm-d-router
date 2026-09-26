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
	"net/http"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:gomega

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// B1-U04: the two legs of an SGLang shared-frontend P/D request must keep
// independent ranks.
//
// SGLang's contract (sglang/srt/managers/io_struct.py, GenerateReqInput):
//   - routed_dp_rank: the rank that must execute THIS leg;
//   - disagg_prefill_dp_rank: a hint on the decode leg naming the prefill rank
//     that will send the KV.
//
// Both default to null, so rank 0 and "absent" stay distinguishable, and
// routed_dp_rank is validated against the engine's dp size.
const (
	b1FieldRoutedDPRank        = "routed_dp_rank"
	b1FieldDisaggPrefillDPRank = "disagg_prefill_dp_rank"
)

// sendSidecarRequest posts a request with the prefill header and returns both
// legs' captured bodies and headers.
func sendSidecarRequest(testInfo *sidecarTestInfo, body map[string]any) (prefill, decode map[string]any,
	prefillHdr, decodeHdr http.Header,
) {
	raw, err := json.Marshal(body)
	Expect(err).ToNot(HaveOccurred())

	req, err := http.NewRequest(http.MethodPost,
		"http://"+testInfo.proxy.addr.String()+reqcommon.PathCompletions, bytes.NewReader(raw))
	Expect(err).ToNot(HaveOccurred())
	req.Header.Add(routing.PrefillEndpointHeader,
		testInfo.prefillBackend.URL[len("http://"):])

	rp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	defer rp.Body.Close()
	Expect(rp.StatusCode).To(Equal(http.StatusOK))

	Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))
	Eventually(func() int { return len(testInfo.decodeHandler.GetCompletionRequests()) }).Should(Equal(1))

	return testInfo.prefillHandler.GetCompletionRequests()[0],
		testInfo.decodeHandler.GetCompletionRequests()[0],
		testInfo.prefillHandler.GetCompletionHeaders()[0],
		testInfo.decodeHandler.GetCompletionHeaders()[0]
}

var _ = Describe("SGLang shared-frontend rank contract (B1-U04)", func() {
	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		testInfo = sidecarConnectionTestSetup(KVConnectorSGLang)
		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			Expect(testInfo.proxy.Start(testInfo.ctx)).To(Succeed())
			testInfo.stoppedCh <- struct{}{}
		}()
		<-testInfo.proxy.readyCh
	})

	AfterEach(func() {
		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("forwards the client's independent prefill and decode ranks unchanged", func() {
		prefill, decode, _, _ := sendSidecarRequest(testInfo, map[string]any{
			"model":                    "Qwen/Qwen2-0.5B",
			"prompt":                   "Hello",
			"max_tokens":               8,
			b1FieldRoutedDPRank:        2,
			b1FieldDisaggPrefillDPRank: 1,
		})

		Expect(prefill).To(HaveKeyWithValue(b1FieldRoutedDPRank, BeNumerically("==", 2)))
		Expect(prefill).To(HaveKeyWithValue(b1FieldDisaggPrefillDPRank, BeNumerically("==", 1)))
		Expect(decode).To(HaveKeyWithValue(b1FieldRoutedDPRank, BeNumerically("==", 2)))
		Expect(decode).To(HaveKeyWithValue(b1FieldDisaggPrefillDPRank, BeNumerically("==", 1)))
	})

	It("does not invent a rank for a request that carries none", func() {
		prefill, decode, prefillHdr, decodeHdr := sendSidecarRequest(testInfo, map[string]any{
			"model":      "Qwen/Qwen2-0.5B",
			"prompt":     "Hello",
			"max_tokens": 8,
		})

		Expect(prefill).ToNot(HaveKey(b1FieldRoutedDPRank),
			"an unranked request must not be pinned to rank 0")
		Expect(decode).ToNot(HaveKey(b1FieldRoutedDPRank))
		Expect(prefill).ToNot(HaveKey(b1FieldDisaggPrefillDPRank))
		Expect(decode).ToNot(HaveKey(b1FieldDisaggPrefillDPRank))

		// The sidecar's own DP pinning is header based (MoRI-IO/Offloading mode).
		// SGLang's rank fields are body fields, so no header may be synthesised
		// for them either.
		Expect(prefillHdr.Get(requestHeaderDataParallelRank)).To(BeEmpty())
		Expect(decodeHdr.Get(requestHeaderDataParallelRank)).To(BeEmpty())
	})

	It("lets each leg carry its own rank without the other leg's rank overwriting it", func() {
		// The EPP selects prefill rank 3 and decode rank 0. Both are ranks; the
		// request must not collapse them into one value, and decode rank 0 must
		// not be dropped as "unset".
		prefill, decode, _, _ := sendSidecarRequest(testInfo, map[string]any{
			"model":                    "Qwen/Qwen2-0.5B",
			"prompt":                   "Hello",
			"max_tokens":               8,
			b1FieldRoutedDPRank:        0,
			b1FieldDisaggPrefillDPRank: 3,
		})

		Expect(prefill).To(HaveKeyWithValue(b1FieldDisaggPrefillDPRank, BeNumerically("==", 3)))
		Expect(decode).To(HaveKeyWithValue(b1FieldRoutedDPRank, BeNumerically("==", 0)),
			"decode rank 0 is a selection, not an absent rank")
		Expect(decode).To(HaveKeyWithValue(b1FieldDisaggPrefillDPRank, BeNumerically("==", 3)))
	})

	It("leaves engine-side rank validation to the engine", func() {
		// The router does not know the engine's dp size, so an out-of-range rank
		// must reach SGLang unchanged and fail there (ValueError) rather than be
		// silently rewritten to rank 0 by the proxy.
		_, decode, _, _ := sendSidecarRequest(testInfo, map[string]any{
			"model":             "Qwen/Qwen2-0.5B",
			"prompt":            "Hello",
			"max_tokens":        8,
			b1FieldRoutedDPRank: 99,
		})

		Expect(decode).To(HaveKeyWithValue(b1FieldRoutedDPRank, BeNumerically("==", 99)),
			"an out-of-range rank must not be silently coerced")
	})
})
