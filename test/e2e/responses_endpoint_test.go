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

package e2e

import (
	"fmt"
	"net/http"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/test/e2e/utils"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
)

// responsesTextInput is the prompt every Responses spec sends. Long enough to
// clear the prefix-based-pd-decider nonCachedTokens threshold so a P/D setup
// disaggregates on a cold cache rather than serving from decode alone.
const responsesTextInput = "Describe in detail what a router does when it disaggregates an inference request across pods."

var _ = ginkgo.Describe("P/D gateway /v1/responses", ginkgo.Ordered, testWrapper(func() {
	// The sidecar refuses a stateful Responses field in readJSONBody, before
	// any connector runs, so the refusal has to be visible end-to-end: the
	// client sees a 400 naming the field and no worker sees the request.
	ginkgo.It("refuses stateful fields without dispatching to a worker", func() {
		nsName := getNamespace()

		prefillReplicas, decodeReplicas := 1, 1
		createModelServersPDSharedStorage(decodeReplicas)
		standalone.Create(standaloneConfig(), pdConfig, 1, 8000)

		prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
		gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

		prefillBefore := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
		decodeBefore := utils.GetPodRequestCount(testConfig, nsName, decodePods[0])
		gomega.Expect(prefillBefore).To(gomega.BeNumerically(">=", 0), "prefill pod metrics unreadable, cannot prove a request was not dispatched")
		gomega.Expect(decodeBefore).To(gomega.BeNumerically(">=", 0), "decode pod metrics unreadable, cannot prove a request was not dispatched")

		// One case per field RejectStatefulResponsesFields refuses. file_id is
		// nested in an input content part rather than sent at the top level.
		cases := []struct {
			field string
			body  map[string]any
		}{
			{reqcommon.FieldPreviousResponseID, map[string]any{
				"model": simModelName, "input": responsesTextInput,
				reqcommon.FieldPreviousResponseID: "resp-e2e-123",
			}},
			{reqcommon.FieldConversation, map[string]any{
				"model": simModelName, "input": responsesTextInput,
				reqcommon.FieldConversation: "conv-e2e-123",
			}},
			{reqcommon.FieldBackground, map[string]any{
				"model": simModelName, "input": responsesTextInput,
				reqcommon.FieldBackground: true,
			}},
			{reqcommon.FieldFileID, map[string]any{
				"model": simModelName,
				"input": []any{map[string]any{
					"role": "user",
					"content": []any{map[string]any{
						"type": "input_image", reqcommon.FieldFileID: "file-e2e-123",
					}},
				}},
			}},
		}

		for _, tc := range cases {
			ginkgo.By("POST /v1/responses carrying " + tc.field)
			resp, raw := doResponses(tc.body)
			gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusBadRequest),
				"expected 400 for %s, got status=%d body=%s", tc.field, resp.StatusCode, string(raw))
			gomega.Expect(string(raw)).To(gomega.ContainSubstring(tc.field),
				"error should name the unsupported field: %s", string(raw))
		}

		gomega.Expect(utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])).To(gomega.Equal(prefillBefore),
			"prefill pod received a request the sidecar should have refused")
		gomega.Expect(utils.GetPodRequestCount(testConfig, nsName, decodePods[0])).To(gomega.Equal(decodeBefore),
			"decode pod received a request the sidecar should have refused")
	})

	// A supported Responses request has to disaggregate like chat completions:
	// the sidecar routes it through disaggregatedPrefillHandler rather than
	// falling through to the decoder catch-all.
	ginkgo.It("routes a text Responses request to the prefill pod", func() {
		nsName := getNamespace()

		prefillReplicas, decodeReplicas := 1, 1
		createModelServersPDSharedStorage(decodeReplicas)
		standalone.Create(standaloneConfig(), pdConfig, 1, 8000)

		prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
		gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

		prefillBefore := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count before: %d", prefillBefore))

		ginkgo.By("POST a supported /v1/responses body")
		resp, raw := doResponses(map[string]any{
			"model":             simModelName,
			"input":             responsesTextInput,
			"max_output_tokens": 20,
		})
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
			"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

		prefillAfter := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count after: %d", prefillAfter))
		gomega.Expect(prefillAfter).To(gomega.BeNumerically(">", prefillBefore),
			"prefill pod should have received the Responses request; the sidecar must route "+
				"/v1/responses through disaggregatedPrefillHandler, not the decoder catch-all")
	})
}))

var _ = ginkgo.Describe("E/P/D gateway /v1/responses encoder-cache fanout", ginkgo.Ordered, testWrapper(func() {
	// Pending on EPP-side multimodal detection (#3003). The disagg profile handler
	// selects the encode profile from PromptTokens.MultiModalFeatures, and the
	// estimate tokenizer backend reports none for a Responses body: it covers
	// ChatCompletions and Messages, while a Responses body falls through to
	// estimateBytes, which marshals input without reporting features. With no
	// encode profile the gateway sends no encoder endpoint header, so the
	// sidecar never runs the fanout and the request succeeds through prefill
	// and decode with the encode pod idle.
	ginkgo.PIt("primes the encoder from Responses input_image content", func() {
		nsName := getNamespace()

		encodeReplicas, prefillReplicas, decodeReplicas := 1, 1, 1
		createModelServersEPDDisagg(encodeReplicas, prefillReplicas, decodeReplicas)
		standalone.Create(standaloneConfig(), epdConfig, 1, 8000)

		encodePods := utils.GetPodNames(testConfig, encodeSelector, nsName)
		gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))

		encodeBefore := utils.GetPodRequestCount(testConfig, nsName, encodePods[0])
		ginkgo.By(fmt.Sprintf("encode request count before: %d", encodeBefore))

		// A Responses input_image carries its URL as a bare string on the part,
		// unlike chat completions' nested image_url object.
		ginkgo.By("POST /v1/responses with an input_image part")
		resp, raw := doResponses(map[string]any{
			"model": simModelName,
			"input": []any{map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Describe what you see."},
					map[string]any{"type": "input_image", "image_url": testImageURL},
				},
			}},
			"max_output_tokens": 20,
		})
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
			"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

		encodeAfter := utils.GetPodRequestCount(testConfig, nsName, encodePods[0])
		ginkgo.By(fmt.Sprintf("encode request count after: %d", encodeAfter))
		gomega.Expect(encodeAfter).To(gomega.BeNumerically(">", encodeBefore),
			"encode pod should have been primed from the Responses input_image; the fanout must "+
				"read input rather than messages when the client sends a Responses request")
	})
}))

// doResponses POSTs a Responses body to /v1/responses through the gateway.
func doResponses(body map[string]any) (*http.Response, []byte) {
	return doRequest(reqcommon.PathResponses, mustMarshal(body))
}
