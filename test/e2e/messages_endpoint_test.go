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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/test/e2e/utils/k8s"
	"github.com/llm-d/llm-d-router/test/e2e/utils/metrics"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

var _ = ginkgo.Describe("P/D gateway /v1/messages", ginkgo.Ordered, testWrapper(func() {
	// The sidecar serves /v1/messages from the same disaggregatedPrefillHandler
	// loop as /v1/responses, so a Messages request has to reach the prefill pod
	// rather than fall through to the decoder catch-all.
	ginkgo.It("routes a text Messages request to the prefill pod", func() {
		nsName := getNamespace()

		prefillReplicas, decodeReplicas := 1, 1
		createModelServersPDSharedStorage(decodeReplicas)
		standalone.Create(standaloneConfig(), pdConfig, 1, 8000)

		prefillPods, decodePods := k8s.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
		gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

		prefillBefore := metrics.GetPodRequestCount(testConfig, nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count before: %d", prefillBefore))

		ginkgo.By("POST a /v1/messages body")
		resp, raw := doMessages(messagesBody(simModelName, textBlockTurn(reqcommon.RoleUser, pdTextInput)))
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
			"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

		prefillAfter := metrics.GetPodRequestCount(testConfig, nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count after: %d", prefillAfter))
		gomega.Expect(prefillAfter).To(gomega.BeNumerically(">", prefillBefore),
			"prefill pod should have received the Messages request; the sidecar must route "+
				"/v1/messages through disaggregatedPrefillHandler, not the decoder catch-all")
	})
}))

var _ = ginkgo.Describe("E/P/D gateway /v1/messages encoder-cache fanout", ginkgo.Ordered, testWrapper(func() {
	// An Anthropic image block is not a chat content part, and the encoder is
	// posted at /v1/chat/completions. Each case fixes one rule the reshaping
	// applies, so a regression in it changes whether the encoder is primed.
	ginkgo.DescribeTable("primes the encoder per the Messages block rules",
		func(primed bool, turns ...map[string]any) {
			nsName := getNamespace()

			encodeReplicas, prefillReplicas, decodeReplicas := 1, 1, 1
			createModelServersEPDDisagg(encodeReplicas, prefillReplicas, decodeReplicas)
			standalone.Create(standaloneConfig(), epdConfig, 1, 8000)

			encodePods := k8s.GetPodNames(testConfig, encodeSelector, nsName)
			gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))

			encodeBefore := metrics.GetPodRequestCount(testConfig, nsName, encodePods[0])
			ginkgo.By(fmt.Sprintf("encode request count before: %d", encodeBefore))

			resp, raw := doMessages(messagesBody(simModelName, turns...))
			gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
				"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

			encodeAfter := metrics.GetPodRequestCount(testConfig, nsName, encodePods[0])
			ginkgo.By(fmt.Sprintf("encode request count after: %d", encodeAfter))
			if primed {
				gomega.Expect(encodeAfter).To(gomega.BeNumerically(">", encodeBefore),
					"encode pod should have been primed from the Messages image block")
				return
			}
			gomega.Expect(encodeAfter).To(gomega.Equal(encodeBefore),
				"encode pod should not have been primed")
		},
		// A user turn's own content is the array vLLM loads images from.
		ginkgo.Entry("user turn image block", true,
			imageBlockTurn(reqcommon.RoleUser, testImageURL)),
		// messagesPartArrays returns no array for a system turn, so vLLM keeps
		// nothing but text there and priming would buy an encoder call for an
		// image the model never sees.
		ginkgo.Entry("system turn image block", false,
			imageBlockTurn(reqcommon.RoleSystem, testImageURL)),
		// A user turn's tool_result content is loaded one level deep.
		ginkgo.Entry("user turn tool_result image block", true,
			toolResultImageTurn(testImageURL)),
		// An assistant turn carries no tool_result array, so the nested image
		// stays unprimed while the turn's own content is still collected.
		ginkgo.Entry("assistant turn tool_result image block", false,
			map[string]any{
				reqcommon.FieldRole: "assistant",
				reqcommon.FieldContent: []any{map[string]any{
					reqcommon.FieldType:    reqcommon.BlockTypeToolResult,
					reqcommon.FieldContent: []any{imageBlock(testImageURL)},
				}},
			}),
	)
}))

// renderLogSettleWindow is how long the EPP log has to stay free of a
// token-producer failure after the request returns.
const renderLogSettleWindow = 10 * time.Second

var _ = ginkgo.Describe("gateway /v1/messages native rendering", ginkgo.Ordered, testWrapper(func() {
	// messagesRenderMode is left at its "auto" default so the spec covers
	// discovery: the plugin probes /v1/messages/render once and falls back to the
	// deprecated chat conversion on 404 or 405. Pinning "native" would skip that
	// probe and leave the fallback unreachable.
	//
	// A DataProducer failure does not fail the request: the director logs it and
	// schedules on. Both a renderer that cannot serve the route and a silent
	// downgrade to the conversion therefore show up only in the EPP log.
	ginkgo.It("renders a Messages request through the renderer without falling back", func() {
		nsName := getNamespace()

		createModelServersDecodeKV(1)
		router := standalone.Create(standaloneConfig(), kvExternalTokenizerConfig(), 1, 8000)

		_, decodePods := k8s.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(decodePods).Should(gomega.HaveLen(1))

		ginkgo.By("POST a /v1/messages body against the token-producer render backend")
		resp, raw := doMessages(messagesBody(kvModelName, textBlockTurn(reqcommon.RoleUser, pdTextInput)))
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
			"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

		eppPods := k8s.GetPodNames(testConfig, router.Selector, nsName)
		gomega.Expect(eppPods).ShouldNot(gomega.BeEmpty())

		// Consistently, not Eventually: both assertions are about a line that
		// must never appear, and Eventually passes on its first poll before a
		// late line could land. The director logs a producer failure before it
		// responds, so the window only has to outlast log delivery.
		gomega.Consistently(func(g gomega.Gomega) {
			logs := testutils.PodLogs(testConfig, nsName, eppPods[0])
			g.Expect(logs).ShouldNot(gomega.BeEmpty(), "EPP logs unreadable, cannot prove rendering succeeded")
			g.Expect(logLinesContaining(logs, "failed to prepare per request data")).To(gomega.BeEmpty(),
				"the token-producer failed; native Messages rendering did not reach the renderer")
			g.Expect(logLinesContaining(logs, "messagesRenderMode=legacy is deprecated")).To(gomega.BeEmpty(),
				"discovery fell back to the deprecated conversion; the renderer did not serve /v1/messages/render")
		}, renderLogSettleWindow, testConfig.Interval).Should(gomega.Succeed())
	})
}))

// doMessages POSTs a Messages body to /v1/messages through the gateway.
func doMessages(body map[string]any) (*http.Response, []byte) {
	return doRequest(reqcommon.PathMessages, mustMarshal(body))
}

// messagesBody wraps turns in a Messages request for model. The Anthropic schema
// requires max_tokens; echoOutputTokens keeps it above the prompt length, which
// echo mode refuses to answer below.
func messagesBody(model string, turns ...map[string]any) map[string]any {
	messages := make([]any, 0, len(turns))
	for _, turn := range turns {
		messages = append(messages, turn)
	}
	return map[string]any{
		"model":                  model,
		"messages":               messages,
		reqcommon.FieldMaxTokens: echoOutputTokens,
	}
}

// textBlockTurn is a turn carrying a single text block.
func textBlockTurn(role, text string) map[string]any {
	return map[string]any{
		reqcommon.FieldRole: role,
		reqcommon.FieldContent: []any{map[string]any{
			reqcommon.FieldType: "text",
			"text":              text,
		}},
	}
}

// imageBlock is an Anthropic image block naming its image by URL.
func imageBlock(url string) map[string]any {
	return map[string]any{
		reqcommon.FieldType: reqcommon.BlockTypeImage,
		reqcommon.FieldSource: map[string]any{
			reqcommon.FieldType: reqcommon.ImageSourceTypeURL,
			reqcommon.FieldURL:  url,
		},
	}
}

// imageBlockTurn is a turn carrying a text block and an image block.
func imageBlockTurn(role, url string) map[string]any {
	turn := textBlockTurn(role, "Describe what you see.")
	turn[reqcommon.FieldContent] = append(turn[reqcommon.FieldContent].([]any), imageBlock(url))
	return turn
}

// toolResultImageTurn is a user turn whose tool_result content holds an image.
func toolResultImageTurn(url string) map[string]any {
	return map[string]any{
		reqcommon.FieldRole: reqcommon.RoleUser,
		reqcommon.FieldContent: []any{map[string]any{
			reqcommon.FieldType:    reqcommon.BlockTypeToolResult,
			reqcommon.FieldContent: []any{imageBlock(url)},
		}},
	}
}

// logLinesContaining returns the log lines holding marker. Asserting on the
// matches rather than the whole log keeps a failure's output to the offending
// lines, which Gomega prints in full where it truncates a pod's entire log.
func logLinesContaining(logs, marker string) []string {
	var found []string
	for _, line := range strings.Split(logs, "\n") {
		if strings.Contains(line, marker) {
			found = append(found, line)
		}
	}
	return found
}
