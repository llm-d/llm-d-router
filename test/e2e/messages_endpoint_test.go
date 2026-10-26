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
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/test/e2e/utils/k8s"
	"github.com/llm-d/llm-d-router/test/e2e/utils/metrics"
	"github.com/llm-d/llm-d-router/test/e2e/utils/response"
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

var _ = ginkgo.Describe("Messages API", ginkgo.Ordered, testWrapper(func() {
	for _, topology := range []string{"aggregated", "P/D"} {
		ginkgo.Describe(topology+" routing", func() {
			var prefillPods, decodePods []string
			var metricsURL string

			ginkgo.BeforeEach(func() {
				config := simpleConfig
				if topology == "P/D" {
					createModelServersPDSharedStorage(1)
					config = pdConfig
				} else {
					createModelServersDecode(1)
				}
				standalone.Create(standaloneConfig(), config, 1, 8000)
				prefillPods, decodePods = k8s.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, getNamespace())
				gomega.Expect(decodePods).To(gomega.HaveLen(1))
				if topology == "P/D" {
					gomega.Expect(prefillPods).To(gomega.HaveLen(1))
				}
				metricsURL = fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())
			})

			for _, stream := range []bool{false, true} {
				format := "JSON"
				if stream {
					format = "SSE"
				}
				for _, apiType := range []reqcommon.APIType{reqcommon.APITypeMessages, reqcommon.APITypeChatCompletions} {
					ginkgo.It(fmt.Sprintf("preserves %s %s responses and metrics", apiType.String(), format), func() {
						prompt := uuid.NewString() + " " + pdTextInput
						body := map[string]any{"model": simModelName, "max_tokens": echoOutputTokens, "stream": stream, "messages": []any{map[string]string{"role": "user", "content": prompt}}}
						headers := map[string]string{}
						if apiType == reqcommon.APITypeMessages {
							body["model"] = "messages-client-model"
							headers[metadata.ModelNameRewriteKey] = simModelName
						} else if stream {
							body["stream_options"] = map[string]bool{"include_usage": true}
						}
						before := apiSurfaceMetrics(metricsURL)
						prefillBefore := 0
						if topology == "P/D" {
							prefillBefore = metrics.GetPodRequestCount(testConfig, getNamespace(), prefillPods[0])
							gomega.Expect(prefillBefore).To(gomega.BeNumerically(">=", 0))
						}
						ginkgo.By(fmt.Sprintf("POST %s stream=%t", apiType.Path(), stream))
						resp, raw := doRequest(apiType.Path(), mustMarshal(body), headers)
						gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK), "response: %s", raw)
						ns, pod, _ := extractInferenceHeaders(resp)
						gomega.Expect(ns).To(gomega.Equal(getNamespace()))
						gomega.Expect(pod).To(gomega.Equal(decodePods[0]))
						var usage map[string]int
						if apiType == reqcommon.APITypeMessages {
							reply, err := decodeMessagesResponse(raw, stream)
							gomega.Expect(err).NotTo(gomega.HaveOccurred())
							gomega.Expect(reply.ID).To(gomega.HavePrefix("msg_"))
							gomega.Expect(reply.Type).To(gomega.Equal("message"))
							gomega.Expect(reply.Role).To(gomega.Equal("assistant"))
							gomega.Expect(reply.Model).To(gomega.Equal(body["model"]))
							gomega.Expect(reply.Content).To(gomega.HaveLen(1))
							gomega.Expect(reply.Content[0].Type).To(gomega.Equal("text"))
							gomega.Expect(reply.Content[0].Text).To(gomega.Equal(prompt))
							gomega.Expect(reply.StopReason).To(gomega.BeElementOf("end_turn", "max_tokens"))
							usage = reply.Usage
							gomega.Expect(usage["input_tokens"]).To(gomega.BeNumerically(">", 0))
							gomega.Expect(usage["output_tokens"]).To(gomega.BeNumerically(">", 1))
						} else if stream {
							gomega.Expect(string(raw)).To(gomega.ContainSubstring("data: [DONE]"))
							gomega.Expect(response.ExtractFinishReasonFromStreaming(string(raw))).To(gomega.Equal("stop"))
						} else {
							var reply struct {
								Choices []struct {
									Message struct {
										Content string `json:"content"`
									} `json:"message"`
								} `json:"choices"`
							}
							gomega.Expect(json.Unmarshal(raw, &reply)).To(gomega.Succeed())
							gomega.Expect(reply.Choices).To(gomega.HaveLen(1))
							gomega.Expect(reply.Choices[0].Message.Content).To(gomega.Equal(prompt))
						}
						if topology == "P/D" {
							gomega.Expect(metrics.GetPodRequestCount(testConfig, getNamespace(), prefillPods[0])).To(gomega.BeNumerically(">", prefillBefore), "the native request must reach prefill")
						}
						surface := apiType.String()
						gomega.Eventually(func(g gomega.Gomega) {
							after := apiSurfaceMetrics(metricsURL)
							for _, name := range []string{"request_total", "request_duration_seconds_count", "request_size_bytes_count", "response_size_bytes_count", "request_input_tokens_count", "request_output_tokens_count", "request_ttft_seconds_count", "request_processing_duration_seconds_count", "response_processing_duration_seconds_count"} {
								g.Expect(after[surface][name]-before[surface][name]).To(gomega.Equal(float64(1)), name)
							}
							g.Expect(after[surface]["request_running"]).To(gomega.BeZero())
							g.Expect(after[surface]["request_error_total"]).To(gomega.Equal(before[surface]["request_error_total"]))
							other := reqcommon.APITypeMessages.String()
							if apiType == reqcommon.APITypeMessages {
								other = reqcommon.APITypeChatCompletions.String()
							}
							g.Expect(after[other]["request_total"]).To(gomega.Equal(before[other]["request_total"]), "surfaces must remain separate")
							if usage != nil {
								input := usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]
								g.Expect(after[surface]["request_input_tokens_sum"] - before[surface]["request_input_tokens_sum"]).To(gomega.Equal(float64(input)))
								g.Expect(after[surface]["request_output_tokens_sum"] - before[surface]["request_output_tokens_sum"]).To(gomega.Equal(float64(usage["output_tokens"])))
								cachedObservations := float64(0)
								if usage["cache_read_input_tokens"] > 0 {
									cachedObservations = 1
								}
								g.Expect(after[surface]["request_cached_tokens_count"] - before[surface]["request_cached_tokens_count"]).To(gomega.Equal(cachedObservations))
								g.Expect(after[surface]["request_cached_tokens_sum"] - before[surface]["request_cached_tokens_sum"]).To(gomega.Equal(float64(usage["cache_read_input_tokens"])))
							}
						}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
					})
				}
			}

			for _, failureCase := range []struct {
				name      string
				body      []byte
				processed float64
			}{
				{"malformed JSON", []byte(`{"model":`), 0},
				{"missing max_tokens", mustMarshal(map[string]any{"model": simModelName, "messages": []any{map[string]string{"role": "user", "content": pdTextInput}}}), 1},
			} {
				ginkgo.It("attributes Messages errors for "+failureCase.name, func() {
					before := apiSurfaceMetrics(metricsURL)
					resp, raw := doRequest(reqcommon.PathMessages, failureCase.body)
					gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusBadRequest), "response: %s", raw)
					var failure struct {
						Type  string `json:"type"`
						Error struct {
							Type string `json:"type"`
						} `json:"error"`
					}
					gomega.Expect(json.Unmarshal(raw, &failure)).To(gomega.Succeed())
					gomega.Expect(failure.Type).To(gomega.Equal("error"))
					if failureCase.processed == 0 {
						gomega.Expect(failure.Error.Type).To(gomega.Equal("invalid_request_error"))
					} else {
						gomega.Expect(string(raw)).To(gomega.ContainSubstring("max_tokens"))
					}
					gomega.Eventually(func(g gomega.Gomega) {
						after := apiSurfaceMetrics(metricsURL)
						surface := reqcommon.APITypeMessages.String()
						chat := reqcommon.APITypeChatCompletions.String()
						g.Expect(after[surface]["request_error_total"] - before[surface]["request_error_total"]).To(gomega.Equal(float64(1)))
						g.Expect(after[surface]["request_total"] - before[surface]["request_total"]).To(gomega.Equal(failureCase.processed))
						g.Expect(after[chat]["request_error_total"]).To(gomega.Equal(before[chat]["request_error_total"]))
					}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
				})
			}
		})
	}
}))

type messagesResponse struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      map[string]int `json:"usage"`
}

func decodeMessagesResponse(raw []byte, streaming bool) (messagesResponse, error) {
	var reply messagesResponse
	if !streaming {
		err := json.Unmarshal(raw, &reply)
		return reply, err
	}
	var types []string
	var text strings.Builder
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var event struct {
			Type         string           `json:"type"`
			Message      messagesResponse `json:"message"`
			ContentBlock struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content_block"`
			Delta struct {
				Type       string `json:"type"`
				Text       string `json:"text"`
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage map[string]int `json:"usage"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &event); err != nil {
			return reply, err
		}
		types = append(types, event.Type)
		switch event.Type {
		case "message_start":
			reply = event.Message
		case "content_block_start":
			text.WriteString(event.ContentBlock.Text)
		case "content_block_delta":
			text.WriteString(event.Delta.Text)
		case "message_delta":
			reply.StopReason = event.Delta.StopReason
			for key, value := range event.Usage {
				if reply.Usage == nil {
					reply.Usage = map[string]int{}
				}
				reply.Usage[key] = value
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return reply, err
	}
	if len(types) < 2 || types[0] != "message_start" || types[len(types)-1] != "message_stop" {
		return reply, fmt.Errorf("incomplete Messages stream: %v", types)
	}
	for _, required := range []string{"content_block_start", "content_block_delta", "content_block_stop", "message_delta"} {
		found := false
		for _, eventType := range types {
			if eventType == required {
				found = true
			}
		}
		if !found {
			return reply, fmt.Errorf("Messages stream missing %s", required)
		}
	}
	reply.Content = append(reply.Content, struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{Type: "text", Text: text.String()})
	return reply, nil
}

func apiSurfaceMetrics(metricsURL string) map[string]map[string]float64 {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(strings.NewReader(strings.Join(metrics.GetMetrics(metricsURL), "\n") + "\n"))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	values := map[string]map[string]float64{}
	for name, family := range families {
		if !strings.HasPrefix(name, "llm_d_epp_") {
			continue
		}
		name = strings.TrimPrefix(name, "llm_d_epp_")
		for _, metric := range family.GetMetric() {
			surface := ""
			for _, label := range metric.GetLabel() {
				if label.GetName() == "api_surface" {
					surface = label.GetValue()
				}
			}
			if surface == "" {
				continue
			}
			if values[surface] == nil {
				values[surface] = map[string]float64{}
			}
			if metric.GetCounter() != nil {
				values[surface][name] += metric.GetCounter().GetValue()
			}
			if metric.GetGauge() != nil {
				values[surface][name] += metric.GetGauge().GetValue()
			}
			if metric.GetHistogram() != nil {
				values[surface][name+"_count"] += float64(metric.GetHistogram().GetSampleCount())
				values[surface][name+"_sum"] += metric.GetHistogram().GetSampleSum()
			}
		}
	}
	return values
}
