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

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	corev1 "k8s.io/api/core/v1"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/test/e2e/utils"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
)

var _ = ginkgo.Describe("Messages API", ginkgo.Ordered, testWrapper(func() {
	for _, topology := range []string{"aggregated", "P/D"} {
		ginkgo.It("preserves native JSON/SSE responses and observability through "+topology+" routing", func() {
			config := simpleConfig
			if topology == "P/D" {
				createModelServersPDSharedStorage(1)
				config = pdConfig
			} else {
				createModelServersDecode(1)
			}
			cfg := standaloneConfig()
			cfg.Tracing = true
			router := standalone.Create(cfg, config, 1, 8000)
			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, getNamespace())
			gomega.Expect(decodePods).To(gomega.HaveLen(1))
			if topology == "P/D" {
				gomega.Expect(prefillPods).To(gomega.HaveLen(1))
			}
			leader := standalone.WaitForReadyLeader(cfg, 1, cfg.Namespace, router.Selector)
			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())
			for _, stream := range []bool{false, true} {
				for _, apiType := range []reqcommon.APIType{reqcommon.APITypeMessages, reqcommon.APITypeChatCompletions} {
					prompt := uuid.NewString() + " " + responsesTextInput
					body := map[string]any{"model": simModelName, "max_tokens": echoOutputTokens, "stream": stream, "messages": []any{map[string]string{"role": "user", "content": prompt}}}
					traceID := strings.ReplaceAll(uuid.NewString(), "-", "")
					headers := map[string]string{"traceparent": "00-" + traceID + "-0123456789abcdef-01"}
					if apiType == reqcommon.APITypeMessages {
						body["model"] = "messages-client-model"
						headers[metadata.ModelNameRewriteKey] = simModelName
					} else if stream {
						body["stream_options"] = map[string]bool{"include_usage": true}
					}
					before := apiSurfaceMetrics(metricsURL)
					prefillBefore := 0
					if topology == "P/D" {
						prefillBefore = utils.GetPodRequestCount(testConfig, getNamespace(), prefillPods[0])
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
						gomega.Expect(utils.ExtractFinishReasonFromStreaming(string(raw))).To(gomega.Equal("stop"))
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
						gomega.Expect(utils.GetPodRequestCount(testConfig, getNamespace(), prefillPods[0])).To(gomega.BeNumerically(">", prefillBefore), "the native request must reach prefill")
					}
					surface := apiType.String()
					gomega.Eventually(func(g gomega.Gomega) {
						after := apiSurfaceMetrics(metricsURL)
						for _, name := range []string{"request_total", "request_duration_seconds_count", "request_size_bytes_count", "response_size_bytes_count", "request_input_tokens_count", "request_output_tokens_count", "request_ttft_seconds_count", "scheduler_e2e_duration_seconds_count", "scheduler_attempts_total", "request_processing_duration_seconds_count", "response_processing_duration_seconds_count"} {
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
					gomega.Eventually(func() (string, error) {
						logs, err := testConfig.KubeCli.CoreV1().Pods(getNamespace()).GetLogs(leader.Name, &corev1.PodLogOptions{Container: "epp"}).DoRaw(testConfig.Context)
						if err != nil {
							return "", err
						}
						return requestTraceSurface(logs, traceID), nil
					}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Equal(surface), "the sampled request trace must retain API attribution")
				}
			}
			for _, failureCase := range []struct {
				body      []byte
				processed float64
			}{
				{[]byte(`{"model":`), 0},
				{mustMarshal(map[string]any{"model": simModelName, "messages": []any{map[string]string{"role": "user", "content": responsesTextInput}}}), 1},
			} {
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
	families, err := parser.TextToMetricFamilies(strings.NewReader(strings.Join(utils.GetMetrics(metricsURL), "\n") + "\n"))
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

func requestTraceSurface(logs []byte, traceID string) string {
	for _, block := range strings.Split("\n"+string(logs), "\n{\n")[1:] {
		end := strings.Index(block, "\n}")
		if end < 0 {
			continue
		}
		var span struct {
			Name        string
			SpanContext struct{ TraceID string }
			Attributes  []struct {
				Key   string
				Value struct{ Value any }
			}
		}
		if json.Unmarshal([]byte("{\n"+block[:end]+"\n}"), &span) != nil || span.Name != "request" || span.SpanContext.TraceID != traceID {
			continue
		}
		for _, attribute := range span.Attributes {
			if attribute.Key == string(semconv.LLMDEPPAPISurfaceKey) {
				surface, _ := attribute.Value.Value.(string)
				return surface
			}
		}
	}
	return ""
}
