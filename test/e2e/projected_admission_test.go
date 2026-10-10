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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/test/e2e/utils"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
)

const (
	projectedHTTPBlockTokens          = 64
	projectedHTTPCapacityTokens       = 10 * projectedHTTPBlockTokens
	projectedHTTPIncumbentTokens      = 6 * projectedHTTPBlockTokens
	projectedHTTPIncomingTokens       = 5 * projectedHTTPBlockTokens
	projectedHTTPDecodeCapacityTokens = 10 * projectedHTTPCapacityTokens
)

var _ = ginkgo.Describe("Projected token admission", ginkgo.Ordered, testWrapper(func() {
	ginkgo.DescribeTable("waits for projected capacity and releases on incumbent cancellation", func(mode string) {
		fixture := newProjectedHTTPFixture(mode, "10s", false)
		first, second, loaded := fixture.seedPair()
		pending := startProjectedHTTPRequest(projectedHTTPIncomingTokens)
		expectProjectedHTTPQueued(pending, loaded.attempts)
		first.cancel()
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && slices.Equal(s.tokenValues(), []float64{projectedHTTPIncomingTokens, projectedHTTPCapacityTokens}) &&
				s.requestTotal() == 2 && s.attempts == loaded.attempts+1
		})
		second.cancel()
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return slices.Equal(s.tokenValues(), []float64{projectedHTTPIncomingTokens}) && s.requestTotal() == 1
		})
		response := awaitProjectedHTTP(pending)
		gomega.Expect(response.err).NotTo(gomega.HaveOccurred())
		gomega.Expect(response.status).To(gomega.Equal(http.StatusOK), string(response.body))
		waitMS, err := strconv.ParseFloat(response.headers.Get(metadata.FlowQueueDurationHeaderKey), 64)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Expect(waitMS).To(gomega.BeNumerically(">=", 200))
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && len(s.tokenValues()) == 0 && s.requestTotal() == 0 && s.attempts == loaded.attempts+1
		})
	}, ginkgo.Entry("tokens", "tokens"), ginkgo.Entry("hybrid", "hybrid"))

	ginkgo.It("removes a canceled queued request without scheduling or accounting it", func() {
		fixture := newProjectedHTTPFixture("tokens", "10s", false)
		first, second, loaded := fixture.seedPair()
		pending := startProjectedHTTPRequest(projectedHTTPIncomingTokens)
		expectProjectedHTTPQueued(pending, loaded.attempts)
		pending.cancel()
		gomega.Expect(errors.Is(awaitProjectedHTTP(pending).err, context.Canceled)).To(gomega.BeTrue())
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && s.attempts == loaded.attempts && slices.Equal(s.tokenValues(), []float64{projectedHTTPIncumbentTokens, projectedHTTPCapacityTokens})
		})
		first.cancel()
		second.cancel()
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.requestTotal() == 0 && len(s.tokenValues()) == 0 && s.attempts == loaded.attempts
		})
	})

	ginkgo.It("expires projected-capacity waits with a queue TTL response", func() {
		fixture := newProjectedHTTPFixture("tokens", "2s", false)
		_, _, loaded := fixture.seedPair()
		pending := startProjectedHTTPRequest(projectedHTTPIncomingTokens)
		expectProjectedHTTPQueued(pending, loaded.attempts)
		response := awaitProjectedHTTP(pending)
		gomega.Expect(response.err).NotTo(gomega.HaveOccurred())
		gomega.Expect(response.status).To(gomega.Equal(http.StatusTooManyRequests), string(response.body))
		gomega.Expect(response.headers.Get(errcommon.RequestDroppedReasonHeaderKey)).To(gomega.Equal(string(errcommon.RequestDroppedReasonTTLExpired)))
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && s.attempts == loaded.attempts && slices.Equal(s.tokenValues(), []float64{projectedHTTPIncumbentTokens, projectedHTTPCapacityTokens})
		})
	})

	ginkgo.It("waits for required prefill while decode still has capacity", func() {
		fixture := newProjectedHTTPFixture("tokens", "10s", true)
		incumbent := startProjectedHTTPRequest(projectedHTTPIncumbentTokens)
		loaded := waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return slices.Equal(s.tokenValues(), []float64{projectedHTTPIncumbentTokens, projectedHTTPIncumbentTokens}) && s.requestTotal() == 2
		})
		prefillPods, _ := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, getNamespace())
		gomega.Expect(prefillPods).To(gomega.HaveLen(1))
		prefillBefore, err := fixture.promptTokens(prefillPods[0])
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		pending := startProjectedHTTPRequest(projectedHTTPIncomingTokens)
		expectProjectedHTTPQueued(pending, loaded.attempts)
		incumbent.cancel()
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && slices.Equal(s.tokenValues(), []float64{projectedHTTPIncomingTokens, projectedHTTPIncomingTokens}) &&
				s.requestTotal() == 2 && s.attempts == loaded.attempts+1
		})
		response := awaitProjectedHTTP(pending)
		gomega.Expect(response.err).NotTo(gomega.HaveOccurred())
		gomega.Expect(response.status).To(gomega.Equal(http.StatusOK), string(response.body))
		waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
			return s.queue == 0 && s.requestTotal() == 0 && s.attempts == loaded.attempts+1
		})
		// A canceled simulator request can still finish and increment counters.
		// Its tokens alone do not prove that the pending request ran prefill.
		gomega.Eventually(func() (float64, error) {
			current, err := fixture.promptTokens(prefillPods[0])
			return current - prefillBefore, err
		}, 10*time.Second, 100*time.Millisecond).Should(gomega.Or(
			gomega.Equal(float64(projectedHTTPIncomingTokens)),
			gomega.Equal(float64(projectedHTTPIncomingTokens+projectedHTTPIncumbentTokens))))
	})
}))

type projectedHTTPFixture struct {
	simulatorPorts map[string]int32
}

func newProjectedHTTPFixture(mode, ttl string, disaggregate bool) *projectedHTTPFixture {
	if disaggregate {
		createModelServersPDSharedStorage(1)
	} else {
		createModelServersDecode(2)
	}
	standalone.Create(standaloneConfig(), projectedHTTPConfig(mode, ttl, disaggregate), 1, 8000)
	fixture := &projectedHTTPFixture{simulatorPorts: map[string]int32{}}
	for _, pod := range utils.GetPods(testConfig, podSelector, getNamespace()) {
		for _, container := range pod.Spec.Containers {
			if container.Name != "vllm" {
				continue
			}
			for _, port := range container.Ports {
				if port.Name == "http" || port.Name == "prefill-http" {
					fixture.simulatorPorts[pod.Name] = port.ContainerPort
				}
			}
		}
	}
	gomega.Expect(fixture.simulatorPorts).To(gomega.HaveLen(2))
	ginkgo.DeferCleanup(func() error { return fixture.setLatency("0s") })
	// Prompt-only accounting ends at the first response chunk. TTFT keeps the
	// token contribution active; a long stream after its first chunk would not.
	gomega.Expect(fixture.setLatency("30s")).To(gomega.Succeed())
	return fixture
}

func (f *projectedHTTPFixture) setLatency(ttft string) error {
	body, err := json.Marshal(map[string]any{
		"latency-calculator": "constant", "time-to-first-token": ttft,
		"time-to-first-token-std-dev": "0s", "inter-token-latency": "0s",
		"inter-token-latency-std-dev": "0s", "kv-cache-transfer-latency": "0s",
		"kv-cache-transfer-latency-std-dev": "0s", "time-factor-under-load": 1,
	})
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(testConfig.Context, 10*time.Second)
	defer cancel()
	var errs []error
	for pod, port := range f.simulatorPorts {
		_, err := testConfig.KubeCli.CoreV1().RESTClient().Post().Namespace(getNamespace()).
			Resource("pods").Name(fmt.Sprintf("%s:%d", pod, port)).SubResource("proxy").Suffix("admin/config").
			SetHeader("Content-Type", "application/json").Body(body).DoRaw(ctx)
		if err != nil {
			errs = append(errs, fmt.Errorf("set simulator latency on %s: %w", pod, err))
		}
	}
	return errors.Join(errs...)
}

func (f *projectedHTTPFixture) seedPair() (*projectedHTTPCall, *projectedHTTPCall, projectedHTTPMetrics) {
	first := startProjectedHTTPRequest(projectedHTTPIncumbentTokens)
	waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
		return slices.Equal(s.tokenValues(), []float64{projectedHTTPIncumbentTokens})
	})
	second := startProjectedHTTPRequest(projectedHTTPCapacityTokens)
	loaded := waitProjectedHTTPMetrics(func(s projectedHTTPMetrics) bool {
		return slices.Equal(s.tokenValues(), []float64{projectedHTTPIncumbentTokens, projectedHTTPCapacityTokens}) && s.requestTotal() == 2
	})
	return first, second, loaded
}

func (f *projectedHTTPFixture) promptTokens(pod string) (float64, error) {
	ctx, cancel := context.WithTimeout(testConfig.Context, 2*time.Second)
	defer cancel()
	raw, err := testConfig.KubeCli.CoreV1().RESTClient().Get().Namespace(getNamespace()).
		Resource("pods").Name(fmt.Sprintf("%s:%d", pod, f.simulatorPorts[pod])).SubResource("proxy").Suffix("metrics").DoRaw(ctx)
	if err != nil {
		return 0, err
	}
	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(bytes.NewReader(raw))
	if err != nil {
		return 0, err
	}
	var tokens float64
	for _, metric := range families["vllm:prompt_tokens_total"].GetMetric() {
		tokens += metric.GetCounter().GetValue()
	}
	return tokens, nil
}

type projectedHTTPCall struct {
	cancel context.CancelFunc
	done   chan struct{}
	result projectedHTTPResult
}

type projectedHTTPResult struct {
	status  int
	headers http.Header
	body    []byte
	err     error
}

func startProjectedHTTPRequest(tokens int) *projectedHTTPCall {
	ids := make([]int, tokens)
	for i := range ids {
		ids[i] = 1000 + i
	}
	id := uuid.NewString()
	body := mustMarshal(map[string]any{"model": simModelName, "token_ids": ids, "cache_salt": id,
		"sampling_params": map[string]any{"max_tokens": 1}, "stream": false})
	ctx, cancel := context.WithTimeout(testConfig.Context, 90*time.Second)
	call := &projectedHTTPCall{cancel: cancel, done: make(chan struct{})}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://localhost:%d%s", getPort(), vLLMGeneratePath), bytes.NewReader(body))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Request-ID", id)
	ginkgo.DeferCleanup(func() {
		cancel()
		gomega.Eventually(call.done, 5*time.Second).Should(gomega.BeClosed())
	})
	go func() {
		defer close(call.done)
		response, err := (&http.Client{Timeout: 90 * time.Second}).Do(request)
		if err != nil {
			call.result.err = err
			return
		}
		defer func() { _ = response.Body.Close() }()
		call.result.status, call.result.headers = response.StatusCode, response.Header.Clone()
		call.result.body, call.result.err = io.ReadAll(response.Body)
	}()
	return call
}

func awaitProjectedHTTP(call *projectedHTTPCall) projectedHTTPResult {
	gomega.Eventually(call.done, 75*time.Second).Should(gomega.BeClosed())
	return call.result
}

func expectProjectedHTTPQueued(call *projectedHTTPCall, attempts float64) {
	gomega.Eventually(func() error {
		state, err := readProjectedHTTPMetrics()
		select {
		case <-call.done:
			return fmt.Errorf("request completed before queue wait: HTTP=%d error=%v body=%q; metrics=%+v metricsError=%v",
				call.result.status, call.result.err, call.result.body, state, err)
		default:
		}
		if err != nil {
			return err
		}
		if state.queue != 1 || state.attempts != attempts {
			return fmt.Errorf("waiting for queue=1 attempts=%v; last metrics=%+v; HTTP pending", attempts, state)
		}
		return nil
	}, 10*time.Second, 25*time.Millisecond).Should(gomega.Succeed())
	gomega.Consistently(func() error {
		select {
		case <-call.done:
			state, err := readProjectedHTTPMetrics()
			return fmt.Errorf("request completed during queue wait: HTTP=%d error=%v body=%q; metrics=%+v metricsError=%v",
				call.result.status, call.result.err, call.result.body, state, err)
		default:
			return nil
		}
	}, 250*time.Millisecond, 25*time.Millisecond).Should(gomega.Succeed())
	state, err := readProjectedHTTPMetrics()
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(state.queue).To(gomega.Equal(float64(1)))
	gomega.Expect(state.attempts).To(gomega.Equal(attempts))
}

type projectedHTTPMetrics struct {
	tokens   map[string]float64
	requests map[string]float64
	queue    float64
	attempts float64
}

func (s projectedHTTPMetrics) tokenValues() []float64 {
	values := []float64{}
	for _, tokens := range s.tokens {
		if tokens != 0 {
			values = append(values, tokens)
		}
	}
	slices.Sort(values)
	return values
}

func (s projectedHTTPMetrics) requestTotal() float64 {
	var total float64
	for _, requests := range s.requests {
		total += requests
	}
	return total
}

func waitProjectedHTTPMetrics(check func(projectedHTTPMetrics) bool) projectedHTTPMetrics {
	var latest projectedHTTPMetrics
	gomega.Eventually(func() error {
		var err error
		latest, err = readProjectedHTTPMetrics()
		if err != nil {
			return err
		}
		if !check(latest) {
			return fmt.Errorf("EPP admission metrics did not reach the expected state; last metrics=%+v", latest)
		}
		return nil
	}, 10*time.Second, 25*time.Millisecond).Should(gomega.Succeed())
	return latest
}

func readProjectedHTTPMetrics() (projectedHTTPMetrics, error) {
	state := projectedHTTPMetrics{tokens: map[string]float64{}, requests: map[string]float64{}}
	response, err := (&http.Client{Timeout: 2 * time.Second}).Get(fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort()))
	if err != nil {
		return state, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return state, fmt.Errorf("metrics returned HTTP %d", response.StatusCode)
	}
	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(response.Body)
	if err != nil {
		return state, err
	}
	for name, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			switch name {
			case metricsutil.LLMDRouterEndpointPickerSubsystem + "_inflight_tokens":
				if labels["namespace"] == getNamespace() {
					state.tokens[labels["endpoint_name"]] += metric.GetGauge().GetValue()
				}
			case metricsutil.LLMDRouterEndpointPickerSubsystem + "_inflight_requests":
				if labels["namespace"] == getNamespace() {
					state.requests[labels["endpoint_name"]] += metric.GetGauge().GetValue()
				}
			case metricsutil.LLMDRouterEndpointPickerSubsystem + "_flow_control_queue_size":
				state.queue += metric.GetGauge().GetValue()
			case metricsutil.LLMDRouterEndpointPickerSubsystem + "_scheduler_attempts_total":
				state.attempts += metric.GetCounter().GetValue()
			}
		}
	}
	return state, nil
}

func projectedHTTPConfig(mode, ttl string, disaggregate bool) string {
	common := fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates: [flowControl]
plugins:
- type: vllmhttp-parser
- type: token-producer
- type: approx-prefix-cache-producer
  parameters:
    blockSizeTokens: %d
    autoTune: false
- type: inflight-load-producer
  parameters:
    addEstimatedOutputTokens: false
- type: decode-filter
- type: max-score-picker
- name: projected-capacity
  type: concurrency-detector
  parameters:
    concurrencyMode: %s
    maxConcurrency: 100
    maxTokenConcurrency: %d
    headroom: 0
    failOpen: false
`, projectedHTTPBlockTokens, mode, projectedHTTPCapacityTokens)
	profiles := `- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: projected-capacity
  - pluginRef: max-score-picker
`
	detector := "projected-capacity"
	if disaggregate {
		profiles = fmt.Sprintf(`- type: prefill-filter
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: always-disagg-pd-decider
- name: decode-capacity
  type: concurrency-detector
  parameters:
    concurrencyMode: tokens
    maxTokenConcurrency: %d
    headroom: 0
    failOpen: false
- name: staged-capacity
  type: max-saturation-detector
  parameters:
    detectors: [projected-capacity, decode-capacity]
    stages:
      projected-capacity: [prefill]
      decode-capacity: [decode]
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: projected-capacity
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: decode-capacity
  - pluginRef: max-score-picker
`, projectedHTTPDecodeCapacityTokens)
		detector = "staged-capacity"
	}
	return common + profiles + fmt.Sprintf(`requestHandler:
  parsers:
  - pluginRef: vllmhttp-parser
flowControl:
  maxRequests: "100"
  maxBytes: "10Mi"
  defaultRequestTTL: %q
  saturationDetector:
    pluginRef: %s
`, ttl, detector)
}
