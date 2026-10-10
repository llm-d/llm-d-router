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

package metrics

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// metricsScrapeRetryTimeout is the budget for connection and non-200 retries.
// Registry presence (llm_d_epp_info) is waited on by the caller's
// Eventually(ReadyTimeout). Nesting ReadyTimeout here would consume that
// outer budget on a single scrape, including when READY_TIMEOUT is overridden.
const (
	metricsScrapeRetryTimeout  = 10 * time.Second
	metricsScrapeRetryInterval = time.Second
)

// GetMetrics fetches Prometheus metrics from metricsURL.
// Transient connection and non-200 errors are retried for metricsScrapeRetryTimeout.
// HTTP 200 can still be controller-runtime boilerplate before llm_d_epp_info is registered.
// Callers that need the EPP registry must Eventually until that series (or the
// specific counter they assert) is present. GetMetrics does not wait for it.
func GetMetrics(metricsURL string) []string {
	deadline := time.Now().Add(metricsScrapeRetryTimeout)
	var lastErr error
	for {
		body, err := scrapeMetrics(metricsURL)
		if err == nil {
			return strings.Split(string(body), "\n")
		}
		lastErr = err
		if time.Now().After(deadline) {
			gomega.Expect(lastErr).ShouldNot(gomega.HaveOccurred())
			return nil
		}
		time.Sleep(metricsScrapeRetryInterval)
	}
}

func scrapeMetrics(metricsURL string) ([]byte, error) {
	resp, err := http.Get(metricsURL) //#nosec G107 -- metricsURL is built from test-controlled config, not request-derived
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// GetCounterMetric fetches the current value of a Prometheus counter metric from the given metrics URL.
// Missing series parse as 0. Connection retries live in GetMetrics.
//
//nolint:unparam // metricName may vary in future test cases
func GetCounterMetric(metricsURL, metricName, labelMatch string) int {
	for _, line := range GetMetrics(metricsURL) {
		if strings.HasPrefix(line, metricName) && strings.Contains(line, labelMatch) {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				valFloat, err := strconv.ParseFloat(fields[len(fields)-1], 64)
				gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
				return int(valFloat)
			}
		}
	}
	return 0
}

// WaitForEPPToDiscoverPods blocks until the EPP's llm_d_epp_ready_endpoints
// gauge for poolName reports at least one pod, indicating the InferencePool
// controller has finished its initial pod discovery. The EPP reports gRPC
// health as SERVING as soon as the pool is set, even if pod discovery found
// zero endpoints, so readiness alone does not guarantee the datastore is
// populated. Polling the gauge avoids routing a real request through the
// EPP, which would otherwise be recorded as a routing decision and skew
// tests that assert exact decision-type counts.
func WaitForEPPToDiscoverPods(cfg *testutils.TestConfig, metricsPort int, poolName string) {
	ginkgo.By("Waiting for EPP to discover pool members")
	metricsURL := fmt.Sprintf("http://localhost:%d/metrics", metricsPort)
	labelMatch := fmt.Sprintf(`name="%s"`, poolName)
	gomega.Eventually(func() int {
		return GetCounterMetric(metricsURL, "llm_d_epp_ready_endpoints", labelMatch)
	}, cfg.ReadyTimeout, time.Second).Should(gomega.BeNumerically(">", 0), "EPP should discover pool members within the ready timeout")
}

// GetPodRequestCount gets the total vLLM request count from a pod's metrics endpoint.
func GetPodRequestCount(cfg *testutils.TestConfig, nsName, podName string) int {
	ginkgo.By("Getting request count from pod: " + podName)

	// Use Kubernetes API proxy to access the metrics endpoint
	output, err := cfg.KubeCli.CoreV1().RESTClient().
		Get().
		Namespace(nsName).
		Resource("pods").
		Name(podName + ":8000").
		SubResource("proxy").
		Suffix("metrics").
		DoRaw(cfg.Context)
	if err != nil {
		ginkgo.By(fmt.Sprintf("Warning: Could not get metrics from pod %s: %v", podName, err))
		return -1
	}

	return parseRequestCountFromMetrics(string(output))
}

func parseRequestCountFromMetrics(metricsOutput string) int {
	// Look for vllm:e2e_request_latency_seconds_count{model_name="food-review"} <count>
	lines := strings.Split(metricsOutput, "\n")
	for _, line := range lines {
		if strings.Contains(line, "vllm:e2e_request_latency_seconds_count") &&
			strings.Contains(line, "food-review") {
			// Extract the count value after the last space
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				count, err := strconv.Atoi(parts[len(parts)-1])
				if err == nil {
					return count
				}
			}
		}
	}
	return 0
}
