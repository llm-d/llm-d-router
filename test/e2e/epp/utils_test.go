/*
Copyright 2026 The Kubernetes Authors.

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

package epp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-inference-scheduler/apix/v1alpha2"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metadata"
	igwtestutils "github.com/llm-d/llm-d-inference-scheduler/test/utils/igw"
)

const (
	firstPort             = 8000
	numPorts              = 2
	maxConcurrentRequests = 2 // prevent hammering Envoy and backend
	maxRetries            = 5
	backoff               = 5 * time.Second
	batches               = 20
	apiCompletions        = "/completions"
	apiChatCompletions    = "/chat/completions"
	apiEmbeddings         = "/embeddings"

	curlPodName = "curl" // name of both the curl pod and its container

	testPrompt = "Write as if you were a critic: San Francisco"

	// statusTrailerFmt is the -w format that appends the HTTP status code to curl's
	// stdout as "HTTP_STATUS=<code>". Prefer matching this over the header status line
	// ("HTTP/1.1 200 OK") since HTTP/2 drops the reason phrase ("HTTP/2 200").
	statusTrailerFmt = "\nHTTP_STATUS=%{http_code}\n"
	statusOK         = "HTTP_STATUS=200"
	statusNotFound   = "HTTP_STATUS=404"
)

// newInferenceObjective creates an InferenceObjective in the given namespace for igwtestutils.
func newInferenceObjective(ns string) *v1alpha2.InferenceObjective {
	return igwtestutils.MakeModelWrapper(types.NamespacedName{Name: "inferenceobjective-sample", Namespace: ns}).
		SetPriority(2).
		SetPoolRef(modelServerName).
		Obj()
}

// verifyTrafficRouting contains the logic for the "Should route traffic to target model servers" test.
func verifyTrafficRouting() {
	ginkgo.By("Verifying traffic routing")
	for _, t := range []struct {
		api              string
		promptOrMessages any
	}{
		{
			api:              apiCompletions,
			promptOrMessages: testPrompt,
		},
		{
			api: apiChatCompletions,
			promptOrMessages: []map[string]any{
				{
					"role":    "user",
					"content": testPrompt,
				},
			},
		},
		{
			api: apiChatCompletions,
			promptOrMessages: []map[string]any{
				{
					"role":    "user",
					"content": testPrompt,
				},
				{"role": "assistant", "content": "Okay, let's see..."},
				{"role": "user", "content": "Now summarize your thoughts."},
			},
		},
		{
			api:              apiEmbeddings,
			promptOrMessages: "The food was delicious and the service was great.",
		},
		{
			api:              apiEmbeddings,
			promptOrMessages: []string{"First sentence to embed.", "Second sentence to embed."},
		},
	} {
		ginkgo.By(fmt.Sprintf("Verifying connectivity through the inference extension with %s api and prompt/messages: %v", t.api, t.promptOrMessages))

		// Skip embeddings API if server returns 404 (not all models support embeddings).
		if t.api == apiEmbeddings {
			probeCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, t.api, t.promptOrMessages, false)
			probeResp, probeErr := igwtestutils.ExecCommandInPod(testConfig, curlPodName, curlPodName, probeCmd)
			if probeErr == nil && strings.Contains(probeResp, statusNotFound) {
				ginkgo.Skip("Skipping " + apiEmbeddings + ": server returned 404 (embeddings may not be supported by this model)")
			}
		}

		// Expected ports and client-facing model name (response model is rewritten back to the incoming name)
		expectedPort := generateSequence(firstPort, numPorts)
		expectedModel := []string{modelName}

		// Observed ports and InferenceObjective target models
		actualModel := make(map[string]int)
		actualPort := make(map[int]int)

		// Send curl requests to verify routing to all target ports in the InferencePool.
		// Run a small batch per retry (e.g., 5) to keep the test active
		for i := range batches {
			uniqueID := time.Now().UnixNano()
			dynamicHashValue := fmt.Sprintf("Nonce-%d", uniqueID)
			currentPromptOrMessages := t.promptOrMessages // Start with the original

			// Check if the payload is a slice of maps (e.g., for /chat/completions)
			if originalMessages, ok := currentPromptOrMessages.([]map[string]any); ok {
				nonceMsg := map[string]any{
					"role":    "system",
					"content": fmt.Sprintf("TestNonce: %s-%d", dynamicHashValue, i),
				}

				currentPromptOrMessages = append([]map[string]any{nonceMsg}, originalMessages...)
			} else if originalString, ok := t.promptOrMessages.(string); ok {
				currentPromptOrMessages = fmt.Sprintf("[TestNonce: %s-%d] %s", dynamicHashValue, i, originalString)
			} else if originalStrings, ok := t.promptOrMessages.([]string); ok {
				// For embeddings with array input, prepend a unique string so each request is distinct.
				withNonce := make([]string, 0, len(originalStrings)+1)
				withNonce = append(withNonce, fmt.Sprintf("[TestNonce: %s-%d]", dynamicHashValue, i))
				withNonce = append(withNonce, originalStrings...)
				currentPromptOrMessages = withNonce
			} else {
				currentPromptOrMessages = t.promptOrMessages
			}

			curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, t.api, currentPromptOrMessages, false)

			var resp string
			var err error
			// Repeatedly send a message until we get a successful response.
			for attempt := 0; attempt <= maxRetries; attempt++ {
				resp, err = igwtestutils.ExecCommandInPod(testConfig, curlPodName, curlPodName, curlCmd)
				if err == nil && strings.Contains(resp, statusOK) {
					break // Success!
				}

				if attempt < maxRetries {
					time.Sleep(backoff)
				}
			}

			gomega.Expect(err).ToNot(gomega.HaveOccurred(), "Expected curl command to succeed")
			gomega.Expect(resp).To(gomega.ContainSubstring(statusOK), "Expected HTTP 200 response")

			for _, m := range expectedModel {
				if strings.Contains(resp, m) {
					actualModel[m] = 0
				}
			}
			for _, p := range expectedPort {
				if strings.Contains(resp, fmt.Sprintf("x-inference-port: %d", p)) {
					actualPort[p] = 0
				}
			}
		}

		gotModel := make([]string, 0, len(actualModel))
		for m := range actualModel {
			gotModel = append(gotModel, m)
		}
		gotPort := make([]int, 0, len(actualPort))
		for p := range actualPort {
			gotPort = append(gotPort, p)
		}

		ginkgo.GinkgoWriter.Printf("Port distribution: %v\n", actualPort)
		ginkgo.GinkgoWriter.Printf("Model distribution: %v\n", actualModel)

		gomega.Expect(gotModel).To(gomega.BeComparableTo(expectedModel, cmpopts.SortSlices(func(a, b string) bool { return a < b })))
		gomega.Expect(gotPort).To(gomega.BeComparableTo(expectedPort, cmpopts.SortSlices(func(a, b int) bool { return a < b })))
	}
}

// verifyMetrics contains the logic for the "Should expose EPP metrics after generating traffic" test.
func verifyMetrics() {
	ginkgo.By("Verifying metrics exposure")

	// Generate traffic by sending requests through the inference extension.
	ginkgo.By("Generating traffic through the inference extension")
	curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, apiCompletions, testPrompt, true)

	// Run the curl command multiple times to generate some metrics data.

	semaphore := make(chan struct{}, maxConcurrentRequests)
	execFn := func(cmd []string) (string, error) {
		return igwtestutils.ExecCommandInPod(testConfig, curlPodName, curlPodName, cmd)
	}

	errorGood := generateTraffic(curlCmd, batches, semaphore, execFn, backoff, statusOK)
	gomega.Expect(errorGood).NotTo(gomega.HaveOccurred(), "Expected good traffic generation to succeed")

	// Modify the curl command to generate some error metrics.
	// Non-200 responses are expected here; "" accepts any exec-success so server-side
	// errors don't cause the traffic-generation step itself to fail.
	curlCmd[len(curlCmd)-1] = "invalid input"
	errorBad := generateTraffic(curlCmd, batches, semaphore, execFn, backoff, "")
	gomega.Expect(errorBad).NotTo(gomega.HaveOccurred(), "Expected bad traffic generation to succeed")

	// looks like a flaky test, will investigate separately
	ginkgo.By("Verifying that all expected metrics are present.")

	// Now scrape metrics from the EPP endpoint via the curl pod.
	ginkgo.By("Scraping metrics from the EPP endpoint and verifying all backends were hit")
	podIP := findReadyPod().Status.PodIP

	// Get the authorization token for reading metrics.
	token := ""
	gomega.Eventually(func(g gomega.Gomega) {
		t, err := getMetricsReaderToken(testConfig.K8sClient)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(t).NotTo(gomega.BeEmpty())
		token = t
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

	// Construct the metric scraping curl command using Pod IP.
	metricScrapeCmd := getMetricsScrapeCommand(podIP, token)

	modelServerPods, err := getPodsByLabel(testConfig.Context, testConfig.K8sClient, testConfig.NsName, "app", modelServerName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Expected to find model server pods")

	// Define the metrics we expect to see
	preset := []string{ //nolint:prealloc
		"inference_objective_request_total",
		"inference_objective_request_error_total",
		"inference_objective_request_duration_seconds",
		"inference_objective_normalized_time_per_output_token_seconds",
		"inference_objective_request_sizes",
		"inference_objective_response_sizes",
		"inference_objective_input_tokens",
		"inference_objective_output_tokens",
		"inference_pool_average_kv_cache_utilization",
		"inference_pool_average_queue_size",
		"inference_pool_per_pod_queue_size",
		"inference_objective_running_requests",
		"inference_pool_ready_pods",
		"inference_extension_info",

		// llm_d metrics
		"llm_d_router_epp_request_total",
		"llm_d_router_epp_request_error_total",
		"llm_d_router_epp_request_duration_seconds",
		"llm_d_router_epp_normalized_time_per_output_token_seconds",
		"llm_d_router_epp_request_sizes",
		"llm_d_router_epp_response_sizes",
		"llm_d_router_epp_input_tokens",
		"llm_d_router_epp_output_tokens",
		"llm_d_router_epp_average_kv_cache_utilization",
		"llm_d_router_epp_average_queue_size",
		"llm_d_router_epp_per_endpoint_queue_size",
		"llm_d_router_epp_running_requests",
		"llm_d_router_epp_ready_endpoints",
		"llm_d_router_epp_info",
	}
	expectedMetrics := make([]string, 0, len(preset)+len(modelServerPods)*numPorts*2)
	expectedMetrics = append(expectedMetrics, preset...)

	for _, modelServerPod := range modelServerPods {
		for rank := range numPorts {
			metricQueueSize := fmt.Sprintf(
				"inference_pool_per_pod_queue_size{model_server_pod=\"%s-rank-%d\",name=\"%s\"}",
				modelServerPod.Name,
				rank,
				modelServerName,
			)
			expectedMetrics = append(expectedMetrics, metricQueueSize)

			metricQueueSizeNew := fmt.Sprintf(
				"llm_d_router_epp_per_endpoint_queue_size{model_server_endpoint=\"%s-rank-%d\",name=\"%s\"}",
				modelServerPod.Name,
				rank,
				modelServerName,
			)
			expectedMetrics = append(expectedMetrics, metricQueueSizeNew)
		}
	}

	gomega.Eventually(func() error {
		// Execute the metrics scrape command inside the curl pod.
		resp, err := igwtestutils.ExecCommandInPod(testConfig, curlPodName, curlPodName, metricScrapeCmd)
		if err != nil {
			return err
		}
		if !strings.Contains(resp, statusOK) {
			return fmt.Errorf("expected HTTP 200, got: %s", resp)
		}
		// Check if all expected metrics are present in the metrics output.
		for _, metric := range expectedMetrics {
			if !strings.Contains(resp, metric) {
				return fmt.Errorf("expected metric %s not found in metrics output", metric)
			}
		}
		return nil
	}, testConfig.ReadyTimeout, curlInterval).Should(gomega.Succeed())
}

func getMetricsReaderToken(k8sClient client.Client) (string, error) {
	secret := &corev1.Secret{}
	err := k8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: metricsReaderSecretName}, secret)
	if err != nil {
		return "", err
	}
	return string(secret.Data["token"]), nil
}

// findReadyPod finds the first EPP pod that has a "Ready" status condition.
// It's used to target the leader pod in an HA setup.
func findReadyPod() *corev1.Pod {
	var readyPod *corev1.Pod
	gomega.Eventually(func(g gomega.Gomega) {
		podList := &corev1.PodList{}
		err := testConfig.K8sClient.List(testConfig.Context, podList, client.InNamespace(testConfig.NsName), client.MatchingLabels{"app": inferExtName})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		foundReadyPod := false
		for i := range podList.Items {
			pod := &podList.Items[i]
			for _, cond := range pod.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
					g.Expect(pod.Status.PodIP).NotTo(gomega.BeEmpty(), "Ready pod must have an IP")
					readyPod = pod
					foundReadyPod = true
					break // break inner loop
				}
			}
			if foundReadyPod {
				break // break outer loop
			}
		}
		g.Expect(foundReadyPod).To(gomega.BeTrue(), "No ready EPP pod found")
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
	return readyPod
}

// getMetricsScrapeCommand returns the command to scrape the /metrics endpoint.
func getMetricsScrapeCommand(podIP, token string) []string {
	return []string{
		"curl", "-i", "--max-time", strconv.Itoa((int)(6 * curlTimeout.Seconds())),
		"-w", statusTrailerFmt,
		"-H", "Authorization: Bearer " + token, fmt.Sprintf("http://%s:%d/metrics", podIP, 9090),
	}
}

// getCurlCommand returns the command, as a slice of strings, for curl'ing
// the test model server at the given name, namespace, port, and model name.
// This command gets executed by a dummy pod that communicates with Envoy
func getCurlCommand(name, ns, port, model string, timeout time.Duration, api string, promptOrMessages any, streaming bool) []string {
	body := map[string]any{
		"model":       model,
		"max_tokens":  100,
		"temperature": 0,
	}
	switch api {
	case apiCompletions:
		body["prompt"] = promptOrMessages
	case apiChatCompletions:
		body["messages"] = promptOrMessages
	case apiEmbeddings:
		body["input"] = promptOrMessages
		delete(body, "max_tokens")
		delete(body, "temperature")
	}
	if streaming && api != apiEmbeddings {
		body["stream"] = true
		body["stream_options"] = map[string]any{
			"include_usage": true,
		}
	}
	b, err := json.Marshal(body)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	return []string{
		"curl",
		"-i",
		"-w",
		statusTrailerFmt,
		"--max-time",
		strconv.Itoa((int)(timeout.Seconds())),
		fmt.Sprintf("%s.%s.svc:%s/v1%s", name, ns, port, api),
		"-H",
		"Content-Type: application/json",
		"-H",
		"Cache-Control: no-cache",
		"-H",
		fmt.Sprintf("%v: inferenceobjective-sample", metadata.ObjectiveKey),
		"-H",
		fmt.Sprintf("%v: %s", metadata.ModelNameRewriteKey, targetModelName),
		"-H",
		"Connection: close",
		"-d",
		string(b),
	}
}

// buildContainerPorts constructs a slice of corev1.ContainerPort starting from 'start' with 'count' ports.
func buildContainerPorts(start int, count int) []corev1.ContainerPort {
	ports := make([]corev1.ContainerPort, count)
	for i := range count {
		portNum := int32(start + i)
		ports[i] = corev1.ContainerPort{
			Name:          fmt.Sprintf("http-%d", portNum),
			ContainerPort: portNum,
			Protocol:      corev1.ProtocolTCP,
		}
	}
	return ports
}

// buildTargetPorts constructs a slice of v1.Port starting from 'start' with 'count' ports.
func buildTargetPorts(start int, count int) []v1.Port {
	ports := make([]v1.Port, count)
	for i := range count {
		ports[i] = v1.Port{
			Number: v1.PortNumber(start + i),
		}
	}
	return ports
}

// waitForDeploymentRollout waits until the Deployment has completed its update.
// It ensures that the new version is fully rolled out and available.
func waitForDeploymentRollout(tc *igwtestutils.TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("Waiting for Deployment %s/%s to complete rollout", deploy.Namespace, deploy.Name))

	key := types.NamespacedName{Name: deploy.Name, Namespace: deploy.Namespace}

	gomega.Eventually(func() error {
		currentDeploy := &appsv1.Deployment{}
		if err := tc.K8sClient.Get(tc.Context, key, currentDeploy); err != nil {
			return err
		}

		if currentDeploy.Generation > currentDeploy.Status.ObservedGeneration {
			return errors.New("deployment generation not observed yet")
		}

		desiredReplicas := *currentDeploy.Spec.Replicas

		if currentDeploy.Status.UpdatedReplicas < desiredReplicas {
			return fmt.Errorf("waiting for updated replicas: %d/%d", currentDeploy.Status.UpdatedReplicas, desiredReplicas)
		}

		if currentDeploy.Status.AvailableReplicas < desiredReplicas {
			return fmt.Errorf("waiting for available replicas: %d/%d", currentDeploy.Status.AvailableReplicas, desiredReplicas)
		}

		if currentDeploy.Status.Replicas > desiredReplicas {
			return fmt.Errorf("waiting for old replicas to terminate: %d > %d", currentDeploy.Status.Replicas, desiredReplicas)
		}

		return nil
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed(), "Deployment failed to roll out within timeout")

	ginkgo.By("Deployment rollout complete")
}

// deploymentReadyCondition is the testable core of waitForDeploymentReady.
// It checks replica counts AND that no pod owned by the deployment is still terminating,
// since terminating pods count toward ReadyReplicas during their graceful-shutdown window.
func deploymentReadyCondition(tc *igwtestutils.TestConfig, key types.NamespacedName) error {
	current := &appsv1.Deployment{}
	if err := tc.K8sClient.Get(tc.Context, key, current); err != nil {
		return err
	}

	if current.Status.ReadyReplicas == 0 {
		return errors.New("no replicas are ready yet")
	}

	if current.Spec.Replicas != nil && *current.Spec.Replicas != current.Status.Replicas {
		return fmt.Errorf("status replicas (%d) has not converged to spec (%d)",
			current.Status.Replicas, *current.Spec.Replicas)
	}

	if current.Status.Replicas != current.Status.ReadyReplicas {
		return fmt.Errorf("replicas mismatch: expected %d, got %d ready",
			current.Status.Replicas, current.Status.ReadyReplicas)
	}

	// Terminating pods retain Ready=True during graceful shutdown, inflating ReadyReplicas.
	// Explicitly wait for them to disappear before declaring the deployment ready.
	podList := &corev1.PodList{}
	if err := tc.K8sClient.List(tc.Context, podList,
		client.InNamespace(key.Namespace),
		client.MatchingLabels(current.Spec.Selector.MatchLabels),
	); err != nil {
		return fmt.Errorf("listing pods: %w", err)
	}
	for i := range podList.Items {
		if podList.Items[i].DeletionTimestamp != nil {
			return fmt.Errorf("pod %s is still terminating", podList.Items[i].Name)
		}
	}

	return nil
}

// waitForDeploymentReady waits for the Deployment to have all replicas ready.
func waitForDeploymentReady(tc *igwtestutils.TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("waiting for Deployment %s/%s to be ready", deploy.Namespace, deploy.Name))

	key := types.NamespacedName{Name: deploy.Name, Namespace: deploy.Namespace}

	gomega.Eventually(func() error {
		return deploymentReadyCondition(tc, key)
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
}

// generateTraffic sends multiple concurrent requests using the provided curl command.
// expectedStatus is matched against the HTTP_STATUS trailer appended by -w (e.g. statusOK).
// Pass "" to accept any exec-success regardless of HTTP status (use when intentionally
// producing server-side errors, e.g. to populate error metrics).
func generateTraffic(
	curlCmd []string,
	batches int,
	semaphore chan struct{},
	execFn func([]string) (string, error),
	retryDelay time.Duration,
	expectedStatus string,
) error {
	var wg sync.WaitGroup
	errorCh := make(chan error, batches)

	for i := range batches {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(requestNum int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			var err error
			for attempt := 0; attempt <= maxRetries; attempt++ {
				var resp string
				resp, err = execFn(curlCmd)
				if err == nil && (expectedStatus == "" || strings.Contains(resp, expectedStatus)) {
					return
				}
				if err == nil {
					err = fmt.Errorf("unexpected response (wanted %q): %.200s", expectedStatus, resp)
				}

				if attempt < maxRetries {
					time.Sleep(retryDelay)
				}
			}

			errorCh <- fmt.Errorf("request %d failed: %w", requestNum, err)
		}(i)
	}

	wg.Wait()
	close(errorCh)

	failures := make([]error, 0, batches)
	for err := range errorCh {
		failures = append(failures, err)
	}

	if len(failures) > 0 {
		return fmt.Errorf("found %d failed requests: %v", len(failures), failures)
	}

	return nil
}

// getPodsByLabel lists pods in a given namespace that have a specific label key-value pair.
func getPodsByLabel(ctx context.Context, k8sClient client.Client, namespace, labelKey, labelValue string) ([]corev1.Pod, error) {
	podList := &corev1.PodList{}
	labels := map[string]string{labelKey: labelValue}

	listOptions := []client.ListOption{
		client.InNamespace(namespace),
		client.MatchingLabels(labels),
	}

	if err := k8sClient.List(ctx, podList, listOptions...); err != nil {
		return nil, fmt.Errorf("failed to list pods with label %s=%s in namespace %s: %w", labelKey, labelValue, namespace, err)
	}
	return podList.Items, nil
}

// generateSequence generates a sequence of integers starting from 'start' with 'count' numbers.
func generateSequence(start int, count int) []int {
	nums := make([]int, count)
	for i := range count {
		nums[i] = start + i
	}
	return nums
}
