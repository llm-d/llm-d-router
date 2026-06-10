package e2e

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

const (
	deploymentKind = "deployment"
)

// getMetricValue scrapes the EPP /metrics endpoint and returns the sum of all
// series for metricName whose label set matches every entry in filters.
// Returns 0 on scrape error or no match.
func getMetricValue(metricName string, filters map[string]string) float64 {
	resp, err := http.Get(fmt.Sprintf("http://localhost:%s/metrics", metricsPort))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0
	}
	var total float64
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, metricName+"{") && !strings.HasPrefix(line, metricName+" ") {
			continue
		}
		match := true
		for k, v := range filters {
			if !strings.Contains(line, fmt.Sprintf("%s=%q", k, v)) {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		f, err := strconv.ParseFloat(parts[len(parts)-1], 64)
		if err == nil {
			total += f
		}
	}
	return total
}

func scaleDeployment(objects []string, increment int) {
	direction := "up"
	absIncrement := increment
	if increment < 0 {
		direction = "down"
		absIncrement = -increment
	}

	for _, kindAndName := range objects {
		split := strings.Split(kindAndName, "/")
		if strings.ToLower(split[0]) == deploymentKind {
			ginkgo.By(fmt.Sprintf("Scaling the deployment %s %s by %d", split[1], direction, absIncrement))
			scale, err := testConfig.KubeCli.AppsV1().Deployments(nsName).GetScale(testConfig.Context, split[1], v1.GetOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

			scale.Spec.Replicas += int32(increment)
			_, err = testConfig.KubeCli.AppsV1().Deployments(nsName).UpdateScale(testConfig.Context, split[1], scale, v1.UpdateOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}
	}
	podsInDeploymentsReady(objects)
}

// getModelServerPods Returns the list of Prefill and Decode vLLM pods separately
func getModelServerPods(podLabels, prefillLabels, decodeLabels map[string]string) ([]string, []string) {
	ginkgo.By("Getting Model server pods")

	pods := getPods(podLabels)

	prefillValidator, err := apilabels.ValidatedSelectorFromSet(prefillLabels)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	decodeValidator, err := apilabels.ValidatedSelectorFromSet(decodeLabels)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	prefillPods := []string{}
	decodePods := []string{}

	for _, pod := range pods {
		podLabels := apilabels.Set(pod.Labels)
		switch {
		case prefillValidator.Matches(podLabels):
			prefillPods = append(prefillPods, pod.Name)
		case decodeValidator.Matches(podLabels):
			decodePods = append(decodePods, pod.Name)
		default:
			// If not labelled at all, it's a decode pod
			notFound := true
			for decodeKey := range decodeLabels {
				if _, ok := pod.Labels[decodeKey]; ok {
					notFound = false
					break
				}
			}
			if notFound {
				decodePods = append(decodePods, pod.Name)
			}
		}
	}

	return prefillPods, decodePods
}

func getPods(labels map[string]string) []corev1.Pod {
	podList := corev1.PodList{}
	selector := apilabels.SelectorFromSet(labels)
	err := testConfig.K8sClient.List(testConfig.Context, &podList, &client.ListOptions{LabelSelector: selector})
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	pods := []corev1.Pod{}
	for _, pod := range podList.Items {
		if pod.DeletionTimestamp == nil {
			pods = append(pods, pod)
		}
	}

	return pods
}

// getPodNames returns the names of all running pods matching the given label selector.
func getPodNames(labels map[string]string) []string {
	pods := getPods(labels)
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
}

func podsInDeploymentsReady(objects []string) {
	isDeploymentReady := func(deploymentName string) bool {
		var deployment appsv1.Deployment
		err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: nsName, Name: deploymentName}, &deployment)
		ginkgo.By(fmt.Sprintf("Waiting for deployment %q to be ready (err: %v): replicas=%#v, status=%#v", deploymentName, err, *deployment.Spec.Replicas, deployment.Status))
		return err == nil && *deployment.Spec.Replicas == deployment.Status.Replicas &&
			deployment.Status.Replicas == deployment.Status.ReadyReplicas
	}

	for _, kindAndName := range objects {
		split := strings.Split(kindAndName, "/")
		if strings.ToLower(split[0]) == deploymentKind {
			gomega.Eventually(isDeploymentReady).
				WithArguments(split[1]).
				WithPolling(interval).
				WithTimeout(readyTimeout).
				Should(gomega.BeTrue())
		}
	}
}

func runKustomize(kustomizeDir string) []string {
	// Use "kubectl kustomize" rather than the standalone "kustomize" binary.
	// CI/dev environments guarantee kubectl but may not have kustomize installed
	// (see Makefile.tools.mk check-kustomize target).
	command := exec.Command("kubectl", "kustomize", kustomizeDir)
	session, err := gexec.Start(command, nil, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
	return strings.Split(string(session.Out.Contents()), "\n---")
}

// removeEmptyArgs strips YAML list items that are empty strings after variable
// substitution (e.g. '- ""' produced when VLLM_EXTRA_ARGS_* is unset).
func removeEmptyArgs(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		lines := strings.Split(input, "\n")
		filtered := make([]string, 0, len(lines))
		for _, line := range lines {
			if strings.TrimSpace(line) == `- ""` {
				continue
			}
			if strings.TrimSpace(line) == `-` {
				continue
			}
			filtered = append(filtered, line)
		}
		outputs[idx] = strings.Join(filtered, "\n")
	}
	return outputs
}

// removeEmptyLabels strips YAML lines like "llm-d.ai/role: " where the value
// is empty after variable substitution. Kubernetes accepts empty-value labels,
// but the test pod-selector logic treats the key's presence as meaningful.
func removeEmptyLabels(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		lines := strings.Split(input, "\n")
		filtered := make([]string, 0, len(lines))
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			// Skip lines like "llm-d.ai/role:" (key with empty value after TrimSpace)
			if strings.HasSuffix(trimmed, ":") {
				if strings.Contains(trimmed, "llm-d.ai/role") {
					continue
				}
			}
			filtered = append(filtered, line)
		}
		outputs[idx] = strings.Join(filtered, "\n")
	}
	return outputs
}

func isModelReal(modelName string) bool {
	url := "https://huggingface.co/api/models/" + modelName

	resp, err := http.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

// removeRenderSidecar takes a slice of YAML strings (each may contain multiple
// objects separated by "---") and returns the same slice with the vllm-render
// container and the model-cache volume stripped from any Deployment.
func removeRenderSidecar(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		docs := strings.Split(input, "\n---")
		rendered := make([]string, 0, len(docs))
		for _, doc := range docs {
			if strings.TrimSpace(doc) == "" {
				continue
			}
			rendered = append(rendered, filterDocument(doc))
		}
		outputs[idx] = strings.Join(rendered, "\n---\n")
	}
	return outputs
}

func filterDocument(doc string) string {
	obj := &unstructured.Unstructured{}
	err := yaml.Unmarshal([]byte(doc), &obj.Object)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	if len(obj.Object) == 0 {
		return doc
	}
	if obj.GetKind() == "Deployment" {
		removePodSpecListItem(obj, "containers", "vllm-render")
		removePodSpecListItem(obj, "initContainers", "vllm-render")
		removePodSpecListItem(obj, "volumes", "model-cache")
	}
	out, err := yaml.Marshal(obj.Object)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	return strings.TrimRight(string(out), "\n")
}

// swapToSimRenderSidecar rewrites the vllm-render sidecar to run the inference
// sim, which emits mm_features that text-only vLLM does not.
func swapToSimRenderSidecar(inputs []string, simImage, modelName string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		docs := strings.Split(input, "\n---")
		rendered := make([]string, 0, len(docs))
		for _, doc := range docs {
			if strings.TrimSpace(doc) == "" {
				continue
			}
			rendered = append(rendered, swapRenderSidecarInDoc(doc, simImage, modelName))
		}
		outputs[idx] = strings.Join(rendered, "\n---\n")
	}
	return outputs
}

func swapRenderSidecarInDoc(doc, simImage, modelName string) string {
	obj := &unstructured.Unstructured{}
	err := yaml.Unmarshal([]byte(doc), &obj.Object)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	if len(obj.Object) == 0 || obj.GetKind() != "Deployment" {
		return doc
	}
	path := []string{"spec", "template", "spec", "containers"}
	containers, found, err := unstructured.NestedSlice(obj.Object, path...)
	if err != nil || !found {
		return doc
	}
	for i, raw := range containers {
		c, ok := raw.(map[string]any)
		if !ok || c["name"] != "vllm-render" {
			continue
		}
		c["image"] = simImage
		delete(c, "command")
		// --log-http: surface inbound URL/method so CI failures are diagnosable.
		c["args"] = []any{"--port=8000", "--model=" + modelName, "--log-http"}
		delete(c, "volumeMounts")
		containers[i] = c
	}
	err = unstructured.SetNestedSlice(obj.Object, containers, path...)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	removePodSpecListItem(obj, "volumes", "model-cache")
	out, err := yaml.Marshal(obj.Object)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	return strings.TrimRight(string(out), "\n")
}

func removePodSpecListItem(obj *unstructured.Unstructured, fieldName, itemName string) {
	path := []string{"spec", "template", "spec", fieldName}
	items, found, err := unstructured.NestedSlice(obj.Object, path...)
	if err != nil || !found {
		return
	}
	filtered := make([]any, 0, len(items))
	for _, item := range items {
		if m, ok := item.(map[string]any); ok && m["name"] == itemName {
			continue
		}
		filtered = append(filtered, item)
	}
	if len(filtered) == 0 {
		unstructured.RemoveNestedField(obj.Object, path...)
		return
	}
	err = unstructured.SetNestedSlice(obj.Object, filtered, path...)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
}

func substituteMany(inputs []string, substitutions map[string]string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		output := input
		for key, value := range substitutions {
			output = strings.ReplaceAll(output, key, value)
		}
		outputs[idx] = output
	}
	return outputs
}

// getCounterMetric fetches the current value of a Prometheus counter metric from the given metrics URL.
// Retries on transient connection errors (e.g. the previous EPP pod is still terminating).
//
//nolint:unparam // metricName may vary in future test cases
func getCounterMetric(metricsURL, metricName, labelMatch string) int {
	var body []byte
	gomega.Eventually(func() error {
		resp, err := http.Get(metricsURL)
		if err != nil {
			return err
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status %d", resp.StatusCode)
		}
		body, err = io.ReadAll(resp.Body)
		return err
	}, 10*time.Second, 1*time.Second).Should(gomega.Succeed())

	metricsText := string(body)
	for _, line := range strings.Split(metricsText, "\n") {
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

// extractFinishReason extracts the finish_reason field from a JSON response string.
func extractFinishReason(jsonStr string) string {
	// Simple extraction - look for "finish_reason":"value" pattern
	idx := strings.Index(jsonStr, `"finish_reason":"`)
	if idx == -1 {
		// Try with null value
		if strings.Contains(jsonStr, `"finish_reason":null`) {
			return "null"
		}
		return ""
	}
	start := idx + len(`"finish_reason":"`)
	end := strings.Index(jsonStr[start:], `"`)
	if end == -1 {
		return ""
	}
	return jsonStr[start : start+end]
}

// extractFinishReasonFromStreaming extracts the finish_reason from the last SSE data chunk.
func extractFinishReasonFromStreaming(sseData string) string {
	// Find the last "finish_reason" that is not null
	lines := strings.Split(sseData, "\n")
	lastFinishReason := ""
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			fr := extractFinishReason(line)
			if fr != "" && fr != "null" {
				lastFinishReason = fr
			}
		}
	}
	return lastFinishReason
}

// getPodRequestCount gets the total vLLM request count from a pod's metrics endpoint.
func getPodRequestCount(podName string) int {
	ginkgo.By("Getting request count from pod: " + podName)

	// Use Kubernetes API proxy to access the metrics endpoint
	output, err := testConfig.KubeCli.CoreV1().RESTClient().
		Get().
		Namespace(nsName).
		Resource("pods").
		Name(podName + ":8000").
		SubResource("proxy").
		Suffix("metrics").
		DoRaw(testConfig.Context)
	if err != nil {
		ginkgo.By(fmt.Sprintf("Warning: Could not get metrics from pod %s: %v", podName, err))
		return -1
	}

	return parseRequestCountFromMetrics(string(output))
}

// parseRequestCountFromMetrics extracts the request count from Prometheus metrics output.
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

// dumpPodsAndLogs dumps all pod statuses and their logs to the Ginkgo writer.
// Call this before cleanup to insure the information is available when CI tests fail.
func dumpPodsAndLogs() {
	if testConfig == nil || testConfig.KubeCli == nil {
		ginkgo.GinkgoWriter.Println("Skipping pod dump: cluster not initialized")
		return
	}

	ginkgo.GinkgoWriter.Printf("\n=== Dumping pod states and logs (namespace: %s) ===\n", nsName)

	ctx, cancel := context.WithTimeout(testConfig.Context, 30*time.Second)
	defer cancel()

	pods, err := testConfig.KubeCli.CoreV1().Pods(nsName).List(ctx, v1.ListOptions{})
	if err != nil {
		ginkgo.GinkgoWriter.Printf("Failed to list pods: %v\n", err)
		return
	}

	ginkgo.GinkgoWriter.Printf("Total pods found: %d\n\n", len(pods.Items))

	// Print summary table (like kubectl get pods)
	ginkgo.GinkgoWriter.Printf("%-55s %-8s %-22s %-8s %-6s\n", "NAME", "READY", "STATUS", "RESTARTS", "AGE")
	for i := range pods.Items {
		pod := &pods.Items[i]
		ready, total := 0, len(pod.Spec.Containers)
		restarts := int32(0)
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				ready++
			}
			restarts += cs.RestartCount
		}
		status := string(pod.Status.Phase)
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.State.Waiting != nil {
				status = cs.State.Waiting.Reason
				break
			}
		}
		age := ""
		if !pod.CreationTimestamp.IsZero() {
			d := time.Since(pod.CreationTimestamp.Time).Round(time.Second)
			age = d.String()
		}
		ginkgo.GinkgoWriter.Printf("%-55s %-8s %-22s %-8d %-6s\n",
			pod.Name, fmt.Sprintf("%d/%d", ready, total), status, restarts, age)
	}
	ginkgo.GinkgoWriter.Println()

	for i := range pods.Items {
		pod := &pods.Items[i]
		ginkgo.GinkgoWriter.Printf("--- Pod: %s | Phase: %s | Node: %s ---\n",
			pod.Name, pod.Status.Phase, pod.Spec.NodeName)

		for _, cs := range pod.Status.InitContainerStatuses {
			printContainerStatus("init", cs)
		}
		for _, cs := range pod.Status.ContainerStatuses {
			printContainerStatus("container", cs)
		}

		restarted := map[string]bool{}
		for _, cs := range pod.Status.InitContainerStatuses {
			if cs.RestartCount > 0 {
				restarted[cs.Name] = true
			}
		}
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.RestartCount > 0 {
				restarted[cs.Name] = true
			}
		}

		for _, c := range pod.Spec.InitContainers {
			if restarted[c.Name] {
				dumpContainerLogs(ctx, pod.Name, c.Name, true)
			}
			dumpContainerLogs(ctx, pod.Name, c.Name, false)
		}
		for _, c := range pod.Spec.Containers {
			if restarted[c.Name] {
				dumpContainerLogs(ctx, pod.Name, c.Name, true)
			}
			dumpContainerLogs(ctx, pod.Name, c.Name, false)
		}
	}
	ginkgo.GinkgoWriter.Println("=== End of pod dump ===")
}

func printContainerStatus(kind string, cs corev1.ContainerStatus) {
	status := fmt.Sprintf("  [%s] %s | ready=%v restarts=%d", kind, cs.Name, cs.Ready, cs.RestartCount)
	if cs.State.Waiting != nil {
		status += " | Waiting: " + cs.State.Waiting.Reason
	}
	if cs.State.Terminated != nil {
		status += fmt.Sprintf(" | Terminated: %s (exit %d)", cs.State.Terminated.Reason, cs.State.Terminated.ExitCode)
	}
	ginkgo.GinkgoWriter.Println(status)
}

func dumpContainerLogs(ctx context.Context, podName, containerName string, previous bool) {
	tailLines := int64(100)
	req := testConfig.KubeCli.CoreV1().Pods(nsName).GetLogs(podName, &corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tailLines,
		Previous:  previous,
	})
	stream, err := req.Stream(ctx)
	if err != nil {
		ginkgo.GinkgoWriter.Printf("  [logs] %s/%s: failed to stream logs: %v\n", podName, containerName, err)
		return
	}
	defer func() {
		if err := stream.Close(); err != nil {
			ginkgo.GinkgoWriter.Printf("  [logs] %s/%s: failed to close log stream: %v\n", podName, containerName, err)
		}
	}()

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, stream); err != nil {
		ginkgo.GinkgoWriter.Printf("  [logs] %s/%s: failed to read logs: %v\n", podName, containerName, err)
		return
	}
	label := "last 100 lines"
	if previous {
		label = "previous instance, last 100 lines"
	}
	ginkgo.GinkgoWriter.Printf("  [logs] %s/%s (%s):\n%s\n", podName, containerName, label, buf.String())
}
