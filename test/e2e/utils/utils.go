/*
Copyright 2025 The llm-d Authors.

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

package utils

import (
	"bytes"
	"context"
	"errors"
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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/client"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const deploymentKind = "deployment"

func ScaleDeployment(cfg *testutils.TestConfig, nsName string, objects []string, increment int) {
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
			scale, err := cfg.KubeCli.AppsV1().Deployments(nsName).GetScale(cfg.Context, split[1], metav1.GetOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())

			scale.Spec.Replicas += int32(increment)
			_, err = cfg.KubeCli.AppsV1().Deployments(nsName).UpdateScale(cfg.Context, split[1], scale, metav1.UpdateOptions{})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}
	}
	PodsInDeploymentsReady(cfg, nsName, objects)
}

// GetModelServerPods returns the list of Prefill and Decode vLLM pods separately.
func GetModelServerPods(cfg *testutils.TestConfig, podLabels, prefillLabels, decodeLabels map[string]string, nsName string) ([]string, []string) {
	ginkgo.By("Getting Model server pods")

	pods := GetPods(cfg, podLabels, nsName)

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

func GetPods(cfg *testutils.TestConfig, labels map[string]string, nsName string) []corev1.Pod {
	podList := corev1.PodList{}
	selector := apilabels.SelectorFromSet(labels)
	err := cfg.K8sClient.List(cfg.Context, &podList,
		&client.ListOptions{LabelSelector: selector, Namespace: nsName})
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	pods := []corev1.Pod{}
	for _, pod := range podList.Items {
		if pod.DeletionTimestamp == nil {
			pods = append(pods, pod)
		}
	}

	return pods
}

// GetPodNames returns the names of all running pods matching the given label selector.
func GetPodNames(cfg *testutils.TestConfig, labels map[string]string, nsName string) []string {
	pods := GetPods(cfg, labels, nsName)
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
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

func PodsInDeploymentsReady(cfg *testutils.TestConfig, nsName string, objects []string) {
	isDeploymentReady := func(deploymentName string) bool {
		var deployment appsv1.Deployment
		err := cfg.K8sClient.Get(cfg.Context, types.NamespacedName{Namespace: nsName, Name: deploymentName}, &deployment)
		ginkgo.By(fmt.Sprintf("Waiting for deployment %q to be ready (err: %v): replicas=%#v, status=%#v", deploymentName, err, *deployment.Spec.Replicas, deployment.Status))
		return err == nil && *deployment.Spec.Replicas == deployment.Status.Replicas &&
			deployment.Status.Replicas == deployment.Status.ReadyReplicas
	}

	for _, kindAndName := range objects {
		split := strings.Split(kindAndName, "/")
		if strings.ToLower(split[0]) == deploymentKind {
			gomega.Eventually(isDeploymentReady).
				WithArguments(split[1]).
				WithPolling(cfg.Interval).
				WithTimeout(cfg.ReadyTimeout).
				Should(gomega.BeTrue())
		}
	}
}

func RunKustomize(kustomizeDir string) []string {
	// Use "kubectl kustomize" rather than the standalone "kustomize" binary.
	// CI/dev environments guarantee kubectl but may not have kustomize installed
	// (see Makefile.tools.mk check-kustomize target).
	command := exec.Command("kubectl", "kustomize", kustomizeDir)
	session, err := gexec.Start(command, nil, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
	return strings.Split(string(session.Out.Contents()), "\n---")
}

// RemoveEmptyArgs strips YAML list items that are empty strings after variable
// substitution (e.g. '- ""' produced when VLLM_EXTRA_ARGS_* is unset).
func RemoveEmptyArgs(inputs []string) []string {
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

// RemoveEmptyLabels strips YAML lines like "llm-d.ai/role: " where the value
// is empty after variable substitution. Kubernetes accepts empty-value labels,
// but the test pod-selector logic treats the key's presence as meaningful.
func RemoveEmptyLabels(inputs []string) []string {
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

func SubstituteMany(inputs []string, substitutions map[string]string) []string {
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

// GetMetrics fetches the current Prometheus metrics from the given metrics URL.
// Retries on transient connection errors (e.g. the previous EPP pod is still terminating).
func GetMetrics(metricsURL string) []string {
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

	return strings.Split(string(body), "\n")
}

// GetCounterMetric fetches the current value of a Prometheus counter metric from the given metrics URL.
// Retries on transient connection errors (e.g. the previous EPP pod is still terminating).
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

// ExtractFinishReason extracts the finish_reason field from a JSON response string.
func ExtractFinishReason(jsonStr string) string {
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

// ExtractFinishReasonFromStreaming extracts the finish_reason from the last SSE data chunk.
func ExtractFinishReasonFromStreaming(sseData string) string {
	// Find the last "finish_reason" that is not null
	lines := strings.Split(sseData, "\n")
	lastFinishReason := ""
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			fr := ExtractFinishReason(line)
			if fr != "" && fr != "null" {
				lastFinishReason = fr
			}
		}
	}
	return lastFinishReason
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

// DecodeCaseObjects decodes a multi-document YAML stream into unstructured
// objects and binds them to namespace.
func DecodeCaseObjects(data []byte, namespace string) ([]*unstructured.Unstructured, error) {
	var objects []*unstructured.Unstructured
	decoder := k8syaml.NewYAMLOrJSONDecoder(bytes.NewReader(data), 4096)
	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if errors.Is(err, io.EOF) {
				return objects, nil
			}
			return nil, fmt.Errorf("decode case resource: %w", err)
		}
		if len(obj.Object) == 0 {
			continue
		}
		obj.SetNamespace(namespace)
		objects = append(objects, obj)
	}
}

// CaseResources tracks the objects one test case created so cleanup can delete
// exactly that set and wait for their Pods to terminate.
type CaseResources struct {
	Client  client.Client
	created []*unstructured.Unstructured
}

// Create creates objects in order and records each object that was created.
func (r *CaseResources) Create(ctx context.Context, objects []*unstructured.Unstructured) error {
	for _, obj := range objects {
		if err := r.Client.Create(ctx, obj); err != nil {
			return fmt.Errorf("create %s/%s: %w", obj.GetKind(), obj.GetName(), err)
		}
		r.created = append(r.created, obj.DeepCopy())
	}
	return nil
}

// Delete removes the recorded objects newest-first with UID preconditions so a
// same-named object created later is never deleted.
func (r *CaseResources) Delete(ctx context.Context) error {
	var errs []error
	for i := len(r.created) - 1; i >= 0; i-- {
		obj := r.created[i]
		uid := obj.GetUID()
		err := r.Client.Delete(ctx, obj, client.PropagationPolicy(metav1.DeletePropagationForeground),
			client.Preconditions{UID: &uid})
		if err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("delete %s/%s: %w", obj.GetKind(), obj.GetName(), err))
		}
	}
	return errors.Join(errs...)
}

// Deleted reports whether every recorded object is gone and no Pod from a
// recorded Deployment still exists.
func (r *CaseResources) Deleted(ctx context.Context) (bool, error) {
	for _, obj := range r.created {
		current := &unstructured.Unstructured{}
		current.SetGroupVersionKind(obj.GroupVersionKind())
		err := r.Client.Get(ctx, client.ObjectKeyFromObject(obj), current)
		if err == nil && current.GetUID() == obj.GetUID() {
			return false, nil
		}
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		if obj.GetKind() == "Deployment" {
			selector, _, err := unstructured.NestedStringMap(obj.Object, "spec", "selector", "matchLabels")
			if err != nil {
				return false, err
			}
			pods := &corev1.PodList{}
			if err := r.Client.List(ctx, pods, client.InNamespace(obj.GetNamespace()), client.MatchingLabels(selector)); err != nil {
				return false, err
			}
			if len(pods.Items) != 0 {
				return false, nil
			}
		}
	}
	return true, nil
}

// DeferCaseCleanup registers case cleanup that runs stop, dumps diagnostics on
// failure when keepOnFailure is set, deletes the recorded resources, and waits
// until they are fully terminated.
func DeferCaseCleanup(cfg *testutils.TestConfig, keepOnFailure bool, resources *CaseResources, namespace string, stop func()) {
	ginkgo.DeferCleanup(func() {
		if stop != nil {
			stop()
		}
		if ginkgo.CurrentSpecReport().Failed() && keepOnFailure {
			testutils.DumpPodsAndLogs(cfg, namespace)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), cfg.ReadyTimeout)
		defer cancel()
		gomega.Expect(resources.Delete(ctx)).To(gomega.Succeed())
		gomega.Eventually(func() (bool, error) {
			return resources.Deleted(ctx)
		}, cfg.ReadyTimeout, cfg.Interval).Should(gomega.BeTrue(), "case resources and their Pods must terminate before names are reused")
	})
}
