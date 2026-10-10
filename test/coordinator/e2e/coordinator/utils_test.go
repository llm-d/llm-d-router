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

package coordinate2e

import (
	"fmt"
	"strings"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestParseEnvoyProfileRoutes(t *testing.T) {
	g := gomega.NewWithT(t)
	logs := `[envoy] epp-profile=encode id=req-1-enc-0 decision-id=decision-1 upstream=10.0.0.1:8000
[envoy] epp-profile=encode id=req-1-enc-1 decision-id=decision-1 upstream=10.0.0.1:8000
[envoy] epp-profile=encode id=req-1-enc-0 decision-id=decision-1 upstream=10.0.0.1:8000
[envoy] epp-profile=prefill id=req-1 decision-id=decision-1 upstream=10.0.0.2:8000
[envoy] epp-profile=decode id=req-1 decision-id=decision-1 upstream=10.0.0.3:8000
[envoy] epp-profile=encode id=req-2-enc-0 decision-id=decision-2 upstream=10.0.0.1:8000
[envoy] epp-profile=prefill id=req-1-enc-0 decision-id=decision-1 upstream=10.0.0.2:8000
[envoy] epp-profile=encode id=req-1-enc-2 decision-id=decision-1 upstream=-
[envoy] epp-profile=- id=req-1 decision-id=- upstream=10.0.0.4:8080`
	roles := map[string]map[string]bool{"encode": nil, "prefill": nil, "decode": nil}
	g.Expect(parseEnvoyProfileRoutes(logs, roles, "req-1")).To(gomega.Equal(map[string][]envoyProfileRoute{
		"encode": {
			{requestID: "req-1-enc-0", revisionDecisionID: "decision-1", upstream: "10.0.0.1"},
			{requestID: "req-1-enc-1", revisionDecisionID: "decision-1", upstream: "10.0.0.1"},
			{requestID: "req-1-enc-0", revisionDecisionID: "decision-1", upstream: "10.0.0.1"},
		},
		"prefill": {{requestID: "req-1", revisionDecisionID: "decision-1", upstream: "10.0.0.2"}},
		"decode":  {{requestID: "req-1", revisionDecisionID: "decision-1", upstream: "10.0.0.3"}},
	}))
}

// roleSelector returns the pod selector for a single model-server role.
func roleSelector(role string) map[string]string {
	return map[string]string{"llm-d.ai/role": role}
}

// Model-server pod selectors keyed by the llm-d.ai/role label.
var (
	encodeSelector  = roleSelector("encode")
	prefillSelector = roleSelector("prefill")
	decodeSelector  = roleSelector("decode")
)

// listRolePods returns all non-terminating pods matching the labels.
func listRolePods(labels map[string]string) []corev1.Pod {
	podList := corev1.PodList{}
	selector := apilabels.SelectorFromSet(labels)
	err := testConfig.K8sClient.List(testConfig.Context, &podList,
		&client.ListOptions{Namespace: getNamespace(), LabelSelector: selector})
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	pods := make([]corev1.Pod, 0, len(podList.Items))
	for _, pod := range podList.Items {
		if pod.DeletionTimestamp == nil {
			pods = append(pods, pod)
		}
	}
	return pods
}

// getPodNames returns the names of all non-terminating pods matching the labels.
func getPodNames(labels map[string]string) []string {
	pods := listRolePods(labels)
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
}

// podIPs returns the set of pod IPs of all non-terminating pods matching the
// labels. Pods without an assigned IP are omitted.
func podIPs(labels map[string]string) map[string]bool {
	pods := listRolePods(labels)
	ips := make(map[string]bool, len(pods))
	for _, pod := range pods {
		if pod.Status.PodIP != "" {
			ips[pod.Status.PodIP] = true
		}
	}
	return ips
}

// podsInDeploymentsReady waits until every Deployment named in objects reports
// all replicas ready in nsName. Non-Deployment entries are ignored.
func podsInDeploymentsReady(nsName string, objects []string) {
	isDeploymentReady := func(deploymentName string) bool {
		var deployment appsv1.Deployment
		err := testConfig.K8sClient.Get(testConfig.Context,
			types.NamespacedName{Namespace: nsName, Name: deploymentName}, &deployment)
		if err != nil || deployment.Spec.Replicas == nil {
			return false
		}
		ginkgo.By(fmt.Sprintf("Waiting for deployment %q to be ready: replicas=%d, status=%#v",
			deploymentName, *deployment.Spec.Replicas, deployment.Status))
		return *deployment.Spec.Replicas == deployment.Status.Replicas &&
			deployment.Status.Replicas == deployment.Status.ReadyReplicas
	}

	for _, kindAndName := range objects {
		split := strings.Split(kindAndName, "/")
		if len(split) == 2 && strings.EqualFold(split[0], "Deployment") {
			gomega.Eventually(isDeploymentReady).
				WithArguments(split[1]).
				WithPolling(defaultInterval).
				WithTimeout(readyTimeout).
				Should(gomega.BeTrue())
		}
	}
}
