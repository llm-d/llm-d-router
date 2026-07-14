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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apilabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Model-server pod selectors keyed by the llm-d.ai/role label.
var (
	encodeSelector  = map[string]string{"llm-d.ai/role": "encode"}
	prefillSelector = map[string]string{"llm-d.ai/role": "prefill"}
	decodeSelector  = map[string]string{"llm-d.ai/role": "decode"}
)

// getPodNames returns the names of all non-terminating pods matching the labels.
func getPodNames(labels map[string]string) []string {
	podList := corev1.PodList{}
	selector := apilabels.SelectorFromSet(labels)
	err := testConfig.K8sClient.List(testConfig.Context, &podList,
		&client.ListOptions{Namespace: nsName, LabelSelector: selector})
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	names := make([]string, 0, len(podList.Items))
	for _, pod := range podList.Items {
		if pod.DeletionTimestamp == nil {
			names = append(names, pod.Name)
		}
	}
	return names
}

// podsInDeploymentsReady waits until every Deployment named in objects reports
// all replicas ready. Non-Deployment entries are ignored.
func podsInDeploymentsReady(objects []string) {
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
