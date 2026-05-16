/*
Copyright 2024 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

const (
	argPort             = "--port"
	argDataParallelSize = "--data-parallel-size"
)

var _ = ginkgo.Describe("InferencePool", func() {
	var infObjective *v1alpha2.InferenceObjective
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(testConfig)

		ginkgo.By("Modifying deployment using local image for testing (temporary).")
		deploy := &appsv1.Deployment{}
		key := types.NamespacedName{Name: modelServerName, Namespace: testConfig.NsName}

		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, key, deploy)
			if err != nil {
				return err
			}

			// Instead of hardcoding arguments, we can instead replace the arguments that need
			// to be changed, preserving any others that may exist.
			var newArgs []string
			skipNext := false
			for _, arg := range deploy.Spec.Template.Spec.Containers[0].Args {
				if skipNext {
					skipNext = false
					continue
				}
				// If this is one of the arguments we are updating, skip it AND its value
				if arg == argPort || arg == argDataParallelSize {
					skipNext = true
					continue
				}
				newArgs = append(newArgs, arg)
			} // contains only the args we want to keep

			// add new arguments to open proper ports
			newArgs = append(newArgs, argPort, strconv.Itoa(firstPort))
			newArgs = append(newArgs, argDataParallelSize, strconv.Itoa(numPorts))
			deploy.Spec.Template.Spec.Containers[0].Args = newArgs
			deploy.Spec.Template.Spec.Containers[0].Ports = buildContainerPorts(firstPort, numPorts)

			return testConfig.K8sClient.Update(testConfig.Context, deploy)

		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		waitForDeploymentRollout(testConfig, deploy)

		pool := &v1.InferencePool{}
		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, key, pool)
			if err != nil {
				return err
			}

			pool.Spec.TargetPorts = buildTargetPorts(firstPort, numPorts)

			return testConfig.K8sClient.Update(testConfig.Context, pool)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		ginkgo.By("Restarting EPP to force configuration reload")
		// We delete the EPP *POD*, not the deployment. The Deployment will recreate it immediately.
		// This forces the new EPP process to read the Multi-Port InferencePool from scratch.
		eppLabels := client.MatchingLabels{"app": inferExtName}
		gomega.Expect(testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.Pod{}, client.InNamespace(testConfig.NsName), eppLabels)).To(gomega.Succeed())

		// Wait for the new EPP to be ready
		eppDeploy := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: inferExtName, Namespace: testConfig.NsName}}
		waitForDeploymentReady(testConfig, eppDeploy)

		ginkgo.By("Creating an InferenceObjective resource")
		infObjective = newInferenceObjective(testConfig.NsName)
		gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, infObjective)).To(gomega.Succeed())

		ginkgo.By("Ensuring the InferenceObjective resource exists in the namespace")
		gomega.Eventually(func() error {
			return testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjective.Namespace, Name: infObjective.Name}, infObjective)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting the InferenceObjective test resource.")
		cleanupInferObjectiveResources()
		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjective.Namespace, Name: infObjective.Name}, infObjective)
			if err == nil {
				return errors.New("InferenceObjective resource still exists")
			}
			if k8serrors.IsNotFound(err) {
				return nil
			}
			return err
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		ginkgo.By("Restoring vLLM Deployment and InferencePool.")
		key := types.NamespacedName{Name: modelServerName, Namespace: testConfig.NsName}

		// Restore InferencePool
		pool := &v1.InferencePool{}
		gomega.Eventually(func() error {
			if err := testConfig.K8sClient.Get(testConfig.Context, key, pool); err != nil {
				return err
			}
			pool.Spec.TargetPorts = []v1.Port{{Number: 8000}}
			return testConfig.K8sClient.Update(testConfig.Context, pool)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		// Restore Deployment Args
		deploy := &appsv1.Deployment{}
		gomega.Eventually(func() error {
			if err := testConfig.K8sClient.Get(testConfig.Context, key, deploy); err != nil {
				return err
			}

			// Filter out the custom args we added in BeforeEach
			var originalArgs []string
			skipNext := false
			for _, arg := range deploy.Spec.Template.Spec.Containers[0].Args {
				if skipNext {
					skipNext = false
					continue
				}
				if arg == argPort || arg == argDataParallelSize {
					skipNext = true
					continue
				}
				originalArgs = append(originalArgs, arg)
			}
			deploy.Spec.Template.Spec.Containers[0].Args = originalArgs

			// Restore container ports to just 8000
			deploy.Spec.Template.Spec.Containers[0].Ports = []corev1.ContainerPort{
				{Name: "http-8000", ContainerPort: 8000, Protocol: corev1.ProtocolTCP},
			}

			return testConfig.K8sClient.Update(testConfig.Context, deploy)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		// Wait for rollback to finish.
		waitForDeploymentRollout(testConfig, deploy)
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
			verifyTrafficRouting()
		})

		ginkgo.It("Should expose EPP metrics after generating traffic", func() {
			verifyMetrics()
		})
	})

	ginkgo.When("Leader election is enabled", func() {
		ginkgo.It("Should elect one leader and have other pods as not ready", func() {
			if !leaderElectionEnabled {
				ginkgo.Skip("Leader election is not enabled for this test run, skipping.")
			}

			ginkgo.By("Verifying that exactly one EPP pod is ready")
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				err := testConfig.K8sClient.List(testConfig.Context, podList, client.InNamespace(testConfig.NsName), client.MatchingLabels{"app": inferExtName})
				g.Expect(err).NotTo(gomega.HaveOccurred())

				// The deployment should have 3 replicas for leader election.
				g.Expect(podList.Items).To(gomega.HaveLen(3))

				readyPods := 0
				for _, pod := range podList.Items {
					for _, cond := range pod.Status.Conditions {
						if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
							readyPods++
						}
					}
				}
				g.Expect(readyPods).To(gomega.Equal(1), "Expected exactly one pod to be ready")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
		})

		ginkgo.It("Should successfully failover and serve traffic after the leader pod is deleted", func() {
			if !leaderElectionEnabled {
				ginkgo.Skip("Leader election is not enabled for this test run, skipping.")
			}

			ginkgo.By("STEP 1: Verifying initial leader is working correctly before failover")
			verifyTrafficRouting()
			verifyMetrics()

			ginkgo.By("STEP 2: Finding and deleting the current leader pod")
			oldLeaderPod := findReadyPod()
			ginkgo.By("Found initial leader pod: " + oldLeaderPod.Name)

			ginkgo.By(fmt.Sprintf("Deleting leader pod %s to trigger failover", oldLeaderPod.Name))
			gomega.Expect(testConfig.K8sClient.Delete(testConfig.Context, oldLeaderPod)).To(gomega.Succeed())

			ginkgo.By("STEP 3: Waiting for a new leader to be elected")
			// The deployment controller will create a new pod. We need to wait for the total number of pods
			// to be back to 3, and for one of the other pods to become the new leader.
			deploy := &appsv1.Deployment{}
			gomega.Eventually(func() error {
				return testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: inferExtName}, deploy)
			}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

			// Wait for one replica to become ready again.
			igwtestutils.DeploymentReadyReplicas(testConfig, deploy, 1)

			// Also wait for the total number of replicas to be back to 3.
			gomega.Eventually(func(g gomega.Gomega) {
				d := &appsv1.Deployment{}
				err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: inferExtName}, d)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(d.Status.Replicas).To(gomega.Equal(int32(3)), "Deployment should have 3 replicas")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())

			ginkgo.By("STEP 4: Verifying a new, different leader is elected")
			var newLeaderPod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				// Find the current ready pod.
				newLeaderPod = findReadyPod()

				// Ensure the new leader is not the same as the one we just deleted.
				// This guards against a race condition where we might find the old leader
				// before its status is updated to NotReady.
				g.Expect(newLeaderPod.Name).NotTo(gomega.Equal(oldLeaderPod.Name), "The new leader should not be the same as the old deleted leader")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
			ginkgo.By("Found new leader pod: " + newLeaderPod.Name)

			ginkgo.By("STEP 5: Verifying the new leader is working correctly after failover")
			verifyTrafficRouting()
			verifyMetrics()
		})
	})
})
