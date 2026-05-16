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

package e2e

import (
	"fmt"
	"os/exec"
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive

	"github.com/llm-d/llm-d-router/test/sidecar/utils"
)

const (
	namespace   = "hc4ai-operator"
	qwenPodName = "qwen2-0--5b"
)

var _ = Describe("Sidecar", Ordered, func() {
	// After each test, check for failures and collect logs, events,
	// and pod descriptions for debugging.
	AfterEach(func() {
		specReport := CurrentSpecReport()
		if specReport.Failed() {
			By("Fetching qwen pod logs")
			cmd := exec.Command("kubectl", "logs", qwenPodName, "-n", namespace)
			controllerLogs, err := utils.Run(cmd)
			if err == nil {
				_, _ = fmt.Fprintf(GinkgoWriter, "qwen logs:\n %s", controllerLogs)
			} else {
				_, _ = fmt.Fprintf(GinkgoWriter, "Failed to get qwen logs: %s", err)
			}
		}

		By("Fetching Kubernetes events")
		cmd := exec.Command("kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp")
		eventsOutput, err := utils.Run(cmd)
		if err == nil {
			_, _ = fmt.Fprintf(GinkgoWriter, "Kubernetes events:\n%s", eventsOutput)
		} else {
			_, _ = fmt.Fprintf(GinkgoWriter, "Failed to get Kubernetes events: %s", err)
		}

		By("Fetching controller manager pod description")
		cmd = exec.Command("kubectl", "describe", "pod", qwenPodName, "-n", namespace)
		podDescription, err := utils.Run(cmd)
		if err == nil {
			fmt.Println("Pod description:\n", podDescription)
		} else {
			fmt.Println("Failed to describe controller pod")
		}
	})

	SetDefaultEventuallyTimeout(20 * time.Second)
	SetDefaultEventuallyPollingInterval(time.Second)

	Context("Qwen", func() {
		It("should run successfully", func() {
			By("validating that the qwen pod is running as expected")

			verifyQwenUp := func(g Gomega) {
				cmd := exec.Command("kubectl", "get",
					"pods", qwenPodName, "-o", "jsonpath={.status.phase}",
					"-n", namespace,
				)
				output, err := utils.Run(cmd)
				g.Expect(err).NotTo(HaveOccurred())
				g.Expect(output).To(Equal("Running"), "Incorrect qwen pod status")
			}

			Eventually(verifyQwenUp).Should(Succeed())
		})
	})
})
