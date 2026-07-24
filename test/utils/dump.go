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

package utils

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/onsi/ginkgo/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// dumpTimeout bounds the pod list and log streaming during a dump.
	dumpTimeout = 30 * time.Second
	// dumpTailLines is the number of trailing log lines fetched per container.
	dumpTailLines = int64(200)
	// dumpLimitBytes caps the log bytes fetched per container.
	dumpLimitBytes = int64(1 << 20) // 1MiB
)

// dumpConfig controls how much container log DumpPodsAndLogs fetches.
type dumpConfig struct {
	tailLines  int64
	limitBytes int64
}

// DumpOption overrides DumpPodsAndLogs defaults.
type DumpOption func(*dumpConfig)

// WithFullLogs streams complete container logs instead of the default tail, for
// callers that need the entire log (e.g. a gateway access log). A non-positive
// tail/limit disables that cap.
func WithFullLogs() DumpOption {
	return func(c *dumpConfig) { c.tailLines, c.limitBytes = 0, 0 }
}

// DumpPodsAndLogs prints pod statuses and container logs for the given namespace
// to the Ginkgo writer. Call this before cleanup to ensure the information is
// available when CI tests fail.
func DumpPodsAndLogs(cfg *TestConfig, nsName string, opts ...DumpOption) {
	if cfg == nil || cfg.KubeCli == nil {
		ginkgo.GinkgoWriter.Println("Skipping pod dump: cluster not initialized")
		return
	}

	dc := dumpConfig{tailLines: dumpTailLines, limitBytes: dumpLimitBytes}
	for _, opt := range opts {
		opt(&dc)
	}

	ginkgo.GinkgoWriter.Printf("\n=== Dumping pod states and logs (namespace: %s) ===\n", nsName)

	ctx, cancel := context.WithTimeout(cfg.Context, dumpTimeout)
	defer cancel()

	pods, err := cfg.KubeCli.CoreV1().Pods(nsName).List(ctx, metav1.ListOptions{})
	if err != nil {
		ginkgo.GinkgoWriter.Printf("Failed to list pods: %v\n", err)
		return
	}

	ginkgo.GinkgoWriter.Printf("Total pods found: %d\n\n", len(pods.Items))

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
				dumpContainerLogs(ctx, cfg, pod.Namespace, pod.Name, c.Name, true, dc)
			}
			dumpContainerLogs(ctx, cfg, pod.Namespace, pod.Name, c.Name, false, dc)
		}
		for _, c := range pod.Spec.Containers {
			if restarted[c.Name] {
				dumpContainerLogs(ctx, cfg, pod.Namespace, pod.Name, c.Name, true, dc)
			}
			dumpContainerLogs(ctx, cfg, pod.Namespace, pod.Name, c.Name, false, dc)
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

func dumpContainerLogs(ctx context.Context, cfg *TestConfig, nsName, podName, containerName string, previous bool, dc dumpConfig) {
	logOpts := &corev1.PodLogOptions{Container: containerName, Previous: previous}
	if dc.tailLines > 0 {
		logOpts.TailLines = &dc.tailLines
	}
	if dc.limitBytes > 0 {
		logOpts.LimitBytes = &dc.limitBytes
	}
	req := cfg.KubeCli.CoreV1().Pods(nsName).GetLogs(podName, logOpts)
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
	scope := "full log"
	if dc.tailLines > 0 {
		scope = fmt.Sprintf("last %d lines", dc.tailLines)
	}
	if previous {
		scope = "previous instance, " + scope
	}
	ginkgo.GinkgoWriter.Printf("  [logs] %s/%s (%s):\n%s\n", podName, containerName, scope, buf.String())
}
