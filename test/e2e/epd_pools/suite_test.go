/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// Package epd_pools runs end-to-end tests against the e-p-d-pools env: one
// InferencePool per phase (encode, prefill, decode) behind a single gateway
// that routes on the EPP-Phase header. The coordinator and its vllm-render
// sidecar are deployed but bypassed — tests POST directly to the gateway,
// emitting the same per-stage payloads the coordinator would (see
// coordinator/docs/communication.md). Mirrors the lifecycle of
// test/e2e/e2e_suite_test.go: BeforeSuite creates a kind cluster, exec's
// scripts/kind-dev-env.sh with DISAGG_POOLS_TOPOLOGY=true to bring the env
// up, and ReportAfterSuite tears the cluster down (gated by
// E2E_KEEP_CLUSTER_ON_FAILURE).
package epd_pools

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	k8slog "sigs.k8s.io/controller-runtime/pkg/log"
	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	infextv1a2 "github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/epp/util/env"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	// kindClusterName is the name of the kind cluster created for the
	// e-p-d-pools e2e suite. Distinct from the single-pool e2e cluster
	// (`e2e-tests`) and the dev cluster (`llm-d-router-dev`).
	kindClusterName = "e2e-pools-tests"

	defaultReadyTimeout = 10 * time.Minute
	defaultInterval     = time.Second * 2

	// defaultCoordinatorHostPort is the host port mapped to the coordinator
	// NodePort 30081. The coordinator is deployed for parity with prod but
	// is bypassed by these tests; the mapping is kept so the env script
	// (which provisions the NodePort) succeeds.
	defaultCoordinatorHostPort = 30081

	// defaultGatewayHostPort is the host port mapped to the gateway
	// NodePort 30080. Tests POST directly here, emitting the same
	// per-stage payloads the coordinator would (see coordinator
	// communication.md).
	defaultGatewayHostPort = 30080

	// envSetupTimeout caps the kind-dev-env.sh exec — Istio install + image
	// pulls + model loads can take several minutes on a cold run.
	envSetupTimeout = 30 * time.Minute
)

var (
	port        string = env.GetEnvString("E2E_PORT", fmt.Sprintf("%d", defaultCoordinatorHostPort), ginkgo.GinkgoLogr)
	gatewayPort string = env.GetEnvString("E2E_GATEWAY_PORT", fmt.Sprintf("%d", defaultGatewayHostPort), ginkgo.GinkgoLogr)
	metricsPort string = env.GetEnvString("E2E_METRICS_PORT", "32090", ginkgo.GinkgoLogr)

	testConfig *testutils.TestConfig

	keepClusterOnFailure = env.GetEnvBool("E2E_KEEP_CLUSTER_ON_FAILURE", false, ginkgo.GinkgoLogr)

	containerRuntime    = env.GetEnvString("CONTAINER_RUNTIME", "docker", ginkgo.GinkgoLogr)
	eppImage            = env.GetEnvString("EPP_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:dev", ginkgo.GinkgoLogr)
	vllmSimImage        = env.GetEnvString("VLLM_IMAGE", "ghcr.io/llm-d/llm-d-inference-sim:v0.9.0", ginkgo.GinkgoLogr)
	sideCarImage        = env.GetEnvString("SIDECAR_IMAGE", "ghcr.io/llm-d/llm-d-router-disagg-sidecar:dev", ginkgo.GinkgoLogr)
	vllmRenderImage     = env.GetEnvString("VLLM_RENDER_IMAGE", "vllm/vllm-openai-cpu:v0.21.0", ginkgo.GinkgoLogr)
	coordinatorImage    = env.GetEnvString("COORDINATOR_IMAGE", "ghcr.io/revit13/llm-d-coordinator:dev", ginkgo.GinkgoLogr)
	downloaderHTTPImage = env.GetEnvString("MOCK_DOWNLOADER_HTTP_IMAGE", "python:3.10-slim", ginkgo.GinkgoLogr)
	downloaderInitImage = env.GetEnvString("MOCK_DOWNLOADER_INIT_IMAGE", "busybox:1.36", ginkgo.GinkgoLogr)
	modelName           = env.GetEnvString("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct", ginkgo.GinkgoLogr)

	nsName     = env.GetEnvString("NAMESPACE", "default", ginkgo.GinkgoLogr)
	k8sContext = env.GetEnvString("K8S_CONTEXT", "", ginkgo.GinkgoLogr)

	readyTimeout = env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr)
	interval     = defaultInterval

	// gatewayBaseURL is the host-side base URL the tests POST to. The
	// gateway routes per-phase based on the EPP-Phase header.
	gatewayBaseURL = fmt.Sprintf("http://localhost:%s", gatewayPort)
)

func TestEPDPoolsE2E(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "EPD Pools E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	if k8sContext == "" {
		setupK8sCluster()
	}
	testConfig = testutils.NewTestConfig(nsName, k8sContext)
	setupK8sClient()

	ginkgo.By("Bringing up the e-p-d-pools env via scripts/kind-dev-env.sh")
	bringUpEnv()
})

// ReportAfterSuite mirrors test/e2e/e2e_suite_test.go's pattern: it fires
// regardless of pass/fail (including BeforeSuite failures) and gates cluster
// teardown on E2E_KEEP_CLUSTER_ON_FAILURE.
var _ = ginkgo.ReportAfterSuite("cleanup", func(report ginkgo.Report) {
	shouldKeep := keepClusterOnFailure && !report.SuiteSucceeded
	if k8sContext != "" {
		// Caller manages cluster lifecycle (e.g. running against an
		// existing OpenShift cluster); only kind-created clusters are
		// owned by this suite.
		return
	}
	if shouldKeep {
		ginkgo.By("Keeping kind cluster " + kindClusterName + " due to suite failure (E2E_KEEP_CLUSTER_ON_FAILURE=true)")
		return
	}
	ginkgo.By("Deleting kind cluster " + kindClusterName)
	command := exec.Command("kind", "delete", "cluster", "--name", kindClusterName)
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	if err != nil {
		ginkgo.GinkgoLogr.Error(err, "Failed to delete kind cluster")
		return
	}
	gomega.Eventually(session).WithTimeout(60 * time.Second).Should(gexec.Exit())
})

// setupK8sCluster creates the kind cluster and loads every image the
// e-p-d-pools env needs. Mirrors test/e2e/e2e_suite_test.go's
// setupK8sCluster but uses our own cluster name and image set.
func setupK8sCluster() {
	command := exec.Command("kind", "create", "cluster", "--name", kindClusterName, "--config", "-")
	stdin, err := command.StdinPipe()
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	go func() {
		defer func() {
			err := stdin.Close()
			gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		}()
		clusterConfig := strings.ReplaceAll(kindClusterConfig, "${PORT}", port)
		clusterConfig = strings.ReplaceAll(clusterConfig, "${GATEWAY_PORT}", gatewayPort)
		clusterConfig = strings.ReplaceAll(clusterConfig, "${METRICS_PORT}", metricsPort)
		_, err := io.WriteString(stdin, clusterConfig)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}()
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))

	// Pre-load images so kind-dev-env.sh's pull/load loop is a no-op.
	for _, img := range []string{
		vllmSimImage,
		eppImage,
		sideCarImage,
		vllmRenderImage,
		coordinatorImage,
		downloaderHTTPImage,
		downloaderInitImage,
	} {
		kindLoadImage(img)
	}
}

// kindLoadImage copies a host image into the kind cluster. Uses
// `docker save | docker exec ctr import` to sidestep kind's
// --all-platforms behaviour, which fails when only the target arch's
// layers are cached locally. Mirrors test/e2e/e2e_suite_test.go.
func kindLoadImage(image string) {
	ginkgo.By(fmt.Sprintf("Loading %s into the cluster %s using %s", image, kindClusterName, containerRuntime))
	if containerRuntime == "docker" {
		nodeName := kindClusterName + "-control-plane"
		save := exec.Command("docker", "save", image)
		importCmd := exec.Command("docker", "exec", "--privileged", "-i", nodeName,
			"ctr", "--namespace=k8s.io", "images", "import", "--digests", "--snapshotter=overlayfs", "-")
		pipe, err := save.StdoutPipe()
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		importCmd.Stdin = pipe
		importCmd.Stdout = ginkgo.GinkgoWriter
		importCmd.Stderr = ginkgo.GinkgoWriter
		gomega.Expect(save.Start()).ShouldNot(gomega.HaveOccurred())
		gomega.Expect(importCmd.Start()).ShouldNot(gomega.HaveOccurred())
		gomega.Expect(save.Wait()).ShouldNot(gomega.HaveOccurred())
		gomega.Expect(importCmd.Wait()).ShouldNot(gomega.HaveOccurred())
		return
	}
	command := exec.Command("kind", "--name", kindClusterName, "load", "docker-image", image)
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
}

func setupK8sClient() {
	k8sCfg, err := config.GetConfigWithContext(k8sContext)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.ExpectWithOffset(1, k8sCfg).NotTo(gomega.BeNil())

	gomega.Expect(clientgoscheme.AddToScheme(testConfig.Scheme)).To(gomega.Succeed())
	gomega.Expect(infextv1.Install(testConfig.Scheme)).To(gomega.Succeed())
	gomega.Expect(apiextv1.AddToScheme(testConfig.Scheme)).To(gomega.Succeed())
	gomega.Expect(infextv1a2.Install(testConfig.Scheme)).To(gomega.Succeed())

	testConfig.CreateCli()
	k8slog.SetLogger(ginkgo.GinkgoLogr)
}

// bringUpEnv exec's scripts/kind-dev-env.sh with DISAGG_POOLS_TOPOLOGY=true
// and the suite's cluster name. The script reuses the existing cluster
// (created by setupK8sCluster) and applies CRDs, Istio, the e-p-d-pools
// env, the per-phase EPPs, the HTTPRoutes, and the cleanup of base-kind-istio's
// catch-all single-pool stack.
func bringUpEnv() {
	cmd := exec.Command("./scripts/kind-dev-env.sh")
	// Resolve to the repo root from test/e2e/epd_pools (3 levels up).
	cmd.Dir = "../../.."
	cmd.Env = append(os.Environ(),
		"CLUSTER_NAME="+kindClusterName,
		fmt.Sprintf("GATEWAY_HOST_PORT=%s", gatewayPort),
		fmt.Sprintf("COORDINATOR_HOST_PORT=%s", port),
		"DISAGG_POOLS_TOPOLOGY=true",
		"MODEL_NAME="+modelName,
		"VLLM_IMAGE="+vllmSimImage,
		"EPP_IMAGE="+eppImage,
		"SIDECAR_IMAGE="+sideCarImage,
		"VLLM_RENDER_IMAGE="+vllmRenderImage,
		"COORDINATOR_IMAGE="+coordinatorImage,
		"MOCK_DOWNLOADER_HTTP_IMAGE="+downloaderHTTPImage,
		"MOCK_DOWNLOADER_INIT_IMAGE="+downloaderInitImage,
		"CONTAINER_RUNTIME="+containerRuntime,
	)
	cmd.Stdout = ginkgo.GinkgoWriter
	cmd.Stderr = ginkgo.GinkgoWriter
	session, err := gexec.Start(cmd, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred(), "kind-dev-env.sh failed to start")
	gomega.Eventually(session).WithTimeout(envSetupTimeout).Should(gexec.Exit(0), "kind-dev-env.sh did not finish successfully")
}

// kindClusterConfig is piped into `kind create cluster --config -`. Maps
// containerPort 30080 (gateway) and 30081 (coordinator NodePort, as
// configured by scripts/kind-dev-env.sh's extraPortMappings) to host ports.
const kindClusterConfig = `
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- image: kindest/node:v1.31.12
  extraPortMappings:
  - containerPort: 30080
    hostPort: ${GATEWAY_PORT}
    protocol: TCP
  - containerPort: 30081
    hostPort: ${PORT}
    protocol: TCP
  - containerPort: 32090
    hostPort: ${METRICS_PORT}
    protocol: TCP
`
