/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

// Package epd_pools runs end-to-end tests against the e-p-d-pools
// topology: one InferencePool per phase (encode, prefill, decode), each with
// its own EPP. A standalone Envoy in front of all three EPPs routes traffic
// based on the EPP-Phase header — no Istio, no Gateway/HTTPRoute CRDs. Tests
// stand in for the coordinator, POSTing per-stage payloads (per
// coordinator/communication.md) directly to the gateway.
//
// Mirrors the lifecycle of test/e2e/single_pool/e2e_suite_test.go: BeforeSuite
// creates a kind cluster, applies CRDs + per-phase EPPs/Pools/vLLM workers +
// the hand-rolled Envoy from testdata/, and ReportAfterSuite tears the cluster
// down (gated by E2E_KEEP_CLUSTER and E2E_KEEP_CLUSTER_ON_FAILURE).
package epdpools

import (
	"fmt"
	"io"
	"os/exec"
	"strconv"
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
	kindClusterName = "e2e-pools-tests"

	defaultReadyTimeout    = 10 * time.Minute
	defaultInterval        = time.Second * 2
	defaultGatewayHostPort = 30080

	poolNameBase = "qwen3-vl-2b-instruct-inference-pool"
	eppName      = "e2e-epp"

	encodeEPPManifest   = "../../../deploy/components/inference-gateway/epd-pools/encode/epp.yaml"
	encodePoolManifest  = "../../../deploy/components/inference-gateway/epd-pools/encode/inference-pool.yaml"
	prefillEPPManifest  = "../../../deploy/components/inference-gateway/epd-pools/prefill/epp.yaml"
	prefillPoolManifest = "../../../deploy/components/inference-gateway/epd-pools/prefill/inference-pool.yaml"
	decodeEPPManifest   = "../../../deploy/components/inference-gateway/epd-pools/decode/epp.yaml"
	decodePoolManifest  = "../../../deploy/components/inference-gateway/epd-pools/decode/inference-pool.yaml"

	vllmEnvKustomizeDir = "../../../deploy/environments/dev/e-p-d"

	encodeEPPConfigPath  = "../../../deploy/config/sim-epp-encode-config.yaml"
	prefillEPPConfigPath = "../../../deploy/config/sim-epp-prefill-config.yaml"
	decodeEPPConfigPath  = "../../../deploy/config/sim-epp-decode-config.yaml"

	envoyManifest    = "testdata/envoy.yaml"
	crdKustomizePath = "../../../config/crd"
)

var (
	gatewayPort = env.GetEnvString("E2E_GATEWAY_PORT", strconv.Itoa(defaultGatewayHostPort), ginkgo.GinkgoLogr)
	metricsPort = env.GetEnvString("E2E_METRICS_PORT", "32090", ginkgo.GinkgoLogr)

	testConfig *testutils.TestConfig

	keepClusterOnFailure = env.GetEnvBool("E2E_KEEP_CLUSTER_ON_FAILURE", false, ginkgo.GinkgoLogr)
	keepCluster          = env.GetEnvBool("E2E_KEEP_CLUSTER", false, ginkgo.GinkgoLogr)

	containerRuntime = env.GetEnvString("CONTAINER_RUNTIME", "docker", ginkgo.GinkgoLogr)
	eppImage         = env.GetEnvString("EPP_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:dev", ginkgo.GinkgoLogr)
	vllmSimImage     = env.GetEnvString("VLLM_IMAGE", "ghcr.io/revit13/llm-d-inference-sim:dev", ginkgo.GinkgoLogr)
	vllmRenderImage  = env.GetEnvString("VLLM_RENDER_IMAGE", "vllm/vllm-openai-cpu:v0.21.0", ginkgo.GinkgoLogr)
	sideCarImage     = env.GetEnvString("SIDECAR_IMAGE", "ghcr.io/llm-d/llm-d-router-disagg-sidecar:dev", ginkgo.GinkgoLogr)
	modelName        = env.GetEnvString("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct", ginkgo.GinkgoLogr)

	nsName     = env.GetEnvString("NAMESPACE", "default", ginkgo.GinkgoLogr)
	k8sContext = env.GetEnvString("K8S_CONTEXT", "", ginkgo.GinkgoLogr)

	readyTimeout = env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr)

	gatewayBaseURL = "http://localhost:" + gatewayPort

	portForwardSession *gexec.Session
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

	if k8sContext != "" {
		ginkgo.By("Reusing existing cluster (K8S_CONTEXT set); skipping env apply")
		return
	}
	setupInfra()
})

var _ = ginkgo.AfterSuite(func() {
	if k8sContext != "" && portForwardSession != nil {
		portForwardSession.Terminate()
	}
})

var _ = ginkgo.ReportAfterSuite("cleanup", func(report ginkgo.Report) {
	if k8sContext != "" {
		return
	}
	if keepCluster {
		ginkgo.By("Keeping kind cluster " + kindClusterName + " (E2E_KEEP_CLUSTER=true)")
		return
	}
	if keepClusterOnFailure && !report.SuiteSucceeded {
		ginkgo.By("Keeping kind cluster " + kindClusterName + " due to suite failure (E2E_KEEP_CLUSTER_ON_FAILURE=true)")
		return
	}
	/*ginkgo.By("Deleting kind cluster " + kindClusterName)
	command := exec.Command("kind", "delete", "cluster", "--name", kindClusterName)
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	if err != nil {
		ginkgo.GinkgoLogr.Error(err, "Failed to delete kind cluster")
		return
	}
	gomega.Eventually(session).WithTimeout(60 * time.Second).Should(gexec.Exit())*/
})

func setupK8sCluster() {
	command := exec.Command("kind", "create", "cluster", "--name", kindClusterName, "--config", "-")
	stdin, err := command.StdinPipe()
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	go func() {
		defer func() {
			err := stdin.Close()
			gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		}()
		clusterConfig := strings.ReplaceAll(kindClusterConfig, "${GATEWAY_PORT}", gatewayPort)
		clusterConfig = strings.ReplaceAll(clusterConfig, "${METRICS_PORT}", metricsPort)
		_, err := io.WriteString(stdin, clusterConfig)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}()
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))

	for _, img := range []string{vllmSimImage, vllmRenderImage, eppImage, sideCarImage} {
		kindLoadImage(img)
	}
}

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

const kindClusterConfig = `
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- image: kindest/node:v1.31.12
  extraPortMappings:
  - containerPort: 30080
    hostPort: ${GATEWAY_PORT}
    protocol: TCP
  - containerPort: 32090
    hostPort: ${METRICS_PORT}
    protocol: TCP
`
