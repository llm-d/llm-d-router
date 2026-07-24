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

// Package coordinate2e runs end-to-end tests for the coordinator service
// against the e-p-d-pools topology: one InferencePool per phase (encode,
// prefill, decode), each with its own EPP, a hand-rolled standalone Envoy
// routing on EPP-Phase header, and the coordinator deployed as a pod.
// No Istio, no Gateway/HTTPRoute CRDs.
package coordinate2e

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	k8slog "sigs.k8s.io/controller-runtime/pkg/log"
	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	infextv1a2 "github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/epp/util/env"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	kindClusterName = "e2e-coordinator-tests"

	defaultReadyTimeout    = 10 * time.Minute
	defaultInterval        = time.Second * 2
	defaultGatewayHostPort = 30080

	poolNameBase = "qwen3-vl-2b-instruct-inference-pool"
	eppName      = "e2e-epp"

	encodeEPPManifest   = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/encode/epp.yaml"
	encodePoolManifest  = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/encode/inference-pool.yaml"
	prefillEPPManifest  = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/prefill/epp.yaml"
	prefillPoolManifest = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/prefill/inference-pool.yaml"
	decodeEPPManifest   = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/decode/epp.yaml"
	decodePoolManifest  = "../../../../deploy/coordinator/components/inference-gateway/epd-pools/decode/inference-pool.yaml"

	epdPoolsKustomizeDir    = "../../../../deploy/coordinator/environments/dev/epd-pools"
	coordinatorComponentDir = "../../../../deploy/coordinator/components/coordinator"
	rendererComponentDir    = "../../../../deploy/coordinator/components/vllm-render"

	envoyManifest = "testdata/envoy.yaml"

	crdGIEPath = "../../../../deploy/components/crds-gie"

	baseRbacManifest = "../../../../deploy/coordinator/components/inference-gateway/base/rbac.yaml"
)

var (
	baseGatewayPort = env.GetEnvInt("E2E_GATEWAY_PORT", defaultGatewayHostPort, ginkgo.GinkgoLogr)

	testConfig *testutils.TestConfig

	keepClusterOnFailure = env.GetEnvBool("E2E_KEEP_CLUSTER_ON_FAILURE", false, ginkgo.GinkgoLogr)
	printLogs            = env.GetEnvBool("E2E_PRINT_LOGS", false, ginkgo.GinkgoLogr)

	containerRuntime = env.GetEnvString("CONTAINER_RUNTIME", "docker", ginkgo.GinkgoLogr)
	eppImage         = env.GetEnvString("EPP_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:dev", ginkgo.GinkgoLogr)
	vllmSimImage     = env.GetEnvString("VLLM_IMAGE", "ghcr.io/llm-d/llm-d-inference-sim:v0.10.2", ginkgo.GinkgoLogr)
	vllmRenderImage  = env.GetEnvString("VLLM_RENDER_IMAGE", vllmSimImage, ginkgo.GinkgoLogr)
	vllmRenderPort   = env.GetEnvString("VLLM_RENDER_PORT", "8082", ginkgo.GinkgoLogr)
	coordinatorImage = env.GetEnvString("COORDINATOR_IMAGE", "", ginkgo.GinkgoLogr)
	modelName        = env.GetEnvString("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct", ginkgo.GinkgoLogr)

	numProcesses = env.GetEnvInt("E2E_NUM_PROCS", 1, ginkgo.GinkgoLogr)

	// baseNsName is the base of the namespace in which the K8S objects will be created.
	baseNsName = env.GetEnvString("NAMESPACE", testutils.DefaultNsName(numProcesses, "e2e-coordinator"), ginkgo.GinkgoLogr)
	k8sContext = env.GetEnvString("K8S_CONTEXT", "", ginkgo.GinkgoLogr)

	readyTimeout = env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr)

	portForwardSessions []*gexec.Session
	rendererObjects     []string
	stableInfraObjects  []string
	createdNameSpace    bool
)

func TestCoordinatorE2E(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Coordinator E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	gomega.Expect(coordinatorImage).NotTo(gomega.BeEmpty(), "COORDINATOR_IMAGE must be set")

	testutils.RequireParallelProcessesMatch(numProcesses)

	if k8sContext == "" {
		setupK8sCluster()
	}
	testConfig = testutils.NewTestConfig(k8sContext)
	setupK8sClient()
	setupNameSpace()

	// Base infra (CRDs, RBAC, Envoy) is created here on suite-owned kind clusters.
	// With K8S_CONTEXT set, base infra is assumed pre-deployed; the per-test
	// workload (EPPs, pools, vLLM workers, coordinator) is created in the test body.
	if k8sContext == "" {
		setupInfra()
	} else {
		// Base infra (including Envoy) is pre-deployed; forward the gateway so
		// the test can post to it. The kind nodePort mapping is unavailable here.
		startPortForward("service/envoy", strconv.Itoa(getGatewayPort()), "8081")
	}

	rendererObjects = createRenderer()

	// Coordinator and EPP Services/RBAC are created once and kept stable across
	// specs (see createStableInfra).
	createStableInfra()
})

var _ = ginkgo.ReportAfterSuite("cleanup", func(report ginkgo.Report) {
	if !report.SuiteSucceeded {
		for idx := range numProcesses {
			testutils.DumpPodsAndLogs(testConfig, testutils.NamespaceForProcess(baseNsName, numProcesses, idx+1))
		}
	}

	if k8sContext == "" && keepClusterOnFailure && !report.SuiteSucceeded {
		ginkgo.By("Keeping kind cluster " + kindClusterName + " due to suite failure (E2E_KEEP_CLUSTER_ON_FAILURE=true)")
		return
	}
	nsName := getNamespace()
	if len(rendererObjects) > 0 {
		testutils.DeleteObjects(testConfig, rendererObjects, nsName)
	}
	if len(stableInfraObjects) > 0 {
		testutils.DeleteObjects(testConfig, stableInfraObjects, nsName)
	}
	for _, session := range portForwardSessions {
		session.Terminate()
	}
	if createdNameSpace {
		err := testConfig.KubeCli.CoreV1().Namespaces().Delete(testConfig.Context, nsName, metav1.DeleteOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}
	if k8sContext != "" {
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

// startPortForward forwards a local port to the given target (e.g.
// "deployment/llm-d-coordinator" or "service/envoy"). Used when running against
// an existing cluster (K8S_CONTEXT set), where the kind nodePort mapping is not
// available. Sessions are tracked for teardown in AfterSuite.
func startPortForward(target, localPort, remotePort string) {
	command := exec.Command("kubectl", "port-forward", target,
		localPort+":"+remotePort,
		"--context="+k8sContext, "--namespace="+getNamespace())
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	portForwardSessions = append(portForwardSessions, session)
}

func setupK8sCluster() {
	// extraPortMappings is substituted into `extraPortMappings: ${EXTRA_PORT_MAPPINGS}` in the Kind
	// cluster configuration below; keep its indentation in sync with testutils.BuildExtraPortMappings.
	extraPortMappings := testutils.BuildExtraPortMappings(numProcesses,
		[2]int{defaultGatewayHostPort, baseGatewayPort},
	)

	command := exec.Command("kind", "create", "cluster", "--name", kindClusterName, "--config", "-")
	stdin, err := command.StdinPipe()
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	go func() {
		defer func() {
			err := stdin.Close()
			gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		}()
		clusterConfig := strings.ReplaceAll(kindClusterConfig, "${EXTRA_PORT_MAPPINGS}", extraPortMappings)
		_, err := io.WriteString(stdin, clusterConfig)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}()
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))

	images := []string{vllmSimImage, eppImage, coordinatorImage}
	if vllmRenderImage != vllmSimImage {
		images = append(images, vllmRenderImage)
	}
	for _, img := range images {
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

// getGatewayPort returns the envoy gateway's NodePort for this process. See testutils.ProcessPort.
func getGatewayPort() int {
	return testutils.ProcessPort(baseGatewayPort)
}

// getNamespace returns the namespace being used by the current process. Each
// parallel process is assigned its own namespace to provide isolation between
// the tests running in it. See testutils.Namespace.
func getNamespace() string {
	return testutils.Namespace(baseNsName, numProcesses)
}

func gatewayBaseURL() string {
	return testutils.LocalhostURL(getGatewayPort())
}

const kindClusterConfig = `
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- image: kindest/node:v1.31.12
  extraPortMappings:${EXTRA_PORT_MAPPINGS}
`
