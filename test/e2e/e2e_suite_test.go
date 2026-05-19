package e2e

import (
	"fmt"
	"io"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	corev1 "k8s.io/api/core/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
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
	// kindClusterName is the name of the Kind cluster created for e2e tests.
	kindClusterName = "e2e-tests"
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
	// xInferPoolManifest is the manifest for the inference pool CRD with 'inference.networking.x-k8s.io' group.
	gieCrdsKustomize = "../../deploy/components/crds-gie"
	// inferExtManifest is the manifest for the inference extension test resources.
	inferExtManifest = "../../deploy/components/inference-gateway/inference-pools.yaml"
	// simModelName is the test model name.
	simModelName = "food-review"
	// kvModelName is the model name used in KV tests.
	kvModelName = "Qwen/Qwen2.5-1.5B-Instruct"
	// envoyManifest is the manifest for the envoy proxy test resources.
	envoyManifest = "../../deploy/environments/dev/e2e-infra/envoy.yaml"
	// eppManifest is the manifest for the deployment of the EPP
	eppManifest = "../../deploy/components/inference-gateway/deployment.yaml"
	// rbacManifest is the manifest for the EPP's RBAC resources.
	rbacManifest = "../../deploy/components/inference-gateway/rbac.yaml"
	// serviceAccountManifest is the manifest for the EPP's service account resources.
	serviceAccountManifest = "../../deploy/components/inference-gateway/service-accounts.yaml"
	// servicesManifest is the manifest for the EPP's service resources.
	servicesManifest = "../../deploy/environments/dev/e2e-infra/services.yaml"
)

var (
	port        string = env.GetEnvString("E2E_PORT", "30080", ginkgo.GinkgoLogr)
	metricsPort string = env.GetEnvString("E2E_METRICS_PORT", "32090", ginkgo.GinkgoLogr)

	testConfig *testutils.TestConfig

	// keepClusterOnFailure skips kind cluster deletion when the suite fails.
	// Set E2E_KEEP_CLUSTER_ON_FAILURE=true to enable.
	keepClusterOnFailure = env.GetEnvBool("E2E_KEEP_CLUSTER_ON_FAILURE", false, ginkgo.GinkgoLogr)

	containerRuntime  = env.GetEnvString("CONTAINER_RUNTIME", "docker", ginkgo.GinkgoLogr)
	eppImage          = env.GetEnvString("EPP_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:dev", ginkgo.GinkgoLogr)
	vllmSimImage      = env.GetEnvString("VLLM_IMAGE", "ghcr.io/llm-d/llm-d-inference-sim:v0.8.2", ginkgo.GinkgoLogr)
	sideCarImage      = env.GetEnvString("SIDECAR_IMAGE", "ghcr.io/llm-d/llm-d-router-disagg-sidecar:dev", ginkgo.GinkgoLogr)
	udsTokenizerImage = env.GetEnvString("UDS_TOKENIZER_IMAGE", "ghcr.io/llm-d/llm-d-uds-tokenizer:dev", ginkgo.GinkgoLogr)
	vllmRenderImage   = env.GetEnvString("VLLM_RENDER_IMAGE", "vllm/vllm-openai-cpu:v0.19.1", ginkgo.GinkgoLogr)
	// nsName is the namespace in which the K8S objects will be created
	nsName = env.GetEnvString("NAMESPACE", "default", ginkgo.GinkgoLogr)

	// k8sContext is the Kubernetes context to work with
	k8sContext = env.GetEnvString("K8S_CONTEXT", "", ginkgo.GinkgoLogr)

	readyTimeout = env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr)
	interval     = defaultInterval

	crdObjects            []string
	envoyObjects          []string
	rbacObjects           []string
	serviceAccountObjects []string
	serviceObjects        []string
	infPoolObjects        []string
	createdNameSpace      bool

	portForwardSession    *gexec.Session
	eppPortForwardSession *gexec.Session
)

func TestEndToEnd(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	if k8sContext == "" {
		setupK8sCluster()
	}
	testConfig = testutils.NewTestConfig(nsName, k8sContext)
	setupK8sClient()
	setupNameSpace()
	createCRDs()
	createEnvoy()
	infraSubs := map[string]string{
		"${EPP_NAME}": "e2e-epp",
	}
	rbacYamls := substituteMany(testutils.ReadYaml(rbacManifest), infraSubs)
	rbacObjects = testutils.CreateObjsFromYaml(testConfig, rbacYamls)
	saYamls := substituteMany(testutils.ReadYaml(serviceAccountManifest), infraSubs)
	serviceAccountObjects = testutils.CreateObjsFromYaml(testConfig, saYamls)
	serviceObjects = testutils.ApplyYAMLFile(testConfig, servicesManifest)

	// Prevent failure in tests due to InferencePool not existing before the test
	infPoolObjects = createInferencePool(1, false)
})

var _ = ginkgo.AfterSuite(func() {
	// Stop port-forwards when using an existing cluster context; they must be
	// terminated before the process exits regardless of pass/fail status.
	if k8sContext != "" {
		if portForwardSession != nil {
			portForwardSession.Terminate()
		}
		if eppPortForwardSession != nil {
			eppPortForwardSession.Terminate()
		}
	}
})

// ReportAfterSuite receives the full suite report and uses report.SuiteSucceeded
// to detect any failure, including failures in BeforeSuite/AfterSuite.
// This is preferred over a suiteFailed flag tracked via ReportAfterEach because
// ReportAfterEach only fires for individual specs and would miss setup/teardown failures.
var _ = ginkgo.ReportAfterSuite("cleanup", func(report ginkgo.Report) {
	if !report.SuiteSucceeded {
		dumpPodsAndLogs()
	}

	shouldKeep := keepClusterOnFailure && !report.SuiteSucceeded
	if k8sContext == "" {
		if shouldKeep {
			ginkgo.By("Keeping kind cluster " + kindClusterName + " due to suite failure (E2E_KEEP_CLUSTER_ON_FAILURE=true)")
		} else {
			// delete kind cluster we created
			ginkgo.By("Deleting kind cluster " + kindClusterName)
			command := exec.Command("kind", "delete", "cluster", "--name", kindClusterName)
			session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
			if err != nil {
				ginkgo.GinkgoLogr.Error(err, "Failed to delete kind cluster")
			} else {
				gomega.Eventually(session).WithTimeout(60 * time.Second).Should(gexec.Exit())
			}
		}
	} else {
		// Used an existing Kubernetes context, clean up created resources
		if shouldKeep {
			ginkgo.By("Keeping created Kubernetes objects due to suite failure (E2E_KEEP_CLUSTER_ON_FAILURE=true)")
		} else {
			ginkgo.By("Deleting created Kubernetes objects")
			testutils.DeleteObjects(testConfig, infPoolObjects)
			testutils.DeleteObjects(testConfig, serviceObjects)
			testutils.DeleteObjects(testConfig, serviceAccountObjects)
			testutils.DeleteObjects(testConfig, rbacObjects)
			testutils.DeleteObjects(testConfig, envoyObjects)
			testutils.DeleteObjects(testConfig, crdObjects)

			if createdNameSpace {
				ginkgo.By("Deleting namespace " + nsName)
				err := testConfig.KubeCli.CoreV1().Namespaces().Delete(testConfig.Context, nsName, metav1.DeleteOptions{})
				gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
			}
		}
	}
})

// Create the Kubernetes cluster for the E2E tests and load the local images
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
		clusterConfig = strings.ReplaceAll(clusterConfig, "${METRICS_PORT}", metricsPort)
		_, err := io.WriteString(stdin, clusterConfig)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}()
	session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))

	kindLoadImage(vllmSimImage)
	kindLoadImage(eppImage)
	kindLoadImage(sideCarImage)
	kindLoadImage(udsTokenizerImage)
	kindLoadImage(vllmRenderImage)
}

func kindLoadImage(image string) {
	ginkgo.By(fmt.Sprintf("Loading %s into the cluster %s using %s", image, kindClusterName, containerRuntime))

	if containerRuntime == "docker" {
		// Use docker save | ctr import to avoid KIND's --all-platforms flag which
		// fails when only the target architecture layers are locally cached.
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
	} else {
		command := exec.Command("kind", "--name", kindClusterName, "load", "docker-image", image)
		session, err := gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
	}
}

func setupK8sClient() {
	k8sCfg, err := config.GetConfigWithContext(k8sContext)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.ExpectWithOffset(1, k8sCfg).NotTo(gomega.BeNil())

	err = clientgoscheme.AddToScheme(testConfig.Scheme)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = infextv1.Install(testConfig.Scheme)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = apiextv1.AddToScheme(testConfig.Scheme)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	err = infextv1a2.Install(testConfig.Scheme)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	testConfig.CreateCli()

	k8slog.SetLogger(ginkgo.GinkgoLogr)
}

// setupNameSpace sets up the specified namespace if it doesn't exist
func setupNameSpace() {
	if nsName == "default" {
		return
	}
	_, err := testConfig.KubeCli.CoreV1().Namespaces().Get(testConfig.Context, nsName, metav1.GetOptions{})
	if err == nil {
		return
	}
	gomega.Expect(errors.IsNotFound(err)).To(gomega.BeTrue())

	ginkgo.By("Creating namespace " + nsName)
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: nsName,
		},
	}
	_, err = testConfig.KubeCli.CoreV1().Namespaces().Create(testConfig.Context, namespace, metav1.CreateOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	createdNameSpace = true
}

// createCRDs creates the Inference Extension CRDs used for testing.
func createCRDs() {
	crds := runKustomize(gieCrdsKustomize)
	crdObjects = testutils.CreateObjsFromYaml(testConfig, crds)
}

func createEnvoy() {
	manifests := testutils.ReadYaml(envoyManifest)
	manifests = substituteMany(manifests, map[string]string{"${NAMESPACE}": nsName})
	ginkgo.By("Creating envoy proxy resources from manifest: " + envoyManifest)
	envoyObjects = testutils.CreateObjsFromYaml(testConfig, manifests)

	if k8sContext != "" {
		envoyName := ""
		for _, obj := range envoyObjects {
			splitObj := strings.Split(obj, "/")
			if strings.ToLower(splitObj[0]) == "deployment" {
				envoyName = splitObj[1]
			}
		}
		gomega.Expect(envoyName).ToNot(gomega.BeEmpty())

		command := exec.Command("kubectl", "port-forward", "deployment/"+envoyName, port+":8081",
			"--context="+k8sContext, "--namespace="+nsName)
		var err error
		portForwardSession, err = gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}
}

func createInferencePool(numTargetPorts int, toDelete bool) []string {
	poolName := simModelName + "-inference-pool"

	if toDelete {
		objName := []string{"inferencepool/" + poolName}
		testutils.DeleteObjects(testConfig, objName)
	}

	infPoolYaml := testutils.ReadYaml(inferExtManifest)
	// targetPorts is substituted into `targetPorts: ${TARGET_PORTS}` in inference-pools.yaml.
	// Each item must use 2-space indentation to match that field's level in the YAML.
	// If the field is ever reindented in inference-pools.yaml, update the format string here too.
	var targetPortsBuilder strings.Builder
	for idx := range numTargetPorts {
		fmt.Fprintf(&targetPortsBuilder, "\n  - number: %d", 8000+idx)
	}
	targetPorts := targetPortsBuilder.String()
	infPoolYaml = substituteMany(infPoolYaml,
		map[string]string{
			"${POOL_NAME}":    poolName,
			"${EPP_NAME}":     "e2e-epp",
			"${TARGET_PORTS}": targetPorts,
		})

	return testutils.CreateObjsFromYaml(testConfig, infPoolYaml)
}

func startEPPMetricsPortForward() {
	pods, err := testConfig.KubeCli.CoreV1().Pods(nsName).List(testConfig.Context, metav1.ListOptions{
		LabelSelector: "app=e2e-epp",
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(pods.Items).NotTo(gomega.BeEmpty())

	eppPodName := pods.Items[0].Name
	command := exec.Command("kubectl", "port-forward", "pod/"+eppPodName, metricsPort+":9090",
		"--context="+k8sContext, "--namespace="+nsName)
	eppPortForwardSession, err = gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	// Give it a moment to establish
	time.Sleep(3 * time.Second)
}

const kindClusterConfig = `
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- image: kindest/node:v1.31.12
  extraPortMappings:
  - containerPort: 30080
    hostPort: ${PORT}
    protocol: TCP
  - containerPort: 30081
    hostPort: 30081
    protocol: TCP
  - containerPort: 32090
    hostPort: ${METRICS_PORT}
    protocol: TCP
`
