package e2e

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
	corev1 "k8s.io/api/core/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
	// eppName is the value of the app label on the EPP pods
	eppName = "e2e-epp"
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
	// crdKustomizePath is the kustomize path for all CRDs (upstream GIE + local llm-d.ai).
	crdKustomizePath = "../../config/crd"
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
	// renderManifest is the manifest for the standalone vLLM render deployment and service.
	renderManifest = "../../deploy/environments/dev/e2e-infra/vllm-render.yaml"

	// defaultPort is the envoy gateway's NodePort.
	defaultPort        = 30080
	defaultMetricsPort = 32090

	// CI shards scheduler e2e specs with label filters.
	extendedTestLabel      = "Extended"
	disruptiveTestLabel    = "Disruptive"
	sharedStorageTestLabel = "SharedStorage"
	metricsTestLabel       = "Metrics"
	deprecatedPDTestLabel  = "DeprecatedPD"
	disaggTestLabel        = "Disagg"
)

var (
	basePort        = env.GetEnvInt("E2E_PORT", defaultPort, ginkgo.GinkgoLogr)
	baseMetricsPort = env.GetEnvInt("E2E_METRICS_PORT", defaultMetricsPort, ginkgo.GinkgoLogr)

	testConfig *testutils.TestConfig

	// keepClusterOnFailure skips kind cluster deletion when the suite fails.
	// Set E2E_KEEP_CLUSTER_ON_FAILURE=true to enable.
	keepClusterOnFailure = env.GetEnvBool("E2E_KEEP_CLUSTER_ON_FAILURE", false, ginkgo.GinkgoLogr)

	containerRuntime = env.GetEnvString("CONTAINER_RUNTIME", "docker", ginkgo.GinkgoLogr)
	eppImage         = env.GetEnvString("EPP_IMAGE", "ghcr.io/llm-d/llm-d-router-endpoint-picker:dev", ginkgo.GinkgoLogr)
	vllmSimImage     = env.GetEnvString("VLLM_IMAGE", "ghcr.io/llm-d/llm-d-inference-sim:v0.10.2", ginkgo.GinkgoLogr)
	sideCarImage     = env.GetEnvString("SIDECAR_IMAGE", "ghcr.io/llm-d/llm-d-router-disagg-sidecar:dev", ginkgo.GinkgoLogr)
	vllmRenderImage  = env.GetEnvString("VLLM_RENDER_IMAGE", "vllm/vllm-openai-cpu:v0.21.0", ginkgo.GinkgoLogr)
	vllmRenderPort   = env.GetEnvString("VLLM_RENDER_PORT", "8082", ginkgo.GinkgoLogr)
	loadRenderImage  = env.GetEnvBool("LOAD_VLLM_RENDER_IMAGE", true, ginkgo.GinkgoLogr)

	// numProcesses is the number of parallel processes that will be used by the tests.
	// Theorectically this value could be gotten from the Ginkgo Suite Configuration.
	// However, that needs to be done while the tests are running and has shown to not
	// always allow the baseNsName field to be set correctly.
	numProcesses = env.GetEnvInt("E2E_NUM_PROCS", 1, ginkgo.GinkgoLogr)
	// baseNsName is the base of the namespace in which the K8S objects will be created
	baseNsName = env.GetEnvString("NAMESPACE", testutils.DefaultNsName(numProcesses, "e2e"), ginkgo.GinkgoLogr)

	// k8sContext is the Kubernetes context to work with
	k8sContext = env.GetEnvString("K8S_CONTEXT", "", ginkgo.GinkgoLogr)

	readyTimeout = env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr)
	interval     = defaultInterval

	crdObjects    []string
	renderObjects []string

	eppPortForwardSession *gexec.Session
)

func TestEndToEnd(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	testutils.RequireParallelProcessesMatch(numProcesses)

	if k8sContext == "" {
		setupK8sCluster()
	}
	testConfig = testutils.NewTestConfig(k8sContext)
	setupK8sClient()
	createCRDs()
	renderObjects = createRender(getNamespace())
})

// ReportAfterSuite receives the full suite report and uses report.SuiteSucceeded
// to detect any failure, including failures in BeforeSuite/AfterSuite.
// This is preferred over a suiteFailed flag tracked via ReportAfterEach because
// ReportAfterEach only fires for individual specs and would miss setup/teardown failures.
var _ = ginkgo.ReportAfterSuite("cleanup", func(report ginkgo.Report) {
	if !report.SuiteSucceeded {
		for idx := range numProcesses {
			testutils.DumpPodsAndLogs(testConfig, testutils.NamespaceForProcess(baseNsName, numProcesses, idx+1))
		}
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
			testutils.DeleteObjects(testConfig, renderObjects, getNamespace())
			testutils.DeleteObjects(testConfig, crdObjects, "")
		}
	}
})

// Create the Kubernetes cluster for the E2E tests and load the local images
func setupK8sCluster() {
	// extraPortMappings is substituted into `extraPortMappings: ${EXTRA_PORT_MAPPINGS}` in the Kind
	// cluster configuration below; keep its indentation in sync with testutils.BuildExtraPortMappings.
	extraPortMappings := testutils.BuildExtraPortMappings(numProcesses,
		[2]int{defaultPort, basePort},
		[2]int{defaultMetricsPort, baseMetricsPort},
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

	kindLoadImage(vllmSimImage)
	kindLoadImage(eppImage)
	kindLoadImage(sideCarImage)
	if loadRenderImage && vllmRenderImage != vllmSimImage {
		kindLoadImage(vllmRenderImage)
	}
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
	_, err := testConfig.KubeCli.CoreV1().Namespaces().Get(testConfig.Context, getNamespace(), metav1.GetOptions{})
	if err == nil {
		return
	}
	gomega.Expect(errors.IsNotFound(err)).To(gomega.BeTrue())

	ginkgo.By("Creating namespace " + getNamespace())
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: getNamespace(),
		},
	}
	_, err = testConfig.KubeCli.CoreV1().Namespaces().Create(testConfig.Context, namespace, metav1.CreateOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	createdNameSpace = true

	ginkgo.By("Ensuring namespace exists: " + getNamespace())
	testutils.EventuallyExists(testConfig, func() error {
		return testConfig.K8sClient.Get(testConfig.Context,
			types.NamespacedName{Name: getNamespace()}, &corev1.Namespace{})
	})
}

// createCRDs creates the Inference Extension CRDs used for testing.
func createCRDs() {
	crds := runKustomize(crdKustomizePath)
	crdObjects = testutils.CreateObjsFromYaml(testConfig, crds, "")
}

func createEnvoy(nsName string) ([]string, *gexec.Session) {
	infraSubs := map[string]string{
		"${NAMESPACE}":       nsName,
		"${ENVOY_NODE_PORT}": strconv.Itoa(30080 + 100*(ginkgo.GinkgoParallelProcess()-1)),
	}
	manifests := testutils.ReadYaml(envoyManifest)
	manifests = substituteMany(manifests, infraSubs)
	ginkgo.By("Creating envoy proxy resources from manifest: " + envoyManifest)
	envoyObjects := testutils.CreateObjsFromYaml(testConfig, manifests, nsName)
	var portForwardSession *gexec.Session

	if k8sContext != "" {
		envoyName := ""
		for _, obj := range envoyObjects {
			splitObj := strings.Split(obj, "/")
			if strings.ToLower(splitObj[0]) == "deployment" {
				envoyName = splitObj[1]
			}
		}
		gomega.Expect(envoyName).ToNot(gomega.BeEmpty())

		command := exec.Command("kubectl", "port-forward", "deployment/"+envoyName, strconv.Itoa(getPort())+":8081",
			"--context="+k8sContext, "--namespace="+getNamespace())
		var err error
		portForwardSession, err = gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}
	return envoyObjects, portForwardSession
}

func createInferencePool(numTargetPorts int) []string {
	poolName := simModelName + "-inference-pool"
	nsName := getNamespace()

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

	return testutils.CreateObjsFromYaml(testConfig, infPoolYaml, nsName)
}

// startEPPMetricsPortForward is a no-op outside an existing-cluster run (k8sContext
// unset) and safe to call repeatedly; it starts at most one port-forward session,
// reused until AfterSuite terminates it.
func startEPPMetricsPortForward() {
	if k8sContext == "" || eppPortForwardSession != nil {
		return
	}

	pods, err := testConfig.KubeCli.CoreV1().Pods(getNamespace()).List(testConfig.Context, metav1.ListOptions{
		LabelSelector: "app=e2e-epp",
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(pods.Items).NotTo(gomega.BeEmpty())

	eppPodName := pods.Items[0].Name
	command := exec.Command("kubectl", "port-forward", "pod/"+eppPodName, strconv.Itoa(getMetricsPort())+":9090",
		"--context="+k8sContext, "--namespace="+getNamespace())
	eppPortForwardSession, err = gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	// Give it a moment to establish
	time.Sleep(3 * time.Second)
}

// getPort returns the envoy service's NodePort for this process. See testutils.ProcessPort.
func getPort() int {
	return testutils.ProcessPort(basePort)
}

// getMetricsPort returns the EPP's metrics NodePort for this process. See testutils.ProcessPort.
func getMetricsPort() int {
	return testutils.ProcessPort(baseMetricsPort)
}

// getNamespace returns the namespace being used by the current process. Each
// parallel process is assigned its own namespace to provide isolation between
// the tests running in it. See testutils.Namespace.
func getNamespace() string {
	return testutils.Namespace(baseNsName, numProcesses)
}

const kindClusterConfig = `
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- image: kindest/node:v1.31.12
  extraPortMappings:${EXTRA_PORT_MAPPINGS}
`
