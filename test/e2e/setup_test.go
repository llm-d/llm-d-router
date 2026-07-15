package e2e

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/sidecar/proxy"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func createModelServersFromKustomize(kustomizeDir string, extra map[string]string) []string {
	nsName := getNamespace()
	subs := map[string]string{
		"${MODEL_NAME}":              simModelName,
		"${POOL_NAME}":               poolName,
		"${VLLM_IMAGE}":              vllmSimImage,
		"${SIDECAR_IMAGE}":           sideCarImage,
		"${VLLM_DATA_PARALLEL_SIZE}": "1",
		"${VLLM_SIM_MODE}":           "echo",
		"${KV_CACHE_ENABLED}":        "false",
		"${DECODE_ROLE}":             "",
		"${EPP_NAME}":                "e2e-epp",
		"${NAMESPACE}":               nsName,
		"${HF_TOKEN}":                os.Getenv("HF_TOKEN"),
		"${VLLM_EXTRA_ARGS_E}":       "--force-dummy-tokenizer",
		"${VLLM_EXTRA_ARGS_P}":       "--force-dummy-tokenizer",
		"${VLLM_EXTRA_ARGS_D}":       "--force-dummy-tokenizer",
		"${VLLM_RENDER_URL}":         fmt.Sprintf("http://vllm-render.%s.svc.cluster.local:%s", nsName, vllmRenderPort),
		"${VLLM_RENDER_PORT}":        vllmRenderPort,
	}
	for k, v := range extra {
		subs[k] = v
	}

	manifests := runKustomize(kustomizeDir)
	manifests = substituteMany(manifests, subs)
	// Remove labels with empty values (produced when ${DECODE_ROLE} is empty)
	manifests = removeEmptyLabels(manifests)
	manifests = removeEmptyArgs(manifests)
	objects := testutils.CreateObjsFromYaml(testConfig, manifests, nsName)
	podsInDeploymentsReady(nsName, objects)
	return objects
}

func createModelServersDecode(replicas int) []string {
	return createModelServersFromKustomize(epdDeploymentDir, map[string]string{
		"${KV_CACHE_ENABLED}":     "false",
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(replicas),
	})
}

func createModelServersDecodeKV(replicas int) []string {
	return createModelServersFromKustomize(epdDeploymentDir, map[string]string{
		"${MODEL_NAME}":           kvModelName,
		"${KV_CACHE_ENABLED}":     "true",
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(replicas),
	})
}

func createModelServersDecodeDP(replicas int) []string {
	return createModelServersFromKustomize("../../deploy/components/vllm-decode", map[string]string{
		"${VLLM_REPLICA_COUNT_D}":    strconv.Itoa(replicas),
		"${VLLM_DATA_PARALLEL_SIZE}": "2",
		"${DECODE_ROLE}":             "decode",
		"${VLLM_EXTRA_ARGS_D}":       "--mode=echo",
	})
}

func createModelServersPDWithConnector(prefillReplicas, decodeReplicas int, connector string) []string {
	return createModelServersFromKustomize(pdDisaggDir, map[string]string{
		"${KV_CACHE_ENABLED}":     "false",
		"${CONNECTOR_TYPE}":       connector,
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(decodeReplicas),
		"${VLLM_REPLICA_COUNT_P}": strconv.Itoa(prefillReplicas),
	})
}

func createModelServersPDNixlV2(prefillReplicas, decodeReplicas int) []string {
	return createModelServersPDWithConnector(prefillReplicas, decodeReplicas, proxy.KVConnectorNIXLV2)
}

func createModelServersPDSharedStorage(decodeReplicas int) []string {
	return createModelServersPDWithConnector(1, decodeReplicas, proxy.KVConnectorSharedStorage)
}

func createModelServersPDMooncake(decodeReplicas int) []string {
	return createModelServersPDWithConnector(1, decodeReplicas, proxy.KVConnectorMooncake)
}

// createModelServersEpDDisagg creates model server resources for E/PD (encode + prefill/decode) testing.
func createModelServersEpDDisagg(encodeReplicas, decodeReplicas int) []string {
	return createModelServersFromKustomize(ePdDisaggDir, map[string]string{
		"${EC_CONNECTOR_TYPE}":    proxy.ECExampleConnector,
		"${VLLM_REPLICA_COUNT_E}": strconv.Itoa(encodeReplicas),
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(decodeReplicas),
	})
}

// createModelServersEPDDisagg creates model server resources for E/P/D (encode/prefill/decode) testing.
func createModelServersEPDDisagg(encodeReplicas, prefillReplicas, decodeReplicas int) []string {
	return createModelServersFromKustomize(ePDDisaggDir, map[string]string{
		"${KV_CONNECTOR_TYPE}":    proxy.KVConnectorSharedStorage,
		"${EC_CONNECTOR_TYPE}":    proxy.ECExampleConnector,
		"${VLLM_REPLICA_COUNT_E}": strconv.Itoa(encodeReplicas),
		"${VLLM_REPLICA_COUNT_P}": strconv.Itoa(prefillReplicas),
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(decodeReplicas),
	})
}

// createModelServersEncodeOnly creates encode-only pods (no prefill, no decode).
func createModelServersEncodeOnly(replicas int) []string {
	return createModelServersFromKustomize(encodeOnlyDir, map[string]string{
		"${EC_CONNECTOR_TYPE}":    "",
		"${VLLM_REPLICA_COUNT_E}": strconv.Itoa(replicas),
	})
}

// createModelServersPrefillOnly creates prefill-only pods (no encode, no decode).
func createModelServersPrefillOnly(replicas int) []string {
	return createModelServersFromKustomize(prefillOnlyDir, map[string]string{
		"${KV_CONNECTOR_TYPE}":    "",
		"${VLLM_REPLICA_COUNT_P}": strconv.Itoa(replicas),
	})
}

// createModelServersEPDUnified creates model server resources for EPD (one deployment for encode/prefill/decode) testing.
func createModelServersEPDUnified(replicas int) []string {
	return createModelServersFromKustomize(epdDeploymentDir, map[string]string{
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(replicas),
		"${DECODE_ROLE}":          "encode-prefill-decode",
	})
}

func createRender(nsName string) []string {
	renderYamls := substituteMany(testutils.ReadYaml(renderManifest),
		map[string]string{
			"${MODEL_NAME}":        kvModelName,
			"${VLLM_RENDER_IMAGE}": vllmRenderImage,
			"${VLLM_RENDER_PORT}":  vllmRenderPort,
		})
	objects := testutils.CreateObjsFromYaml(testConfig, renderYamls, nsName)
	podsInDeploymentsReady(nsName, objects)
	return objects
}

func createEndPointPicker(eppConfig string) []string {
	objects := createEndPointPickerHelper(eppConfig, 1, false, true)
	podsInDeploymentsReady(getNamespace(), objects)

	// Envoy registers the EPP as a healthy ext_proc upstream asynchronously.
	// "no healthy upstream" returns HTTP 500 with empty body; any non-empty
	// response (200 or 500-with-body) means EPP is reachable from Envoy.
	ginkgo.By("Waiting for gateway to be ready")
	gomega.Eventually(func() bool {
		resp, err := http.Get(fmt.Sprintf("http://localhost:%d/v1/models", getPort()))
		if err != nil {
			return false
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return resp.StatusCode == http.StatusOK || len(body) > 0
	}, readyTimeout, 2*time.Second).Should(gomega.BeTrue(), "gateway should be ready within the ready timeout")

	waitForEPPToDiscoverPods(poolName)

	return objects
}

func createEndPointPickerHelper(eppConfig string, replicas int, isLeaderElectionEnabled bool, waitForReady bool) []string {
	nsName := getNamespace()
	configMap := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "epp-config",
			Namespace: getNamespace(),
		},
		Data: map[string]string{"epp-config.yaml": eppConfig},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, configMap)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	objects := make([]string, 1, 10)
	objects[0] = "ConfigMap/epp-config"

	eppYamls := testutils.ReadYaml(eppManifest)
	eppYamls = substituteMany(eppYamls,
		map[string]string{
			"${EPP_NAME}":               eppName,
			"${EPP_IMAGE}":              eppImage,
			"${NAMESPACE}":              nsName,
			"${POOL_NAME}":              simModelName + "-inference-pool",
			"${METRICS_ENDPOINT_AUTH}":  "false",
			"${EPP_REPLICA_COUNT}":      strconv.Itoa(replicas),
			"${ENABLE_LEADER_ELECTION}": strconv.FormatBool(isLeaderElectionEnabled),
		})
	eppYamls = appendEppArgs(eppYamls, eppExtraArgs)

	if waitForReady {
		return append(objects, testutils.CreateObjsFromYaml(testConfig, eppYamls, nsName)...)
	}
	objs := testutils.CreateUnstructuredObjs(testConfig, eppYamls)
	objects = append(objects, testutils.CreateObjsWithVerifier(testConfig, objs, nsName, func(kind string, clientObj client.Object) {})...)

	gomega.Eventually(func() error {
		_, _, err := tryCompletion(simplePrompt, simModelName)
		return err
	}, readyTimeout, 1*time.Second).Should(gomega.Succeed())

	return objects
}

func usesTokenProducer(eppConfig string) bool {
	cfg, _, err := configloader.LoadRawConfig([]byte(eppConfig), ginkgo.GinkgoLogr)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	for _, plugin := range cfg.Plugins {
		if plugin.Type == tokenizer.PluginType {
			return true
		}
	}
	return false
}

// testWrapper wraps tests with the setup and teardown code needed.
// It is used as a wrapper of the function passed to ginkgo.When calls that
// setup the tests. It is important that the gink.Ordered decorator is used
// to enable the use of the ginkgo.BeforeAll and ginkgo.AfterAll functions
// to inject the setup and teardown code.
func testWrapper(test func()) func() {
	var (
		nsName           = getNamespace()
		createdNameSpace bool

		rbacObjects           []string
		serviceAccountObjects []string
		serviceObjects        []string
		envoyObjects          []string
		portForwardSession    *gexec.Session
	)
	return func() {
		ginkgo.BeforeAll(func() {
			createdNameSpace = setupNameSpace()

			envoyObjects, portForwardSession = createEnvoy(nsName)

			infraSubs := map[string]string{
				"${EPP_NAME}":          "e2e-epp",
				"${METRICS_NODE_PORT}": strconv.Itoa(32090 + 100*(ginkgo.GinkgoParallelProcess()-1)),
			}
			rbacYamls := substituteMany(testutils.ReadYaml(rbacManifest), infraSubs)
			rbacObjects = testutils.CreateObjsFromYaml(testConfig, rbacYamls, nsName)
			saYamls := substituteMany(testutils.ReadYaml(serviceAccountManifest), infraSubs)
			serviceAccountObjects = testutils.CreateObjsFromYaml(testConfig, saYamls, nsName)
			svcYamls := substituteMany(testutils.ReadYaml(servicesManifest), infraSubs)
			serviceObjects = testutils.CreateObjsFromYaml(testConfig, svcYamls, nsName)
		})

		ginkgo.AfterEach(func() {
			// The starting of the EPP can launch a port-forwarder to access the metrics port.
			if eppPortForwardSession != nil {
				eppPortForwardSession.Terminate()
				eppPortForwardSession = nil
			}
		})

		ginkgo.AfterAll(func() {
			// Only cleanup if the test succeeded
			if !ginkgo.CurrentSpecReport().Failed() {
				testutils.DeleteObjects(testConfig, rbacObjects, nsName)
				testutils.DeleteObjects(testConfig, serviceObjects, nsName)
				testutils.DeleteObjects(testConfig, serviceAccountObjects, nsName)
				if portForwardSession != nil {
					portForwardSession.Terminate()
					portForwardSession = nil
				}
				testutils.DeleteObjects(testConfig, envoyObjects, nsName)

				if createdNameSpace {
					ginkgo.By("Deleting namespace " + getNamespace())
					err := testConfig.KubeCli.CoreV1().Namespaces().Delete(testConfig.Context, getNamespace(), metav1.DeleteOptions{})
					gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
					gomega.Eventually(func() bool {
						_, err := testConfig.KubeCli.CoreV1().Namespaces().Get(testConfig.Context, getNamespace(), metav1.GetOptions{})
						return apierrors.IsNotFound(err)
					}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.BeTrue())
				}
			} else {
				// The test failed
				dumpPodsAndLogs(getNamespace())
			}
		})

		test()
	}
}
