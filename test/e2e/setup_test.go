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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	configloader "github.com/llm-d/llm-d-router/pkg/epp/config/loader"
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
		"${VLLM_RENDER_IMAGE}":       vllmRenderImage,
		"${SIDECAR_IMAGE}":           sideCarImage,
		"${VLLM_DATA_PARALLEL_SIZE}": "1",
		"${VLLM_SIM_MODE}":           "echo",
		"${KV_CACHE_ENABLED}":        "false",
		"${DECODE_ROLE}":             "",
		"${EPP_NAME}":                "e2e-epp",
		"${NAMESPACE}":               nsName,
		"${HF_TOKEN}":                os.Getenv("HF_TOKEN"),
		"${VLLM_EXTRA_ARGS_E}":       "",
		"${VLLM_EXTRA_ARGS_P}":       "",
		"${VLLM_EXTRA_ARGS_D}":       "",
	}
	for k, v := range extra {
		subs[k] = v
	}

	manifests := runKustomize(kustomizeDir)
	manifests = substituteMany(manifests, subs)
	// Remove labels with empty values (produced when ${DECODE_ROLE} is empty)
	manifests = removeEmptyLabels(manifests)
	manifests = removeEmptyArgs(manifests)
	// remove render sidecar if model is simulated
	if !isModelReal(subs["${MODEL_NAME}"]) {
		manifests = removeRenderSidecar(manifests)
	}
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

// createModelServersEPDUnified creates model server resources for EPD (one deployment for encode/prefill/decode) testing.
func createModelServersEPDUnified(replicas int) []string {
	return createModelServersFromKustomize(epdDeploymentDir, map[string]string{
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(replicas),
		"${DECODE_ROLE}":          "encode-prefill-decode",
	})
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
			"${EPP_NAME}":          eppName,
			"${EPP_IMAGE}":         eppImage,
			"${VLLM_RENDER_IMAGE}": vllmRenderImage,
			// The render sidecar needs a real, fetchable model. Sim tests
			// don't query it; the cost is paying weights-load on every EPP.
			"${MODEL_NAME}":             kvModelName,
			"${NAMESPACE}":              nsName,
			"${POOL_NAME}":              simModelName + "-inference-pool",
			"${METRICS_ENDPOINT_AUTH}":  "false",
			"${EPP_REPLICA_COUNT}":      strconv.Itoa(replicas),
			"${ENABLE_LEADER_ELECTION}": strconv.FormatBool(isLeaderElectionEnabled),
		})
	if !usesTokenProducer(eppConfig) {
		eppYamls = removeRenderSidecar(eppYamls)
	}
	eppYamls = appendEppArgs(eppYamls, eppExtraArgs)

	if waitForReady {
		return append(objects, testutils.CreateObjsFromYaml(testConfig, eppYamls, nsName)...)
	}
	objs := testutils.CreateUnstructuredObjs(testConfig, eppYamls)
	return append(objects, testutils.CreateObjsWithVerifier(testConfig, objs, nsName, func(kind string, clientObj client.Object) {})...)
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
