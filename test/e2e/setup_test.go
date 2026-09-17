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

package e2e

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"github.com/llm-d/llm-d-router/pkg/sidecar/proxy"
	"github.com/llm-d/llm-d-router/test/e2e/utils"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// standaloneConfig passes the suite's per-process settings to the standalone
// router helpers.
func standaloneConfig() standalone.Config {
	return standalone.Config{
		TestConfig:    testConfig,
		Namespace:     getNamespace(),
		EPPImage:      eppImage,
		HTTPPort:      getPort(),
		MetricsPort:   getMetricsPort(),
		K8sContext:    k8sContext,
		PodSelector:   podSelector,
		ReleaseName:   poolName,
		KeepOnFailure: keepClusterOnFailure,
	}
}

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
		"${EPP_NAME}":                eppName,
		"${NAMESPACE}":               nsName,
		"${HF_TOKEN}":                os.Getenv("HF_TOKEN"),
		"${VLLM_EXTRA_ARGS_E}":       "--force-dummy-tokenizer",
		"${VLLM_EXTRA_ARGS_P}":       "--force-dummy-tokenizer",
		"${VLLM_EXTRA_ARGS_D}":       "--force-dummy-tokenizer",
		"${VLLM_RENDER_URL}":         fmt.Sprintf("http://vllm-render.%s.svc.cluster.local:%s", baseNsName, vllmRenderPort),
		"${VLLM_RENDER_PORT}":        vllmRenderPort,
	}
	for k, v := range extra {
		subs[k] = v
	}

	manifests := utils.RunKustomize(kustomizeDir)
	manifests = utils.SubstituteMany(manifests, subs)
	// Remove labels with empty values (produced when ${DECODE_ROLE} is empty)
	manifests = utils.RemoveEmptyLabels(manifests)
	manifests = utils.RemoveEmptyArgs(manifests)
	objects, err := utils.DecodeCaseObjects([]byte(strings.Join(manifests, "\n---\n")), nsName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	resources := &utils.CaseResources{Client: testConfig.K8sClient}
	utils.DeferCaseCleanup(testConfig, keepClusterOnFailure, resources, nsName, nil)
	gomega.Expect(resources.Create(testConfig.Context, objects)).To(gomega.Succeed())
	names := make([]string, len(objects))
	for i, obj := range objects {
		names[i] = obj.GetKind() + "/" + obj.GetName()
	}
	utils.PodsInDeploymentsReady(testConfig, nsName, names)
	return names
}

func createModelServersDecode(replicas int) []string {
	return createModelServersFromKustomize(epdDeploymentDir, map[string]string{
		"${KV_CACHE_ENABLED}":     "false",
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(replicas),
	})
}

func createModelServersDecodeKV(replicas int) {
	createModelServersFromKustomize(epdDeploymentDir, map[string]string{
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

func createModelServersPDSharedStorage(decodeReplicas int) {
	createModelServersPDWithConnector(1, decodeReplicas, proxy.KVConnectorSharedStorage)
}

func createModelServersPDMooncake(decodeReplicas int) {
	createModelServersPDWithConnector(1, decodeReplicas, proxy.KVConnectorMooncake)
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
	renderYamls := utils.SubstituteMany(testutils.ReadYaml(renderManifest),
		map[string]string{
			"${MODEL_NAME}":        kvModelName,
			"${VLLM_RENDER_IMAGE}": vllmRenderImage,
			"${VLLM_RENDER_PORT}":  vllmRenderPort,
		})
	objects := testutils.CreateObjsFromYaml(testConfig, renderYamls, nsName)
	utils.PodsInDeploymentsReady(testConfig, nsName, objects)
	return objects
}

// testWrapper requires an Ordered group so BeforeAll can register namespace
// cleanup that runs after per-case resource cleanup.
func testWrapper(test func()) func() {
	return func() {
		ginkgo.BeforeAll(func() {
			nsName := getNamespace()
			createdNameSpace := setupNameSpace()
			ginkgo.DeferCleanup(func() {
				if ginkgo.CurrentSpecReport().Failed() && keepClusterOnFailure {
					testutils.DumpPodsAndLogs(testConfig, nsName)
				} else if createdNameSpace {
					deleteNameSpace(nsName)
				}
			})
		})

		test()
	}
}
