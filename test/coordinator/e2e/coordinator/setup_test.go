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

package coordinate2e

import (
	"net/http"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/test/coordinator/e2e/internal/e2eutil"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// setupNameSpace creates the test namespace if it does not already exist and
// records whether it was created so AfterSuite can delete it on cleanup.
func setupNameSpace() {
	if nsName == "default" {
		return
	}
	_, err := testConfig.KubeCli.CoreV1().Namespaces().Get(testConfig.Context, nsName, metav1.GetOptions{})
	if err == nil {
		return
	}
	gomega.Expect(apierrors.IsNotFound(err)).To(gomega.BeTrue())

	ginkgo.By("Creating namespace " + nsName)
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: nsName,
		},
	}
	_, err = testConfig.KubeCli.CoreV1().Namespaces().Create(testConfig.Context, ns, metav1.CreateOptions{})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	createdNameSpace = true
}

// setupInfra installs the base infra shared across tests: Gateway API + GIE
// CRDs, the epp-reader Role, and Envoy. Runs only on suite-owned kind clusters;
// with K8S_CONTEXT set the caller is responsible for having base infra in place.
// The per-test workload (EPPs, InferencePools, vLLM workers, coordinator) is
// created in the test body.
func setupInfra() {
	createCRDs()

	ginkgo.By("Applying shared Role/epp-reader from " + baseRbacManifest)
	_ = testutils.CreateObjsFromYaml(testConfig, testutils.ReadYaml(baseRbacManifest))

	ginkgo.By("Applying Envoy from " + envoyManifest)
	applyManifest(envoyManifest, map[string]string{
		"${NAMESPACE}": nsName,
		"${EPP_NAME}":  eppName,
	})
}

// createCRDs installs the GIE CRDs used for testing.
func createCRDs() {
	ginkgo.By("Installing GIE CRDs from " + crdGIEPath)
	gieCRDs := e2eutil.RunKustomize(crdGIEPath)
	_ = testutils.CreateObjsFromYaml(testConfig, gieCRDs)
}

// createEndPointPicker creates the scheduling ConfigMap and EPP Deployment (plus
// its ServiceAccount, RoleBinding, and Service) for the given phase from the
// supplied EPP config and waits for the EPP Deployment to become ready. Returns
// the created object ids for cleanup.
func createEndPointPicker(phase, config string) []string {
	manifest := map[string]string{
		"encode":  encodeEPPManifest,
		"prefill": prefillEPPManifest,
		"decode":  decodeEPPManifest,
	}[phase]

	cmName := "epp-config-" + phase
	createEPPConfigMap(cmName, config)

	objects := make([]string, 1, 8)
	objects[0] = "ConfigMap/" + cmName
	objects = append(objects, applyManifest(manifest, eppSubstitutions())...)
	podsInDeploymentsReady(objects)
	return objects
}

// createInferencePool creates the InferencePool for the given phase. When
// toDelete is set, the existing pool is removed first so the test starts clean.
func createInferencePool(phase string, toDelete bool) []string {
	manifest := map[string]string{
		"encode":  encodePoolManifest,
		"prefill": prefillPoolManifest,
		"decode":  decodePoolManifest,
	}[phase]

	if toDelete {
		deletePoolIfExists(poolNameBase + "-" + phase)
	}

	docs := testutils.ReadYaml(manifest)
	docs = e2eutil.SubstituteMany(docs, eppSubstitutions())
	return testutils.CreateObjsFromYaml(testConfig, docs)
}

// deletePoolIfExists removes the named InferencePool when present so a rerun
// against a persistent cluster starts clean. testutils.DeleteObjects asserts
// the object exists, so a fresh cluster needs the existence check first.
func deletePoolIfExists(name string) {
	pool := &inferenceapi.InferencePool{}
	err := testConfig.K8sClient.Get(testConfig.Context,
		types.NamespacedName{Namespace: testConfig.NsName, Name: name}, pool)
	if apierrors.IsNotFound(err) {
		return
	}
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "checking InferencePool %s", name)
	testutils.DeleteObjects(testConfig, []string{"InferencePool/" + name})
}

// createModelServers deploys the vLLM encode/prefill/decode workers from the
// epd-pools kustomize environment with the given per-type replica counts and
// waits for their Deployments to be ready.
func createModelServers(encodeReplicas, prefillReplicas, decodeReplicas int) []string {
	subs := allSubstitutions()
	subs["${VLLM_REPLICA_COUNT_E}"] = strconv.Itoa(encodeReplicas)
	subs["${VLLM_REPLICA_COUNT_P}"] = strconv.Itoa(prefillReplicas)
	subs["${VLLM_REPLICA_COUNT_D}"] = strconv.Itoa(decodeReplicas)

	docs := e2eutil.RunKustomize(epdPoolsKustomizeDir)
	docs = e2eutil.SubstituteMany(docs, subs)
	docs = e2eutil.RemoveEmptyArgs(docs)
	docs = e2eutil.RemoveEmptyLabels(docs)
	objects := testutils.CreateObjsFromYaml(testConfig, docs)
	podsInDeploymentsReady(objects)
	return objects
}

// createCoordinator builds the coordinator ConfigMap from the given pipeline
// config, deploys the coordinator component (Deployment + Service + SA), starts a
// port-forward when running against an existing cluster, and waits for readiness.
func createCoordinator(config string) []string {
	coordinatorYAML := e2eutil.SubstituteMany([]string{config}, map[string]string{
		"${NAMESPACE}":        nsName,
		"${VLLM_RENDER_PORT}": vllmRenderPort,
	})[0]
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "llm-d-coordinator-config",
			Namespace: nsName,
		},
		Data: map[string]string{"coordinator.yaml": coordinatorYAML},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, cm)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "creating coordinator ConfigMap")
	}
	objects := make([]string, 1, 8)
	objects[0] = "ConfigMap/llm-d-coordinator-config"

	docs := e2eutil.RunKustomize(coordinatorComponentDir)
	docs = e2eutil.FilterKinds(docs, "ConfigMap")
	docs = e2eutil.SubstituteMany(docs, coordinatorSubstitutions())
	docs = e2eutil.RemoveEmptyArgs(docs)
	objects = append(objects, testutils.CreateObjsFromYaml(testConfig, docs)...)

	podsInDeploymentsReady(objects)
	if k8sContext != "" {
		startPortForward("deployment/llm-d-coordinator", coordinatorPort, "8080")
	}
	waitForCoordinatorReady()
	return objects
}

// waitForCoordinatorReady polls /readyz on the coordinator until HTTP 200 via
// two paths: the direct NodePort and the Envoy gateway's default route. Both
// must succeed before the test sends inference requests.
//
// The NodePort check confirms the coordinator pod is up. The gateway check
// confirms Envoy's STRICT_DNS coordinator cluster has resolved, which can lag
// because the coordinator Service is created per-test while Envoy starts once
// at suite setup.
func waitForCoordinatorReady() {
	ginkgo.By("Waiting for coordinator to be ready (direct)")
	gomega.Eventually(func() bool {
		return pollReady(coordinatorBaseURL + "/readyz")
	}, readyTimeout, defaultInterval).Should(gomega.BeTrue(), "coordinator should be ready within the ready timeout")

	ginkgo.By("Waiting for coordinator to be reachable via gateway")
	gomega.Eventually(func() bool {
		return pollReady(gatewayBaseURL + "/readyz")
	}, readyTimeout, defaultInterval).Should(gomega.BeTrue(), "coordinator should be reachable via gateway within the ready timeout")
}

// pollReady reports whether a GET on url returns HTTP 200 within the client timeout.
func pollReady(url string) bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func createEPPConfigMap(name, content string) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: nsName,
		},
		Data: map[string]string{"epp-config.yaml": content},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, cm)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "creating ConfigMap %s", name)
	}
}

func applyManifest(path string, subs map[string]string) []string {
	docs := testutils.ReadYaml(path)
	docs = e2eutil.SubstituteMany(docs, subs)
	docs = e2eutil.RemoveEmptyArgs(docs)
	return testutils.CreateObjsFromYaml(testConfig, docs)
}

func eppSubstitutions() map[string]string {
	return map[string]string{
		"${EPP_NAME}":              eppName,
		"${POOL_NAME}":             poolNameBase,
		"${EPP_IMAGE}":             eppImage,
		"${NAMESPACE}":             nsName,
		"${METRICS_ENDPOINT_AUTH}": "false",
	}
}

// allSubstitutions returns the substitution map for the epd-pools kustomize
// environment (vLLM workers only).
func allSubstitutions() map[string]string {
	return map[string]string{
		"${POOL_NAME}":               poolNameBase,
		"${MODEL_NAME}":              modelName,
		"${VLLM_IMAGE}":              vllmSimImage,
		"${VLLM_DATA_PARALLEL_SIZE}": "1",
		"${VLLM_REPLICA_COUNT_E}":    "1",
		"${VLLM_REPLICA_COUNT_P}":    "1",
		"${VLLM_REPLICA_COUNT_D}":    "1",
		"${VLLM_EXTRA_ARGS_E}":       "",
		"${VLLM_EXTRA_ARGS_P}":       "",
		"${VLLM_EXTRA_ARGS_D}":       "",
		"${KV_CONNECTOR_TYPE}":       "",
		"${EC_CONNECTOR_TYPE}":       "",
		"${CONNECTOR_TYPE}":          "",
		"${VLLM_SIM_MODE}":           "echo",
		"${KV_CACHE_ENABLED}":        "false",
		"${HF_TOKEN}":                "",
		"${EPP_NAME}":                eppName,
		"${NAMESPACE}":               nsName,
		"${DECODE_ROLE}":             "decode",
	}
}

// coordinatorSubstitutions returns the substitution map for the coordinator
// component manifests.
func coordinatorSubstitutions() map[string]string {
	return map[string]string{
		"${COORDINATOR_IMAGE}": coordinatorImage,
	}
}

// rendererSubstitutions returns the substitution map for the vllm-render
// component manifests.
func rendererSubstitutions() map[string]string {
	return map[string]string{
		"${VLLM_RENDER_IMAGE}": vllmRenderImage,
		"${VLLM_RENDER_PORT}":  vllmRenderPort,
		"${MODEL_NAME}":        modelName,
	}
}

// createRenderer deploys the vllm-render component and waits for readiness.
func createRenderer() []string {
	ginkgo.By("Deploying vllm-render")
	docs := e2eutil.RunKustomize(rendererComponentDir)
	docs = e2eutil.SubstituteMany(docs, rendererSubstitutions())
	docs = e2eutil.RemoveEmptyArgs(docs)
	objects := testutils.CreateObjsFromYaml(testConfig, docs)
	podsInDeploymentsReady(objects)
	return objects
}
