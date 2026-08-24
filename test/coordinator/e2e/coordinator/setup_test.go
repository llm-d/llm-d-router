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
	"fmt"
	"net/http"
	"strconv"
	"strings"
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
	nsName := getNamespace()
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
// CRDs and Envoy. Runs only on suite-owned kind clusters; with K8S_CONTEXT set
// the caller is responsible for having base infra in place. The per-test
// workload (EPPs, InferencePools, vLLM workers, coordinator) is created in the
// test body; the EPP's RBAC/ServiceAccount/Service come from createStableInfra.
func setupInfra() {
	nsName := getNamespace()
	createCRDs()

	manifest := envoyManifest
	infraSubs := map[string]string{
		"${NAMESPACE}": nsName,
		"${EPP_NAME}":  eppName,
	}
	if threeEPP {
		manifest = envoy3EPPManifest
		infraSubs = map[string]string{
			"${NAMESPACE}":        nsName,
			"${EPP_NAME_ENCODE}":  eppNameEncode,
			"${EPP_NAME_PREFILL}": eppNamePrefill,
			"${EPP_NAME_DECODE}":  eppNameDecode,
		}
	}
	ginkgo.By("Applying Envoy routing ConfigMap from " + manifest)
	applyManifest(manifest, infraSubs)
	ginkgo.By("Applying shared Envoy Deployment and Service from " + sharedEnvoyManifest)
	applyManifest(sharedEnvoyManifest, infraSubs)
}

// createCRDs installs the GIE CRDs used for testing.
func createCRDs() {
	ginkgo.By("Installing GIE CRDs from " + crdGIEPath)
	gieCRDs := e2eutil.RunKustomize(crdGIEPath)
	_ = testutils.CreateObjsFromYaml(testConfig, gieCRDs, "")
}

// createEndPointPickers creates each EPP's scheduling ConfigMap and Deployment
// for the active topology and waits for the Deployments to become ready. The
// single-EPP topology creates one EPP from eppConfig; the 3-EPP topology creates
// one per role, each from its role config with a per-role ConfigMap. Each EPP's
// ServiceAccount, RoleBinding, and Service are created once by createStableInfra.
// Returns all created object ids for cleanup.
func createEndPointPickers() []string {
	epps := eppsToCreate()
	objects := make([]string, 0, len(epps)*2)
	for _, e := range epps {
		objects = append(objects, createOneEndPointPicker(e)...)
	}
	return objects
}

// createOneEndPointPicker creates a single EPP's ConfigMap and Deployment. In
// the 3-EPP topology each role gets its own ConfigMap (epp-config-<role>) and the
// shared Deployment's config volume is retargeted to it (see renameEPPConfigVolume),
// so the three EPPs do not share one ConfigMap.
func createOneEndPointPicker(e roleEPP) []string {
	cmName := "epp-config"
	if threeEPP {
		cmName = "epp-config-" + e.role
	}
	createEPPConfigMap(cmName, e.config)

	objects := make([]string, 1, 8)
	objects[0] = "ConfigMap/" + cmName
	// eppManifest is the EPP Deployment only; its Service, ServiceAccount, and
	// RBAC are created once by createStableInfra.
	docs := testutils.ReadYaml(eppManifest)
	docs = e2eutil.SubstituteMany(docs, eppSubstitutionsFor(e.eppName, e.poolName))
	if threeEPP {
		docs = renameEPPConfigVolume(docs, cmName)
	}
	objects = append(objects, testutils.CreateObjsFromYaml(testConfig, docs, getNamespace())...)
	podsInDeploymentsReady(objects)
	return objects
}

// renameEPPConfigVolume retargets the EPP Deployment's config volume, volume
// mount, and ConfigMap reference from the shared "epp-config" name to cmName so
// each 3-EPP role mounts its own ConfigMap. It matches only the "name: epp-config"
// line endings, leaving the "/etc/epp/epp-config.yaml" mount path and ConfigMap
// key (still keyed "epp-config.yaml") untouched. The match count is asserted so a
// formatting change in the shared manifest fails here instead of silently
// no-opping and leaving all three roles pointed at the wrong ConfigMap.
func renameEPPConfigVolume(docs []string, cmName string) []string {
	const oldRef = "name: epp-config\n"
	out := make([]string, len(docs))
	matches := 0
	for i, d := range docs {
		matches += strings.Count(d, oldRef)
		out[i] = strings.ReplaceAll(d, oldRef, "name: "+cmName+"\n")
	}
	gomega.Expect(matches).To(gomega.Equal(3),
		"expected 3 %q references (volume, volumeMount, configMap) in the EPP Deployment; the shared manifest format may have changed", oldRef)
	return out
}

// createInferencePool creates the InferencePool(s) for the active topology: one
// pool covering all three worker roles (single-EPP), or one role-scoped pool per
// role (3-EPP). When toDelete is set, the existing pool(s) are removed first so
// the test starts clean.
func createInferencePool(toDelete bool) []string {
	nsName := getNamespace()

	if toDelete {
		for _, name := range poolNames() {
			deletePoolIfExists(name)
		}
	}

	subs := eppSubstitutionsFor(eppName, poolNameBase)
	// TARGET_PORTS is a YAML block-sequence fragment for the pool manifest's
	// targetPorts field, at that field's 2-space indentation. The coordinator's
	// vLLM workers all listen on 8000.
	subs["${TARGET_PORTS}"] = "\n  - number: 8000"
	manifest := poolManifest
	if threeEPP {
		manifest = pool3EPPManifest
		subs["${EPP_NAME_ENCODE}"] = eppNameEncode
		subs["${EPP_NAME_PREFILL}"] = eppNamePrefill
		subs["${EPP_NAME_DECODE}"] = eppNameDecode
		subs["${POOL_NAME_ENCODE}"] = poolNameEncode
		subs["${POOL_NAME_PREFILL}"] = poolNamePrefill
		subs["${POOL_NAME_DECODE}"] = poolNameDecode
	}
	docs := testutils.ReadYaml(manifest)
	docs = e2eutil.SubstituteMany(docs, subs)
	return testutils.CreateObjsFromYaml(testConfig, docs, nsName)
}

// deletePoolIfExists removes the named InferencePool when present so a rerun
// against a persistent cluster starts clean. testutils.DeleteObjects asserts
// the object exists, so a fresh cluster needs the existence check first.
func deletePoolIfExists(name string) {
	nsName := getNamespace()
	pool := &inferenceapi.InferencePool{}
	err := testConfig.K8sClient.Get(testConfig.Context,
		types.NamespacedName{Namespace: nsName, Name: name}, pool)
	if apierrors.IsNotFound(err) {
		return
	}
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "checking InferencePool %s", name)
	testutils.DeleteObjects(testConfig, []string{"InferencePool/" + name}, nsName)
}

// createModelServers deploys the vLLM encode/prefill/decode workers from the
// coordinator-epd kustomize environment with the given per-type replica counts and
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
	objects := testutils.CreateObjsFromYaml(testConfig, docs, getNamespace())
	podsInDeploymentsReady(objects)
	return objects
}

// createCoordinator builds the coordinator ConfigMap from the given pipeline
// config, deploys the coordinator Deployment, and waits for readiness. Its
// Service and ServiceAccount are created once by createStableInfra.
func createCoordinator(config string) []string {
	nsName := getNamespace()
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
	// Service and ServiceAccount are created once by createStableInfra; recreate
	// only the Deployment per spec.
	docs = e2eutil.FilterKinds(docs, "ConfigMap", "Service", "ServiceAccount")
	docs = e2eutil.SubstituteMany(docs, coordinatorSubstitutions())
	docs = e2eutil.RemoveEmptyArgs(docs)
	objects = append(objects, testutils.CreateObjsFromYaml(testConfig, docs, nsName)...)

	podsInDeploymentsReady(objects)
	waitForCoordinatorReady()
	return objects
}

// waitForCoordinatorReady polls /readyz through Envoy until it returns 200,
// confirming the freshly recreated coordinator pod is reachable through the
// gateway before the test sends its request. The gateway Service is stable
// across specs (see createStableInfra), so this waits only for the new pod to
// appear behind it. (podsInDeploymentsReady already confirms the coordinator
// pod itself is ready.)
func waitForCoordinatorReady() {
	ginkgo.By("Waiting for coordinator to be reachable via gateway")
	gomega.Eventually(func() bool {
		return pollReady(gatewayBaseURL() + "/readyz")
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
			Namespace: getNamespace(),
		},
		Data: map[string]string{"epp-config.yaml": content},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, cm)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "creating ConfigMap %s", name)
	}
}

// applyManifest reads a manifest, substitutes vars, and creates the objects.
// It does not strip empty args: the manifests it applies (Envoy, the EPP's
// Deployment/RBAC/ServiceAccount/Service) carry no empty ${VLLM_EXTRA_ARGS_*}
// placeholders, and rbac.yaml's core API group ("") is a legitimate `- ""` that
// RemoveEmptyArgs would wrongly drop. The vLLM workers, which do need arg
// stripping, go through createModelServers instead.
func applyManifest(path string, subs map[string]string) []string {
	docs := testutils.ReadYaml(path)
	docs = e2eutil.SubstituteMany(docs, subs)
	return testutils.CreateObjsFromYaml(testConfig, docs, getNamespace())
}

// createStableInfra creates the coordinator and EPP Services, ServiceAccounts,
// and RoleBindings once, up front. It appends each created id to
// stableInfraObjects as it goes rather than returning them at the end, so a
// partial failure still leaves the already-created objects tracked for suite
// teardown. Envoy fronts the Services via STRICT_DNS clusters and outlives the
// per-spec workload; recreating a Service each spec would rotate its ClusterIP and
// force Envoy to re-resolve, so only the Deployments behind them churn per spec.
func createStableInfra() {
	docs := e2eutil.RunKustomize(coordinatorComponentDir)
	docs = e2eutil.FilterKinds(docs, "ConfigMap", "Deployment")
	docs = e2eutil.SubstituteMany(docs, coordinatorSubstitutions())
	docs = e2eutil.RemoveEmptyArgs(docs)
	stableInfraObjects = append(stableInfraObjects, testutils.CreateObjsFromYaml(testConfig, docs, getNamespace())...)

	// Each EPP's RBAC, ServiceAccount, and Service come from the shared
	// inference-gateway component's split files; the Deployment is recreated per
	// spec in createEndPointPickers. In the 3-EPP topology this runs once per role.
	for _, e := range eppsToCreate() {
		subs := eppSubstitutionsFor(e.eppName, e.poolName)
		for _, manifest := range []string{eppRbacManifest, eppServiceAccountManifest, eppServicesManifest} {
			stableInfraObjects = append(stableInfraObjects, applyManifest(manifest, subs)...)
		}
	}
}

func eppSubstitutionsFor(name, pool string) map[string]string {
	return map[string]string{
		"${EPP_NAME}":               name,
		"${POOL_NAME}":              pool,
		"${EPP_IMAGE}":              eppImage,
		"${NAMESPACE}":              getNamespace(),
		"${METRICS_ENDPOINT_AUTH}":  "false",
		"${EPP_REPLICA_COUNT}":      "1",
		"${ENABLE_LEADER_ELECTION}": "false",
	}
}

// vllmExtraArgs formats one or more flags to fill a single `- ${VLLM_EXTRA_ARGS_*}`
// list item in the kustomize-rendered worker manifests. SubstituteMany does raw
// text replacement, so each extra flag is emitted as its own list item at the
// placeholder's 8-space indent, yielding a distinct argv element.
func vllmExtraArgs(flags ...string) string {
	return strings.Join(flags, "\n        - ")
}

// allSubstitutions returns the substitution map for the coordinator-epd kustomize
// environment (vLLM workers only).
func allSubstitutions() map[string]string {
	// The pipeline base64-inlines each image into the request body, and the dummy
	// tokenizer counts that blob as text: the largest test image is ~97k tokens,
	// far past the simulator's default 1024-token context. Every vLLM role raises
	// --max-model-len so the encode sub-request is not rejected as over-length.
	vllmArgs := vllmExtraArgs("--force-dummy-tokenizer", "--max-model-len=131072")
	// ${EPP_NAME} is the zmq endpoint the decode workers publish KV events to. In
	// the 3-EPP topology that is the decode EPP's Service.
	workerEPPName := eppName
	if threeEPP {
		workerEPPName = eppNameDecode
	}
	return map[string]string{
		"${POOL_NAME}":               poolNameBase,
		"${MODEL_NAME}":              modelName,
		"${VLLM_IMAGE}":              vllmSimImage,
		"${VLLM_RENDER_URL}":         fmt.Sprintf("http://vllm-render.%s.svc:%s", getNamespace(), vllmRenderPort),
		"${VLLM_DATA_PARALLEL_SIZE}": "1",
		"${VLLM_REPLICA_COUNT_E}":    "1",
		"${VLLM_REPLICA_COUNT_P}":    "1",
		"${VLLM_REPLICA_COUNT_D}":    "1",
		"${VLLM_EXTRA_ARGS_E}":       vllmArgs,
		"${VLLM_EXTRA_ARGS_P}":       vllmArgs,
		"${VLLM_EXTRA_ARGS_D}":       vllmArgs,
		"${KV_CONNECTOR_TYPE}":       "",
		"${EC_CONNECTOR_TYPE}":       "",
		"${CONNECTOR_TYPE}":          "",
		"${VLLM_SIM_MODE}":           "echo",
		"${KV_CACHE_ENABLED}":        "false",
		"${HF_TOKEN}":                "",
		"${EPP_NAME}":                workerEPPName,
		"${NAMESPACE}":               getNamespace(),
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
	docs := testutils.ReadYaml(rendererManifest)
	docs = e2eutil.SubstituteMany(docs, rendererSubstitutions())
	docs = e2eutil.RemoveEmptyArgs(docs)
	objects := testutils.CreateObjsFromYaml(testConfig, docs, getNamespace())
	podsInDeploymentsReady(objects)
	return objects
}
