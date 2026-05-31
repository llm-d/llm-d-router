/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package epdpools

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// setupInfra installs the CRDs, per-phase EPP scheduling ConfigMaps, the per-
// phase EPPs / InferencePools / vLLM workers, and the hand-rolled Envoy. Runs
// only on suite-owned clusters; with K8S_CONTEXT set the caller is responsible
// for already having all of this in place.
func setupInfra() {
	ginkgo.By("Installing CRDs from " + crdKustomizePath)
	crds := runKustomize(crdKustomizePath)
	_ = testutils.CreateObjsFromYaml(testConfig, crds)

	ginkgo.By("Creating per-phase EPP scheduling ConfigMaps")
	createEPPConfigMap("epp-config-encode", encodeEPPConfigPath)
	createEPPConfigMap("epp-config-prefill", prefillEPPConfigPath)
	createEPPConfigMap("epp-config-decode", decodeEPPConfigPath)

	ginkgo.By("Applying per-phase InferencePools")
	applyManifest(encodePoolManifest, eppSubstitutions())
	applyManifest(prefillPoolManifest, eppSubstitutions())
	applyManifest(decodePoolManifest, eppSubstitutions())

	ginkgo.By("Applying per-phase EPP Deployments + RBAC + Services")
	applyManifest(encodeEPPManifest, eppSubstitutions())
	applyManifest(prefillEPPManifest, eppSubstitutions())
	applyManifest(decodeEPPManifest, eppSubstitutions())

	ginkgo.By("Applying per-phase vLLM workers from " + vllmEnvKustomizeDir)
	vllmDocs := runKustomize(vllmEnvKustomizeDir)
	vllmDocs = substituteMany(vllmDocs, vllmSubstitutions())
	vllmDocs = removeEmptyArgs(vllmDocs)
	vllmDocs = removeEmptyLabels(vllmDocs)
	_ = testutils.CreateObjsFromYaml(testConfig, vllmDocs)

	ginkgo.By("Applying Envoy from " + envoyManifest)
	envoyObjects := applyManifest(envoyManifest, map[string]string{
		"${NAMESPACE}": nsName,
		"${EPP_NAME}":  eppName,
	})

	if k8sContext != "" {
		envoyDeploy := ""
		for _, obj := range envoyObjects {
			parts := strings.Split(obj, "/")
			if strings.EqualFold(parts[0], "deployment") {
				envoyDeploy = parts[1]
			}
		}
		gomega.Expect(envoyDeploy).ToNot(gomega.BeEmpty(), "envoy Deployment not found in applied objects")
		ginkgo.By("Starting kubectl port-forward to Envoy for K8S_CONTEXT mode")
		command := exec.Command("kubectl", "port-forward", "deployment/"+envoyDeploy, gatewayPort+":8081",
			"--context="+k8sContext, "--namespace="+nsName)
		var err error
		portForwardSession, err = gexec.Start(command, ginkgo.GinkgoWriter, ginkgo.GinkgoWriter)
		gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	}

	waitForGatewayReady()
}

// waitForGatewayReady polls the gateway until the EPP ext_proc pipeline is
// wired up. It sends a minimal POST with EPP-Phase: encode so the request
// matches an Envoy route. Any non-connection-error response with a body
// (even 400) means Envoy reached the EPP successfully.
func waitForGatewayReady() {
	ginkgo.By("Waiting for gateway to be ready")
	probeBody := []byte(`{"token_ids":[1],"sampling_params":{"max_tokens":1}}`)
	gomega.Eventually(func() bool {
		req, err := http.NewRequest(http.MethodPost,
			fmt.Sprintf("http://localhost:%s/inference/v1/generate", gatewayPort),
			bytes.NewReader(probeBody))
		if err != nil {
			return false
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("EPP-Phase", "encode")

		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return false
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return len(body) > 0
	}, readyTimeout, 2*time.Second).Should(gomega.BeTrue(), "gateway should be ready within the ready timeout")
}

func createEPPConfigMap(name, path string) {
	yamlContents := strings.Join(testutils.ReadYaml(path), "\n---\n")
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: nsName,
		},
		Data: map[string]string{"epp-config.yaml": yamlContents},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, cm)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "creating ConfigMap %s", name)
	}
}

func applyManifest(path string, subs map[string]string) []string {
	docs := testutils.ReadYaml(path)
	docs = substituteMany(docs, subs)
	docs = removeEmptyArgs(docs)
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

func vllmSubstitutions() map[string]string {
	return map[string]string{
		"${POOL_NAME}":               poolNameBase,
		"${MODEL_NAME}":              modelName,
		"${VLLM_IMAGE}":              vllmSimImage,
		"${VLLM_RENDER_IMAGE}":       vllmRenderImage,
		"${SIDECAR_IMAGE}":           sideCarImage,
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
