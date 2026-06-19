package e2e

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const ssrfTestLabel = "SSRF"

// createModelServersPDSSRF creates model servers with SSRF protection enabled on the sidecar.
func createModelServersPDSSRF(prefillReplicas, decodeReplicas int) []string {
	// Apply SSRF RBAC (Role + RoleBinding for InferencePool/Pod watch).
	rbacDocs := testutils.ReadYaml("../../deploy/components/vllm-decode/rbac-ssrf.yaml")
	for i, doc := range rbacDocs {
		rbacDocs[i] = strings.ReplaceAll(doc, "${POOL_NAME}", poolName)
		rbacDocs[i] = strings.ReplaceAll(rbacDocs[i], "${NAMESPACE}", nsName)
	}
	testutils.CreateObjsFromYaml(testConfig, rbacDocs)

	return createModelServersFromKustomize(pdDisaggDir, map[string]string{
		"${KV_CACHE_ENABLED}":     "false",
		"${CONNECTOR_TYPE}":       "nixlv2",
		"${VLLM_REPLICA_COUNT_D}": strconv.Itoa(decodeReplicas),
		"${VLLM_REPLICA_COUNT_P}": strconv.Itoa(prefillReplicas),
		// Enable SSRF protection on the sidecar (bool flag must be separate from value flags)
		"${SIDECAR_EXTRA_ARGS}":      "--inference-pool=" + poolName,
		"${SIDECAR_EXTRA_ARGS_BOOL}": "--enable-ssrf-protection",
	})
}

// sendRequestWithPrefillHeader sends a chat completion request with the X-Prefiller-Host-Port header
// set to the specified value. Returns the HTTP status code.
func sendRequestWithPrefillHeader(prefillHeader string) int {
	body := fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"test"}],"max_tokens":10}`, simModelName)
	req, err := http.NewRequest(http.MethodPost,
		"http://localhost:"+port+"/v1/chat/completions",
		strings.NewReader(body))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	req.Header.Set("Content-Type", "application/json")
	if prefillHeader != "" {
		req.Header.Set("X-Prefiller-Host-Port", prefillHeader)
	}

	resp, err := http.DefaultClient.Do(req)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer func() {
		_ = resp.Body.Close()
	}()

	_, err = io.ReadAll(resp.Body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	return resp.StatusCode
}

var _ = ginkgo.Describe("SSRF Protection", ginkgo.Label(ssrfTestLabel), func() {
	ginkgo.When("Sidecar has SSRF protection enabled", func() {
		ginkgo.It("should allow requests with valid prefill header targeting an allowed pod", func() {
			infPoolObjects = createInferencePool(1, true)

			prefillReplicas := 1
			decodeReplicas := 1
			modelServers := createModelServersPDSSRF(prefillReplicas, decodeReplicas)

			epp := createEndPointPicker(pdConfig)

			// Use pod IPs (not names) because the sidecar needs to connect to the target
			// and pod names are not DNS-resolvable outside a headless service.
			prefillPodIPs := getPodIPs(prefillSelector)
			gomega.Expect(prefillPodIPs).Should(gomega.HaveLen(prefillReplicas))

			// Send a request with a valid prefill header targeting an allowed pod.
			// The header format is "host:port" where host is the pod IP and port is from the InferencePool targetPorts.
			// Retry with Eventually because the sidecar's pod informer starts asynchronously and needs
			// time to discover pods and populate the allowlist after the proxy server begins listening.
			validHeader := prefillPodIPs[0] + ":8000"
			gomega.Eventually(func() int {
				return sendRequestWithPrefillHeader(validHeader)
			}, "30s", "1s").Should(gomega.Equal(http.StatusOK),
				"Request with valid prefill header should be allowed")

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})

		ginkgo.It("should block requests with invalid port in prefill header", func() {
			infPoolObjects = createInferencePool(1, true)

			prefillReplicas := 1
			decodeReplicas := 1
			modelServers := createModelServersPDSSRF(prefillReplicas, decodeReplicas)

			epp := createEndPointPicker(pdConfig)

			prefillPodIPs := getPodIPs(prefillSelector)
			gomega.Expect(prefillPodIPs).Should(gomega.HaveLen(prefillReplicas))

			// Send a request with an invalid port (9999) that is not in the InferencePool targetPorts
			invalidPortHeader := prefillPodIPs[0] + ":9999"
			statusCode := sendRequestWithPrefillHeader(invalidPortHeader)
			gomega.Expect(statusCode).Should(gomega.Equal(http.StatusForbidden),
				"Request with invalid port should be blocked")

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})

		ginkgo.It("should block requests with invalid host in prefill header", func() {
			infPoolObjects = createInferencePool(1, true)

			prefillReplicas := 1
			decodeReplicas := 1
			modelServers := createModelServersPDSSRF(prefillReplicas, decodeReplicas)

			epp := createEndPointPicker(pdConfig)

			// Send a request with an invalid host that is not in the allowlist
			invalidHostHeader := "192.168.99.99:8000"
			statusCode := sendRequestWithPrefillHeader(invalidHostHeader)
			gomega.Expect(statusCode).Should(gomega.Equal(http.StatusForbidden),
				"Request with invalid host should be blocked")

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})
})
