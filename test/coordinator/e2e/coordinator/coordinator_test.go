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
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const requestTimeout = 60 * time.Second

// testImageURL and testImageURL2 are publicly accessible images used to
// exercise multimodal requests that trigger the encode stage.
const testImageURL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
const testImageURL2 = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg"

var (
	// allSteps lists the full pipeline steps for multimodal requests.
	allSteps = []string{"replace-media-urls", "render", "encode", "prefill", "decode"}
	// textOnlySteps lists the pipeline steps for text-only requests.
	textOnlySteps = []string{"prefill", "decode"}
)

var _ = ginkgo.Describe("Coordinator pipeline", func() {
	ginkgo.It("routes a text only chat completion end-to-end", func() {
		runCoordinatorPipeline([]byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`,
			modelName,
		)), textOnlySteps, 0)
	})

	ginkgo.It("routes a multimodal image chat completion end-to-end", func() {
		runCoordinatorPipeline([]byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"max_tokens":150}`,
			modelName, testImageURL,
		)), allSteps, 1)
	})

	ginkgo.It("routes a multimodal chat completion with two images end-to-end", func() {
		runCoordinatorPipeline([]byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"image_url","image_url":{"url":%q},"uuid":"image-1"},{"type":"text","text":"What is in these two images?"}]}],"max_tokens":150}`,
			modelName, testImageURL, testImageURL2,
		)), allSteps, 2)
	})
})

// runCoordinatorPipeline deploys the e-p-d topology and coordinator, posts the
// given chat-completion body, asserts a 200 with a non-empty body, verifies
// that the coordinator logs show all expected pipeline steps completed, then
// tears the workload down. expectedImages is the number of images in the
// request; when > 0 the encoder ec_transfer_params count is also verified.
func runCoordinatorPipeline(body []byte, expectedSteps []string, expectedImages int) {
	var (
		coordinator  []string
		modelServers []string
		decodeEPP    []string
		prefillEPP   []string
		encodeEPP    []string
		decodePool   []string
		prefillPool  []string
		encodePool   []string
	)

	// Registered first → runs last (LIFO), after the log dump below.
	ginkgo.DeferCleanup(func() {
		if keepClusterOnFailure && ginkgo.CurrentSpecReport().Failed() {
			return
		}
		testutils.DeleteObjects(testConfig, coordinator)
		testutils.DeleteObjects(testConfig, modelServers)
		testutils.DeleteObjects(testConfig, decodeEPP)
		testutils.DeleteObjects(testConfig, prefillEPP)
		testutils.DeleteObjects(testConfig, encodeEPP)
		testutils.DeleteObjects(testConfig, decodePool)
		testutils.DeleteObjects(testConfig, prefillPool)
		testutils.DeleteObjects(testConfig, encodePool)
	})

	// Dump coordinator logs on failure, or always when E2E_PRINT_COORDINATOR_LOGS is
	// set. Registered second → runs first (LIFO), so the deployment still exists.
	ginkgo.DeferCleanup(func() {
		if !ginkgo.CurrentSpecReport().Failed() && !printCoordinatorLogs {
			return
		}
		args := []string{"logs", "deployment/llm-d-coordinator",
			"-c", "coordinator", "--namespace=" + nsName}
		if k8sContext != "" {
			args = append(args, "--context="+k8sContext)
		}
		out, err := exec.Command("kubectl", args...).CombinedOutput()
		if err != nil {
			fmt.Fprintf(ginkgo.GinkgoWriter, "\n--- coordinator logs (kubectl error: %v) ---\n%s\n---\n", err, string(out))
		} else {
			fmt.Fprintf(ginkgo.GinkgoWriter, "\n--- coordinator logs ---\n%s\n---\n", string(out))
		}
	})

	// Pools first so each EPP can resolve its --pool-name.
	encodePool = createInferencePool("encode", true)
	prefillPool = createInferencePool("prefill", true)
	decodePool = createInferencePool("decode", true)
	expectAllPoolsExist()

	encodeEPP = createEndPointPicker("encode", encodeEPPConfig)
	prefillEPP = createEndPointPicker("prefill", prefillEPPConfig)
	decodeEPP = createEndPointPicker("decode", decodeEPPConfig)

	encodeReplicas, prefillReplicas, decodeReplicas := 1, 1, 1
	modelServers = createModelServers(encodeReplicas, prefillReplicas, decodeReplicas)

	encodePods := getPodNames(encodeSelector)
	prefillPods := getPodNames(prefillSelector)
	decodePods := getPodNames(decodeSelector)
	gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
	gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
	gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

	coordinator = createCoordinator(coordinatorConfigNIXL)

	req, err := http.NewRequest(http.MethodPost,
		gatewayBaseURL+"/v1/chat/completions",
		bytes.NewReader(body))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
		"coordinator returned non-200: body=%s", string(raw))
	gomega.Expect(raw).NotTo(gomega.BeEmpty(), "coordinator returned empty body")

	verifyCoordinatorSteps(expectedSteps, expectedImages)
}

// verifyCoordinatorSteps fetches the coordinator pod logs and asserts that
// every expected step has a "step complete" log entry. When expectedImages > 0,
// it also verifies that the encode step reported the correct number of
// ec_transfer_params entries via the "merged encode response" total.
func verifyCoordinatorSteps(expectedSteps []string, expectedImages int) {
	ginkgo.By("Verifying coordinator logs contain all pipeline steps")

	args := []string{"logs", "deployment/llm-d-coordinator",
		"-c", "coordinator", "--namespace=" + nsName}
	if k8sContext != "" {
		args = append(args, "--context="+k8sContext)
	}

	out, err := exec.Command("kubectl", args...).CombinedOutput()
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred(),
		"failed to fetch coordinator logs: %s", string(out))

	logs := string(out)
	for _, step := range expectedSteps {
		stepField := `"step":"` + step + `"`

		// Verify the step was actually completed (not just started).
		completed := false
		for _, line := range strings.Split(logs, "\n") {
			if strings.Contains(line, `"msg":"step complete"`) &&
				strings.Contains(line, stepField) {
				completed = true
				break
			}
		}
		gomega.Expect(completed).To(gomega.BeTrue(),
			"coordinator logs have no 'step complete' entry for step %q", step)
	}

	if expectedImages > 0 {
		ginkgo.By("Verifying encode returned ec_transfer_params for all images")
		// The encoder logs "merged encode response","total":<N> as it
		// accumulates ec_transfer_params from each sub-request, and
		// "all sub-requests complete","count":<N> when done. Match msg and the
		// count field independently per line: structured-log fields such as
		// x-request-id sit between them, so a contiguous substring would miss.
		mergedMarker := `"msg":"merged encode response"`
		totalField := fmt.Sprintf(`"total":%d`, expectedImages)
		merged := false
		for _, line := range strings.Split(logs, "\n") {
			if strings.Contains(line, mergedMarker) &&
				strings.Contains(line, totalField) {
				merged = true
				break
			}
		}
		gomega.Expect(merged).To(gomega.BeTrue(),
			"coordinator logs missing merged encode response with total=%d", expectedImages)

		countMarker := `"msg":"all sub-requests complete"`
		countField := fmt.Sprintf(`"count":%d`, expectedImages)
		found := false
		for _, line := range strings.Split(logs, "\n") {
			if strings.Contains(line, countMarker) &&
				strings.Contains(line, countField) {
				found = true
				break
			}
		}
		gomega.Expect(found).To(gomega.BeTrue(),
			"coordinator logs missing 'all sub-requests complete' with count=%d", expectedImages)
	}
}
