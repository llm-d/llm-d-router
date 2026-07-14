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

	ginkgo.It("routes a multimodal chat completion with an inline base64 image end-to-end", func() {
		runCoordinatorPipeline([]byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"max_tokens":150}`,
			modelName, inlineImageDataURI,
		)), allSteps, 1)
	})

	ginkgo.It("routes a multimodal chat completion with one inline and one remote image end-to-end", func() {
		runCoordinatorPipeline([]byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"image_url","image_url":{"url":%q},"uuid":"image-1"},{"type":"text","text":"Describe what you see in both images."}]}],"max_tokens":150}`,
			modelName, inlineImageDataURI, testImageURL,
		)), allSteps, 2)
	})

})

// inlineImageDataURI is a 64x64 solid-color PNG encoded as a base64 data URI.
// It exercises the inline data: branch of replace-media-urls, which the
// remote-URL specs never reach, while still flowing through encode/prefill/decode.
const inlineImageDataURI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAaUlEQVR4nOzPUQkAIRQAweMwx+sfxViG8GMQdhLsrj3zvezXAbca0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0E4AAAD//9Q1AYfjlntsAAAAAElFTkSuQmCC"

// runCoordinatorPipeline deploys the e-p-d topology and coordinator, posts the
// given chat-completion body, asserts a 200 with a non-empty body, verifies
// that the coordinator logs show all expected pipeline steps completed, then
// tears the workload down. expectedImages is the number of images in the
// request; when > 0 the encoder log assertions are also verified.
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
		testutils.DeleteObjects(testConfig, coordinator, nsName)
		testutils.DeleteObjects(testConfig, modelServers, nsName)
		testutils.DeleteObjects(testConfig, decodeEPP, nsName)
		testutils.DeleteObjects(testConfig, prefillEPP, nsName)
		testutils.DeleteObjects(testConfig, encodeEPP, nsName)
		testutils.DeleteObjects(testConfig, decodePool, nsName)
		testutils.DeleteObjects(testConfig, prefillPool, nsName)
		testutils.DeleteObjects(testConfig, encodePool, nsName)
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
	verifyCoordinatorSteps(expectedSteps, expectedImages, true, true)
}

// verifyCoordinatorSteps fetches the coordinator pod logs and asserts that
// every expected step has a "step complete" log entry. When kvNIXL is set it
// also asserts kv_transfer_params on the prefill and decode legs, and when
// expectedImages > 0 it asserts the encode step completed all image
// sub-requests, plus, when ecNIXL is set, the ec_transfer_params total via the
// "merged encode response" marker. The kv/ec params surface in the logs only
// for the NIXL connectors, so those checks are gated accordingly.
func verifyCoordinatorSteps(expectedSteps []string, expectedImages int, kvNIXL, ecNIXL bool) {
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
		gomega.Expect(logHasLine(logs, `"msg":"step complete"`, stepField)).To(gomega.BeTrue(),
			"coordinator logs have no 'step complete' entry for step %q", step)
	}

	if kvNIXL {
		// kv_transfer_params surfaces on three legs of the NIXL handshake.
		// Prefill request: the coordinator forwards kv_transfer_params in the
		// outgoing prefill request body (gateway/client.go "request body" trace).
		// Prefill response: the prefill server returns kv_transfer_params with
		// do_remote_prefill=true; use that field to distinguish the prefill
		// response from encode responses, which carry "kv_transfer_params":null.
		// Decode: the request goes out via a reverse proxy the gateway client
		// never sees, so the kv connector's "preparing decode kv params" trace,
		// which always sets do_remote_prefill=true, is where the decode leg surfaces.
		ginkgo.By("Verifying kv_transfer_params forwarded on the prefill request")
		gomega.Expect(logHasLine(logs, `"msg":"request body"`, `"epp-phase":"prefill"`, `"kv_transfer_params"`)).To(gomega.BeTrue(),
			"coordinator logs have no prefill request body carrying kv_transfer_params")

		ginkgo.By("Verifying kv_transfer_params in the prefill response")
		gomega.Expect(logHasLine(logs, `"msg":"response body"`, `"do_remote_prefill":true`)).To(gomega.BeTrue(),
			"coordinator logs have no prefill response body carrying kv_transfer_params with do_remote_prefill=true")

		ginkgo.By("Verifying kv_transfer_params on the decode leg")
		gomega.Expect(logHasLine(logs, `"msg":"preparing decode kv params"`, `"do_remote_prefill":true`)).To(gomega.BeTrue(),
			"coordinator logs have no decode kv_transfer_params with do_remote_prefill=true")
	}

	if expectedImages > 0 {
		// The encode step fans out one sub-request per image; this marker is
		// logged by the step itself, so it holds for any ec connector.
		ginkgo.By("Verifying encode completed all image sub-requests")
		gomega.Expect(logHasLine(logs, `"msg":"all sub-requests complete"`, fmt.Sprintf(`"count":%d`, expectedImages))).To(gomega.BeTrue(),
			"coordinator logs missing 'all sub-requests complete' with count=%d", expectedImages)

		if ecNIXL {
			// The NIXL ec connector merges one ec_transfer_params entry per image
			// ("merged encode response","total":N), then the merged set is carried
			// on the prefill request body.
			ginkgo.By("Verifying ec_transfer_params merged for all images")
			gomega.Expect(logHasLine(logs, `"msg":"merged encode response"`, fmt.Sprintf(`"total":%d`, expectedImages))).To(gomega.BeTrue(),
				"coordinator logs missing merged encode response with total=%d", expectedImages)

			ginkgo.By("Verifying ec_transfer_params forwarded on the prefill request")
			gomega.Expect(logHasLine(logs, `"msg":"request body"`, `"epp-phase":"prefill"`, `"ec_transfer_params"`)).To(gomega.BeTrue(),
				"coordinator logs have no prefill request body carrying ec_transfer_params")
		}
	}
}

// logHasLine reports whether any single line in logs contains all of substrs.
func logHasLine(logs string, substrs ...string) bool {
	for _, line := range strings.Split(logs, "\n") {
		matched := true
		for _, s := range substrs {
			if !strings.Contains(line, s) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}
