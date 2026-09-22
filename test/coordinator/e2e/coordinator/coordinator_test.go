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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
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
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`,
			modelName,
		)), textOnlySteps, 0, tokenLimits{})
	})

	ginkgo.It("forwards the client token limits to decode and caps them on prefill and encode", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"min_tokens":3,"max_tokens":5,"max_completion_tokens":100}`,
			modelName, inlineImageDataURI,
		)), allSteps, 1, tokenLimits{min: 3, max: 5, maxCompletion: 100})
	})

	// Passthrough disabled collapses the chat request to the generate wire format
	// on the encode and prefill requests.
	ginkgo.It("forwards the client token limits to decode and caps them on prefill and encode with OpenAI passthrough disabled", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"min_tokens":3,"max_tokens":5,"max_completion_tokens":100}`,
			modelName, inlineImageDataURI,
		)), allSteps, 1, tokenLimits{min: 3, max: 5, maxCompletion: 100}, coordinatorConfigNIXLGenerate)
	})

	ginkgo.It("routes a multimodal image chat completion end-to-end", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"max_tokens":150}`,
			modelName, testImageURL,
		)), allSteps, 1, tokenLimits{})
	})

	ginkgo.It("routes a multimodal chat completion with two images end-to-end", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"image_url","image_url":{"url":%q},"uuid":"image-1"},{"type":"text","text":"What is in these two images?"}]}],"max_tokens":150}`,
			modelName, testImageURL, testImageURL2,
		)), allSteps, 2, tokenLimits{})
	})

	ginkgo.It("routes a multimodal chat completion with an inline base64 image end-to-end", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"text","text":"Describe what you see."}]}],"max_tokens":150}`,
			modelName, inlineImageDataURI,
		)), allSteps, 1, tokenLimits{})
	})

	ginkgo.It("routes a multimodal chat completion with one inline and one remote image end-to-end", func() {
		runCoordinatorPipeline(reqcommon.PathChatCompletions, []byte(fmt.Sprintf(
			`{"model":%q,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q},"uuid":"image-0"},{"type":"image_url","image_url":{"url":%q},"uuid":"image-1"},{"type":"text","text":"Describe what you see in both images."}]}],"max_tokens":150}`,
			modelName, inlineImageDataURI, testImageURL,
		)), allSteps, 2, tokenLimits{})
	})

})

// inlineImageDataURI is a 64x64 solid-color PNG encoded as a base64 data URI.
// It exercises the inline data: branch of replace-media-urls, which the
// remote-URL specs never reach, while still flowing through encode/prefill/decode.
const inlineImageDataURI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAaUlEQVR4nOzPUQkAIRQAweMwx+sfxViG8GMQdhLsrj3zvezXAbca0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0E4AAAD//9Q1AYfjlntsAAAAAElFTkSuQmCC"

// tokenLimits are the output limits a request sends. A zero field means the
// request omits that limit, so its value is not asserted on the decode request.
type tokenLimits struct{ min, max, maxCompletion int }

// runCoordinatorPipeline deploys the e-p-d topology and coordinator, posts the
// given body to path (e.g. /v1/chat/completions or /inference/v1/generate),
// asserts a 200 with a non-empty body, verifies that the coordinator logs show
// all expected pipeline steps completed, then tears the workload down.
// expectedImages is the number of images in the request; when > 0 the encoder
// log assertions are also verified. A non-zero limits.max additionally asserts
// the token-limit contract described on verifyTokenLimits.
// coordinatorConfig, when supplied, overrides the default coordinatorConfigNIXL
// pipeline config for the coordinator deployment.
func runCoordinatorPipeline(path string, body []byte, expectedSteps []string, expectedImages int, limits tokenLimits, coordinatorConfig ...string) {
	nsName := getNamespace()
	var (
		coordinator  []string
		modelServers []string
		epp          []string
		pool         []string
	)

	// Registered first → runs last (LIFO), after the log dump below.
	ginkgo.DeferCleanup(func() {
		if keepClusterOnFailure && ginkgo.CurrentSpecReport().Failed() {
			return
		}
		testutils.DeleteObjects(testConfig, coordinator, nsName)
		testutils.DeleteObjects(testConfig, modelServers, nsName)
		testutils.DeleteObjects(testConfig, epp, nsName)
		testutils.DeleteObjects(testConfig, pool, nsName)
	})

	// Dump all pod logs (coordinator, EPPs, Envoy, workers) on failure, or always
	// when E2E_PRINT_LOGS is set. Registered second → runs first (LIFO), so the
	// pods still exist.
	ginkgo.DeferCleanup(func() {
		if !ginkgo.CurrentSpecReport().Failed() && !printLogs {
			return
		}
		testutils.DumpPodsAndLogs(testConfig, nsName, testutils.WithFullLogs())
	})

	// Pool first so the EPP can resolve its --pool-name.
	pool = createInferencePool(true)
	expectPoolExists()

	epp = createEndPointPickers()

	encodeReplicas, prefillReplicas, decodeReplicas := 1, 1, 1
	modelServers = createModelServers(encodeReplicas, prefillReplicas, decodeReplicas)

	encodePods := getPodNames(encodeSelector)
	prefillPods := getPodNames(prefillSelector)
	decodePods := getPodNames(decodeSelector)
	gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
	gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
	gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

	cfg := coordinatorConfigNIXL
	if len(coordinatorConfig) > 0 {
		cfg = coordinatorConfig[0]
	}
	coordinator = createCoordinator(cfg)

	req, err := http.NewRequest(http.MethodPost,
		gatewayBaseURL()+path,
		bytes.NewReader(body))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	req.Header.Set("Content-Type", "application/json")
	// Envoy's pipeline listener preserves a client-supplied x-request-id
	// (preserve_external_request_id) and the coordinator propagates it to every
	// pipeline request, so a unique id here scopes the per-role routing check to this
	// one request. Envoy is shared infra whose access log accumulates across specs,
	// so an unscoped parse would match earlier specs' requests on since-recycled IPs.
	reqID := uuid.NewString()
	req.Header.Set("X-Request-Id", reqID)

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
		"coordinator returned non-200: body=%s", string(raw))
	gomega.Expect(raw).NotTo(gomega.BeEmpty(), "coordinator returned empty body")

	logs := fetchCoordinatorLogs(nsName)
	verifyCoordinatorSteps(logs, expectedSteps, expectedImages, true, true)
	// The native generate leg, and a chat leg with OpenAI passthrough disabled,
	// speak the generate wire format. On that leg the transfer params sit at the
	// top level of the request body, not nested under sampling_params.extra_args.
	// verifyCoordinatorSteps only substring-matches the prefill body and would
	// still pass with the params nested, so this parses the structured http_body
	// to pin the field's location.
	if path == reqcommon.PathVLLMGenerate || cfg == coordinatorConfigNIXLGenerate {
		verifyToplevelTransferParams(logs)
	}
	if threeEPP {
		verifyPerRoleRouting(nsName, slices.Contains(expectedSteps, "encode"), reqID)
	}
	if limits.max > 0 {
		// Prefill is always capped; encode is capped only when the request carries
		// images, since it fires one sub-request per image.
		capSteps := []string{"prefill"}
		if expectedImages > 0 {
			capSteps = append(capSteps, "encode")
		}
		// Mirrors resolveFormat: the pipeline sends generate for a native generate
		// request and for a chat request with passthrough disabled, chat otherwise.
		requestsSpeakChat := path != reqcommon.PathVLLMGenerate && cfg != coordinatorConfigNIXLGenerate
		verifyTokenLimits(logs, limits, requestsSpeakChat, capSteps)
	}
}

// verifyCoordinatorSteps asserts that the coordinator logs show every expected
// step has a "step complete" log entry. When kvNIXL is set it
// also asserts kv_transfer_params on the prefill and decode requests, and when
// expectedImages > 0 it asserts the encode step completed all image
// sub-requests, plus, when ecNIXL is set, the ec_transfer_params total via the
// "merged encode response" marker. The kv/ec params surface in the logs only
// for the NIXL connectors, so those checks are gated accordingly.
func verifyCoordinatorSteps(logs string, expectedSteps []string, expectedImages int, kvNIXL, ecNIXL bool) {
	ginkgo.By("Verifying coordinator logs contain all pipeline steps")

	for _, step := range expectedSteps {
		stepField := `"step":"` + step + `"`
		gomega.Expect(logHasLine(logs, `"body":"step complete"`, stepField)).To(gomega.BeTrue(),
			"coordinator logs have no 'step complete' entry for step %q", step)
	}

	if kvNIXL {
		// kv_transfer_params surfaces in three places in the NIXL handshake.
		// Prefill request: the coordinator forwards kv_transfer_params in the
		// outgoing prefill request body (gateway/client.go "request body" trace).
		// Prefill response: the prefill server returns kv_transfer_params with
		// do_remote_prefill=true; use that field to distinguish the prefill
		// response from encode responses, which carry "kv_transfer_params":null.
		// Decode: the request goes out via a reverse proxy the gateway client
		// never sees, so the kv connector's "preparing decode kv params" trace,
		// which always sets do_remote_prefill=true, is where the decode request surfaces.
		ginkgo.By("Verifying kv_transfer_params forwarded on the prefill request")
		gomega.Expect(logHasLine(logs, `"body":"request body"`, `"epp-profile":"prefill"`, `"kv_transfer_params"`)).To(gomega.BeTrue(),
			"coordinator logs have no prefill request body carrying kv_transfer_params")

		ginkgo.By("Verifying kv_transfer_params in the prefill response")
		gomega.Expect(logHasLine(logs, `"body":"response body"`, `"do_remote_prefill":true`)).To(gomega.BeTrue(),
			"coordinator logs have no prefill response body carrying kv_transfer_params with do_remote_prefill=true")

		ginkgo.By("Verifying kv_transfer_params on the decode request")
		gomega.Expect(logHasLine(logs, `"body":"preparing decode kv params"`, `"do_remote_prefill":true`)).To(gomega.BeTrue(),
			"coordinator logs have no decode kv_transfer_params with do_remote_prefill=true")
	}

	if expectedImages > 0 {
		// The encode step fans out one sub-request per image; this marker is
		// logged by the step itself, so it holds for any ec connector.
		ginkgo.By("Verifying encode completed all image sub-requests")
		gomega.Expect(logHasLine(logs, `"body":"all sub-requests complete"`, fmt.Sprintf(`"count":%d`, expectedImages))).To(gomega.BeTrue(),
			"coordinator logs missing 'all sub-requests complete' with count=%d", expectedImages)

		if ecNIXL {
			// The NIXL ec connector merges one ec_transfer_params entry per image
			// ("merged encode response","total":N), then the merged set is carried
			// on the prefill request body.
			ginkgo.By("Verifying ec_transfer_params merged for all images")
			gomega.Expect(logHasLine(logs, `"body":"merged encode response"`, fmt.Sprintf(`"total":%d`, expectedImages))).To(gomega.BeTrue(),
				"coordinator logs missing merged encode response with total=%d", expectedImages)

			ginkgo.By("Verifying ec_transfer_params forwarded on the prefill request")
			gomega.Expect(logHasLine(logs, `"body":"request body"`, `"epp-profile":"prefill"`, `"ec_transfer_params"`)).To(gomega.BeTrue(),
				"coordinator logs have no prefill request body carrying ec_transfer_params")
		}
	}
}

// verifyToplevelTransferParams pins the generate leg's transfer-param location
// from the coordinator's own log output: the prefill request body carries
// kv_transfer_params at the top level and its sampling_params has no
// extra_args, so a regression that nests the params under
// sampling_params.extra_args fails. verifyCoordinatorSteps only substring-
// matches the prefill body, which still passes with the params nested, so this
// parses the structured http_body the gateway logs at TRACE (log_level 5).
func verifyToplevelTransferParams(logs string) {
	ginkgo.By("Verifying the generate prefill body carries top-level kv_transfer_params")
	body := parsePrefillGenerateBody(logs)

	kv, ok := body["kv_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"generate prefill body has no top-level kv_transfer_params object: %v", body)
	gomega.Expect(kv).NotTo(gomega.BeEmpty(),
		"generate prefill body top-level kv_transfer_params is empty: %v", body)

	sampling, ok := body["sampling_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"generate prefill body has no sampling_params object: %v", body)
	gomega.Expect(sampling).NotTo(gomega.HaveKey("extra_args"),
		"generate prefill body nests transfer params under sampling_params.extra_args: %v", body)
}

// parsePrefillGenerateBody returns the prefill leg's outgoing request body from
// the coordinator log: the gateway's TRACE "request body" record whose
// epp-profile is prefill, with its http_body field decoded. The body is the
// redacted map the gateway logs (long strings collapsed), which keeps every
// structural field this check reads.
func parsePrefillGenerateBody(logs string) map[string]any {
	prefillLine := ""
	for _, line := range strings.Split(logs, "\n") {
		if strings.Contains(line, `"epp-profile":"prefill"`) && strings.Contains(line, `"http_body":{`) {
			prefillLine = line
			break
		}
	}
	gomega.Expect(prefillLine).NotTo(gomega.BeEmpty(),
		"coordinator logs have no prefill request body carrying an http_body field (is the coordinator at log_level 5?)")

	bodyJSON := extractJSONObject(prefillLine, `"http_body":`)
	var body map[string]any
	gomega.Expect(json.Unmarshal([]byte(bodyJSON), &body)).To(gomega.Succeed(),
		"prefill http_body is not valid JSON: %s", bodyJSON)
	return body
}

// extractJSONObject returns the JSON object that follows key (e.g.
// `"http_body":`) in s. It locates key anywhere in s and brace-scans forward
// with awareness of string literals, so braces inside quoted values and any
// fields logged after the object do not confuse the result.
func extractJSONObject(s, key string) string {
	i := strings.Index(s, key)
	gomega.Expect(i).To(gomega.BeNumerically(">=", 0), "log line has no %s field: %s", key, s)
	start := i + len(key)
	for start < len(s) && s[start] != '{' {
		start++
	}
	gomega.Expect(start).To(gomega.BeNumerically("<", len(s)), "no JSON object follows %s: %s", key, s)

	depth := 0
	inString, escaped := false, false
	for j := start; j < len(s); j++ {
		c := s[j]
		switch {
		case escaped:
			escaped = false
		case inString:
			switch c {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
		case c == '"':
			inString = true
		case c == '{':
			depth++
		case c == '}':
			depth--
		}
		if depth == 0 {
			return s[start : j+1]
		}
	}
	return s[start:]
}

// verifyPerRoleRouting asserts the Envoy EPP-Profile dispatch (envoy-3-epp.yaml)
// delivered each pipeline request to a worker of its own role. The Envoy access log
// records the EPP-Profile value and the upstream pod for every request, so each
// request must land on a pod whose role matches its profile. This is echo-mode
// independent and catches a misrouted or swapped EPP-Profile route, which a
// status-code check (all workers echo a plausible 200) cannot.
//
// reqID scopes the access-log parse to this request (see parseEnvoyProfileRoutes).
//
// The check is on where each request landed, not how many: the coordinator propagates
// the one shared request id (reqID) to every request, so the per-image encode sub-requests
// are indistinguishable in the access log. Their count is already asserted from the
// coordinator logs (see "all sub-requests complete" in verifyCoordinatorSteps);
// here we require the encode profile to appear only when the pipeline runs the
// encode step (expectEncode), and prefill and decode to always appear. The
// generate path carries images but encodes inline on prefill, so it runs no
// encode request despite the images.
func verifyPerRoleRouting(nsName string, expectEncode bool, reqID string) {
	ginkgo.By("Verifying each pipeline request was routed to its own role's worker")

	roleIPs := map[string]map[string]bool{}
	for _, e := range eppsToCreate() {
		roleIPs[e.role] = podIPs(roleSelector(e.role))
	}

	// Envoy flushes access logs asynchronously, so poll until every expected role
	// request is recorded before asserting where each was routed.
	gomega.Eventually(func(g gomega.Gomega) {
		routes := parseEnvoyProfileRoutes(fetchDeploymentLogs(nsName, "envoy", "envoy"), roleIPs, reqID)
		for _, e := range eppsToCreate() {
			role := e.role
			if role == "encode" && !expectEncode {
				g.Expect(routes[role]).To(gomega.BeEmpty(),
					"request produced an encode request in the Envoy access log but the pipeline runs no encode step")
				continue
			}
			g.Expect(routes[role]).ToNot(gomega.BeEmpty(),
				"no %s request recorded in the Envoy access log", role)
			for _, upstream := range routes[role] {
				g.Expect(roleIPs[role]).To(gomega.HaveKey(upstream),
					"%s request routed to upstream %s, not a %s-role pod; EPP-Profile routing is wrong",
					role, upstream, role)
			}
		}
	}, readyTimeout, defaultInterval).Should(gomega.Succeed())
}

// parseEnvoyProfileRoutes extracts the per-role pipeline requests from the Envoy access
// log (format defined in envoy-3-epp.yaml). It returns, per EPP-Profile value in
// roles, the upstream pod IPs Envoy routed those requests to. Only lines carrying
// reqID are considered: Envoy is shared across specs and its access log
// accumulates, so scoping by request id keeps earlier specs' requests (on
// since-recycled pod IPs) out. Lines whose profile is not a known role (the
// external client request and readiness probes take the default route) are ignored.
func parseEnvoyProfileRoutes(logs string, roles map[string]map[string]bool, reqID string) map[string][]string {
	out := map[string][]string{}
	for _, line := range strings.Split(logs, "\n") {
		if !strings.Contains(line, "[envoy]") {
			continue
		}
		fields := map[string]string{}
		for _, tok := range strings.Fields(line) {
			if k, v, ok := strings.Cut(tok, "="); ok {
				fields[k] = v
			}
		}
		if fields["id"] != reqID {
			continue
		}
		profile := fields["epp-profile"]
		if _, ok := roles[profile]; !ok {
			continue
		}
		// upstream is host:port; keep the IP. Envoy renders an unassigned
		// upstream as "-" (an interim flush entry for an in-flight request, or a
		// request that never reached a backend); skip those.
		upstream := fields["upstream"]
		if upstream == "" || upstream == "-" {
			continue
		}
		if i := strings.LastIndex(upstream, ":"); i >= 0 {
			upstream = upstream[:i]
		}
		out[profile] = append(out[profile], upstream)
	}
	return out
}

// fetchDeploymentLogs returns the named container's logs across all pods of the
// given Deployment.
func fetchDeploymentLogs(nsName, deployment, container string) string {
	args := []string{"logs", "deployment/" + deployment,
		"-c", container, "--namespace=" + nsName}
	if k8sContext != "" {
		args = append(args, "--context="+k8sContext)
	}

	out, err := exec.Command("kubectl", args...).CombinedOutput()
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred(),
		"failed to fetch %s logs: %s", deployment, string(out))
	return string(out)
}

// fetchCoordinatorLogs returns the coordinator container's pod logs.
func fetchCoordinatorLogs(nsName string) string {
	return fetchDeploymentLogs(nsName, "llm-d-coordinator", "coordinator")
}

// verifyTokenLimits asserts the pipeline's token-limit contract: the decode
// request forwards the client's limits unchanged, while the synthetic prefill and
// encode requests (capSteps) cap output to a single token and strip min_tokens.
// CapSingleToken writes every output cap field the request format defines, so a
// chat request always carries max_completion_tokens=1 and a generate request
// never carries the field at all, whatever the client sent. Pipeline request
// bodies surface only at TRACE, so this relies on the coordinator running at
// log_level 5.
func verifyTokenLimits(logs string, limits tokenLimits, requestsSpeakChat bool, capSteps []string) {
	ginkgo.By("Verifying decode request forwards the client min_tokens/max_tokens")
	gomega.Expect(logHasLine(logs, `"body":"request body"`, `"epp-profile":"decode"`,
		fmt.Sprintf(`"min_tokens":%d`, limits.min), fmt.Sprintf(`"max_tokens":%d`, limits.max))).To(gomega.BeTrue(),
		"coordinator logs have no decode request body carrying min_tokens=%d and max_tokens=%d", limits.min, limits.max)

	if limits.maxCompletion > 0 {
		ginkgo.By("Verifying decode request forwards the client max_completion_tokens")
		gomega.Expect(logHasLine(logs, `"body":"request body"`, `"epp-profile":"decode"`,
			fmt.Sprintf(`"max_completion_tokens":%d`, limits.maxCompletion))).To(gomega.BeTrue(),
			"coordinator logs have no decode request body carrying max_completion_tokens=%d", limits.maxCompletion)
	}

	for _, phase := range capSteps {
		phaseField := `"epp-profile":"` + phase + `"`

		ginkgo.By("Verifying " + phase + " request caps max_tokens to 1")
		gomega.Expect(logHasLine(logs, `"body":"request body"`, phaseField, `"max_tokens":1`)).To(gomega.BeTrue(),
			"coordinator logs have no %s request body carrying max_tokens=1", phase)

		ginkgo.By("Verifying " + phase + " request strips min_tokens")
		gomega.Expect(logHasLine(logs, `"body":"request body"`, phaseField, `"min_tokens"`)).To(gomega.BeFalse(),
			"%s request body must not carry min_tokens", phase)

		if requestsSpeakChat {
			ginkgo.By("Verifying " + phase + " request caps max_completion_tokens to 1")
			gomega.Expect(logHasLine(logs, `"body":"request body"`, phaseField, `"max_completion_tokens":1`)).To(gomega.BeTrue(),
				"coordinator logs have no %s request body carrying max_completion_tokens=1", phase)
		} else {
			ginkgo.By("Verifying " + phase + " request drops max_completion_tokens")
			gomega.Expect(logHasLine(logs, `"body":"request body"`, phaseField, `"max_completion_tokens"`)).To(gomega.BeFalse(),
				"%s request body must not carry max_completion_tokens on the generate wire format", phase)
		}
	}
}

// verifyEncodeSkipped asserts the coordinator skipped the encode fan-out on the
// generate path. The skip marker is logged immediately before the step returns,
// so its presence is dispositive: the prefill worker encodes inline from
// kwargs_data and no encode sub-request is issued.
func verifyEncodeSkipped(nsName string) {
	logs := fetchCoordinatorLogs(nsName)

	ginkgo.By("Verifying encode was skipped for the generate request")
	gomega.Expect(logHasLine(logs, `"body":"skipping encode for generate request"`)).To(gomega.BeTrue(),
		"coordinator logs missing 'skipping encode for generate request'")
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
