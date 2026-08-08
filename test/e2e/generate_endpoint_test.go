package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	generatePath = "/inference/v1/generate"

	requestTimeout = 60 * time.Second
)

// imageSpec describes one multimodal image entry as it appears in encode
// and prefill request bodies. Hash, Offset and Length come from the
// (mocked) render stage; tests pick fixed values so the bodies are
// deterministic.
type imageSpec struct {
	Hash   string
	Offset int
	Length int
}

// singleImage is the canonical one-image spec the basic encode/prefill
// tests share. Offset is always 1 in encode requests (right after BOS),
// length matches the placeholder span.
var singleImage = imageSpec{Hash: "e2e-image-hash", Offset: 1, Length: 3}

// twoImages is the canonical two-image spec for fan-out coverage.
// Distinct lengths so the two placeholder spans aren't accidentally identical.
var twoImages = []imageSpec{
	{Hash: "e2e-image-hash-0", Offset: 1, Length: 3},
	{Hash: "e2e-image-hash-1", Offset: 4, Length: 5},
}

var _ = ginkgo.Describe("Direct gateway /inference/v1/generate encode against encode-only", ginkgo.Ordered, testWrapper(func() {
	// Uses single-profile-handler (generateEncodeConfig) so the EPP routes
	// directly to encode pods without requiring a decode stage first.
	ginkgo.It("returns ec_transfer_params for encode bodies", func() {
		nsName := getNamespace()
		infPoolObjects := createInferencePool(1)

		encodeReplicas := 1
		modelServers := createModelServersEncodeOnly(encodeReplicas)
		epp := createEndPointPicker(generateEncodeConfig)

		encodePods := getPodNames(encodeSelector, nsName)
		gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))

		ginkgo.By("Encode_Generate: single-image encode body returns ec_transfer_params")
		{
			resp, raw := doGenerate(encodeBody(singleImage))
			parsed := expectGenerateOK(resp, raw)
			expectECTransferParams(parsed, raw)
		}

		ginkgo.By("TwoImages_Encode: per-image fan-out, every encode body returns ec_transfer_params")
		for _, img := range twoImages {
			resp, raw := doGenerate(encodeBody(img))
			parsed := expectGenerateOK(resp, raw)
			expectECTransferParams(parsed, raw)
		}

		testutils.DeleteObjects(testConfig, epp, nsName)
		testutils.DeleteObjects(testConfig, modelServers, nsName)
		testutils.DeleteObjects(testConfig, infPoolObjects, nsName)
	})
}))

var _ = ginkgo.Describe("Direct gateway /inference/v1/generate prefill against prefill-only", ginkgo.Ordered, testWrapper(func() {
	// Uses single-profile-handler (generatePrefillConfig) so the EPP routes
	// directly to prefill pods without requiring a decode stage first.
	ginkgo.It("returns kv_transfer_params for prefill bodies", func() {
		nsName := getNamespace()
		infPoolObjects := createInferencePool(1)

		prefillReplicas := 1
		modelServers := createModelServersPrefillOnly(prefillReplicas)
		epp := createEndPointPicker(generatePrefillConfig)

		prefillPods := getPodNames(prefillSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))

		ginkgo.By("TwoImages_Prefill: combined two-image prefill body returns kv_transfer_params")
		{
			tokenIDs := []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789}
			resp, raw := doGenerate(prefillBody(tokenIDs, twoImages))
			parsed := expectGenerateOK(resp, raw)
			expectKVTransferParams(parsed, raw)
		}

		testutils.DeleteObjects(testConfig, epp, nsName)
		testutils.DeleteObjects(testConfig, modelServers, nsName)
		testutils.DeleteObjects(testConfig, infPoolObjects, nsName)
	})
}))

// imageFeatures builds the features map that encode and prefill share:
// mm_hashes, mm_placeholders, kwargs_data, all keyed by modality
// ("image"). kwargs_data is a placeholder "AA==" per entry; the
// simulator does not validate it.
func imageFeatures(images []imageSpec) map[string]any {
	hashes := make([]string, len(images))
	placeholders := make([]map[string]any, len(images))
	kwargs := make([]string, len(images))
	for i, img := range images {
		hashes[i] = img.Hash
		placeholders[i] = map[string]any{"offset": img.Offset, "length": img.Length}
		kwargs[i] = "AA=="
	}
	return map[string]any{
		"mm_hashes":       map[string]any{"image": hashes},
		"mm_placeholders": map[string]any{"image": placeholders},
		"kwargs_data":     map[string]any{"image": kwargs},
	}
}

// encodeBody builds a single-image encode request.
// token_ids = [BOS, placeholder*length]; offset is always 1 since each
// encode carries exactly one image.
func encodeBody(image imageSpec) []byte {
	tokenIDs := make([]int, 1+image.Length)
	tokenIDs[0] = 1
	for i := 1; i < len(tokenIDs); i++ {
		tokenIDs[i] = 32000
	}
	body := map[string]any{
		"model":           simModelName,
		"token_ids":       tokenIDs,
		"features":        imageFeatures([]imageSpec{{Hash: image.Hash, Offset: 1, Length: image.Length}}),
		"sampling_params": map[string]any{"max_tokens": 1},
	}
	return mustMarshal(body)
}

// ecTransferEntries builds the per-image entries of
// prefill.ec_transfer_params.image, one map per image keyed by mm_hash.
// Values are dummy NIXL transfer params; the simulator does not
// validate.
func ecTransferEntries(images []imageSpec) []map[string]any {
	entries := make([]map[string]any, len(images))
	for i, img := range images {
		entries[i] = map[string]any{
			img.Hash: map[string]any{
				"peer_host":               "10.0.0.1",
				"peer_port":               5501 + i,
				"size_bytes":              0,
				"nixl_agent_metadata_b64": "",
			},
		}
	}
	return entries
}

// prefillBody builds a prefill request covering every image in one body.
func prefillBody(tokenIDs []int, images []imageSpec) []byte {
	body := map[string]any{
		"request_id":         "e2e-prefill-" + uuid.NewString(),
		"model":              simModelName,
		"token_ids":          tokenIDs,
		"features":           imageFeatures(images),
		"ec_transfer_params": map[string]any{"image": ecTransferEntries(images)},
		"sampling_params": map[string]any{
			"max_tokens": 1,
			"extra_args": map[string]any{
				"kv_transfer_params": map[string]any{"do_remote_decode": true},
			},
		},
	}
	return mustMarshal(body)
}

// doRequest POSTs body to <gateway><path>. Always sets Content-Type and a
// fresh X-Request-ID. Returns the live *http.Response (its body is already
// drained) and the raw response body.
func doRequest(path string, body []byte) (*http.Response, []byte) {
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%d%s", getPort(), path), bytes.NewReader(body))
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "build POST request")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", uuid.NewString())

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "POST %s", req.URL)
	defer func() {
		gomega.Expect(resp.Body.Close()).To(gomega.Succeed())
	}()
	raw, err := io.ReadAll(resp.Body)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "read response body")
	return resp, raw
}

// doGenerate is a thin wrapper over doRequest targeting /inference/v1/generate.
func doGenerate(body []byte) (*http.Response, []byte) {
	return doRequest(generatePath, body)
}

// expectGenerateOK asserts a 2xx status and that the response body parses
// as a JSON object, returning the parsed map for further phase-specific
// assertions.
func expectGenerateOK(resp *http.Response, raw []byte) map[string]any {
	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
		"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))
	var parsed map[string]any
	gomega.Expect(json.Unmarshal(raw, &parsed)).To(gomega.Succeed(),
		"response is not valid JSON: %s", string(raw))
	return parsed
}

// expectECTransferParams asserts the encode response carries a non-empty
// ec_transfer_params object keyed by mm_hash.
func expectECTransferParams(parsed map[string]any, raw []byte) {
	ec, ok := parsed["ec_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed ec_transfer_params in encode response: %s", string(raw))
	gomega.Expect(ec).NotTo(gomega.BeEmpty(),
		"ec_transfer_params is empty: %s", string(raw))
}

// expectKVTransferParams asserts the prefill response carries a non-empty
// kv_transfer_params object (handoff metadata for the decode worker).
func expectKVTransferParams(parsed map[string]any, raw []byte) {
	kv, ok := parsed["kv_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed kv_transfer_params in prefill response: %s", string(raw))
	gomega.Expect(kv).NotTo(gomega.BeEmpty(),
		"kv_transfer_params is empty: %s", string(raw))
}

var _ = ginkgo.Describe("P/D gateway /inference/v1/generate disaggregates via sidecar", ginkgo.Ordered, testWrapper(func() {
	// Regression test for https://github.com/llm-d/llm-d-router/issues/1461:
	// the pd-sidecar previously had no route for /inference/v1/generate, so
	// token-in P/D requests silently fell through to decode-only. This test
	// verifies that the prefill pod receives the generate request when a full
	// P/D setup is used, proving the sidecar routes it through
	// disaggregatedPrefillHandler rather than the decoder catch-all.
	ginkgo.It("routes token-in generate to the prefill pod", func() {
		nsName := getNamespace()
		infPoolObjects := createInferencePool(1)

		prefillReplicas := 1
		decodeReplicas := 1
		modelServers := createModelServersPDSharedStorage(decodeReplicas)
		epp := createEndPointPicker(pdConfig)

		prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
		gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

		prefillCountBefore := getPodRequestCount(nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count before test: %d", prefillCountBefore))

		ginkgo.By("sending /inference/v1/generate through P/D EPP")
		resp, raw := doGenerate(simpleTokenGenerateBody())
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK),
			"non-200 from gateway: status=%d body=%s", resp.StatusCode, string(raw))

		prefillCountAfter := getPodRequestCount(nsName, prefillPods[0])
		ginkgo.By(fmt.Sprintf("prefill request count after test: %d", prefillCountAfter))

		gomega.Expect(prefillCountAfter).To(gomega.BeNumerically(">", prefillCountBefore),
			"prefill pod should have received the generate request; sidecar must route "+
				"/inference/v1/generate through disaggregatedPrefillHandler, not the decoder catch-all")

		testutils.DeleteObjects(testConfig, epp, nsName)
		testutils.DeleteObjects(testConfig, modelServers, nsName)
		testutils.DeleteObjects(testConfig, infPoolObjects, nsName)
	})
}))

// simpleTokenGenerateBody builds a minimal /inference/v1/generate body with
// enough token IDs to exceed the prefix-based-pd-decider nonCachedTokens
// threshold (16) and guarantee P/D routing on a cold cache.
func simpleTokenGenerateBody() []byte {
	tokenIDs := make([]int, 20)
	for i := range tokenIDs {
		tokenIDs[i] = 1000 + i
	}
	return mustMarshal(map[string]any{
		"model":     simModelName,
		"token_ids": tokenIDs,
		"sampling_params": map[string]any{
			"max_tokens": 1,
		},
	})
}

func mustMarshal(v any) []byte {
	b, err := json.Marshal(v)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "json.Marshal failed: %v", v)
	return b
}
