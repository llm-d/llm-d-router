/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package epd_pools

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

const (
	// generatePath is the EPP gateway path used by the coordinator's encode
	// (Stage 4 Option A) and prefill (Stage 5 Option A) requests.
	generatePath = "/inference/v1/generate"

	// requestTimeout caps each direct gateway POST. The simulator backends
	// respond in well under a second; this timeout is generous to absorb
	// gateway/EPP startup jitter.
	requestTimeout = 60 * time.Second
)

// roleLabel selects which phase pool a pod belongs to. Matches the selector
// on each InferencePool (deploy/components/inference-gateway/epd-pools/
// <phase>/inference-pool.yaml). The HTTPRoute EPP-Phase header values use
// the same role strings, so the routing target and the served pod can be
// validated against the same source.
const (
	roleLabel    = bylabel.RoleLabel
	phaseEncode  = bylabel.RoleEncode
	phasePrefill = bylabel.RolePrefill
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
// tests share. Offset/length match communication.md Stage 4 ("offset is
// always 1 in encode requests, length matches the placeholder span").
var singleImage = imageSpec{Hash: "e2e-image-hash", Offset: 1, Length: 3}

// twoImages is the canonical two-image spec for fan-out coverage.
// Offsets follow communication.md's Stage 5 example layout
// ([BOS, img0×3, img1×3, text×2]).
var twoImages = []imageSpec{
	{Hash: "e2e-image-hash-0", Offset: 1, Length: 3},
	{Hash: "e2e-image-hash-1", Offset: 4, Length: 3},
}

var _ = ginkgo.Describe("EPD pools direct gateway", func() {
	ginkgo.It("Encode_Generate: routes /inference/v1/generate with EPP-Phase=encode to the encode pool", func() {
		body := encodeBody(singleImage)
		resp, raw := doGenerate(body, phaseEncode)
		parsed := expectGenerateOK(resp, raw)
		expectServedByRole(resp, phaseEncode)
		expectECTransferParams(parsed, raw)
	})

	ginkgo.It("Prefill_Generate: routes /inference/v1/generate with EPP-Phase=prefill to the prefill pool", func() {
		body := prefillBody([]int{1, 32000, 32000, 32000, 2345, 6789}, []imageSpec{singleImage})
		resp, raw := doGenerate(body, phasePrefill)
		parsed := expectGenerateOK(resp, raw)
		expectServedByRole(resp, phasePrefill)
		expectRemoteDecodeFinish(parsed, raw)
	})

	ginkgo.It("TwoImages_Encode: per-image fan-out routes each request to the encode pool", func() {
		// Stage 4 is one encode request per image — two images = two
		// sequential POSTs. Each must route to the encode pool and
		// return its own ec_transfer_params keyed by mm_hash.
		for _, img := range twoImages {
			body := encodeBody(img)
			resp, raw := doGenerate(body, phaseEncode)
			parsed := expectGenerateOK(resp, raw)
			expectServedByRole(resp, phaseEncode)
			expectECTransferParams(parsed, raw)
		}
	})

	ginkgo.It("TwoImages_Prefill: combined two-image prefill routes to the prefill pool with remote_decode finish", func() {
		// Stage 5 carries all images in one request. token_ids match
		// the placeholder offsets declared in twoImages: 1..3 and 4..6,
		// followed by two text tokens at 7..8.
		tokenIDs := []int{1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789}
		body := prefillBody(tokenIDs, twoImages)
		resp, raw := doGenerate(body, phasePrefill)
		parsed := expectGenerateOK(resp, raw)
		expectServedByRole(resp, phasePrefill)
		expectRemoteDecodeFinish(parsed, raw)
	})

	ginkgo.It("MissingEPPPhase: rejects /inference/v1/generate without an EPP-Phase header", func() {
		// Omit the header — no HTTPRoute matches, so the gateway should
		// return a non-2xx status. The exact code is gateway-dependent
		// (typically 404), so we assert the class only.
		resp, raw := doGenerate(encodeBody(singleImage), "")
		gomega.Expect(resp.StatusCode).To(gomega.SatisfyAll(
			gomega.BeNumerically(">=", 400),
			gomega.BeNumerically("<", 600),
		), "expected error status without EPP-Phase, got %d: %s", resp.StatusCode, string(raw))
	})

	ginkgo.It("InvalidEPPPhase: rejects /inference/v1/generate with an unknown EPP-Phase value", func() {
		resp, raw := doGenerate(encodeBody(singleImage), "bogus")
		gomega.Expect(resp.StatusCode).To(gomega.SatisfyAll(
			gomega.BeNumerically(">=", 400),
			gomega.BeNumerically("<", 600),
		), "expected error status with bogus EPP-Phase, got %d: %s", resp.StatusCode, string(raw))
	})
})

// imageFeatures builds the features map that encode and prefill share:
// mm_hashes, mm_placeholders, kwargs_data — all keyed by modality
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

// encodeBody builds a single-image encode request per Stage 4 Option A
// in communication.md. token_ids = [BOS, placeholder*length]; offset is
// always 1 since each encode carries exactly one image.
func encodeBody(image imageSpec) []byte {
	tokenIDs := make([]int, 1+image.Length)
	tokenIDs[0] = 1
	for i := 1; i < len(tokenIDs); i++ {
		tokenIDs[i] = 32000
	}
	body := map[string]any{
		"model":           modelName,
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

// prefillBody builds a Stage 5 Option A prefill request covering every
// image in one body. tokenIDs must already contain placeholder spans
// matching each image's Offset/Length. Top-level kv_transfer_params is
// the form the simulator honors (see prefill_generate It block above
// for the simulator/real-vLLM divergence).
func prefillBody(tokenIDs []int, images []imageSpec) []byte {
	body := map[string]any{
		"request_id":         "e2e-prefill-" + uuid.NewString(),
		"model":              modelName,
		"token_ids":          tokenIDs,
		"features":           imageFeatures(images),
		"ec_transfer_params": map[string]any{"image": ecTransferEntries(images)},
		"kv_transfer_params": map[string]any{"do_remote_decode": true},
		"sampling_params":    map[string]any{"max_tokens": 1},
	}
	return mustMarshal(body)
}

// doGenerate POSTs body to <gateway>/inference/v1/generate. If phase is
// non-empty it sets EPP-Phase: <phase>; the empty case exercises the
// missing-header negative path. Returns the live *http.Response (closed by
// the caller's defer is unnecessary — body is already drained) and the raw
// response body.
func doGenerate(body []byte, phase string) (*http.Response, []byte) {
	req, err := http.NewRequest(http.MethodPost, gatewayBaseURL+generatePath, bytes.NewReader(body))
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "build POST request")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", uuid.NewString())
	if phase != "" {
		req.Header.Set("EPP-Phase", phase)
	}

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "POST %s phase=%q", req.URL, phase)
	defer func() {
		gomega.Expect(resp.Body.Close()).To(gomega.Succeed())
	}()
	raw, err := io.ReadAll(resp.Body)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "read response body")
	return resp, raw
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

// expectECTransferParams asserts the encode response contains an
// ec_transfer_params object keyed by mm_hash. The simulator (running with
// --mm-encoder-only on encode pods) emits this with dummy values; we
// assert structural shape only.
func expectECTransferParams(parsed map[string]any, raw []byte) {
	ec, ok := parsed["ec_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed ec_transfer_params in encode response: %s", string(raw))
	gomega.Expect(ec).NotTo(gomega.BeEmpty(),
		"ec_transfer_params is empty: %s", string(raw))
}

// expectRemoteDecodeFinish asserts the prefill response carries
// finish_reason="remote_decode" — the prefill stage's signal that it
// computed KV cache and is handing off to a remote decode worker. We use
// this rather than the documented kv_transfer_params object because the
// /inference/v1/generate endpoint has a known propagation bug (see
// coordinator/docs/communication.md Stage 5 Option A) that drops those
// params from the response; the finish_reason still surfaces.
func expectRemoteDecodeFinish(parsed map[string]any, raw []byte) {
	choices, ok := parsed["choices"].([]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed choices in prefill response: %s", string(raw))
	gomega.Expect(choices).NotTo(gomega.BeEmpty(),
		"choices is empty: %s", string(raw))
	first, ok := choices[0].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"first choice is not an object: %s", string(raw))
	gomega.Expect(first["finish_reason"]).To(gomega.Equal("remote_decode"),
		"prefill finish_reason mismatch: %s", string(raw))
}

// expectServedByRole reads the x-inference-pod header set by the EPP, fetches
// the pod, and asserts its llm-d.ai/role label matches the expected phase.
// This is how we verify EPP-Phase routing landed at the right pool — each
// InferencePool selects pods by this label.
func expectServedByRole(resp *http.Response, expectedRole string) {
	podName := resp.Header.Get("x-inference-pod")
	gomega.Expect(podName).NotTo(gomega.BeEmpty(),
		"missing x-inference-pod header; full headers=%v", resp.Header)

	ns := resp.Header.Get("x-inference-namespace")
	if ns == "" {
		ns = testConfig.NsName
	}

	pod := &corev1.Pod{}
	gomega.Expect(testConfig.K8sClient.Get(testConfig.Context,
		types.NamespacedName{Name: podName, Namespace: ns}, pod)).
		To(gomega.Succeed(), "fetch served pod %s/%s", ns, podName)

	gomega.Expect(pod.Labels[roleLabel]).To(gomega.Equal(expectedRole),
		"served pod %s/%s has %s=%q, want %q", ns, podName, roleLabel, pod.Labels[roleLabel], expectedRole)
}

func mustMarshal(v any) []byte {
	b, err := json.Marshal(v)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "json.Marshal failed: %v", v)
	return b
}
