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

var _ = ginkgo.Describe("EPD pools direct gateway", func() {
	ginkgo.It("Encode_Generate: routes /inference/v1/generate with EPP-Phase=encode to the encode pool", func() {
		body := encodeRequestBody()
		resp, raw := doGenerate(body, phaseEncode)
		parsed := expectGenerateOK(resp, raw)
		expectServedByRole(resp, phaseEncode)
		expectECTransferParams(parsed, raw)
	})

	ginkgo.It("Prefill_Generate: routes /inference/v1/generate with EPP-Phase=prefill to the prefill pool", func() {
		body := prefillRequestBody()
		resp, raw := doGenerate(body, phasePrefill)
		parsed := expectGenerateOK(resp, raw)
		expectServedByRole(resp, phasePrefill)
		expectKVTransferParams(parsed, raw)
	})

	ginkgo.It("MissingEPPPhase: rejects /inference/v1/generate without an EPP-Phase header", func() {
		// Omit the header — no HTTPRoute matches, so the gateway should
		// return a non-2xx status. The exact code is gateway-dependent
		// (typically 404), so we assert the class only.
		resp, raw := doGenerate(encodeRequestBody(), "")
		gomega.Expect(resp.StatusCode).To(gomega.SatisfyAll(
			gomega.BeNumerically(">=", 400),
			gomega.BeNumerically("<", 600),
		), "expected error status without EPP-Phase, got %d: %s", resp.StatusCode, string(raw))
	})

	ginkgo.It("InvalidEPPPhase: rejects /inference/v1/generate with an unknown EPP-Phase value", func() {
		resp, raw := doGenerate(encodeRequestBody(), "bogus")
		gomega.Expect(resp.StatusCode).To(gomega.SatisfyAll(
			gomega.BeNumerically(">=", 400),
			gomega.BeNumerically("<", 600),
		), "expected error status with bogus EPP-Phase, got %d: %s", resp.StatusCode, string(raw))
	})
})

// encodeRequestBody mirrors Stage 4 Option A in coordinator/docs/communication.md:
// a single-image encode payload (BOS + placeholder tokens), without text. The
// kwargs_data blob is a placeholder — the simulator does not validate its
// contents; the goal is to verify EPP routing.
func encodeRequestBody() []byte {
	body := map[string]any{
		"token_ids": []int{1, 32000, 32000, 32000},
		"features": map[string]any{
			"mm_hashes":       map[string]any{"image": []string{"e2e-encode-hash"}},
			"mm_placeholders": map[string]any{"image": []map[string]any{{"offset": 1, "length": 3}}},
			"kwargs_data":     map[string]any{"image": []string{"AA=="}},
		},
		"sampling_params": map[string]any{"max_tokens": 1},
	}
	return mustMarshal(body)
}

// prefillRequestBody mirrors Stage 5 Option A in coordinator/docs/
// communication.md: full token sequence, per-image features, EC transfer
// params from the encode stage, and top-level kv_transfer_params with
// do_remote_decode=true. The simulator emits kv_transfer_params in its
// response when the request carries do_remote_decode=true at this location;
// the workaround form (kv_transfer_params under sampling_params.extra_args)
// targets a real-vLLM bug that does not apply here.
func prefillRequestBody() []byte {
	body := map[string]any{
		"request_id": "e2e-prefill-" + uuid.NewString(),
		"model":      modelName,
		"token_ids":  []int{1, 32000, 32000, 32000, 2345, 6789},
		"features": map[string]any{
			"mm_hashes": map[string]any{"image": []string{"e2e-prefill-hash"}},
			"mm_placeholders": map[string]any{"image": []map[string]any{
				{"offset": 1, "length": 3},
			}},
			"kwargs_data": map[string]any{"image": []string{"AA=="}},
		},
		"ec_transfer_params": map[string]any{
			"image": []map[string]any{
				{"e2e-prefill-hash": map[string]any{
					"peer_host":               "10.0.0.1",
					"peer_port":               5501,
					"size_bytes":              0,
					"nixl_agent_metadata_b64": "",
				}},
			},
		},
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
// ec_transfer_params object keyed by mm_hash, per Stage 4 in
// coordinator/docs/communication.md. The simulator (running with
// --mm-encoder-only on encode pods) emits this with dummy values; we
// assert structural shape only.
func expectECTransferParams(parsed map[string]any, raw []byte) {
	ec, ok := parsed["ec_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed ec_transfer_params in encode response: %s", string(raw))
	gomega.Expect(ec).NotTo(gomega.BeEmpty(),
		"ec_transfer_params is empty: %s", string(raw))
}

// expectKVTransferParams asserts the prefill response contains a
// kv_transfer_params object with the fields documented in Stage 5 of
// coordinator/docs/communication.md (block_id, peer_host, peer_port). The
// simulator emits this whenever the request carries
// kv_transfer_params.do_remote_decode=true; values are dummy.
func expectKVTransferParams(parsed map[string]any, raw []byte) {
	kv, ok := parsed["kv_transfer_params"].(map[string]any)
	gomega.Expect(ok).To(gomega.BeTrue(),
		"missing or malformed kv_transfer_params in prefill response: %s", string(raw))
	for _, key := range []string{"block_id", "peer_host", "peer_port"} {
		gomega.Expect(kv).To(gomega.HaveKey(key),
			"kv_transfer_params missing %q: %s", key, string(raw))
	}
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
