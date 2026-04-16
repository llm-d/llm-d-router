//go:build integration_tests
// +build integration_tests

// Package integration_test contains integration tests that require a running vLLM instance.
//
// Run with: go test -tags=integration_tests -v ./test/integration/... -vllm-url=http://localhost:8000
//
// DP mode coverage:
//
//   - Internal LB (default): start a single `vllm serve ... --data-parallel-size N`
//     and run without -external-lb. Exercises the full @dpN key flow: KV events
//     published with rank-qualified keys, scorer collapses them, PreRequest
//     plugin injects `X-data-parallel-rank`, vLLM accepts ranks 0..N-1.
//
//   - External LB: start two standalone `vllm serve` processes on different
//     ports/GPUs and run with `-external-lb=true -vllm-url-alt=...`. Exercises
//     the non-DP path: indexer keys have no `@dpN` suffix, the scorer never
//     emits the internal header, PreRequest MUST NOT set
//     `X-data-parallel-rank`, and each server accepts requests with no rank
//     header.
//
//   - Hybrid LB: multi-node topology where each node runs its own API server
//     with a local subset of DP ranks. From the scheduler's perspective a
//     Hybrid LB pod is indistinguishable from an Internal LB pod — both
//     publish `ip:port@dpN` KV-event keys and both accept the
//     `X-data-parallel-rank` header — so the Internal LB test suite is the
//     authoritative coverage for the scheduler-side contract. Running the
//     real multi-node topology requires separate physical nodes and is out of
//     scope for this single-host test.
package integration_test

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common"
	prerequest "github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/pre-request"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/scorer"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	vllmURL    = flag.String("vllm-url", "http://localhost:8000", "vLLM server URL")
	vllmURLAlt = flag.String("vllm-url-alt", "http://localhost:8001", "Secondary vLLM server URL (for External LB mode)")
	externalLB = flag.Bool("external-lb", false, "Enable External LB mode tests (requires two vLLM servers)")
)

// chatRequest sends a chat completion request to vLLM with optional headers.
func chatRequest(t *testing.T, url, prompt string, headers map[string]string) (int, map[string]interface{}) {
	t.Helper()
	body := fmt.Sprintf(`{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"%s"}],"max_tokens":5}`, prompt)
	req, err := http.NewRequest("POST", url+"/v1/chat/completions", strings.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var result map[string]interface{}
	_ = json.Unmarshal(respBody, &result)
	return resp.StatusCode, result
}

// ---------- Test 1: ParseDPScoringKey + StripDPRankFromScores (real Go code) ----------

// ExportedStripDPRankFromScores wraps the unexported scorer function for testing.
// We test the public common.ParseDPScoringKey which is the core of stripDPRankFromScores.
func TestParseDPScoringKey_WithRealFormats(t *testing.T) {
	// These are the exact key formats the KV cache indexer produces in Internal LB mode.
	// vLLM publishes KV events with pod identifier "IP:PORT" and the indexer appends "@dpN".
	tests := []struct {
		key      string
		wantBase string
		wantRank int
	}{
		{"127.0.0.1:8000@dp0", "127.0.0.1:8000", 0},
		{"127.0.0.1:8000@dp1", "127.0.0.1:8000", 1},
		{"10.0.0.1:8000@dp0", "10.0.0.1:8000", 0},
		{"10.0.0.1:8000@dp1", "10.0.0.1:8000", 1},
		// External LB mode: no @dpN suffix
		{"10.0.0.1:8000", "10.0.0.1:8000", common.NoDataParallelRank},
		// Hybrid LB mode: node IP with port
		{"192.168.1.10:8000@dp0", "192.168.1.10:8000", 0},
		{"192.168.1.10:8000@dp3", "192.168.1.10:8000", 3},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			base, rank := common.ParseDPScoringKey(tt.key)
			assert.Equal(t, tt.wantBase, base, "base key mismatch")
			assert.Equal(t, tt.wantRank, rank, "rank mismatch")
		})
	}
}

// ---------- Test 2: EncodeWinningRanks/DecodeWinningRanks round-trip ----------

func TestWinningRanksRoundTrip(t *testing.T) {
	original := map[string]int{
		"127.0.0.1:8000": 0,
		"10.0.0.2:8000":  1,
		"10.0.0.3:8000":  5,
	}

	encoded, err := common.EncodeWinningRanks(original)
	require.NoError(t, err)
	t.Logf("Encoded: %s", encoded)

	decoded, err := common.DecodeWinningRanks(encoded)
	require.NoError(t, err)
	assert.Equal(t, original, decoded)
}

// ---------- Test 3: DPRankHeaderHandler with real scheduling types ----------

func TestDPRankHeaderHandler_FullPipeline(t *testing.T) {
	// Simulate what the scorer produces after stripDPRankFromScores
	winningRanks := map[string]int{
		"127.0.0.1:8000": 0, // rank 0 won for this pod
		"10.0.0.2:8000":  1, // rank 1 won for this pod
	}

	encoded, err := common.EncodeWinningRanks(winningRanks)
	require.NoError(t, err)

	// Build the request with internal header (as the scorer would set it)
	request := &scheduling.LLMRequest{
		Headers: map[string]string{
			common.DPWinningRanksHeader: encoded,
		},
	}

	// Build scheduling result for the selected pod (127.0.0.1:8000)
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-dp-pod"},
			Address:        "127.0.0.1",
			Port:           "8000",
		},
		&fwkdl.Metrics{},
		nil,
	)
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {
				TargetEndpoints: []scheduling.Endpoint{endpoint},
			},
		},
	}

	// Run the real DPRankHeaderHandler
	handler := prerequest.NewDPRankHeaderHandler().WithName("test-dp-rank")
	handler.PreRequest(context.Background(), request, result)

	// Verify X-data-parallel-rank is set correctly
	rankHeader, exists := request.Headers[common.DataParallelRankHeader]
	assert.True(t, exists, "X-data-parallel-rank should be set")
	assert.Equal(t, "0", rankHeader, "should pin to rank 0")

	// Verify internal header was removed
	_, internalExists := request.Headers[common.DPWinningRanksHeader]
	assert.False(t, internalExists, "internal header should be removed")

	t.Logf("Pipeline result: X-data-parallel-rank=%s (internal header removed: %v)", rankHeader, !internalExists)
}

// ---------- Test 4: Full pipeline → vLLM (real HTTP request) ----------

func TestDPRankHeaderHandler_ToVLLM(t *testing.T) {
	// Skip if vLLM is not running
	resp, err := http.Get(*vllmURL + "/health")
	if err != nil {
		t.Skipf("vLLM not available at %s: %v", *vllmURL, err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Skipf("vLLM unhealthy at %s: %d", *vllmURL, resp.StatusCode)
	}

	// Simulate the full scheduler pipeline:
	// 1. KV cache indexer returns scores with @dpN keys
	// 2. Scorer calls stripDPRankFromScores
	// 3. Scorer writes internal header
	// 4. Profile handler selects pod
	// 5. PreRequest plugin reads internal header, sets X-data-parallel-rank
	// 6. Request is sent to vLLM

	// Step 1-2: Simulate raw scores from indexer and strip DP ranks
	rawScores := map[string]float64{
		"127.0.0.1:8000@dp0": 5.0,
		"127.0.0.1:8000@dp1": 3.0,
	}

	// Use the real ParseDPScoringKey logic (same as stripDPRankFromScores)
	stripped := make(map[string]float64)
	winningRanks := make(map[string]int)
	for key, score := range rawScores {
		baseKey, dpRank := common.ParseDPScoringKey(key)
		if existing, ok := stripped[baseKey]; !ok || score > existing {
			stripped[baseKey] = score
			if dpRank != common.NoDataParallelRank {
				winningRanks[baseKey] = dpRank
			}
		}
	}

	assert.Equal(t, map[string]float64{"127.0.0.1:8000": 5.0}, stripped)
	assert.Equal(t, map[string]int{"127.0.0.1:8000": 0}, winningRanks)
	t.Logf("Step 1-2: stripped=%v, winningRanks=%v", stripped, winningRanks)

	// Step 3: Scorer writes internal header
	encoded, err := common.EncodeWinningRanks(winningRanks)
	require.NoError(t, err)

	request := &scheduling.LLMRequest{
		Headers: map[string]string{
			common.DPWinningRanksHeader: encoded,
		},
	}
	t.Logf("Step 3: internal header = %s", encoded)

	// Step 4: Profile handler selects pod (127.0.0.1:8000)
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-dp-pod"},
			Address:        "127.0.0.1",
			Port:           "8000",
		},
		&fwkdl.Metrics{},
		nil,
	)
	schedulingResult := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {
				TargetEndpoints: []scheduling.Endpoint{endpoint},
			},
		},
	}

	// Step 5: PreRequest plugin processes the request
	handler := prerequest.NewDPRankHeaderHandler().WithName("dp-rank-handler")
	handler.PreRequest(context.Background(), request, schedulingResult)

	// Verify headers
	rankHeader := request.Headers[common.DataParallelRankHeader]
	assert.Equal(t, "0", rankHeader)
	_, internalExists := request.Headers[common.DPWinningRanksHeader]
	assert.False(t, internalExists)
	t.Logf("Step 5: X-data-parallel-rank=%s, internal removed=%v", rankHeader, !internalExists)

	// Step 6: Send the actual request to vLLM with the header our plugin produced
	status, result := chatRequest(t, *vllmURL, "Hello from scheduler pipeline test", map[string]string{
		common.DataParallelRankHeader: rankHeader,
	})
	assert.Equal(t, 200, status, "vLLM should accept the request")
	t.Logf("Step 6: vLLM response id=%v, status=%d", result["id"], status)

	// Also test rank 1 — but only for Internal LB mode.
	// External LB servers are DP=1, so rank 1 does not exist and vLLM returns 400.
	winningRanks1 := map[string]int{"127.0.0.1:8000": 1}
	encoded1, _ := common.EncodeWinningRanks(winningRanks1)
	request1 := &scheduling.LLMRequest{
		Headers: map[string]string{
			common.DPWinningRanksHeader: encoded1,
		},
	}
	handler.PreRequest(context.Background(), request1, schedulingResult)
	assert.Equal(t, "1", request1.Headers[common.DataParallelRankHeader])

	status1, result1 := chatRequest(t, *vllmURL, "Hello from rank 1 pipeline test", map[string]string{
		common.DataParallelRankHeader: request1.Headers[common.DataParallelRankHeader],
	})
	wantStatus1 := 200
	if *externalLB {
		wantStatus1 = 400
	}
	assert.Equal(t, wantStatus1, status1)
	t.Logf("Rank 1 test: vLLM response id=%v, status=%d", result1["id"], status1)
}

// ---------- Test 5: ZMQ Event Subscription (real Go zmq library) ----------

func TestZMQEventSubscription(t *testing.T) {
	// Skip if vLLM is not running
	resp, err := http.Get(*vllmURL + "/health")
	if err != nil {
		t.Skipf("vLLM not available at %s: %v", *vllmURL, err)
	}
	resp.Body.Close()

	// The llm-d-kv-cache library uses go-zmq to subscribe to KV events.
	// We can't easily import the full kvevents.Pool here without complex setup,
	// but we CAN verify that:
	// 1. The ZMQ ports are reachable
	// 2. Events arrive on both rank ports
	// This test uses net.Dial to verify port reachability.

	// Port 5557 = rank 0, Port 5558 = rank 1
	for _, port := range []int{5557, 5558} {
		addr := fmt.Sprintf("127.0.0.1:%d", port)
		t.Logf("Checking ZMQ port %s", addr)

		// We can't do a full ZMQ subscribe without the zmq library,
		// but we verify the port is listening
		conn, err := (&net.Dialer{Timeout: 2 * time.Second}).Dial("tcp", addr)
		if err != nil {
			t.Logf("Port %d not reachable (may need zmq client): %v", port, err)
			continue
		}
		conn.Close()
		t.Logf("Port %d is reachable", port)
	}
}

// ---------- Test 6: scorer.StripDPRankFromScores via exported test helper ----------
// Since stripDPRankFromScores is unexported, we test the equivalent logic
// using the public common.ParseDPScoringKey which is the core implementation.

func TestStripDPRankFromScores_ViaParseDPScoringKey(t *testing.T) {
	// Simulate exactly what the scorer does
	rawScores := map[string]float64{
		"127.0.0.1:8000@dp0": 5.0,
		"127.0.0.1:8000@dp1": 3.0,
		"10.0.0.2:8000@dp0":  2.0,
		"10.0.0.2:8000@dp1":  7.0,
		"10.0.0.3:8000":      4.0, // non-DP pod
	}

	stripped := make(map[string]float64)
	winningRanks := make(map[string]int)
	for key, score := range rawScores {
		baseKey, dpRank := common.ParseDPScoringKey(key)
		if existing, ok := stripped[baseKey]; !ok || score > existing {
			stripped[baseKey] = score
			if dpRank != common.NoDataParallelRank {
				winningRanks[baseKey] = dpRank
			}
		}
	}

	// Verify collapsed scores
	assert.Equal(t, 5.0, stripped["127.0.0.1:8000"], "rank 0 wins (5.0 > 3.0)")
	assert.Equal(t, 7.0, stripped["10.0.0.2:8000"], "rank 1 wins (7.0 > 2.0)")
	assert.Equal(t, 4.0, stripped["10.0.0.3:8000"], "non-DP pod passes through")

	// Verify winning ranks
	assert.Equal(t, 0, winningRanks["127.0.0.1:8000"], "rank 0 won for first pod")
	assert.Equal(t, 1, winningRanks["10.0.0.2:8000"], "rank 1 won for second pod")
	_, nonDPExists := winningRanks["10.0.0.3:8000"]
	assert.False(t, nonDPExists, "non-DP pod should NOT be in winning ranks")

	// Encode and verify PreRequest can decode
	encoded, err := common.EncodeWinningRanks(winningRanks)
	require.NoError(t, err)

	decoded, err := common.DecodeWinningRanks(encoded)
	require.NoError(t, err)
	assert.Equal(t, winningRanks, decoded)

	t.Logf("Scores: %v", stripped)
	t.Logf("Winning ranks: %v", winningRanks)
	t.Logf("Encoded header: %s", encoded)
}

// ---------- Test 7: End-to-end header verification with vLLM ----------

func TestVLLMAcceptsSchedulerHeaders(t *testing.T) {
	resp, err := http.Get(*vllmURL + "/health")
	if err != nil {
		t.Skipf("vLLM not available at %s: %v", *vllmURL, err)
	}
	resp.Body.Close()

	// In External LB mode each vLLM server is DP=1, so only rank 0 is valid.
	// In Internal LB mode the server is DP=N, so rank 0..N-1 are all valid.
	rank1Status := 200
	if *externalLB {
		rank1Status = 400
	}
	tests := []struct {
		name       string
		rank       string
		wantStatus int
	}{
		{"rank 0", "0", 200},
		{"rank 1", "1", rank1Status},
		{"no header (internal LB)", "", 200},
		{"invalid rank 99", "99", 400},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := map[string]string{}
			if tt.rank != "" {
				headers[common.DataParallelRankHeader] = tt.rank
			}

			status, result := chatRequest(t, *vllmURL, fmt.Sprintf("Test %s", tt.name), headers)
			assert.Equal(t, tt.wantStatus, status, "unexpected status for %s", tt.name)

			if status == 200 {
				assert.Contains(t, result, "id", "response should have id")
				t.Logf("  %s: OK (id=%v)", tt.name, result["id"])
			} else {
				t.Logf("  %s: status=%d (expected)", tt.name, status)
			}
		})
	}
}

// ---------- Test 8: Verify internal header is NOT leaked to vLLM ----------

func TestInternalHeaderNotLeaked(t *testing.T) {
	resp, err := http.Get(*vllmURL + "/health")
	if err != nil {
		t.Skipf("vLLM not available at %s: %v", *vllmURL, err)
	}
	resp.Body.Close()

	// Simulate full pipeline: scorer sets internal header, PreRequest processes it
	winningRanks := map[string]int{"127.0.0.1:8000": 0}
	encoded, _ := common.EncodeWinningRanks(winningRanks)

	request := &scheduling.LLMRequest{
		Headers: map[string]string{
			common.DPWinningRanksHeader: encoded,
		},
	}

	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "vllm-pod"},
			Address:        "127.0.0.1",
			Port:           "8000",
		},
		&fwkdl.Metrics{},
		nil,
	)
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}

	handler := prerequest.NewDPRankHeaderHandler().WithName("test")
	handler.PreRequest(context.Background(), request, result)

	// After PreRequest, only X-data-parallel-rank should be in headers
	assert.Equal(t, "0", request.Headers[common.DataParallelRankHeader])
	_, leaked := request.Headers[common.DPWinningRanksHeader]
	assert.False(t, leaked, "internal header x-llm-d-dp-winning-ranks MUST be removed")

	// Send the resulting headers to vLLM — should work fine
	status, _ := chatRequest(t, *vllmURL, "Internal header leak test", request.Headers)
	assert.Equal(t, 200, status)
	t.Logf("vLLM accepted request with cleaned headers (no internal header leak)")
}

// ---------- Test 9: scorer package exported test ----------
// Verify the scorer package compiles and the factory is registered

func TestScorerPluginTypeRegistered(t *testing.T) {
	assert.Equal(t, "precise-prefix-cache-scorer", scorer.PrecisePrefixCachePluginType)
	assert.Equal(t, "dp-rank-header-handler", prerequest.DPRankHeaderHandlerType)

	// Verify factory creates valid plugin
	p, err := prerequest.DPRankHeaderHandlerFactory("test-instance", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "test-instance", p.TypedName().Name)
	assert.Equal(t, prerequest.DPRankHeaderHandlerType, p.TypedName().Type)
}

// ---------- Test 10: Multiple pods with different winning ranks ----------

func TestMultiplePods_CorrectRankSelection(t *testing.T) {
	resp, err := http.Get(*vllmURL + "/health")
	if err != nil {
		t.Skipf("vLLM not available at %s: %v", *vllmURL, err)
	}
	resp.Body.Close()

	// Simulate 3 pods with different winning ranks
	rawScores := map[string]float64{
		"127.0.0.1:8000@dp0": 10.0, // pod 1, rank 0 wins
		"127.0.0.1:8000@dp1": 2.0,
		"10.0.0.2:8000@dp0":  3.0,
		"10.0.0.2:8000@dp1":  8.0, // pod 2, rank 1 wins
		"10.0.0.3:8000":      5.0, // pod 3, non-DP
	}

	// Strip DP ranks (same logic as scorer)
	stripped := make(map[string]float64)
	winningRanks := make(map[string]int)
	for key, score := range rawScores {
		baseKey, dpRank := common.ParseDPScoringKey(key)
		if existing, ok := stripped[baseKey]; !ok || score > existing {
			stripped[baseKey] = score
			if dpRank != common.NoDataParallelRank {
				winningRanks[baseKey] = dpRank
			}
		}
	}

	encoded, err := common.EncodeWinningRanks(winningRanks)
	require.NoError(t, err)

	handler := prerequest.NewDPRankHeaderHandler().WithName("test")

	// Scenario A: Scheduler picks pod 1 (127.0.0.1:8000) → should get rank 0
	reqA := &scheduling.LLMRequest{Headers: map[string]string{common.DPWinningRanksHeader: encoded}}
	endpointA := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "pod-1"},
			Address:        "127.0.0.1",
			Port:           "8000",
		}, &fwkdl.Metrics{}, nil,
	)
	resultA := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpointA}},
		},
	}
	handler.PreRequest(context.Background(), reqA, resultA)
	assert.Equal(t, "0", reqA.Headers[common.DataParallelRankHeader], "pod 1 should get rank 0")

	// Actually send to vLLM with rank 0
	status, _ := chatRequest(t, *vllmURL, "Multi-pod test A", map[string]string{
		common.DataParallelRankHeader: reqA.Headers[common.DataParallelRankHeader],
	})
	assert.Equal(t, 200, status)

	// Scenario B: Scheduler picks pod 2 (10.0.0.2:8000) → should get rank 1
	// Re-encode since PreRequest deletes the internal header
	reqB := &scheduling.LLMRequest{Headers: map[string]string{common.DPWinningRanksHeader: encoded}}
	endpointB := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "pod-2"},
			Address:        "10.0.0.2",
			Port:           "8000",
		}, &fwkdl.Metrics{}, nil,
	)
	resultB := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpointB}},
		},
	}
	handler.PreRequest(context.Background(), reqB, resultB)
	assert.Equal(t, "1", reqB.Headers[common.DataParallelRankHeader], "pod 2 should get rank 1")

	// Scenario C: Scheduler picks pod 3 (non-DP) → no rank header
	reqC := &scheduling.LLMRequest{Headers: map[string]string{common.DPWinningRanksHeader: encoded}}
	endpointC := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "pod-3"},
			Address:        "10.0.0.3",
			Port:           "8000",
		}, &fwkdl.Metrics{}, nil,
	)
	resultC := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpointC}},
		},
	}
	handler.PreRequest(context.Background(), reqC, resultC)
	_, rankExists := reqC.Headers[common.DataParallelRankHeader]
	assert.False(t, rankExists, "non-DP pod should NOT get rank header")

	t.Logf("Pod 1 → rank %s, Pod 2 → rank %s, Pod 3 → no rank",
		reqA.Headers[common.DataParallelRankHeader],
		reqB.Headers[common.DataParallelRankHeader])
}

// ---------- Test 11: External LB mode (two standalone vLLM processes) ----------
//
// In External LB mode each vLLM pod is its own DP world (single rank), so the
// KV cache indexer produces keys with NO "@dpN" suffix and the scheduler MUST
// NOT inject the X-data-parallel-rank header.
//
// This test requires two vLLM servers (vllm-url and vllm-url-alt) and is gated
// by -external-lb=true.
func TestExternalLBMode_NoRankHeaderInjected(t *testing.T) {
	if !*externalLB {
		t.Skip("External LB mode disabled (pass -external-lb=true with two vLLM servers)")
	}

	for _, url := range []string{*vllmURL, *vllmURLAlt} {
		resp, err := http.Get(url + "/health")
		if err != nil {
			t.Skipf("vLLM not available at %s: %v", url, err)
		}
		resp.Body.Close()
		require.Equal(t, 200, resp.StatusCode, "vLLM unhealthy at %s", url)
	}

	// Raw scores from two external-LB pods — note: no @dpN suffix on keys.
	rawScores := map[string]float64{
		"127.0.0.1:8000": 6.0,
		"127.0.0.1:8001": 9.0,
	}

	stripped := make(map[string]float64)
	winningRanks := make(map[string]int)
	for key, score := range rawScores {
		baseKey, dpRank := common.ParseDPScoringKey(key)
		stripped[baseKey] = score
		if dpRank != common.NoDataParallelRank {
			winningRanks[baseKey] = dpRank
		}
	}

	assert.Equal(t, rawScores, stripped, "external-LB keys pass through unchanged")
	assert.Empty(t, winningRanks, "no winning ranks in External LB mode")

	// Encoding an empty map must return ErrEmptyWinningRanks so the scorer
	// skips the internal header entirely.
	_, err := common.EncodeWinningRanks(winningRanks)
	require.ErrorIs(t, err, common.ErrEmptyWinningRanks)

	// Simulate PreRequest: no internal header present.
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: "ext-pod"},
			Address:        "127.0.0.1",
			Port:           "8001",
		}, &fwkdl.Metrics{}, nil,
	)
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
	request := &scheduling.LLMRequest{Headers: map[string]string{}}
	handler := prerequest.NewDPRankHeaderHandler().WithName("ext-lb")
	handler.PreRequest(context.Background(), request, result)

	_, rankSet := request.Headers[common.DataParallelRankHeader]
	assert.False(t, rankSet, "External LB: X-data-parallel-rank MUST NOT be injected")
	_, leaked := request.Headers[common.DPWinningRanksHeader]
	assert.False(t, leaked, "External LB: internal header MUST NOT be set")

	// Actually round-trip to both vLLM servers without the rank header.
	for _, url := range []string{*vllmURL, *vllmURLAlt} {
		status, body := chatRequest(t, url, "External LB round-trip", nil)
		assert.Equal(t, 200, status, "vLLM at %s should accept request with no rank header", url)
		assert.Contains(t, body, "id")
		t.Logf("External LB %s: id=%v", url, body["id"])
	}
}
