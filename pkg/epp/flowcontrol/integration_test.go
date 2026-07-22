/*
Copyright 2026 The Kubernetes Authors.

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

package flowcontrol_test

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/eviction"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/filtering"
	evictionordering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/ordering"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/slodeadline"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// ============================================================================
// Saturation Data Path Tests (producer + detector contracts)
// ============================================================================

// TestConcurrentSaturationReads verifies no data races when multiple goroutines
// read saturation while requests are being tracked and released concurrently.
func TestConcurrentSaturationReads(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	pd := newProducerAndDetector(ctx, t, 100)
	endpoints := []datalayer.Endpoint{pd.ep}

	// Use a start barrier to ensure both goroutines begin concurrently.
	start := make(chan struct{})
	var wg sync.WaitGroup

	// Writer: track and release 200 requests rapidly.
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			schedEp := fwksched.NewEndpoint(pd.epMeta, datalayer.NewMetrics(), nil)
			req := &fwksched.InferenceRequest{
				RequestID: fmt.Sprintf("req-%d", i),
				Body: &fwkrh.InferenceRequestBody{
					TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 10)}},
				},
			}
			result := &fwksched.SchedulingResult{
				ProfileResults: map[string]*fwksched.ProfileRunResult{
					"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
				},
			}
			pd.producer.PreRequest(ctx, req, result)
			req.SchedulingResult = result
			pd.producer.ResponseBody(ctx, req,
				&requestcontrol.Response{EndOfStream: true}, pd.epMeta)
		}
	}()

	// Reader: read saturation 200 times concurrently with the writer.
	// Track violations via atomic counter — require/assert must not be called
	// from non-test goroutines.
	var saturationViolations atomic.Int32
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < 200; i++ {
			sat := pd.detector.Saturation(ctx, endpoints)
			if sat < 0.0 || sat > 1.0 {
				saturationViolations.Add(1)
			}
		}
	}()

	close(start)
	wg.Wait()

	require.Equal(t, int32(0), saturationViolations.Load(),
		"saturation was outside [0.0, 1.0] during concurrent reads")
	require.InDelta(t, 0.0, pd.detector.Saturation(ctx, endpoints), 1e-9,
		"saturation must return to exactly 0 after all concurrent operations complete")
}

// ============================================================================
// Full-Loop Controller Tests (detector wired into dispatch cycle)
// ============================================================================

// TestSaturationFullLoop wires a real InFlightLoadProducer, real concurrency
// detector, and real persistent endpoints into the FlowController's dispatch
// cycle via EndpointCandidates.Locate(). Verifies that the dispatch cycle
// gates on real saturation read through DynamicAttributes.
func TestSaturationFullLoop(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	pd := newProducerAndDetector(ctx, t, 2)

	// EndpointCandidates returns the SAME persistent endpoint objects that
	// Extract() registered DynamicAttributes on. This is the critical contract.
	persistentEndpoints := []datalayer.Endpoint{pd.ep}

	h := newHarness(t, harnessOpts{
		detector: pd.detector,
		endpointCandidates: &contractmocks.MockEndpointCandidates{
			Candidates: persistentEndpoints,
		},
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Simulate 2 in-flight requests (maxConcurrency=2 on 1 endpoint -> saturation=1.0).
	for i := 0; i < 2; i++ {
		schedEp := fwksched.NewEndpoint(pd.epMeta, datalayer.NewMetrics(), nil)
		req := &fwksched.InferenceRequest{
			RequestID: fmt.Sprintf("prefill-%d", i),
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 50)}},
			},
		}
		result := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
			},
		}
		pd.producer.PreRequest(h.ctx, req, result)
	}

	// Verify the detector sees saturation via the real Locate->Saturation path.
	sat := pd.detector.Saturation(h.ctx, persistentEndpoints)
	require.InDelta(t, 1.0, sat, 1e-9,
		"2 in-flight requests with maxConcurrency=2 should report full saturation")

	// Now enqueue a request through the FlowController. Since saturation=1.0,
	// the dispatch cycle should NOT dispatch it -- it should queue.
	results := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(h.ctx, 1*time.Second)
		defer reqCancel()
		req := &testRequest{id: "queued-req", key: key, byteSize: 100, ttl: 1 * time.Second}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "queued-req", outcome: outcome, err: err}
	}()

	// The request has a 1s context timeout. It must NOT dispatch and must
	// return with an eviction/rejection outcome within that timeout.
	select {
	case r := <-results:
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request dispatched despite full saturation -- "+
				"the Locate->Saturation data path is broken (regression of #1474)")
		require.Error(t, r.err, "saturated request should return an error (TTL or deadline)")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not finalize within 3s -- possible dispatch cycle hang under full saturation")
	}
}

// ============================================================================
// Dispatch Ordering Tests (SLO deadline)
// ============================================================================

// TestDispatchOrderingSLODeadline verifies that the SLO deadline ordering policy
// reads the x-llm-d-slo-ttft-ms header from real InferenceRequests.
func TestDispatchOrderingSLODeadline(t *testing.T) {
	t.Parallel()

	handle := testutils.NewTestHandle(t.Context())
	sloPlugin, err := slodeadline.SLODeadlineOrderingPolicyFactory("slo", nil, handle)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		ordering: sloPlugin.(flowcontrol.OrderingPolicy),
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	now := time.Now()

	type reqSpec struct {
		id    string
		sloMs string
	}
	specs := []reqSpec{
		{"loose-10s", "10000"},
		{"tight-500ms", "500"},
		{"mid-2s", "2000"},
	}

	results := make(chan dispatchResult, len(specs))
	for _, s := range specs {
		go func() {
			req := &testRequest{
				id: s.id, key: key, byteSize: 100, ttl: 5 * time.Minute,
				timestamp: now,
				infReq: &fwksched.InferenceRequest{
					RequestID: s.id,
					Headers:   map[string]string{metadata.TTFTSLOHeaderKey: s.sloMs},
				},
			}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: s.id, outcome: outcome, err: err}
			detector.Release()
		}()
		time.Sleep(5 * time.Millisecond)
	}

	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == uint64(len(specs))
	}, time.Second, time.Millisecond, "all requests should be queued before unblocking")
	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < len(specs); i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d", i)
		}
	}

	require.Equal(t, "tight-500ms", dispatchOrder[0], "tightest SLO should dispatch first")
	require.Equal(t, "mid-2s", dispatchOrder[1], "middle SLO should dispatch second")
	require.Equal(t, "loose-10s", dispatchOrder[2], "loosest SLO should dispatch last")
}

// ============================================================================
// Priority and Fairness Tests
// ============================================================================

// TestPriorityBackpressure verifies that high-priority requests dispatch before
// low-priority requests under saturation.
func TestPriorityBackpressure(t *testing.T) {
	t.Parallel()

	handle := testutils.NewTestHandle(t.Context())

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}

	highBand, err := registry.NewPriorityBandConfig(10, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)
	lowBand, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		detector: detector,
		bands:    []*registry.PriorityBandConfig{highBand, lowBand},
	})

	highKey := flowcontrol.FlowKey{ID: "high-flow", Priority: 10}
	lowKey := flowcontrol.FlowKey{ID: "low-flow", Priority: 0}

	results := make(chan dispatchResult, 4)

	enqueue := func(id string, key flowcontrol.FlowKey) {
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
			detector.Release()
		}()
	}

	enqueue("low-1", lowKey)
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == 1
	}, time.Second, time.Millisecond, "low-priority request should be queued")
	enqueue("high-1", highKey)
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == 2
	}, time.Second, time.Millisecond, "both requests should be queued")

	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < 2; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d", i)
		}
	}

	require.Equal(t, "high-1", dispatchOrder[0],
		"high-priority should dispatch before low-priority under backpressure")
	require.Equal(t, "low-1", dispatchOrder[1])
}

// TestFairnessRoundRobin verifies that round-robin fairness rotates dispatch
// across flows within a priority band.
func TestFairnessRoundRobin(t *testing.T) {
	t.Parallel()

	handle := testutils.NewTestHandle(t.Context())
	rrPlugin, err := roundrobin.RoundRobinFairnessPolicyFactory("rr", nil, handle)
	require.NoError(t, err)

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{
		fairness: rrPlugin.(flowcontrol.FairnessPolicy),
		detector: detector,
	})

	flows := []string{"flow-a", "flow-b", "flow-c"}
	const reqsPerFlow = 3
	total := len(flows) * reqsPerFlow

	results := make(chan dispatchResult, total)

	for _, flow := range flows {
		for i := 0; i < reqsPerFlow; i++ {
			id := fmt.Sprintf("%s-req-%d", flow, i)
			key := flowcontrol.FlowKey{ID: flow, Priority: 0}
			go func() {
				req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
				outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
				results <- dispatchResult{id: id, flowID: key.ID, outcome: outcome, err: err}
				detector.Release()
			}()
			time.Sleep(2 * time.Millisecond)
		}
	}

	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == uint64(total)
	}, time.Second, time.Millisecond, "all requests should be queued before unblocking")
	detector.Unblock(1)

	var dispatchOrder []string
	for i := 0; i < total; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatchOrder = append(dispatchOrder, r.flowID)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for dispatch %d of %d", i, total)
		}
	}

	require.Len(t, dispatchOrder, total)

	for round := 0; round < reqsPerFlow; round++ {
		start := round * len(flows)
		end := start + len(flows)
		if end > len(dispatchOrder) {
			break
		}
		chunk := dispatchOrder[start:end]
		seen := map[string]bool{}
		for _, f := range chunk {
			seen[f] = true
		}
		require.Len(t, seen, len(flows),
			"round %d: expected all %d flows in dispatch chunk %v", round, len(flows), chunk)
	}
}

// TestUsageLimitThresholdGatesDispatch verifies that a UsageLimitPolicy with
// threshold < 1.0 triggers HoL blocking at partial saturation.
// With threshold=0.5, the dispatch cycle should block when saturation >= 0.5.
func TestUsageLimitThresholdGatesDispatch(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// maxConcurrency=10 on 1 endpoint. 5 in-flight -> saturation=0.5.
	pd := newProducerAndDetector(ctx, t, 10)

	// Threshold=0.5: HoL blocking triggers at 50% saturation.
	halfThresholdPolicy := usagelimits.NewConstPolicy("half", 0.5)

	h := newHarness(t, harnessOpts{
		detector: pd.detector,
		endpointCandidates: &contractmocks.MockEndpointCandidates{
			Candidates: []datalayer.Endpoint{pd.ep},
		},
		usageLimitPolicy: halfThresholdPolicy,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Drive 5 in-flight requests -> saturation=5/10=0.5 -> meets threshold.
	for i := 0; i < 5; i++ {
		schedEp := fwksched.NewEndpoint(pd.epMeta, datalayer.NewMetrics(), nil)
		req := &fwksched.InferenceRequest{
			RequestID: fmt.Sprintf("inflight-%d", i),
			Body: &fwkrh.InferenceRequestBody{
				TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 10)}},
			},
		}
		result := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
			},
		}
		pd.producer.PreRequest(h.ctx, req, result)
	}

	// saturation=0.5, threshold=0.5: 0.5 >= 0.5 -> HoL blocking should trigger.
	results := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(h.ctx, 500*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "gated-req", key: key, byteSize: 100, ttl: 500 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "gated-req", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should NOT dispatch at saturation=0.5 with threshold=0.5")
		require.Error(t, r.err,
			"gated request should return an error (TTL or deadline)")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not finalize within 3s -- possible dispatch cycle hang under partial saturation")
	}
}

// ============================================================================
// Capacity Enforcement Tests (bytes, requests, global vs band)
// ============================================================================

// TestGlobalAndBandCapacityInteraction verifies that the global MaxRequests
// limit rejects requests even when the per-band limit has capacity.
func TestGlobalAndBandCapacityInteraction(t *testing.T) {
	t.Parallel()

	// Band allows 10 requests, but global allows only 3.
	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector:           detector,
		maxRequests:        3,
		bandMaxRequests:    10,
		endpointCandidates: nonEmptyCandidates(),
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Fill the global capacity with 3 requests that stay queued (blocked detector, long TTL), and
	// wait until the queue observably holds all of them before sending the overflow. Enqueuing
	// everything on a fixed stagger instead would race with goroutine scheduling under CI load.
	queuedCtx, queuedCancel := context.WithCancel(h.ctx)
	defer queuedCancel()
	queued := make(chan dispatchResult, 3)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("queued-req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(queuedCtx, req)
			queued <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == 3
	}, time.Second, time.Millisecond, "3 requests should queue, exhausting the global limit")

	// The global limit (3) is exhausted while the band limit (10) still has room, so further
	// requests must be rejected for capacity.
	overflow := make(chan dispatchResult, 2)
	for i := 0; i < 2; i++ {
		id := fmt.Sprintf("overflow-req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			overflow <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}
	for i := 0; i < 2; i++ {
		select {
		case r := <-overflow:
			require.Equal(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
				"global MaxRequests=3 should reject %s even though the band allows 10", r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for overflow rejection %d", i)
		}
	}

	// Release the queued requests and verify none of them was rejected for capacity.
	queuedCancel()
	for i := 0; i < 3; i++ {
		select {
		case r := <-queued:
			require.NotEqual(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
				"queued request %s should have been admitted, not capacity-rejected", r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for queued request %d after cancellation", i)
		}
	}
}

// TestByteCapacityEnforcement verifies that the per-band byte capacity limit
// rejects requests when their cumulative byte size exceeds the band budget.
func TestByteCapacityEnforcement(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector:           detector,
		bandMaxBytes:       1000,
		endpointCandidates: nonEmptyCandidates(),
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Queue 3 requests of 300 bytes each (900 total, within the 1000-byte budget), waiting until
	// all are observably queued before sending the request that would exceed the budget.
	queuedCtx, queuedCancel := context.WithCancel(h.ctx)
	defer queuedCancel()
	queued := make(chan dispatchResult, 3)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("byte-req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 300, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(queuedCtx, req)
			queued <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalByteSize == 900
	}, time.Second, time.Millisecond, "3x300 bytes should be queued within the 1000-byte budget")

	// A 4th 300-byte request would bring the band to 1200 bytes, so it must be rejected.
	overflow := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "byte-req-overflow", key: key, byteSize: 300, ttl: 5 * time.Minute}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		overflow <- dispatchResult{id: "byte-req-overflow", outcome: outcome, err: err}
	}()
	select {
	case r := <-overflow:
		require.Equal(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
			"request exceeding the band byte budget should be rejected")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for byte-capacity rejection")
	}

	queuedCancel()
	for i := 0; i < 3; i++ {
		select {
		case r := <-queued:
			require.NotEqual(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
				"queued request %s should have been admitted, not capacity-rejected", r.id)
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for queued request %d after cancellation", i)
		}
	}
}

// TestEmptyPoolRejectsAsNoEndpoints verifies the scale-from-zero path: when the candidate pool has no
// endpoints, requests buffer until capacity is exhausted, and the overflow rejection is surfaced as
// RejectedNoEndpoints (mapped to 503) rather than RejectedCapacity (429).
func TestEmptyPoolRejectsAsNoEndpoints(t *testing.T) {
	t.Parallel()

	// Blocked detector prevents dispatch; the pool is intentionally empty (no endpointCandidates set).
	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector:     detector,
		bandMaxBytes: 1000,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Fill the 1000-byte band with 3 requests of 300 bytes, waiting until all are observably
	// queued before sending the request that overflows the budget.
	queuedCtx, queuedCancel := context.WithCancel(h.ctx)
	defer queuedCancel()
	queued := make(chan dispatchResult, 3)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("noep-req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 300, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(queuedCtx, req)
			queued <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalByteSize == 900
	}, time.Second, time.Millisecond, "3x300 bytes should be queued within the 1000-byte budget")

	// The 4th request exceeds the byte budget, but because the pool is empty the rejection must
	// surface as RejectedNoEndpoints (503), not RejectedCapacity (429).
	overflow := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "noep-req-overflow", key: key, byteSize: 300, ttl: 5 * time.Minute}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		overflow <- dispatchResult{id: "noep-req-overflow", outcome: outcome, err: err}
	}()
	select {
	case r := <-overflow:
		require.Equal(t, fcTypes.QueueOutcomeRejectedNoEndpoints, r.outcome,
			"with an empty pool, a full-queue rejection should be RejectedNoEndpoints")
		require.ErrorIs(t, r.err, fcTypes.ErrNoEndpoints,
			"no-endpoints rejection should wrap ErrNoEndpoints")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for no-endpoints rejection")
	}

	queuedCancel()
	for i := 0; i < 3; i++ {
		select {
		case r := <-queued:
			require.NotEqual(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
				"with an empty pool, no rejection should be classified as RejectedCapacity")
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for queued request %d after cancellation", i)
		}
	}
}

// ============================================================================
// Eviction Pipeline Tests
// ============================================================================

// TestEvictionPipeline wires the real eviction components together:
// RequestEvictor + SheddableFilter + PriorityTimeOrdering + ImmediateResponseEvictor.
// Verifies: PreRequest->queue tracking->EvictN->channel closure->cleanup.
func TestEvictionPipeline(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := testutils.NewTestHandle(ctx)

	orderPlugin, err := evictionordering.PriorityThenTimeOrderingFactory("evict-order", nil, handle)
	require.NoError(t, err)
	filterPlugin, err := filtering.SheddableFilterFactory("evict-filter", nil, handle)
	require.NoError(t, err)

	evictor := eviction.NewImmediateResponseEvictor()
	requestEvictor := eviction.NewRequestEvictor(
		orderPlugin.(flowcontrol.EvictionOrderingPolicy),
		filterPlugin.(flowcontrol.EvictionFilterPolicy),
		evictor,
	)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
		Address:        "10.0.0.1",
		Port:           "8000",
	}
	schedEndpoint := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	makeResult := func() *fwksched.SchedulingResult {
		return &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{schedEndpoint}},
			},
		}
	}

	// Sheddable requests have priority < 0.
	sheddableReq := &fwksched.InferenceRequest{
		RequestID:  "shed-1",
		Headers:    map[string]string{reqcommon.RequestIDHeaderKey: "shed-1"},
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	// Non-sheddable requests have priority >= 0.
	protectedReq := &fwksched.InferenceRequest{
		RequestID:  "protect-1",
		Headers:    map[string]string{reqcommon.RequestIDHeaderKey: "protect-1"},
		Objectives: fwksched.RequestObjectives{Priority: 1},
	}

	reqCtx, reqCancel := context.WithCancel(ctx)
	defer reqCancel()

	requestEvictor.PreRequest(reqCtx, sheddableReq, makeResult())
	requestEvictor.PreRequest(reqCtx, protectedReq, makeResult())

	inFlight, evictable := requestEvictor.Stats()
	require.Equal(t, 2, inFlight, "both requests should be tracked in-flight")
	require.Equal(t, 1, evictable,
		"only the sheddable request (priority < 0) should be evictable")

	reg := requestEvictor.EvictionRegistry()
	sheddableCh := reg.Get("shed-1")
	require.NotNil(t, sheddableCh, "sheddable request should have an eviction channel")

	evictedIDs, err := requestEvictor.EvictN(ctx, 1)
	require.NoError(t, err)
	require.Equal(t, []string{"shed-1"}, evictedIDs)

	select {
	case <-sheddableCh:
		// Channel was closed by ImmediateResponseEvictor -- eviction signaled.
	default:
		t.Fatal("eviction channel should be closed after EvictN")
	}

	// Protected request's channel should still be open.
	protectedCh := reg.Get("protect-1")
	require.NotNil(t, protectedCh)
	select {
	case <-protectedCh:
		t.Fatal("protected request's channel should not be closed")
	default:
	}

	// Complete the protected request normally.
	requestEvictor.ResponseBody(ctx, protectedReq,
		&requestcontrol.Response{EndOfStream: true}, epMeta)

	inFlight, evictable = requestEvictor.Stats()
	require.Equal(t, 0, inFlight, "all requests should be cleaned up after eviction and completion")
	require.Equal(t, 0, evictable)
}

// ============================================================================
// Error/Non-Happy-Path Tests (TTL, context cancel)
// ============================================================================

// TestTTLExpiryEvictsQueuedRequest verifies that a request queued under
// saturation is evicted with QueueOutcomeEvictedTTL + ErrTTLExpired when its
// TTL expires.
func TestTTLExpiryEvictsQueuedRequest(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "ttl-req", key: key, byteSize: 100, ttl: 100 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{id: "ttl-req", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.Error(t, r.err)
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"TTL-expired request should be wrapped with ErrEvicted")
		require.ErrorIs(t, r.err, fcTypes.ErrTTLExpired,
			"TTL-expired request should contain ErrTTLExpired")
		require.Equal(t, fcTypes.QueueOutcomeEvictedTTL, r.outcome,
			"TTL-expired request should have outcome QueueOutcomeEvictedTTL")
	case <-time.After(5 * time.Second):
		t.Fatal("request did not return after TTL expiry")
	}
}

// TestCallerContextCancellationEvictsRequest verifies that cancelling the
// caller's context while a request is queued produces the correct eviction
// outcome and error chain.
func TestCallerContextCancellationEvictsRequest(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	reqCtx, reqCancel := context.WithCancel(h.ctx)
	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "cancel-req", key: key, byteSize: 100, ttl: 5 * time.Minute}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "cancel-req", outcome: outcome, err: err}
	}()

	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == 1
	}, time.Second, time.Millisecond, "request should be queued before cancelling")
	reqCancel()

	select {
	case r := <-results:
		require.Error(t, r.err)
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"cancelled request should be wrapped with ErrEvicted")
		require.Equal(t, fcTypes.QueueOutcomeEvictedContextCancelled, r.outcome,
			"cancelled request should have QueueOutcomeEvictedContextCancelled")
	case <-time.After(5 * time.Second):
		t.Fatal("request did not return after context cancellation")
	}
}

// ============================================================================
// Shutdown and Lifecycle Tests
// ============================================================================

// TestConcurrentEnqueueDuringShutdown verifies there are no races or panics
// when requests are being enqueued concurrently with controller shutdown.
func TestConcurrentEnqueueDuringShutdown(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	const numRequests = 50

	results := make(chan dispatchResult, numRequests)
	for i := 0; i < numRequests; i++ {
		id := fmt.Sprintf("req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
	}

	// Cancel mid-flight while goroutines are still enqueuing.
	time.Sleep(10 * time.Millisecond)
	h.cancel()

	for i := 0; i < numRequests; i++ {
		select {
		case r := <-results:
			// Every request must reach a terminal state -- no panics, no hangs.
			require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
				"no request should dispatch (detector is blocked and controller is shutting down)")
			require.Error(t, r.err,
				"every request should receive an error during shutdown")
		case <-time.After(5 * time.Second):
			t.Fatalf("request %d hung during concurrent shutdown", i)
		}
	}
}

// TestGracefulShutdownDrainsQueuedRequests verifies that when the controller's
// context is cancelled (simulating pod termination), all queued requests receive
// a clean eviction outcome rather than hanging or panicking.
func TestGracefulShutdownDrainsQueuedRequests(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector: detector,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}
	const numRequests = 10

	results := make(chan dispatchResult, numRequests)
	for i := 0; i < numRequests; i++ {
		id := fmt.Sprintf("req-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			results <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
		time.Sleep(2 * time.Millisecond)
	}

	// Wait for all requests to queue (detector is blocked).
	require.Eventually(t, func() bool {
		return h.reg.Stats().TotalLen == uint64(numRequests)
	}, time.Second, time.Millisecond, "all requests should be queued before shutdown")

	// Simulate pod termination: cancel the controller context.
	h.cancel()

	var evicted int
	for i := 0; i < numRequests; i++ {
		select {
		case r := <-results:
			require.Error(t, r.err, "queued request should receive an error on shutdown")
			switch r.outcome {
			case fcTypes.QueueOutcomeEvictedOther,
				fcTypes.QueueOutcomeEvictedContextCancelled,
				fcTypes.QueueOutcomeRejectedOther:
				evicted++
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("request %d did not return within 5s of shutdown -- possible hang", i)
		}
	}

	require.Equal(t, numRequests, evicted,
		"all queued requests should be evicted or rejected, not silently dropped")
}

// ============================================================================
// Production Edge Cases
// ============================================================================

// TestZombieCapacityStarvation verifies that TTL-expired items still in the
// queue (zombies) consume capacity until the cleanup sweep runs. If the sweep
// interval is long, new requests are falsely rejected because capacity is held
// by dead items.
func TestZombieCapacityStarvation(t *testing.T) {
	t.Parallel()

	detector := newBlockedDetector()

	h := newHarness(t, harnessOpts{
		detector:           detector,
		bandMaxRequests:    3,
		endpointCandidates: nonEmptyCandidates(),
		controllerCfg: &controller.Config{
			DefaultRequestTTL:        50 * time.Millisecond,
			ExpiryCleanupInterval:    10 * time.Second,
			EnqueueChannelBufferSize: 100,
		},
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Fill capacity with 3 requests that will expire via TTL.
	expired := make(chan dispatchResult, 3)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("zombie-%d", i)
		go func() {
			req := &testRequest{id: id, key: key, byteSize: 100, ttl: 50 * time.Millisecond}
			outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
			expired <- dispatchResult{id: id, outcome: outcome, err: err}
		}()
		time.Sleep(5 * time.Millisecond)
	}

	// Wait for all to expire.
	for i := 0; i < 3; i++ {
		select {
		case r := <-expired:
			require.ErrorIs(t, r.err, fcTypes.ErrTTLExpired)
		case <-time.After(5 * time.Second):
			t.Fatalf("zombie %d did not expire", i)
		}
	}

	// All 3 expired, but cleanup hasn't run (interval=10s).
	// The new request is rejected because zombies still consume capacity
	// in the registry's atomic counters.
	newResult := make(chan dispatchResult, 1)
	go func() {
		reqCtx, reqCancel := context.WithTimeout(h.ctx, 200*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "post-zombie", key: key, byteSize: 100, ttl: 200 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		newResult <- dispatchResult{id: "post-zombie", outcome: outcome, err: err}
	}()

	select {
	case r := <-newResult:
		// Zombie capacity starvation: the 3 expired items still occupy
		// capacity slots until the cleanup sweep reclaims them (interval=10s).
		require.Equal(t, fcTypes.QueueOutcomeRejectedCapacity, r.outcome,
			"post-zombie request should be rejected -- expired items consume capacity until cleanup sweep runs")
	case <-time.After(5 * time.Second):
		t.Fatal("post-zombie request hung")
	}
}

// TestEndpointReregistrationSaturationAccuracy verifies that when an endpoint
// is deleted and re-added (pod cycling), the saturation detector accurately
// reflects the state -- in-flight requests from before the delete should not
// leak into the new tracker.
func TestEndpointReregistrationSaturationAccuracy(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := testutils.NewTestHandle(ctx)

	producerName := "rereg-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"rereg-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: ep,
	}))

	// Track a request on the original endpoint.
	schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	oldReq := &fwksched.InferenceRequest{
		RequestID: "old-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 50)}},
		},
	}
	oldResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
		},
	}
	producer.PreRequest(ctx, oldReq, oldResult)

	sat := detector.Saturation(ctx, []datalayer.Endpoint{ep})
	require.Greater(t, sat, 0.0, "saturation should be nonzero with in-flight request")

	// Simulate pod cycling: delete + re-add.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventDelete, Endpoint: ep,
	}))

	// Re-create the endpoint (simulates new pod with same name).
	newEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: newEp,
	}))

	// The new endpoint should have a fresh tracker -- saturation should be 0.
	newSat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.InDelta(t, 0.0, newSat, 1e-9,
		"re-registered endpoint should have zero saturation (fresh tracker)")

	// Complete the old request -- should not panic or corrupt the new tracker.
	require.NotPanics(t, func() {
		oldReq.SchedulingResult = oldResult
		producer.ResponseBody(ctx, oldReq,
			&requestcontrol.Response{EndOfStream: true}, epMeta)
	})

	// New tracker should still be at zero -- old request's cleanup must not
	// affect the new tracker.
	finalSat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.InDelta(t, 0.0, finalSat, 1e-9,
		"old request completion should not affect re-registered endpoint's tracker")

	// Verify the deleted endpoint's tracker is clean.
	eid := epMeta.NamespacedName.String()
	require.Equal(t, int64(0), producer.GetRequests(eid),
		"deleted endpoint tracker should report 0 requests")

	// Track a NEW request on the re-registered endpoint, then complete it.
	// Verify the counter returns to exactly 0. In this sequential execution
	// the old OnEvicted already fired (no corruption). A negative counter
	// here would indicate the old cleanup path leaked a decrement into
	// the new tracker's lifecycle.
	newSchedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	newReq := &fwksched.InferenceRequest{
		RequestID: "new-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 50)}},
		},
	}
	newResult := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{newSchedEp}},
		},
	}
	producer.PreRequest(ctx, newReq, newResult)
	require.Equal(t, int64(1), producer.GetRequests(eid),
		"new request on re-registered endpoint should be tracked")

	newReq.SchedulingResult = newResult
	producer.ResponseBody(ctx, newReq,
		&requestcontrol.Response{EndOfStream: true}, epMeta)
	require.Equal(t, int64(0), producer.GetRequests(eid),
		"counter must be exactly 0 after new request completes -- negative value indicates old OnEvicted corrupted the new tracker")
}

// TestEndpointIdentityCollisionDuringPodReplacement verifies that when a pod is
// replaced (delete old + add new with same NamespacedName), the stale delete
// event for the old pod does not clear the new pod's tracker.
func TestEndpointIdentityCollisionDuringPodReplacement(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	handle := testutils.NewTestHandle(ctx)

	producerName := "collision-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorCfgJSON := []byte(fmt.Sprintf(
		`{"maxConcurrency": 10, "inFlightLoadProducerName": %q}`, producerName,
	))
	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		"collision-detector", fwkplugin.StrictDecoder(detectorCfgJSON), handle,
	)
	require.NoError(t, err)
	detector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod-1", Namespace: "default"},
	}

	oldEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: oldEp,
	}))

	newEp := datalayer.NewEndpoint(epMeta, datalayer.NewMetrics())
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventAddOrUpdate, Endpoint: newEp,
	}))

	schedEp := fwksched.NewEndpoint(epMeta, datalayer.NewMetrics(), nil)
	req := &fwksched.InferenceRequest{
		RequestID: "new-pod-req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, 50)}},
		},
	}
	result := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{schedEp}},
		},
	}
	producer.PreRequest(ctx, req, result)

	require.Greater(t, detector.Saturation(ctx, []datalayer.Endpoint{newEp}), 0.0,
		"new endpoint should show in-flight load before stale delete")

	// Stale delete for old pod must not clear the new pod's tracker.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{
		Type: datalayer.EventDelete, Endpoint: oldEp,
	}))

	sat := detector.Saturation(ctx, []datalayer.Endpoint{newEp})
	require.Greater(t, sat, 0.0,
		"stale delete for old pod must not clear new pod's tracker")
}

// ============================================================================
// Metrics Emission Tests
// ============================================================================

// queueSizeGaugeSum returns the sum of the llm_d_epp_flow_control_queue_size gauge across all
// label sets, or 0 if the family has no series yet.
func queueSizeGaugeSum(t *testing.T) float64 {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	var sum float64
	for _, f := range families {
		if f.GetName() == "llm_d_epp_flow_control_queue_size" {
			for _, m := range f.GetMetric() {
				sum += m.GetGauge().GetValue()
			}
		}
	}
	return sum
}

// TestFlowControlMetricsEmitted verifies that EnqueueAndWait maintains the queue_size gauge: a
// blocked request holds the gauge > 0 while queued, and the gauge returns to 0 once the request
// finalizes.
//
// This test intentionally does NOT run in parallel: it asserts exact values on the process-global
// metrics registry, so it must not interleave with other tests that enqueue requests. Every other
// test in this package calls t.Parallel() before doing any work, so the sequential phase gives
// this test exclusive access to the registry.
func TestFlowControlMetricsEmitted(t *testing.T) {
	eppmetrics.Register()

	detector := newBlockedDetector()
	h := newHarness(t, harnessOpts{detector: detector})

	key := flowcontrol.FlowKey{ID: "metrics-flow", Priority: 0}

	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "metrics-req", key: key, byteSize: 512, ttl: 5 * time.Minute}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{outcome: outcome, err: err}
	}()

	// The gauge is incremented on the EnqueueAndWait goroutine, so poll for it rather than sleep:
	// a fixed delay races with goroutine scheduling under CI load (#2117). The long TTL keeps the
	// request queued for as long as the poll needs.
	require.Eventually(t, func() bool {
		return queueSizeGaugeSum(t) > 0
	}, 5*time.Second, 5*time.Millisecond,
		"queue_size should be > 0 while a request is actively queued")

	// Unblock the detector so the request finalizes deterministically via dispatch.
	detector.Unblock(1)

	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
	case <-time.After(5 * time.Second):
		t.Fatal("request did not dispatch after detector was unblocked")
	}

	// The gauge decrement runs before EnqueueAndWait returns, so once the result is observed the
	// gauge is already back at 0 -- no polling needed.
	require.Zero(t, queueSizeGaugeSum(t),
		"queue_size should return to 0 after the request finalizes")
}

// ============================================================================
// Agentic Churn and Stress Tests
// ============================================================================

// TestHighConcurrencyFlowChurnNoDeadlock sends requests from many goroutines,
// each with a unique flow ID, while the registry GC runs concurrently.
// Verifies no deadlocks, panics, or data races under contention between
// WithConnection (write lock for new flows) and executeGCCycle (write lock
// for cleanup).
func TestHighConcurrencyFlowChurnNoDeadlock(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(100 * time.Millisecond)
	})

	handle := testutils.NewTestHandle(ctx)
	oPolicy, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory("gs", nil, handle)
	require.NoError(t, err)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxBytes(10_000_000_000),
	)
	require.NoError(t, err)

	// Very aggressive GC to maximize contention with WithConnection.
	regCfg, err := registry.NewConfig(defaults,
		registry.WithPriorityBand(band),
		registry.WithFlowGCTimeout(50*time.Millisecond),
	)
	require.NoError(t, err)

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.RunMaintenanceLoop(ctx)

	detector := newGatedDetector(0)

	fc := controller.NewFlowController(ctx, "contention-test", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    50 * time.Millisecond,
		EnqueueChannelBufferSize: 200,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: detector,
		EndpointCandidates: &contractmocks.MockEndpointCandidates{},
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	time.Sleep(10 * time.Millisecond)

	const numGoroutines = 50
	const reqsPerGoroutine = 20

	results := make(chan dispatchResult, numGoroutines*reqsPerGoroutine)

	for g := 0; g < numGoroutines; g++ {
		go func() {
			for r := 0; r < reqsPerGoroutine; r++ {
				id := fmt.Sprintf("g%d-r%d", g, r)
				key := flowcontrol.FlowKey{ID: id, Priority: 0}
				req := &testRequest{id: id, key: key, byteSize: 100, ttl: 5 * time.Minute}
				outcome, err := fc.EnqueueAndWait(ctx, req)
				results <- dispatchResult{id: id, outcome: outcome, err: err}
			}
		}()
	}

	total := numGoroutines * reqsPerGoroutine
	dispatched := 0
	for i := 0; i < total; i++ {
		select {
		case r := <-results:
			require.NoError(t, r.err,
				"request %s should not error under contention", r.id)
			require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
			dispatched++
		case <-time.After(30 * time.Second):
			t.Fatalf("deadlock detected: only %d/%d requests completed", dispatched, total)
		}
	}

	require.Equal(t, total, dispatched,
		"all requests should dispatch without deadlock under concurrent flow churn + GC")

	// Wait for flow GC to reclaim idle flows.
	time.Sleep(200 * time.Millisecond)

	// Verify aggregate stats are consistent after dispatch + GC.
	stats := reg.Stats()
	require.Equal(t, uint64(0), stats.TotalLen,
		"registry should have 0 queued items after all dispatched + GC")
	require.Equal(t, uint64(0), stats.TotalByteSize,
		"registry should have 0 bytes after all dispatched + GC")
}

// TestEndpointChurnUnderLoad verifies that when endpoints disappear and reappear
// while requests are queued, the dispatch cycle reacts correctly:
//   - Endpoints gone -> Saturation()=1.0 (fail closed) -> requests stay queued
//   - Endpoints return -> Saturation() drops -> requests dispatch
func TestEndpointChurnUnderLoad(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		time.Sleep(50 * time.Millisecond)
	})

	pd := newProducerAndDetector(ctx, t, 100)

	// Mutable endpoint list that the dispatch cycle reads via Locate().
	var endpointsMu atomic.Value
	endpointsMu.Store([]datalayer.Endpoint{pd.ep})

	endpointCandidates := &contractmocks.MockEndpointCandidates{
		LocateFunc: func(_ context.Context, _ map[string]any) []datalayer.Endpoint {
			return endpointsMu.Load().([]datalayer.Endpoint)
		},
	}

	h := newHarness(t, harnessOpts{
		detector:           pd.detector,
		endpointCandidates: endpointCandidates,
	})

	key := flowcontrol.FlowKey{ID: "flow-a", Priority: 0}

	// Phase 1: Endpoints present, no load -> requests should dispatch.
	results := make(chan dispatchResult, 1)
	go func() {
		req := &testRequest{id: "before-churn", key: key, byteSize: 100, ttl: 5 * time.Second}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{id: "before-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should dispatch when endpoints are healthy")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not dispatch with healthy endpoints")
	}

	// Phase 2: Remove all endpoints -> Locate() returns empty -> saturation=1.0.
	endpointsMu.Store([]datalayer.Endpoint{})

	go func() {
		reqCtx, reqCancel := context.WithTimeout(h.ctx, 500*time.Millisecond)
		defer reqCancel()
		req := &testRequest{id: "during-churn", key: key, byteSize: 100, ttl: 500 * time.Millisecond}
		outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
		results <- dispatchResult{id: "during-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		// Pool is empty -> saturation=1.0 -> request queues -> 500ms timeout fires.
		require.NotEqual(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should not dispatch when all endpoints are gone")
		require.Error(t, r.err, "request should return an error when endpoints are gone")
		require.ErrorIs(t, r.err, fcTypes.ErrEvicted,
			"request should be evicted (TTL/deadline) when endpoints are gone, not silently dropped")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not terminate after endpoint removal")
	}

	// Phase 3: Restore endpoints -> requests should dispatch again.
	endpointsMu.Store([]datalayer.Endpoint{pd.ep})

	go func() {
		req := &testRequest{id: "after-churn", key: key, byteSize: 100, ttl: 5 * time.Second}
		outcome, err := h.fc.EnqueueAndWait(h.ctx, req)
		results <- dispatchResult{id: "after-churn", outcome: outcome, err: err}
	}()

	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome,
			"request should dispatch after endpoints are restored")
	case <-time.After(3 * time.Second):
		t.Fatal("request did not dispatch after endpoint restoration")
	}
}
