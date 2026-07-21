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

package benchmark

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	requesthandling "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// BenchmarkFlowController_PerformanceMatrix evaluates throughput across a matrix of variables.
// It systematically evaluates the impact of strict egress limits, priority levels, flow density,
// and concurrent connections.
func BenchmarkFlowController_PerformanceMatrix(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping PerformanceMatrix in short mode")
	}
	limits := []egressConcurrencyLimit{0, 1, 100}
	priorities := []priorityLevels{1, 8}
	flows := []flowCount{10, 5000}
	concurrencies := []ingressConcurrency{10, 5000}

	for _, L := range limits {
		for _, P := range priorities {
			for _, F := range flows {
				for _, W := range concurrencies {
					// Skip illogical boundaries.
					if L == 0 && W > 100 {
						continue // High concurrency is redundant for free-flow.
					}
					if L > 0 && int64(W) <= int64(L) {
						continue // Requires W > L to generate backpressure.
					}

					matrix := benchMatrix{limit: L, priorities: P, flows: F, concurrency: W}
					b.Run(matrix.name(), func(b *testing.B) {
						runMatrixCoordinate(b, matrix)
					})
				}
			}
		}
	}
}

// runMatrixCoordinate executes a single coordinate of the performance hypercube.
func runMatrixCoordinate(b *testing.B, m benchMatrix) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure SUT goroutines are torn down even when the coordinate fails fatally.

	fc, detector := setupBenchmarkHarness(ctx, b, m.priorities, m.limit, nil, nil)

	// Warm up: block on one request until it's dispatched, proving the dispatch loop is live before
	// timing. Release the slot per the loop's immediate-drain protocol (a no-op under free-flow).
	warmup := &benchRequest{key: flowcontrol.FlowKey{ID: "warmup", Priority: 0}, byteSize: 1024}
	if outcome, err := fc.EnqueueAndWait(ctx, warmup); outcome != types.QueueOutcomeDispatched {
		b.Fatalf("warmup request was not dispatched: outcome=%v, err=%v", outcome, err)
	}
	detector.Release()

	reqs := make([]*benchRequest, m.flows)
	for i := 0; i < int(m.flows); i++ {
		// Use the Knuth 32-bit multiplier to deterministically scatter payload sizes (100B - 9KB).
		hash := uint32(i) * 2654435769
		reqs[i] = &benchRequest{
			key: flowcontrol.FlowKey{
				ID:       fmt.Sprintf("flow-%d", i),
				Priority: i % int(m.priorities),
			},
			byteSize: 100 + uint64(hash%9000), // Payload entropy between 100B and 9KB.
		}
	}

	telemetry := newBenchmarkTelemetry()

	// Scale execution threads to match simulated concurrency.
	procs := runtime.GOMAXPROCS(0)
	parallelism := max(int(m.concurrency)/procs, 1)
	b.SetParallelism(parallelism)

	numFlows := int(m.flows)

	// Round up to the next power of two for fast modulo via bitmasking.
	zipfSize := 1
	for zipfSize < numFlows*4 {
		zipfSize <<= 1
	}
	zipfMask := zipfSize - 1
	zipfIndices := make([]int, zipfSize)

	if numFlows > 1 {
		// Pre-compute with a deterministic seed to ensure benchmark consistency.
		// The (1.1, 1.0) parameters bias selections toward lower indices, simulating a "hot tenant".
		rng := rand.New(rand.NewSource(1))
		zipf := rand.NewZipf(rng, 1.1, 1.0, uint64(numFlows-1))
		for i := 0; i < zipfSize; i++ {
			zipfIndices[i] = int(zipf.Uint64())
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	var globalThreadID atomic.Uint64

	b.RunParallel(func(pb *testing.PB) {
		var localTelemetry threadTelemetry

		// Offset the starting index per thread to prevent identical striding over the array.
		// Multiply by a prime to guarantee threads start at different offsets.
		threadID := globalThreadID.Add(1)
		localIdx := int(threadID) * 9973

		for pb.Next() {
			localIdx++

			flowIdx := zipfIndices[localIdx&zipfMask]
			sourceReq := reqs[flowIdx]
			outcome, err := fc.EnqueueAndWait(ctx, sourceReq)

			if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				telemetry.recordError(&localTelemetry)
				if outcome != types.QueueOutcomeDispatched {
					telemetry.recordReject(&localTelemetry)
				}
				continue
			}

			if outcome == types.QueueOutcomeDispatched {
				telemetry.recordDispatch(&localTelemetry)
				if m.limit > 0 {
					// Free capacity instantly to maintain active queue depth (W - L).
					detector.Release()
				}
			}
		}

		telemetry.commit(&localTelemetry)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	telemetry.report(b, elapsed)

	// Every coordinate expects flow (W>L or free-flow), so zero dispatches means the dispatch path
	// wedged mid-run: fail loudly instead of publishing plausible-looking zero rows. Assert on the
	// benchmark goroutine (never inside RunParallel, where FailNow is invalid).
	if telemetry.dispatchCount.Load() == 0 {
		b.Fatalf("coordinate %s dispatched zero requests; the dispatch path is wedged", m.name())
	}

	cancel()                          // Graceful teardown to prevent async skewing of subsequent coordinates.
	time.Sleep(50 * time.Millisecond) // Wait for SUT background goroutines to terminate.
}

// BenchmarkFlowController_TopologyChurn evaluates dynamic provisioning overhead by continuously
// registering new flows, forcing the Registry to acquire write locks.
func BenchmarkFlowController_TopologyChurn(b *testing.B) {
	ctx := b.Context()

	cfg := &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    1 * time.Hour, // Effectively disabled
		EnqueueChannelBufferSize: 100,
	}

	fc, detector := setupBenchmarkHarness(ctx, b, 1, 100, nil, cfg)

	const numKeys = 5000
	preAllocatedReqs := make([]*benchRequest, numKeys)
	for i := range numKeys {
		preAllocatedReqs[i] = &benchRequest{
			key:      flowcontrol.FlowKey{ID: fmt.Sprintf("novel-flow-%d", i), Priority: 0},
			byteSize: 1024,
		}
	}

	var dispatchCount atomic.Int64
	var globalThreadID atomic.Uint64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(100)

	b.RunParallel(func(pb *testing.PB) {
		var localDisp int64

		// Multiply by a prime to guarantee threads start at different modulo offsets, avoiding lockstep
		// contention on the exact same Registry keys.
		localID := int(globalThreadID.Add(1)) * 9973

		for pb.Next() {
			localID++
			req := preAllocatedReqs[localID%numKeys]

			outcome, _ := fc.EnqueueAndWait(ctx, req)
			if outcome == types.QueueOutcomeDispatched {
				localDisp++
				detector.Release()
			}
		}
		dispatchCount.Add(localDisp)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(math.Round(float64(dispatchCount.Load())/elapsed), "d/s")
	}
}

// BenchmarkFlowController_MassCancellation evaluates the cleanup overhead of client abandonment by
// aggressively timing out requests under high load.
func BenchmarkFlowController_MassCancellation(b *testing.B) {
	ctx := b.Context()

	cfg := &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    10 * time.Millisecond, // Hyper-aggressive sweep for benchmark
		EnqueueChannelBufferSize: 100,
	}

	// Use the permanently saturated detector to guarantee all requests queue and definitively rot.
	fc, _ := setupBenchmarkHarness(ctx, b, 1, 100, &alwaysSaturatedDetector{}, cfg)

	var timeoutCount atomic.Int64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(100)

	b.RunParallel(func(pb *testing.PB) {
		var localTimeout int64
		req := &benchRequest{
			key:      flowcontrol.FlowKey{ID: "zombie-flow", Priority: 0},
			byteSize: 1024,
		}

		for pb.Next() {
			reqCtx, reqCancel := context.WithTimeout(ctx, 50*time.Millisecond)
			_, err := fc.EnqueueAndWait(reqCtx, req)
			reqCancel()

			if errors.Is(err, context.DeadlineExceeded) ||
				errors.Is(err, types.ErrTTLExpired) ||
				errors.Is(err, context.Canceled) {
				localTimeout++
			}
		}
		timeoutCount.Add(localTimeout)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(math.Round(float64(timeoutCount.Load())/elapsed), "zombies/s")
	}
}

// BenchmarkFlowController_FullPath measures dispatch throughput and allocation overhead of the
// complete flow-control data path using REAL components: a real InFlightLoadProducer feeding a real
// concurrency SaturationDetector, which gates a real FlowController. Unlike the throughput
// microbenchmarks (which use the mock benchDetector), this captures the cost of the detector's
// per-dispatch DynamicAttribute reads against live producer state in the hot path.
//
// Each iteration runs the full request lifecycle: EnqueueAndWait (admission) -> PreRequest
// (producer records in-flight load) -> ResponseBody StartOfStream / EndOfStream (producer releases
// it). Workers spread across priority bands and mint a unique flow per request for registry churn.
//
// Run with:
//
//	go test -bench=FullPath -run=^$ ./pkg/epp/flowcontrol/benchmark/
//
// Reports d/s (dispatch throughput). Producer-level leak correctness is covered by the inflightload
// producer's own unit tests, not here.
func BenchmarkFlowController_FullPath(b *testing.B) {
	const numPriorities = 4

	ctx := b.Context()
	h := setupFullPathBenchmark(ctx, b, "fullpath", numPriorities)

	sosResp := &requestcontrol.Response{StartOfStream: true}
	eosResp := &requestcontrol.Response{EndOfStream: true}
	profileResults := map[string]*scheduling.ProfileRunResult{
		"decode": {TargetEndpoints: []scheduling.Endpoint{h.schedEp}},
	}

	telemetry := newBenchmarkTelemetry()

	// Concurrency-gated by the real detector: each dispatched request records in-flight load (via
	// PreRequest) and releases it after ResponseBody. Load is held only briefly, so this measures
	// free-flowing dispatch plus the detector's per-cycle DynamicAttribute read cost rather than
	// sustained W>L queuing.
	var globalReqID atomic.Uint64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(max(100/runtime.GOMAXPROCS(0), 1))

	b.RunParallel(func(pb *testing.PB) {
		var local threadTelemetry
		for pb.Next() {
			id := globalReqID.Add(1)
			reqID := fmt.Sprintf("req-%d", id)
			priority := int(id) % numPriorities

			// 1. Admission: FlowController gates the request.
			fcReq := &benchRequest{
				key:      flowcontrol.FlowKey{ID: reqID, Priority: priority},
				byteSize: 512,
			}
			outcome, _ := h.fc.EnqueueAndWait(ctx, fcReq)
			if outcome != types.QueueOutcomeDispatched {
				telemetry.recordReject(&local)
				continue
			}
			telemetry.recordDispatch(&local)

			// 2. Post-scheduling: producer records in-flight load on the endpoint, which the
			//    detector reads to compute saturation.
			infReq := &scheduling.InferenceRequest{
				RequestID: reqID,
				Body:      &requesthandling.InferenceRequestBody{TokenizedPrompt: &requesthandling.TokenizedPrompt{PerPromptTokens: [][]uint32{benchTokenIDs}}},
			}
			schedResult := &scheduling.SchedulingResult{ProfileResults: profileResults}
			h.producer.PreRequest(ctx, infReq, schedResult)

			// 3. Response lifecycle: release the in-flight load.
			infReq.SchedulingResult = schedResult
			h.producer.ResponseBody(ctx, infReq, sosResp, h.epMeta)
			h.producer.ResponseBody(ctx, infReq, eosResp, h.epMeta)
		}
		telemetry.commit(&local)
	})

	b.StopTimer()
	telemetry.report(b, b.Elapsed().Seconds())

	// A fully saturated full-path run must dispatch every request; a rejection
	// means the throughput numbers are skewed. Assert on the benchmark goroutine
	// (never inside RunParallel, where FailNow is invalid).
	if rejects := telemetry.rejectCount.Load(); rejects > 0 {
		b.Fatalf("full-path benchmark saw %d rejected requests; throughput numbers are unreliable", rejects)
	}
}

// benchTokenIDs is a pre-allocated token slice to avoid per-iteration allocation noise.
var benchTokenIDs = make([]uint32, 50)
