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

	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	requesthandling "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils"
)

// BenchmarkFlowController_PerformanceMatrix evaluates throughput across a matrix of variables.
// It systematically evaluates the impact of strict egress limits, data parallelism, priority
// levels, flow density, and concurrent connections.
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

	fc, detector := setupBenchmarkHarness(ctx, b, m.priorities, m.limit, nil, nil)

	// Yield briefly to allow the background supervisor to bootstrap the data plane.
	time.Sleep(10 * time.Millisecond)

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

// fullPathBenchHarness holds shared setup state for full-path benchmarks that
// wire a real InFlightLoadProducer, real concurrency detector, and real
// persistent endpoints into the FlowController.
type fullPathBenchHarness struct {
	fc       *controller.FlowController
	producer *inflightload.InFlightLoadProducer
	epMeta   *fwkdl.EndpointMetadata
	schedEp  scheduling.Endpoint
}

// setupFullPathBenchmark creates the shared harness for benchmarks that exercise
// the complete flow control data path: producer -> detector -> controller -> cleanup.
func setupFullPathBenchmark(ctx context.Context, b *testing.B, name string, numPriorities int) *fullPathBenchHarness {
	b.Helper()
	if numPriorities < 1 {
		numPriorities = 1
	}
	handle := igwtestutils.NewTestHandle(ctx)

	producerName := name + "-producer"
	producerPlugin, err := inflightload.InFlightLoadProducerFactory(
		producerName, fwkplugin.StrictDecoder([]byte(`{}`)), handle,
	)
	if err != nil {
		b.Fatal(err)
	}
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)

	detectorPlugin, err := concurrency.ConcurrencyDetectorFactory(
		name+"-detector",
		fwkplugin.StrictDecoder([]byte(fmt.Sprintf(
			`{"maxConcurrency": 100, "inFlightLoadProducerName": %q}`, producerName,
		))),
		handle,
	)
	if err != nil {
		b.Fatal(err)
	}
	realDetector := detectorPlugin.(flowcontrol.SaturationDetector)

	epMeta := &fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
	}
	ep := fwkdl.NewEndpoint(epMeta, fwkdl.NewMetrics())
	if err := producer.Extract(ctx, fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: ep,
	}); err != nil {
		b.Fatal(err)
	}

	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory(registry.DefaultFairnessPolicyRef, nil, handle)
	if err != nil {
		b.Fatal(err)
	}
	handle.AddPlugin(registry.DefaultFairnessPolicyRef, fPolicy)

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory(registry.DefaultOrderingPolicyRef, nil, handle)
	if err != nil {
		b.Fatal(err)
	}
	handle.AddPlugin(registry.DefaultOrderingPolicyRef, oPolicy)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}
	reg := setupRegistry(ctx, b, defaults, priorityLevels(numPriorities))

	fc := controller.NewFlowController(ctx, name+"-bench", &controller.Config{
		DefaultRequestTTL:        5 * time.Minute,
		ExpiryCleanupInterval:    1 * time.Hour,
		EnqueueChannelBufferSize: 2000,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: realDetector,
		EndpointCandidates: &mocks.MockEndpointCandidates{Candidates: []fwkdl.Endpoint{ep}},
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	time.Sleep(10 * time.Millisecond)

	schedEp := scheduling.NewEndpoint(epMeta, fwkdl.NewMetrics(), nil)

	return &fullPathBenchHarness{
		fc:       fc,
		producer: producer,
		epMeta:   epMeta,
		schedEp:  schedEp,
	}
}

// BenchmarkFlowController_FullPathStress exercises the complete flow control
// data path under realistic production conditions: real InFlightLoadProducer,
// real concurrency detector reading DynamicAttributes from persistent endpoints,
// real FlowController with backpressure from a concurrency-gated detector,
// multiple priority bands, concurrent workers creating unique flows (agentic
// churn), and the full PreRequest -> dispatch -> StartOfStream -> EndOfStream
// lifecycle per request.
//
// What it catches:
//   - PluginState entry leaks (addedTokensEntry not cleaned up by OnEvicted)
//   - DynamicAttribute closure leaks (producer tracker never decremented)
//   - Cross-band counter drift (stats routed to wrong priority band)
//   - Registry flow infrastructure leaks under concurrent churn
//   - Contention-induced allocation growth (lock convoy, sync.Map thrashing)
//
// Run with:
//
//	go test -bench=FullPathStress -benchtime=5000x -count=3 -run=^$ ./pkg/epp/flowcontrol/benchmark/
//
// Stable B/op across runs = no leak. Counter assertion at the end catches
// tracker drift that B/op alone would miss.
func BenchmarkFlowController_FullPathStress(b *testing.B) {
	const numPriorities = 4

	ctx := b.Context()
	h := setupFullPathBenchmark(ctx, b, "stress", numPriorities)

	sosResp := &requestcontrol.Response{StartOfStream: true}
	eosResp := &requestcontrol.Response{EndOfStream: true}
	profileResults := map[string]*scheduling.ProfileRunResult{
		"decode": {TargetEndpoints: []scheduling.Endpoint{h.schedEp}},
	}

	// Concurrency-gated: each dispatched request consumes a slot. Workers
	// release immediately after ResponseBody, creating sustained backpressure
	// (W workers > L limit). This forces real queuing in the dispatch cycle.
	var globalReqID atomic.Uint64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(max(100/runtime.GOMAXPROCS(0), 1))

	b.RunParallel(func(pb *testing.PB) {
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
				b.Fatalf("request %s was not dispatched: %v", reqID, outcome)
			}

			// 2. Post-scheduling: producer tracks the request on the endpoint.
			infReq := &scheduling.InferenceRequest{
				RequestID: reqID,
				Body:      &requesthandling.InferenceRequestBody{TokenizedPrompt: &requesthandling.TokenizedPrompt{TokenIDs: benchTokenIDs}},
			}
			schedResult := &scheduling.SchedulingResult{ProfileResults: profileResults}
			h.producer.PreRequest(ctx, infReq, schedResult)

			// 3. Response lifecycle: release counters.
			infReq.SchedulingResult = schedResult
			h.producer.ResponseBody(ctx, infReq, sosResp, h.epMeta)
			h.producer.ResponseBody(ctx, infReq, eosResp, h.epMeta)
		}
	})

	b.StopTimer()
	b.ReportMetric(float64(globalReqID.Load()), "total-requests")

	finalRequests := h.producer.GetRequests(h.epMeta.NamespacedName.String())
	finalTokens := h.producer.GetTokens(h.epMeta.NamespacedName.String())
	if finalRequests != 0 || finalTokens != 0 {
		b.Errorf("counter leak detected after %d requests across %d priorities: requests=%d, tokens=%d",
			globalReqID.Load(), numPriorities, finalRequests, finalTokens)
	}
}

// benchTokenIDs is a pre-allocated token slice to avoid per-iteration allocation noise.
var benchTokenIDs = make([]uint32, 50)
