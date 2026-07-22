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
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func init() {
	// Silence verbose logging during aggressive scaling benchmarks to prevent I/O blocking.
	log.SetLogger(logr.Discard())
}

// egressConcurrencyLimit (L) defines the maximum capacity of the simulated pool.
type egressConcurrencyLimit int64

// priorityLevels (P) dictates the number of priority bands.
type priorityLevels int

// flowCount (F) dictates the number of active flows.
// Relative to queue depth (W - L), sweeping F shifts load between a few deep queues (stressing
// Ordering policies) and many shallow queues (stressing Fairness policies).
type flowCount int

// ingressConcurrency (W) dictates the volume of simultaneous incoming requests.
type ingressConcurrency int

// benchMatrix defines a single coordinate in the performance hypercube.
type benchMatrix struct {
	limit       egressConcurrencyLimit
	priorities  priorityLevels
	flows       flowCount
	concurrency ingressConcurrency
}

// name returns a human-readable string representation of the matrix coordinate.
func (m benchMatrix) name() string {
	return fmt.Sprintf("L=%03d/P=%03d/F=%06d/W=%05d",
		m.limit, m.priorities, m.flows, m.concurrency)
}

// testDetector exposes an API to manually release downstream capacity during a test run.
type testDetector interface {
	flowcontrol.SaturationDetector
	Release()
}

// benchDetector is a mock SaturationDetector that models a strict concurrency limit (L) without
// parking goroutines. The processor's dispatch loop calls Saturation once per cycle and then
// dispatches at most one item; the benchmark worker calls Release after that item is dispatched.
//
// Saturation optimistically reserves a permit (inFlight++); Release frees it. Under the matrix's
// W>L load the queue is always full, so every grant leads to a dispatch and a Release: inFlight
// accumulates to L and Saturation reports 1.0 once exceeded, which is the backpressure the matrix
// measures.
//
// The complication: the processor also runs dispatchCycle on a 1ms ticker when the queue is EMPTY
// (notably at startup). Those idle grants never dispatch and so are never Released; a naive detector
// would latch saturated forever and deadlock every request until its TTL. We distinguish leaked
// idle grants from genuine load saturation with releaseCount: while releases are
// flowing the saturation is real; if two consecutive saturated reads see zero releases between them
// the outstanding grants are leaked idle permits, so we reclaim one. The 1.0/0.99 split sits below
// usagelimits.DefaultPolicy's 1.0 ceiling, so a grant never trips head-of-line gating.
//
// stuckReads and lastReleaseCount are touched only by the single-threaded dispatch loop (Saturation)
// and so need no synchronization; inFlight and releaseCount are atomic (Release runs on workers).
type benchDetector struct {
	flowcontrol.SaturationDetector
	concurrencyLimit atomic.Int64
	// _ prevents false sharing between concurrencyLimit and the hot atomic counters below.
	_            [48]byte
	inFlight     atomic.Int64
	releaseCount atomic.Int64

	stuckReads       int
	lastReleaseCount int64
}

// stuckReadThreshold is the number of consecutive saturated reads with zero intervening releases
// after which the detector treats the outstanding grants as leaked idle permits and reclaims one.
// >1 so that a single slow Release under load does not cause a spurious reclaim.
//
// The heuristic is deliberately biased toward liveness: if a genuine holder's Release stalls past
// the threshold (e.g. its worker goroutine is descheduled), the reclaim transiently over-admits
// beyond L rather than risking a latched-saturated deadlock. For a CPU-cost benchmark that is the
// right trade: true queueing still reports saturated (1.0) while releases flow, and a slightly
// loose L only shifts the coordinate's operating point, whereas a latch would hang the run.
const stuckReadThreshold = 2

// Release frees the permit held by the dispatch that just completed.
func (d *benchDetector) Release() {
	if d.concurrencyLimit.Load() <= 0 {
		return
	}
	d.inFlight.Add(-1)
	d.releaseCount.Add(1)
}

// Saturation reserves a permit for this cycle, reporting saturated once the limit is exceeded and
// self-healing leaked idle grants (see the type doc).
func (d *benchDetector) Saturation(ctx context.Context, candidates []fwkdl.Endpoint) float64 {
	limit := d.concurrencyLimit.Load()
	if limit <= 0 {
		return 0.0 // Free-flow
	}

	// Optimistically reserve a permit for this cycle.
	if d.inFlight.Add(1) <= limit {
		d.stuckReads = 0
		return 0.99 // Return < 1.0 so the dispatcher proceeds.
	}

	// Capacity exceeded; roll back the speculative reservation.
	d.inFlight.Add(-1)

	rc := d.releaseCount.Load()
	if rc != d.lastReleaseCount {
		// Releases are flowing: this is genuine load saturation, not a leak.
		d.lastReleaseCount = rc
		d.stuckReads = 0
		return 1.0
	}

	// No releases since the last saturated read. After a couple of these the outstanding grants must
	// be leaked idle permits (an empty-queue dispatch tick reserved but never dispatched), so reclaim
	// one to keep the detector from latching.
	d.stuckReads++
	if d.stuckReads >= stuckReadThreshold {
		d.inFlight.Add(-1)
		d.stuckReads = 0
	}
	return 1.0 // Saturated - forces the Flow Control layer to hold the item.
}

// alwaysSaturatedDetector simulates a permanently saturated downstream pool.
type alwaysSaturatedDetector struct {
	flowcontrol.SaturationDetector
}

// Release is a no-op for the permanently saturated mock.
func (d *alwaysSaturatedDetector) Release() {}

// Saturation permanently returns 1.0 (100% saturated) to ensure requests queue.
func (d *alwaysSaturatedDetector) Saturation(ctx context.Context, candidates []fwkdl.Endpoint) float64 {
	return 1.0
}

// benchRequest models an inbound inference request with realistic payload entropy.
type benchRequest struct {
	key      flowcontrol.FlowKey
	byteSize uint64
}

// --- stubs required by FlowControlRequest interface ---
func (r *benchRequest) FlowKey() flowcontrol.FlowKey                   { return r.key }
func (r *benchRequest) ByteSize() uint64                               { return r.byteSize }
func (r *benchRequest) InitialEffectiveTTL() time.Duration             { return 5 * time.Minute }
func (r *benchRequest) ID() string                                     { return "bench-req" }
func (r *benchRequest) GetMetadata() map[string]any                    { return nil }
func (r *benchRequest) InferencePoolName() string                      { return "bench-pool" }
func (r *benchRequest) ModelName() string                              { return "bench-model" }
func (r *benchRequest) TargetModelName() string                        { return "bench-target" }
func (r *benchRequest) InferenceRequest() *scheduling.InferenceRequest { return nil }
func (r *benchRequest) ReceivedTimestamp() time.Time                   { return time.Now() }

// setupRegistry provisions the concrete FlowRegistry.
func setupRegistry(
	b *testing.B,
	defaults registry.PriorityBandPolicyDefaults,
	p priorityLevels,
) contracts.FlowRegistry {
	b.Helper()

	cfgOpts := []registry.ConfigOption{
		registry.WithMaxBytes(0), // Capacity restricted strictly via concurrency (L).
	}

	for i := 0; i < int(p); i++ {
		band, err := registry.NewPriorityBandConfig(
			i, defaults,
			registry.WithBandMaxBytes(10_000_000_000), // Prevent capacity-based rejections.
		)
		if err != nil {
			b.Fatalf("Failed to init priority band %d: %v", i, err)
		}
		cfgOpts = append(cfgOpts, registry.WithPriorityBand(band))
	}

	regCfg, err := registry.NewConfig(defaults, cfgOpts...)
	if err != nil {
		b.Fatalf("Failed to create registry config: %v", err)
	}

	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	// Registry maintenance (GC, priority band sync) runs in the Processor loop started by FlowController.
	return reg
}

// setupRegistryWithDefaultPolicies registers the default FCFS ordering and global-strict fairness
// policies on the handle, then provisions a FlowRegistry with `p` priority bands using them as the
// per-band defaults. Shared by every benchmark harness so the policy wiring lives in one place.
func setupRegistryWithDefaultPolicies(
	b *testing.B,
	handle fwkplugin.Handle,
	p priorityLevels,
) contracts.FlowRegistry {
	b.Helper()

	fPolicy, err := globalstrict.GlobalStrictFairnessPolicyFactory(registry.DefaultFairnessPolicyRef, nil, handle)
	if err != nil {
		b.Fatalf("Failed to create fairness policy: %v", err)
	}
	handle.AddPlugin(registry.DefaultFairnessPolicyRef, fPolicy)

	oPolicy, err := fcfs.FCFSOrderingPolicyFactory(registry.DefaultOrderingPolicyRef, nil, handle)
	if err != nil {
		b.Fatalf("Failed to create ordering policy: %v", err)
	}
	handle.AddPlugin(registry.DefaultOrderingPolicyRef, oPolicy)

	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: oPolicy.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fPolicy.(flowcontrol.FairnessPolicy),
	}

	return setupRegistry(b, defaults, p)
}

// setupBenchmarkHarness creates the standard SUT environment.
func setupBenchmarkHarness(
	ctx context.Context,
	b *testing.B,
	p priorityLevels,
	limit egressConcurrencyLimit,
	customDetector testDetector,
	customCfg *controller.Config,
) (*controller.FlowController, testDetector) {
	b.Helper()
	handle := testutils.NewTestHandle(ctx)

	reg := setupRegistryWithDefaultPolicies(b, handle, p)

	detector := customDetector
	if detector == nil {
		defaultDetector := &benchDetector{}
		defaultDetector.concurrencyLimit.Store(int64(limit))
		detector = defaultDetector
	}

	cfg := customCfg
	if cfg == nil {
		cfg = &controller.Config{
			// Nothing in a throughput benchmark legitimately waits minutes: the worst observed
			// steady-state queue wait is ~10s (L=1, W=5000). A tight TTL bounds the cost of a wedged
			// coordinate to seconds of CI time instead of minutes per cell.
			DefaultRequestTTL:        30 * time.Second,
			ExpiryCleanupInterval:    1 * time.Hour, // Effectively disabled
			EnqueueChannelBufferSize: 2000,
		}
	}

	fc := controller.NewFlowController(ctx, "benchmark", cfg, controller.Deps{
		Registry:           reg,
		SaturationDetector: detector,
		EndpointCandidates: &mocks.MockEndpointCandidates{},
		UsageLimitPolicy:   usagelimits.DefaultPolicy()},
	)

	return fc, detector
}

// fullPathBenchHarness holds shared setup state for full-path benchmarks that wire a real
// InFlightLoadProducer, real concurrency detector, and real persistent endpoints into the
// FlowController.
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
	handle := testutils.NewTestHandle(ctx)

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

	reg := setupRegistryWithDefaultPolicies(b, handle, priorityLevels(numPriorities))

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

	schedEp := scheduling.NewEndpoint(epMeta, fwkdl.NewMetrics(), nil)

	// Warm up: block on one request until it's dispatched, proving the dispatch loop is live before
	// timing. It exercises only the FlowController, not the producer, so no in-flight load is left
	// behind.
	warmup := &benchRequest{key: flowcontrol.FlowKey{ID: "warmup", Priority: 0}, byteSize: 512}
	if outcome, _ := fc.EnqueueAndWait(ctx, warmup); outcome != types.QueueOutcomeDispatched {
		b.Fatalf("full-path warmup did not dispatch: %v", outcome)
	}

	return &fullPathBenchHarness{
		fc:       fc,
		producer: producer,
		epMeta:   epMeta,
		schedEp:  schedEp,
	}
}

// benchmarkTelemetry aggregates benchmark statistics lock-free: threads mutate local structs and
// commit totals once at the end.
type benchmarkTelemetry struct {
	errorCount    atomic.Int64
	dispatchCount atomic.Int64
	rejectCount   atomic.Int64
}

// newBenchmarkTelemetry provisions the global telemetry aggregator.
func newBenchmarkTelemetry() *benchmarkTelemetry {
	return &benchmarkTelemetry{}
}

// threadTelemetry is a thread-local accumulator for benchmark statistics.
type threadTelemetry struct {
	errs int64
	disp int64
	rej  int64
}

// recordDispatch logs a successful dispatch for the thread.
func (t *benchmarkTelemetry) recordDispatch(local *threadTelemetry) {
	local.disp++
}

// recordError tracks system evaluation errors.
func (t *benchmarkTelemetry) recordError(local *threadTelemetry) {
	local.errs++
}

// recordReject logs explicit QueueOutcomeRejected events.
func (t *benchmarkTelemetry) recordReject(local *threadTelemetry) {
	local.rej++
}

// commit adds thread-local statistics to the global atomic counts.
func (t *benchmarkTelemetry) commit(local *threadTelemetry) {
	if local.errs > 0 {
		t.errorCount.Add(local.errs)
	}
	if local.disp > 0 {
		t.dispatchCount.Add(local.disp)
	}
	if local.rej > 0 {
		t.rejectCount.Add(local.rej)
	}
}

// report aggregates the committed globals and issues the standard b.ReportMetric calls.
func (t *benchmarkTelemetry) report(b *testing.B, elapsed float64) {
	if elapsed <= 0 {
		return
	}

	b.ReportMetric(math.Round(float64(t.dispatchCount.Load())/elapsed), "d/s")
	if rejects := t.rejectCount.Load(); rejects > 0 {
		b.ReportMetric(math.Round(float64(rejects)/elapsed), "r/s")
	}
	if errs := t.errorCount.Load(); errs > 0 {
		b.ReportMetric(float64(errs), "errors")
	}
}
