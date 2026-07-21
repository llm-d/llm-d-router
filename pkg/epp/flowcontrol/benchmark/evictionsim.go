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

// This file implements a closed-loop simulator for the eviction/reclamation dynamics benchmark
// (BenchmarkEvictionDynamics). It models an inference pool as a fixed number of capacity units
// occupied by sheddable leases, with configurable sensor lag between a lease's termination and
// the moment the saturation gauge reflects the freed capacity. All flow control components are
// real (FlowController, registry, RequestEvictor, ReclamationController); only the pool and its
// sensor are synthetic. See docs/flow-control-eviction.md for the pacing design under test.
package benchmark

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/eviction"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	evictfiltering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/filtering"
	evictordering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/ordering"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

// sensorProfile models where the saturation sensor sits relative to reality when a lease frees.
type sensorProfile struct {
	name string
	// termLatency is the delay between the eviction signal (EvictCh close) and the stream's
	// termination (the confirmation event).
	termLatency time.Duration
	// freeVisibilityLag is the additional delay between termination and the freed capacity
	// appearing in the gauge (engine GC + metrics scrape + staleness for a utilization-style
	// sensor; ~0 for a concurrency-style sensor whose counters decrement at termination).
	freeVisibilityLag time.Duration
}

var (
	profileConcurrency = sensorProfile{name: "conc", termLatency: 3 * time.Millisecond, freeVisibilityLag: 0}
	profileUtilization = sensorProfile{name: "util", termLatency: 3 * time.Millisecond, freeVisibilityLag: 200 * time.Millisecond}
)

// burstShape describes the high-priority demand applied to the saturated pool.
type burstShape struct {
	name       string
	hpCount    int           // total HP requests issued (rate-shaped when window > 0)
	arrivalGap time.Duration // spacing between HP arrivals (0 = simultaneous burst)
	// serviceTime is how long a dispatched HP request occupies its capacity unit before freeing it
	// through the same lagged path as evictions. 0 = sticky: held until scenario end.
	serviceTime time.Duration
	// window, when non-zero, switches the scenario to timed mode: HP arrivals are generated for
	// the window duration and goodput is measured over it, instead of waiting for an analytic
	// admission count.
	window time.Duration
}

// evictionScenario is one coordinate of the benchmark matrix.
type evictionScenario struct {
	profile  sensorProfile
	grace    time.Duration
	maxRevoc int
	burst    burstShape
	// evictionOff disables reclamation for baseline arms.
	evictionOff bool
	// churnMean, when non-zero, gives sheddable leases a natural completion time drawn
	// deterministically from a truncated exponential-like ladder with this mean.
	churnMean time.Duration
	capacity  int     // C, pool capacity in lease units
	preload   int     // S, sheddable leases occupying the pool at burst start
	hpCeiling float64 // h, the HP band's dispatch ceiling
}

func (sc evictionScenario) name() string {
	mode := fmt.Sprintf("g%d/k%d", sc.grace.Milliseconds(), sc.maxRevoc)
	if sc.evictionOff {
		if sc.churnMean > 0 {
			mode = "off-churn"
		} else {
			mode = "off-nochurn"
		}
	}
	return fmt.Sprintf("%s/%s/%s", sc.profile.name, mode, sc.burst.name)
}

// maxUnitsBelowCeiling returns the largest integer unit count strictly below h*C, i.e. the highest
// occupancy at which the HP band can still dispatch.
func maxUnitsBelowCeiling(capacity int, ceiling float64) int {
	return int(math.Ceil(ceiling*float64(capacity))) - 1
}

// minEvictionsRequired computes the analytic minimum number of sheddable evictions needed for a
// sticky HP burst: preloaded S leases in a pool of capacity C, K HP arrivals, HP ceiling h. The
// k-th admission requires occupancy <= maxUnitsBelowCeiling just before it; admitted HP requests
// are sticky (never free). Returns (minEvictions, admittableHP).
func minEvictionsRequired(preload, hpCount, capacity int, ceiling float64) (minEvictions, admittable int) {
	maxBelow := maxUnitsBelowCeiling(capacity, ceiling)
	if maxBelow < 0 {
		return 0, 0 // Ceiling of zero admits nothing.
	}
	// With every sheddable lease evicted, the k-th admission needs k-1 <= maxBelow.
	admittable = min(hpCount, maxBelow+1)
	if admittable == 0 {
		return 0, 0
	}
	// The last admission needs preload - E + (admittable-1) <= maxBelow.
	minEvictions = max(0, preload+admittable-1-maxBelow)
	minEvictions = min(minEvictions, preload)
	return minEvictions, admittable
}

// syntheticPool is the synthetic pool + sensor. visibleUnits is the gauge numerator: sheddable
// leases whose free is not yet sensor-visible, plus dispatched HP footprint.
type syntheticPool struct {
	// Embedded to satisfy the plugin.Plugin surface of SaturationDetector (same pattern as
	// benchDetector); only Saturation is ever called.
	flowcontrol.SaturationDetector

	capacity int64
	profile  sensorProfile
	evictor  *eviction.RequestEvictor
	epMeta   *fwkdl.EndpointMetadata

	visibleUnits     atomic.Int64
	maxObservedUnits atomic.Int64

	// hpWaiting counts HP requests currently blocked in EnqueueAndWait; sampled at eviction-signal
	// time to attribute evictions fired with no waiting demand.
	hpWaiting    atomic.Int64
	maxHPWaiting atomic.Int64

	evictionsSignaled  atomic.Int64
	evictionsConfirmed atomic.Int64
	evictionsWhileIdle atomic.Int64

	leaseCtx    context.Context
	leaseCancel context.CancelFunc
	wg          sync.WaitGroup
}

var _ flowcontrol.SaturationDetector = (*syntheticPool)(nil)

func newSyntheticPool(sc evictionScenario, evictor *eviction.RequestEvictor) *syntheticPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &syntheticPool{
		capacity: int64(sc.capacity),
		profile:  sc.profile,
		evictor:  evictor,
		epMeta: &fwkdl.EndpointMetadata{
			NamespacedName: types.NamespacedName{Name: "sim-pod", Namespace: "default"},
			Address:        "10.0.0.1",
			Port:           "8000",
		},
		leaseCtx:    ctx,
		leaseCancel: cancel,
	}
}

func (p *syntheticPool) Saturation(context.Context, []fwkdl.Endpoint) float64 {
	return float64(p.visibleUnits.Load()) / float64(p.capacity)
}

// addUnits adjusts the gauge and maintains the overshoot watermark.
func (p *syntheticPool) addUnits(delta int64) {
	v := p.visibleUnits.Add(delta)
	for {
		cur := p.maxObservedUnits.Load()
		if v <= cur || p.maxObservedUnits.CompareAndSwap(cur, v) {
			return
		}
	}
}

// PreloadSheddable occupies the pool with n sheddable leases tracked in the real RequestEvictor,
// each with its own lifecycle goroutine reacting to eviction (or natural churn for baselines).
// churnAfter, when non-nil, returns the lease's natural completion delay (0 = never completes).
func (p *syntheticPool) PreloadSheddable(n int, churnAfter func(i int) time.Duration) {
	for i := range n {
		id := fmt.Sprintf("shed-%d", i)
		req := &fwksched.InferenceRequest{
			RequestID:  id,
			Headers:    map[string]string{reqcommon.RequestIDHeaderKey: id},
			Objectives: fwksched.RequestObjectives{Priority: -1},
		}
		result := &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(p.epMeta, fwkdl.NewMetrics(), nil),
				}},
			},
		}
		p.evictor.PreRequest(p.leaseCtx, req, result)
		evictCh := p.evictor.EvictionRegistry().Get(id)
		p.addUnits(1)

		var churn time.Duration
		if churnAfter != nil {
			churn = churnAfter(i)
		}
		p.wg.Add(1)
		go p.leaseLifecycle(req, evictCh, churn)
	}
}

// leaseLifecycle waits for eviction, natural churn, or teardown, and walks the freed capacity
// through the sensor lag stages.
func (p *syntheticPool) leaseLifecycle(req *fwksched.InferenceRequest, evictCh chan struct{}, churnAfter time.Duration) {
	defer p.wg.Done()

	var churnCh <-chan time.Time
	if churnAfter > 0 {
		timer := time.NewTimer(churnAfter)
		defer timer.Stop()
		churnCh = timer.C
	}

	select {
	case <-evictCh:
		p.evictionsSignaled.Add(1)
		if p.hpWaiting.Load() == 0 {
			p.evictionsWhileIdle.Add(1)
		}
		if !sleepCtx(p.leaseCtx, p.profile.termLatency) {
			return
		}
		// Stream termination: the real cleanup path, which fires the confirmation listener.
		p.evictor.ResponseBody(p.leaseCtx, req, &fwkrc.Response{EndOfStream: true}, p.epMeta)
		p.evictionsConfirmed.Add(1)
	case <-churnCh:
		// Natural completion frees capacity through the same termination path.
		p.evictor.ResponseBody(p.leaseCtx, req, &fwkrc.Response{EndOfStream: true}, p.epMeta)
	case <-p.leaseCtx.Done():
		return
	}

	if !sleepCtx(p.leaseCtx, p.profile.freeVisibilityLag) {
		return
	}
	p.addUnits(-1)
}

// HPEnqueued and HPFinishedWaiting bracket a client's EnqueueAndWait call.
func (p *syntheticPool) HPEnqueued() {
	w := p.hpWaiting.Add(1)
	for {
		cur := p.maxHPWaiting.Load()
		if w <= cur || p.maxHPWaiting.CompareAndSwap(cur, w) {
			return
		}
	}
}

func (p *syntheticPool) HPFinishedWaiting() { p.hpWaiting.Add(-1) }

// HPDispatched claims the dispatched request's footprint immediately (the sensor-lag dimension
// applies to frees only, which is conservative for the pacing question under test). If
// serviceTime > 0 the footprint is released through the lagged path after service completes.
func (p *syntheticPool) HPDispatched(serviceTime time.Duration) {
	p.addUnits(1)
	if serviceTime <= 0 {
		return // Sticky: held until scenario teardown.
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		if !sleepCtx(p.leaseCtx, serviceTime) {
			return
		}
		if !sleepCtx(p.leaseCtx, p.profile.freeVisibilityLag) {
			return
		}
		p.addUnits(-1)
	}()
}

// Close tears the pool down and fails the benchmark if lease goroutines leak.
func (p *syntheticPool) Close(b *testing.B) {
	b.Helper()
	p.leaseCancel()
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		b.Fatal("syntheticPool teardown timed out: lease goroutines leaked")
	}
}

// sleepCtx sleeps for d unless ctx is cancelled first; returns false on cancellation. A zero or
// negative d returns true immediately.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// hpBenchRequest is a FlowControlRequest for the high-priority flow with a unique ID.
type hpBenchRequest struct {
	id  string
	key flowcontrol.FlowKey
}

func (r *hpBenchRequest) FlowKey() flowcontrol.FlowKey                 { return r.key }
func (r *hpBenchRequest) ByteSize() uint64                             { return 1024 }
func (r *hpBenchRequest) InitialEffectiveTTL() time.Duration           { return 5 * time.Minute }
func (r *hpBenchRequest) ID() string                                   { return r.id }
func (r *hpBenchRequest) GetMetadata() map[string]any                  { return nil }
func (r *hpBenchRequest) InferencePoolName() string                    { return "bench-pool" }
func (r *hpBenchRequest) ModelName() string                            { return "bench-model" }
func (r *hpBenchRequest) TargetModelName() string                      { return "bench-target" }
func (r *hpBenchRequest) InferenceRequest() *fwksched.InferenceRequest { return nil }
func (r *hpBenchRequest) ReceivedTimestamp() time.Time                 { return time.Now() }

// setupEvictionBenchHarness builds the real SUT stack for one scenario: registry with a single
// priority-0 band, constant HP ceiling, the synthetic pool as the saturation detector, and the
// full eviction plumbing wired through controller Deps.
func setupEvictionBenchHarness(
	ctx context.Context,
	b *testing.B,
	sc evictionScenario,
) (*controller.FlowController, *syntheticPool) {
	b.Helper()

	defaults := buildPolicyDefaults(ctx, b)
	reg := setupRegistry(b, defaults, 1) // Single band at priority 0.

	orderingPlugin, err := evictordering.PriorityThenTimeOrderingFactory(evictordering.PriorityThenTimeOrderingType, nil, nil)
	if err != nil {
		b.Fatalf("Failed to create eviction ordering policy: %v", err)
	}
	filterPlugin, err := evictfiltering.SheddableFilterFactory(evictfiltering.SheddableFilterType, nil, nil)
	if err != nil {
		b.Fatalf("Failed to create eviction filter policy: %v", err)
	}
	requestEvictor := eviction.NewRequestEvictor(
		orderingPlugin.(flowcontrol.EvictionOrderingPolicy),
		filterPlugin.(flowcontrol.EvictionFilterPolicy),
		eviction.NewImmediateResponseEvictor(),
	)

	pool := newSyntheticPool(sc, requestEvictor)

	cfg := &controller.Config{
		DefaultRequestTTL:           5 * time.Minute,
		ExpiryCleanupInterval:       1 * time.Hour, // Effectively disabled.
		EnqueueChannelBufferSize:    2000,
		EnableEviction:              !sc.evictionOff,
		MaxRevocationsPerDecision:   sc.maxRevoc,
		EvictionConfirmationGrace:   sc.grace,
		EvictionConfirmationTimeout: 10 * time.Second,
	}

	deps := controller.Deps{
		Registry:           reg,
		SaturationDetector: pool,
		EndpointCandidates: &mocks.MockEndpointCandidates{},
		UsageLimitPolicy:   usagelimits.NewConstPolicy("evict-bench", sc.hpCeiling),
	}
	if !sc.evictionOff {
		deps.InFlightEvictor = requestEvictor
	}

	fc := controller.NewFlowController(ctx, "eviction-bench", cfg, deps)
	return fc, pool
}
