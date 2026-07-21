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

package benchmark

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

func TestMinEvictionsRequired(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name                string
		preload, hp, cap    int
		ceiling             float64
		wantMin, wantAdmits int
	}{
		// C=20, h=0.9: maxBelow = 17 (occupancy must be <= 17 to admit).
		{"FullPool_SingleHP", 20, 1, 20, 0.9, 3, 1},
		{"FullPool_Batch32", 20, 32, 20, 0.9, 20, 18},
		{"FullPool_Batch10", 20, 10, 20, 0.9, 12, 10},
		{"HalfPool_SingleHP", 10, 1, 20, 0.9, 0, 1},
		{"IntegralCeiling", 18, 1, 18, 1.0, 1, 1}, // maxBelow = 17; need 18-E <= 17 -> E=1.
		{"FractionalCeiling", 20, 1, 20, 0.95, 2, 1},
		{"ZeroCeiling", 20, 5, 20, 0.0, 0, 0},
		{"EmptyPool", 0, 5, 20, 0.9, 0, 5},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotMin, gotAdmits := minEvictionsRequired(tc.preload, tc.hp, tc.cap, tc.ceiling)
			if gotMin != tc.wantMin || gotAdmits != tc.wantAdmits {
				t.Fatalf("minEvictionsRequired(%d,%d,%d,%v) = (%d,%d), want (%d,%d)",
					tc.preload, tc.hp, tc.cap, tc.ceiling, gotMin, gotAdmits, tc.wantMin, tc.wantAdmits)
			}
		})
	}
}

// scenarioResult holds one scenario run's measurements.
type scenarioResult struct {
	ttfr           time.Duration // burst start -> first HP dispatch (0 = none dispatched)
	p50, p95       time.Duration
	admitted       int64
	elapsed        time.Duration
	evictSignaled  int64
	evictConfirmed int64
	evictHPIdle    int64
	overEvict      int64 // confirmed - analytic minimum (sticky scenarios only; else -1)
	maxHPWaiting   int64
	overshoot      int64
	timedOut       bool
}

const scenarioTimeout = 8 * time.Second

// runScenario executes one full scenario against a fresh harness and returns its measurements.
func runScenario(b *testing.B, sc evictionScenario) scenarioResult {
	b.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fc, pool := setupEvictionBenchHarness(ctx, b, sc)

	var churnFn func(i int) time.Duration
	if sc.churnMean > 0 {
		n := sc.preload
		mean := sc.churnMean
		// Deterministic exponential quantile ladder so churn is identical across runs.
		churnFn = func(i int) time.Duration {
			u := (float64(i) + 0.5) / float64(n)
			return time.Duration(-float64(mean) * math.Log(1.0-u))
		}
	}
	pool.PreloadSheddable(sc.preload, churnFn)

	minEvict, admittable := minEvictionsRequired(sc.preload, sc.burst.hpCount, sc.capacity, sc.hpCeiling)
	timed := sc.burst.window > 0

	var admitted atomic.Int64
	var ttfrNanos atomic.Int64
	var waitsMu sync.Mutex
	waits := make([]time.Duration, 0, sc.burst.hpCount)

	clientCtx, clientCancel := context.WithCancel(ctx)
	defer clientCancel()
	var clientWG sync.WaitGroup
	burstStart := time.Now()

	launch := func(i int) {
		clientWG.Add(1)
		go func() {
			defer clientWG.Done()
			req := &hpBenchRequest{id: fmt.Sprintf("hp-%d", i), key: flowcontrol.FlowKey{ID: "hp-flow", Priority: 0}}
			pool.HPEnqueued()
			t0 := time.Now()
			outcome, err := fc.EnqueueAndWait(clientCtx, req)
			wait := time.Since(t0)
			pool.HPFinishedWaiting()
			if err != nil || outcome != types.QueueOutcomeDispatched {
				return
			}
			pool.HPDispatched(sc.burst.serviceTime)
			ttfrNanos.CompareAndSwap(0, time.Since(burstStart).Nanoseconds())
			admitted.Add(1)
			waitsMu.Lock()
			waits = append(waits, wait)
			waitsMu.Unlock()
		}()
	}

	timedOut := false
	if timed {
		gap := sc.burst.window / time.Duration(sc.burst.hpCount)
		ticker := time.NewTicker(gap)
		windowEnd := time.NewTimer(sc.burst.window)
		i := 0
	genLoop:
		for {
			select {
			case <-ticker.C:
				if i < sc.burst.hpCount {
					launch(i)
					i++
				}
			case <-windowEnd.C:
				break genLoop
			}
		}
		ticker.Stop()
	} else {
		for i := range sc.burst.hpCount {
			launch(i)
			if sc.burst.arrivalGap > 0 {
				time.Sleep(sc.burst.arrivalGap)
			}
		}
		// Baselines with no churn admit nothing; observe a fixed window instead of the full timeout.
		waitBudget := scenarioTimeout
		target := int64(admittable)
		if sc.evictionOff && sc.churnMean == 0 {
			waitBudget = 2 * time.Second
			target = math.MaxInt64
		}
		deadline := time.Now().Add(waitBudget)
		for admitted.Load() < target {
			if time.Now().After(deadline) {
				timedOut = target != math.MaxInt64
				break
			}
			time.Sleep(time.Millisecond)
		}
	}
	elapsed := time.Since(burstStart)

	// Unblock any still-queued HP clients and collect.
	clientCancel()
	clientWG.Wait()

	sort.Slice(waits, func(i, j int) bool { return waits[i] < waits[j] })
	percentile := func(p float64) time.Duration {
		if len(waits) == 0 {
			return 0
		}
		idx := int(p * float64(len(waits)-1))
		return waits[idx]
	}

	res := scenarioResult{
		ttfr:           time.Duration(ttfrNanos.Load()),
		p50:            percentile(0.50),
		p95:            percentile(0.95),
		admitted:       admitted.Load(),
		elapsed:        elapsed,
		evictSignaled:  pool.evictionsSignaled.Load(),
		evictConfirmed: pool.evictionsConfirmed.Load(),
		evictHPIdle:    pool.evictionsWhileIdle.Load(),
		overEvict:      -1,
		maxHPWaiting:   pool.maxHPWaiting.Load(),
		// Overshoot beyond the higher of the preload watermark and the ceiling: any rise above it
		// means dispatch outran the gauge.
		overshoot: max(0, pool.maxObservedUnits.Load()-
			max(int64(sc.preload), int64(math.Ceil(sc.hpCeiling*float64(sc.capacity))))),
		timedOut: timedOut,
	}
	if !timed && !sc.evictionOff {
		res.overEvict = res.evictConfirmed - int64(minEvict)
	}

	// Teardown: stop the controller, then the pool (leases, HP release timers).
	cancel()
	pool.Close(b)
	time.Sleep(50 * time.Millisecond) // Drain processor goroutines (package precedent).
	return res
}

// --- Aggregation over b.N iterations ---

type scenarioAgg struct {
	n       int
	sum     scenarioResult
	anyTO   bool
	sumTTFR time.Duration
}

func (a *scenarioAgg) add(r scenarioResult) {
	a.n++
	a.sumTTFR += r.ttfr
	a.sum.p50 += r.p50
	a.sum.p95 += r.p95
	a.sum.admitted += r.admitted
	a.sum.elapsed += r.elapsed
	a.sum.evictSignaled += r.evictSignaled
	a.sum.evictConfirmed += r.evictConfirmed
	a.sum.evictHPIdle += r.evictHPIdle
	a.sum.overEvict += r.overEvict
	a.sum.maxHPWaiting += r.maxHPWaiting
	a.sum.overshoot += r.overshoot
	a.anyTO = a.anyTO || r.timedOut
}

// evictionRow is one line of the shareable results table.
type evictionRow struct {
	name                                             string
	ttfrMS, p50MS, p95MS, goodput                    float64
	evictConfirmed, overEvict, evictHPIdle, maxQueue float64
	timedOut                                         bool
}

var (
	evictionTableMu sync.Mutex
	evictionTable   []evictionRow
)

func (a *scenarioAgg) report(b *testing.B, name string) {
	n := float64(a.n)
	ms := func(d time.Duration) float64 { return math.Round(float64(d)/float64(time.Millisecond)/n*10) / 10 }
	row := evictionRow{
		name:           name,
		ttfrMS:         ms(a.sumTTFR),
		p50MS:          ms(a.sum.p50),
		p95MS:          ms(a.sum.p95),
		goodput:        math.Round(float64(a.sum.admitted)/a.sum.elapsed.Seconds()*10) / 10,
		evictConfirmed: float64(a.sum.evictConfirmed) / n,
		overEvict:      float64(a.sum.overEvict) / n,
		evictHPIdle:    float64(a.sum.evictHPIdle) / n,
		maxQueue:       float64(a.sum.maxHPWaiting) / n,
		timedOut:       a.anyTO,
	}

	b.ReportMetric(row.ttfrMS, "ttfr-ms")
	b.ReportMetric(row.p50MS, "p50-ms")
	b.ReportMetric(row.p95MS, "p95-ms")
	b.ReportMetric(row.goodput, "goodput-rps")
	b.ReportMetric(row.evictConfirmed, "evict-confirmed")
	if row.overEvict >= 0 {
		b.ReportMetric(row.overEvict, "over-evict")
	}
	b.ReportMetric(row.evictHPIdle, "evict-hp-idle")
	b.ReportMetric(float64(a.sum.overshoot)/n, "overshoot")
	if a.anyTO {
		b.ReportMetric(1, "timed-out")
	}

	evictionTableMu.Lock()
	evictionTable = append(evictionTable, row)
	evictionTableMu.Unlock()
}

// --- The matrix ---

var (
	burstSingle    = burstShape{name: "single", hpCount: 1}
	burstBatch32   = burstShape{name: "batch32", hpCount: 32}
	burstSustained = burstShape{name: "sustained", hpCount: 300, serviceTime: 250 * time.Millisecond, window: 3 * time.Second}
)

// matchedGrace is the grace a deployment would pair with each sensor profile per the design doc.
func matchedGrace(p sensorProfile) time.Duration {
	if p.freeVisibilityLag == 0 {
		return 50 * time.Millisecond
	}
	return 500 * time.Millisecond
}

func evictionScenarios(full bool) []evictionScenario {
	base := evictionScenario{capacity: 20, preload: 20, hpCeiling: 0.9, maxRevoc: 2}
	profiles := []sensorProfile{profileConcurrency, profileUtilization}
	graces := []time.Duration{5 * time.Millisecond, 50 * time.Millisecond, 200 * time.Millisecond, 500 * time.Millisecond}
	ks := []int{1, 2, 5}
	bursts := []burstShape{burstSingle, burstBatch32, burstSustained}

	var out []evictionScenario
	seen := map[string]bool{}
	add := func(sc evictionScenario) {
		if name := sc.name(); !seen[name] {
			seen[name] = true
			out = append(out, sc)
		}
	}

	for _, p := range profiles {
		if full {
			for _, g := range graces {
				for _, k := range ks {
					for _, bu := range bursts {
						sc := base
						sc.profile, sc.grace, sc.maxRevoc, sc.burst = p, g, k, bu
						add(sc)
					}
				}
			}
		} else {
			// Informative diagonal: grace sweep at k=2/batch32; k sweep at matched grace/batch32;
			// single + sustained at matched settings.
			for _, g := range graces {
				sc := base
				sc.profile, sc.grace, sc.burst = p, g, burstBatch32
				add(sc)
			}
			for _, k := range ks {
				sc := base
				sc.profile, sc.grace, sc.maxRevoc, sc.burst = p, matchedGrace(p), k, burstBatch32
				add(sc)
			}
			for _, bu := range []burstShape{burstSingle, burstSustained} {
				sc := base
				sc.profile, sc.grace, sc.burst = p, matchedGrace(p), bu
				add(sc)
			}
			// Over-eviction exposure: low grace on shallow demand. Against the lagged sensor, debits
			// expire before frees become gauge-visible (expect over-eviction); against the leading
			// sensor, near-zero grace should stay clean (expect none).
			lowGrace := 5 * time.Millisecond
			if p.freeVisibilityLag > 0 {
				lowGrace = 50 * time.Millisecond
			}
			{
				sc := base
				sc.profile, sc.grace, sc.burst = p, lowGrace, burstSingle
				add(sc)
			}
		}
		// Baselines: eviction off, with and without natural churn.
		for _, churn := range []time.Duration{0, 500 * time.Millisecond} {
			sc := base
			sc.profile, sc.grace, sc.burst = p, matchedGrace(p), burstBatch32
			sc.evictionOff, sc.churnMean = true, churn
			add(sc)
		}
	}
	return out
}

// BenchmarkEvictionDynamics measures the reclamation controller's pacing behavior end-to-end.
//
// Invocation:
//
//	go test ./pkg/epp/flowcontrol/benchmark/ -run '^$' -bench EvictionDynamics -benchtime=1x -count=5
//
// Set EVICTION_BENCH_FULL=1 for the full matrix and EVICTION_BENCH_TABLE=1 to print a markdown
// summary table after the run.
func BenchmarkEvictionDynamics(b *testing.B) {
	if testing.Short() {
		b.Skip("eviction dynamics benchmark skipped in -short mode")
	}

	for _, sc := range evictionScenarios(os.Getenv("EVICTION_BENCH_FULL") != "") {
		b.Run(sc.name(), func(b *testing.B) {
			agg := &scenarioAgg{}
			for range b.N {
				agg.add(runScenario(b, sc))
			}
			agg.report(b, sc.name())
		})
	}

	if os.Getenv("EVICTION_BENCH_TABLE") != "" {
		printEvictionTable()
	}
}

func printEvictionTable() {
	evictionTableMu.Lock()
	defer evictionTableMu.Unlock()

	var sb strings.Builder
	sb.WriteString("\n| scenario | ttfr(ms) | p50(ms) | p95(ms) | goodput(r/s) | evicted | over-evict | evict-hp-idle | maxq |\n")
	sb.WriteString("|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range evictionTable {
		over := "n/a"
		if r.overEvict >= 0 {
			over = fmt.Sprintf("%.1f", r.overEvict)
		}
		name := r.name
		if r.timedOut {
			name += " (TIMEOUT)"
		}
		fmt.Fprintf(&sb, "| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %s | %.1f | %.0f |\n",
			name, r.ttfrMS, r.p50MS, r.p95MS, r.goodput, r.evictConfirmed, over, r.evictHPIdle, r.maxQueue)
	}
	fmt.Println(sb.String())
}
