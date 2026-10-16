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

package lorascore

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

var t0 = time.Unix(1_000_000, 0)

func at(s float64) time.Time { return t0.Add(time.Duration(s * float64(time.Second))) }

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestDemandDecaysByHalfLife(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(0))
	r0 := d.Rate("a", at(0))
	if r := d.Rate("a", at(10)); !near(r, r0/2) {
		t.Fatalf("Rate after one half-life = %v, want %v", r, r0/2)
	}
}

func TestDemandSteadyStateMatchesArrivalRate(t *testing.T) {
	d := mustDemand(t, 5*time.Second)
	for i := 0; i < 2000; i++ { // 4 req/s for 500 s, 100 half-lives
		d.Observe("a", at(float64(i)*0.25))
	}
	if r := d.Rate("a", at(499.75)); math.Abs(r-4) > 0.3 {
		t.Fatalf("steady-state Rate = %v, want ~4 req/s", r)
	}
}

// Target model names come from clients, so adapters whose demand has decayed
// to nothing must be forgotten or the estimator grows with every name seen.
func TestDemandForgetsDecayedAdapters(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	for i := 0; i < 1000; i++ {
		d.Observe(fmt.Sprintf("stale-%d", i), at(0))
	}
	for i := 0; i < 1000; i++ { // 1 req/s for 1000 s, 100 half-lives later
		d.Observe("live", at(float64(1000+i)))
	}
	if got := len(d.rate) + len(d.last); got > 2*minPruneSize {
		t.Fatalf("tracking %d entries after the stale adapters decayed, want at most %d", got, 2*minPruneSize)
	}
	if s := d.Share("live", at(1999)); !near(s, 1) {
		t.Fatalf("Share(live) = %v, want 1", s)
	}
}

// A client can send a new name with every request. Sweeps must keep running
// while the map grows, or no name is ever forgotten.
func TestDemandForgetsContinuouslyUniqueNames(t *testing.T) {
	d := mustDemand(t, time.Second) // a name is forgotten ~20 s after its request
	for i := 0; i < 10000; i++ {    // 10 new names/s for 1000 s
		d.Observe(fmt.Sprintf("name-%d", i), at(float64(i)*0.1))
	}
	if got := len(d.rate); got > 1000 || len(d.last) != got {
		t.Fatalf("tracking %d rates and %d times after 10000 unique names, want at most 1000 each", got, len(d.last))
	}
}

// The running total must match the sum of the per-adapter estimates, including
// across sweeps and for observations that arrive out of time order.
func TestShareMatchesSumOfRates(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	for i := 0; i < 3000; i++ {
		ts := float64(i) * 0.1
		if i%7 == 0 {
			ts -= 0.35 // a late observation, as concurrent callers can produce
		}
		d.Observe(fmt.Sprintf("a%d", i%(1+i/40)), at(ts))
	}
	now := at(300)
	sum := 0.0
	for a := range d.rate {
		sum += d.Rate(a, now)
	}
	for _, a := range []string{"a0", "a3", "a70"} {
		if got, want := d.Share(a, now), d.Rate(a, now)/sum; math.Abs(got-want) > 1e-9*want {
			t.Fatalf("Share(%s) = %v, want %v", a, got, want)
		}
	}
}

// Score reads the clock before taking the lock, so a concurrent Observe can
// move the total past the time Share is asked about. Shares must still be
// taken at one time and sum to 1, not each come out near 1.
func TestShareBeforeLatestObservation(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(0))
	d.Observe("b", at(100))
	if sum := d.Share("a", at(0)) + d.Share("b", at(0)); !near(sum, 1) {
		t.Fatalf("Share(a)+Share(b) at t=0 = %v, want 1", sum)
	}
	if got, want := d.Share("a", at(0)), d.Share("a", at(100)); got != want {
		t.Fatalf("Share(a) at t=0 = %v, want %v, its value at the latest observation", got, want)
	}
}

// Target model names come from clients, so Share must not cost time in the
// number of names tracked.
func TestShareCostIsIndependentOfTrackedNames(t *testing.T) {
	d := mustDemand(t, time.Hour) // nothing decays away during the test
	for i := 0; i < 100000; i++ {
		d.Observe(fmt.Sprintf("name-%d", i), at(0))
	}
	if n := testing.AllocsPerRun(10, func() { d.Share("name-1", at(1)) }); n != 0 {
		t.Fatalf("Share allocates %v times per call, want 0", n)
	}
}

func TestDemandShare(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(0))
	d.Observe("a", at(0))
	d.Observe("b", at(0))
	if s := d.Share("a", at(0)); !near(s, 2.0/3) {
		t.Fatalf("Share(a) = %v, want 2/3", s)
	}
	if s := d.Share("unseen", at(0)); s != 0 {
		t.Fatalf("Share(unseen) = %v, want 0", s)
	}
}

func setup(t *testing.T) (*residency.Model, *Demand) {
	m := residency.NewModel(3)
	for _, p := range []string{"p0", "p1", "p2"} {
		m.ObserveCapacity(p, 1)
	}
	return m, mustDemand(t, time.Minute)
}

func serve(m *residency.Model, d *Demand, pod, adapter string, s float64) {
	m.Routed(pod, adapter)
	d.Observe(adapter, at(s))
	m.Started(pod, adapter, at(s))
	m.Finished(pod, adapter, at(s), true)
}

func TestGPUBeatsCPUBeatsAbsent(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1) // p0: a on GPU
	serve(m, d, "p1", "a", 1)
	serve(m, d, "p1", "x", 2) // p1: x on GPU, a in CPU
	// p2 knows nothing: a must be loaded.
	s := Score(m, d, "a", false, []string{"p0", "p1", "p2"}, DefaultWeights(), at(3))
	if !(s["p0"] > s["p1"] && s["p1"] > s["p2"]) {
		t.Fatalf("want p0 > p1 > p2, got %v", s)
	}
}

func TestEvictingHotAdapterCostsMore(t *testing.T) {
	m, d := setup(t)
	for i := 0; i < 9; i++ {
		serve(m, d, "p0", "hot", float64(i)*0.1) // p0's GPU slot holds a hot adapter
	}
	serve(m, d, "p1", "cold", 1) // p1's holds a cold one
	s := Score(m, d, "new", false, []string{"p0", "p1"}, DefaultWeights(), at(2))
	if !(s["p1"] > s["p0"]) {
		t.Fatalf("want evicting cold (p1) preferred over hot (p0), got %v", s)
	}
}

func TestEvictionWeightZeroIgnoresVictim(t *testing.T) {
	m, d := setup(t)
	for i := 0; i < 9; i++ {
		serve(m, d, "p0", "hot", float64(i)*0.1)
	}
	serve(m, d, "p1", "cold", 1)
	w := DefaultWeights()
	w.EvictionWeight = 0
	s := Score(m, d, "new", false, []string{"p0", "p1"}, w, at(2))
	if !near(s["p0"], s["p1"]) {
		t.Fatalf("with EvictionWeight 0 both are plain loads, want equal, got %v", s)
	}
}

func TestBlockedPodPenalized(t *testing.T) {
	m, d := setup(t)
	m.Routed("p0", "busy")
	m.Started("p0", "busy", at(1)) // p0's only slot is running
	d.Observe("busy", at(1))
	s := Score(m, d, "new", false, []string{"p0", "p1"}, DefaultWeights(), at(2))
	if !(s["p1"] > s["p0"]) {
		t.Fatalf("want free p1 over blocked p0, got %v", s)
	}
}

func TestBaseModelRequestIsNeutral(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	m.Routed("p1", "busy")
	m.Started("p1", "busy", at(1))
	s := Score(m, d, "base", true, []string{"p0", "p1", "p2"}, DefaultWeights(), at(2))
	for _, p := range []string{"p1", "p2"} {
		if s[p] != s["p0"] {
			t.Fatalf("base-model request must score every pod equally, got %v", s)
		}
	}
}

func TestScoresInUnitRangeAndBestIsOne(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	s := Score(m, d, "a", false, []string{"p0", "p1", "p2"}, DefaultWeights(), at(2))
	for p, v := range s {
		if v < 0 || v > 1 {
			t.Fatalf("score %s = %v outside [0,1]", p, v)
		}
	}
	if s["p0"] != 1 {
		t.Fatalf("zero-cost pod scores %v, want 1", s["p0"])
	}
	if s["p2"] != 0 {
		t.Fatalf("costliest pod scores %v, want 0", s["p2"])
	}
}

func TestAllEqualCostAllScoreOne(t *testing.T) {
	m, d := setup(t)
	s := Score(m, d, "a", false, []string{"p0", "p1"}, DefaultWeights(), at(1))
	if s["p0"] != 1 || s["p1"] != 1 {
		t.Fatalf("equal costs must not discriminate, got %v", s)
	}
}

// A CPU-tier adapter always comes with a GPU victim, so isolate the copy cost
// by turning the eviction term off.
func TestCopyCostAloneOrdersTiers(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	serve(m, d, "p1", "a", 1)
	serve(m, d, "p1", "x", 2)
	w := DefaultWeights()
	w.EvictionWeight = 0
	s := Score(m, d, "a", false, []string{"p0", "p1", "p2"}, w, at(3))
	if !(s["p0"] > s["p1"] && s["p1"] > s["p2"]) {
		t.Fatalf("want GPU > CPU > Absent on copy/load cost alone, got %v", s)
	}
}

// With no zero-cost pod the cheapest must still score 1.
func TestCheapestScoresOneWhenAllCostSomething(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p1", "a", 1)
	serve(m, d, "p1", "x", 2) // p1: a in CPU
	s := Score(m, d, "a", false, []string{"p1", "p2"}, DefaultWeights(), at(3))
	if s["p1"] != 1 || s["p2"] != 0 {
		t.Fatalf("want p1=1 (copy) and p2=0 (load), got %v", s)
	}
}

func TestShareBeforeAnyRequestIsZero(t *testing.T) {
	d := mustDemand(t, time.Minute)
	if s := d.Share("a", at(0)); s != 0 {
		t.Fatalf("Share with no observations = %v, want 0", s)
	}
}

// In vLLM's default max_cpu_loras == max_loras, an absent
// admission drops the GPU victim out of the CPU cache too, so bringing it back
// is a load, not a copy.
func TestDefaultConfigVictimPricedAsLoad(t *testing.T) {
	m := residency.NewModel(0)
	m.ObserveCapacity("p0", 1)
	d := mustDemand(t, time.Minute)
	serve(m, d, "p0", "hot", 1)
	serve(m, d, "p0", "other", 1) // other evicts hot; give hot a known share
	serve(m, d, "p0", "hot", 2)
	adm := m.Admit("p0", "new")
	if adm.CPUVictim != "hot" || adm.GPUVictim != "" {
		t.Fatalf("Admit(new) = %+v, want CPUVictim hot and no GPUVictim", adm)
	}
	w := DefaultWeights()
	share := d.Share("hot", at(3))
	if got, want := cost(adm, d, w, at(3)), w.LoadCost+share*w.LoadCost; !near(got, want) {
		t.Fatalf("cost = %v, want LoadCost + share*LoadCost = %v", got, want)
	}
}

// With a larger CPU cache, an absent admission prices both victims.
func TestBothVictimsPriced(t *testing.T) {
	m := residency.NewModel(2)
	m.ObserveCapacity("p0", 1)
	d := mustDemand(t, time.Minute)
	serve(m, d, "p0", "a", 1)
	serve(m, d, "p0", "b", 2) // GPU = {b}, CPU = {b, a}
	adm := m.Admit("p0", "new")
	if adm.GPUVictim != "b" || adm.CPUVictim != "a" {
		t.Fatalf("Admit(new) = %+v, want GPUVictim b, CPUVictim a", adm)
	}
	w := DefaultWeights()
	want := w.LoadCost + d.Share("b", at(3))*w.CopyCost + d.Share("a", at(3))*w.LoadCost
	if got := cost(adm, d, w, at(3)); !near(got, want) {
		t.Fatalf("cost = %v, want %v", got, want)
	}
}

// A routed request that has not started must not make its adapter look
// resident: it stays Absent, and any preference comes from PendingCost alone,
// so with PendingCost equal to LoadCost the pending pod gains nothing.
func TestPendingRequestDoesNotLookResident(t *testing.T) {
	m, d := setup(t)
	m.Routed("p0", "a")
	if got := m.Tier("p0", "a"); got != residency.Absent {
		t.Fatalf("Tier(p0, a) = %v, want Absent while only pending", got)
	}
	w := DefaultWeights()
	w.PendingCost = w.LoadCost
	s := Score(m, d, "a", false, []string{"p0", "p1"}, w, at(1))
	if s["p0"] != s["p1"] {
		t.Fatalf("with PendingCost = LoadCost the pending pod must gain nothing, got %v", s)
	}
}

func mustDemand(t *testing.T, halfLife time.Duration) *Demand {
	t.Helper()
	d, err := NewDemand(halfLife)
	if err != nil {
		t.Fatalf("NewDemand(%v): %v", halfLife, err)
	}
	return d
}

// A non-positive half-life gives infinite rates and NaN scores.
func TestNewDemandRejectsNonPositiveHalfLife(t *testing.T) {
	for _, h := range []time.Duration{0, -time.Second} {
		if _, err := NewDemand(h); err == nil {
			t.Errorf("NewDemand(%v) = nil error, want an error", h)
		}
	}
}

func TestWeightsValidate(t *testing.T) {
	if err := DefaultWeights().Validate(); err != nil {
		t.Fatalf("DefaultWeights().Validate() = %v", err)
	}
	bad := map[string]func(*Weights){
		"negative copy":     func(w *Weights) { w.CopyCost = -1 },
		"NaN load":          func(w *Weights) { w.LoadCost = math.NaN() },
		"infinite blocked":  func(w *Weights) { w.BlockedCost = math.Inf(1) },
		"negative eviction": func(w *Weights) { w.EvictionWeight = -0.1 },
		"infinite pending":  func(w *Weights) { w.PendingCost = math.Inf(1) },
	}
	for name, mutate := range bad {
		w := DefaultWeights()
		mutate(&w)
		if err := w.Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want an error", name)
		}
	}
}

// Bursty arrivals: while the first request for a cold
// adapter is still loading on p0, later ones must follow it rather than
// scatter across pods.
func TestBurstForColdAdapterFollowsPendingPod(t *testing.T) {
	m, d := setup(t)
	m.Routed("p0", "cold")
	d.Observe("cold", at(1))
	for i := 0; i < 3; i++ {
		s := Score(m, d, "cold", false, []string{"p0", "p1", "p2"}, DefaultWeights(), at(1))
		if !(s["p0"] > s["p1"] && s["p0"] > s["p2"]) {
			t.Fatalf("request %d: want p0 (already loading cold) preferred, got %v", i, s)
		}
		m.Routed("p0", "cold")
	}
}

// A pending pod must not beat a pod that already holds the adapter on GPU.
func TestGPUBeatsPending(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	m.Routed("p1", "a")
	s := Score(m, d, "a", false, []string{"p0", "p1"}, DefaultWeights(), at(2))
	if !(s["p0"] > s["p1"]) {
		t.Fatalf("want GPU p0 over pending p1, got %v", s)
	}
}

// Share must not depend on map iteration order. Rates spanning more than 2^53
// make the floating-point total order-sensitive, so a map-ordered sum would
// differ between calls; the running total is never summed in map order.
func TestShareIsIndependentOfMapOrder(t *testing.T) {
	d := mustDemand(t, time.Second)
	t0 := time.Unix(0, 0)
	d.rate["big"], d.last["big"] = 1e17, t0
	for _, a := range []string{"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"} {
		d.rate[a], d.last[a] = 7, t0
	}
	d.retotal(t0)
	first := d.Share("s1", t0)
	for i := 0; i < 500; i++ {
		if got := d.Share("s1", t0); math.Float64bits(got) != math.Float64bits(first) {
			t.Fatalf("call %d: Share = %v, first call gave %v", i, got, first)
		}
	}
}
