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
	r0 := d.rate("a", at(0))
	if r := d.rate("a", at(10)); !near(r, r0/2) {
		t.Fatalf("Rate after one half-life = %v, want %v", r, r0/2)
	}
}

func TestDemandSteadyStateMatchesArrivalRate(t *testing.T) {
	d := mustDemand(t, 5*time.Second)
	for i := 0; i < 2000; i++ { // 4 req/s for 500 s, 100 half-lives
		d.Observe("a", at(float64(i)*0.25))
	}
	if r := d.rate("a", at(499.75)); math.Abs(r-4) > 0.3 {
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
	d.Observe("live", at(1000)) // 100 half-lives later
	if s := d.share("live", at(1000)); !near(s, 1) {
		t.Fatalf("Share(live) = %v, want 1", s)
	}
	if got := len(d.adapter); got != 1 {
		t.Fatalf("tracking %d adapters after the stale ones decayed, want 1", got)
	}
}

// A client can send a new name with every request faster than names decay.
// The estimator stays at its bound, and its shares stay those of the summed
// rates.
func TestDemandBoundedUnderUniqueNames(t *testing.T) {
	d := mustDemand(t, time.Hour) // nothing decays away during the test
	d.limit = 64
	for i := 0; i < 10000; i++ {
		d.Observe(fmt.Sprintf("name-%d", i), at(float64(i)*1e-3))
	}
	if got := len(d.adapter); got > d.limit {
		t.Fatalf("tracking %d adapters, want at most %d", got, d.limit)
	}
	now := at(10)
	sum := 0.0
	for a := range d.adapter {
		sum += d.rate(a, now)
	}
	for _, a := range []string{"name-9999", "name-9990"} {
		if got, want := d.share(a, now), d.rate(a, now)/sum; want == 0 || math.Abs(got-want) > 1e-9*want {
			t.Fatalf("Share(%s) = %v, want %v", a, got, want)
		}
	}
}

// Past the bound, a new adapter displaces the least requested one.
func TestDemandFullDropsLowestRate(t *testing.T) {
	d := mustDemand(t, time.Minute)
	d.limit = 3
	for i := 0; i < 5; i++ {
		d.Observe("hot", at(0))
	}
	d.Observe("warm", at(0))
	d.Observe("warm", at(0))
	d.Observe("cold", at(0))
	d.Observe("new", at(1))
	if _, ok := d.adapter["cold"]; ok {
		t.Fatalf("cold kept past the bound: %v", d.adapter)
	}
	for _, a := range []string{"hot", "warm", "new"} {
		if _, ok := d.adapter[a]; !ok {
			t.Fatalf("%s dropped, want only cold dropped", a)
		}
	}
}

// Shares read before the latest observation, as a caller that read the clock
// before a concurrent Observe does, still sum to 1.
func TestSharesBeforeLatestObservationSumToOne(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(0))
	d.Observe("b", at(100))
	sh := d.Shares([]string{"a", "b"}, at(0))
	if sum := sh["a"] + sh["b"]; !near(sum, 1) {
		t.Fatalf("Share(a)+Share(b) at t=0 = %v, want 1", sum)
	}
}

// A late observation counts as one request at the adapter's latest time.
func TestDemandLateObservationDoesNotInflate(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(10))
	d.Observe("a", at(0))
	want := mustDemand(t, 10*time.Second)
	want.Observe("a", at(10))
	want.Observe("a", at(10))
	if got, w := d.rate("a", at(10)), want.rate("a", at(10)); !near(got, w) {
		t.Fatalf("Rate after a late observation = %v, want %v", got, w)
	}
}

func TestDemandShare(t *testing.T) {
	d := mustDemand(t, 10*time.Second)
	d.Observe("a", at(0))
	d.Observe("a", at(0))
	d.Observe("b", at(0))
	if s := d.share("a", at(0)); !near(s, 2.0/3) {
		t.Fatalf("Share(a) = %v, want 2/3", s)
	}
	if s := d.share("unseen", at(0)); s != 0 {
		t.Fatalf("Share(unseen) = %v, want 0", s)
	}
}

type estimate = map[string]*residency.Pod

func setup(t *testing.T) (estimate, *Demand) {
	m := estimate{}
	for _, p := range []string{"p0", "p1", "p2"} {
		m[p] = residency.NewPod(1, 3)
	}
	return m, mustDemand(t, time.Minute)
}

// admit is each pod's admission of adapter, as the plugin builds it: a pod
// with no estimate admits as Absent.
func admit(m estimate, adapter string, pods ...string) map[string]residency.Admission {
	out := make(map[string]residency.Admission, len(pods))
	for _, p := range pods {
		out[p] = residency.Admission{Tier: residency.Absent}
		if e := m[p]; e != nil {
			out[p] = e.Admit(adapter)
		}
	}
	return out
}

func serve(m estimate, d *Demand, pod, adapter string, s float64) {
	m[pod].Routed(adapter)
	d.Observe(adapter, at(s))
	m[pod].Started(adapter, at(s))
	m[pod].Finished(adapter, at(s), true)
}

func TestGPUBeatsCPUBeatsAbsent(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1) // p0: a on GPU
	serve(m, d, "p1", "a", 1)
	serve(m, d, "p1", "x", 2) // p1: x on GPU, a in CPU
	// p2 knows nothing: a must be loaded.
	s := Score(admit(m, "a", []string{"p0", "p1", "p2"}...), d, false, DefaultWeights(), at(3))
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
	s := Score(admit(m, "new", []string{"p0", "p1"}...), d, false, DefaultWeights(), at(2))
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
	s := Score(admit(m, "new", []string{"p0", "p1"}...), d, false, w, at(2))
	if !near(s["p0"], s["p1"]) {
		t.Fatalf("with EvictionWeight 0 both are plain loads, want equal, got %v", s)
	}
}

func TestBlockedPodPenalized(t *testing.T) {
	m, d := setup(t)
	m["p0"].Routed("busy")
	m["p0"].Started("busy", at(1)) // p0's only slot is running
	d.Observe("busy", at(1))
	s := Score(admit(m, "new", []string{"p0", "p1"}...), d, false, DefaultWeights(), at(2))
	if !(s["p1"] > s["p0"]) {
		t.Fatalf("want free p1 over blocked p0, got %v", s)
	}
}

func TestBaseModelRequestIsNeutral(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	m["p1"].Routed("busy")
	m["p1"].Started("busy", at(1))
	s := Score(admit(m, "base", []string{"p0", "p1", "p2"}...), d, true, DefaultWeights(), at(2))
	for _, p := range []string{"p1", "p2"} {
		if s[p] != s["p0"] {
			t.Fatalf("base-model request must score every pod equally, got %v", s)
		}
	}
}

func TestScoresInUnitRangeAndBestIsOne(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	s := Score(admit(m, "a", []string{"p0", "p1", "p2"}...), d, false, DefaultWeights(), at(2))
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
	s := Score(admit(m, "a", []string{"p0", "p1"}...), d, false, DefaultWeights(), at(1))
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
	s := Score(admit(m, "a", []string{"p0", "p1", "p2"}...), d, false, w, at(3))
	if !(s["p0"] > s["p1"] && s["p1"] > s["p2"]) {
		t.Fatalf("want GPU > CPU > Absent on copy/load cost alone, got %v", s)
	}
}

// With no zero-cost pod the cheapest must still score 1.
func TestCheapestScoresOneWhenAllCostSomething(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p1", "a", 1)
	serve(m, d, "p1", "x", 2) // p1: a in CPU
	s := Score(admit(m, "a", []string{"p1", "p2"}...), d, false, DefaultWeights(), at(3))
	if s["p1"] != 1 || s["p2"] != 0 {
		t.Fatalf("want p1=1 (copy) and p2=0 (load), got %v", s)
	}
}

func TestShareBeforeAnyRequestIsZero(t *testing.T) {
	d := mustDemand(t, time.Minute)
	if s := d.share("a", at(0)); s != 0 {
		t.Fatalf("Share with no observations = %v, want 0", s)
	}
}

// In vLLM's default max_cpu_loras == max_loras, an absent
// admission drops the GPU victim out of the CPU cache too, so bringing it back
// is a load, not a copy.
func TestDefaultConfigVictimPricedAsLoad(t *testing.T) {
	m := estimate{"p0": residency.NewPod(1, 0)}
	d := mustDemand(t, time.Minute)
	serve(m, d, "p0", "hot", 1)
	serve(m, d, "p0", "other", 1) // other evicts hot; give hot a known share
	serve(m, d, "p0", "hot", 2)
	adm := m["p0"].Admit("new")
	if adm.CPUVictim != "hot" || adm.GPUVictim != "" {
		t.Fatalf("Admit(new) = %+v, want CPUVictim hot and no GPUVictim", adm)
	}
	w := DefaultWeights()
	share := d.share("hot", at(3))
	if got, want := cost(adm, d.Shares([]string{adm.GPUVictim, adm.CPUVictim}, at(3)), w), w.LoadCost+share*w.LoadCost; !near(got, want) {
		t.Fatalf("cost = %v, want LoadCost + share*LoadCost = %v", got, want)
	}
}

// With a larger CPU cache, an absent admission prices both victims.
func TestBothVictimsPriced(t *testing.T) {
	m := estimate{"p0": residency.NewPod(1, 2)}
	d := mustDemand(t, time.Minute)
	serve(m, d, "p0", "a", 1)
	serve(m, d, "p0", "b", 2) // GPU = {b}, CPU = {b, a}
	adm := m["p0"].Admit("new")
	if adm.GPUVictim != "b" || adm.CPUVictim != "a" {
		t.Fatalf("Admit(new) = %+v, want GPUVictim b, CPUVictim a", adm)
	}
	w := DefaultWeights()
	want := w.LoadCost + d.share("b", at(3))*w.CopyCost + d.share("a", at(3))*w.LoadCost
	if got := cost(adm, d.Shares([]string{adm.GPUVictim, adm.CPUVictim}, at(3)), w); !near(got, want) {
		t.Fatalf("cost = %v, want %v", got, want)
	}
}

// A routed request that has not started must not make its adapter look
// resident: it stays Absent, and any preference comes from PendingCost alone,
// so with PendingCost equal to LoadCost the pending pod gains nothing.
func TestPendingRequestDoesNotLookResident(t *testing.T) {
	m, d := setup(t)
	m["p0"].Routed("a")
	if got := m["p0"].Admit("a").Tier; got != residency.Absent {
		t.Fatalf("Tier(p0, a) = %v, want Absent while only pending", got)
	}
	w := DefaultWeights()
	w.PendingCost = w.LoadCost
	s := Score(admit(m, "a", []string{"p0", "p1"}...), d, false, w, at(1))
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
	m["p0"].Routed("cold")
	d.Observe("cold", at(1))
	for i := 0; i < 3; i++ {
		s := Score(admit(m, "cold", []string{"p0", "p1", "p2"}...), d, false, DefaultWeights(), at(1))
		if !(s["p0"] > s["p1"] && s["p0"] > s["p2"]) {
			t.Fatalf("request %d: want p0 (already loading cold) preferred, got %v", i, s)
		}
		m["p0"].Routed("cold")
	}
}

// A pending pod must not beat a pod that already holds the adapter on GPU.
func TestGPUBeatsPending(t *testing.T) {
	m, d := setup(t)
	serve(m, d, "p0", "a", 1)
	m["p1"].Routed("a")
	s := Score(admit(m, "a", []string{"p0", "p1"}...), d, false, DefaultWeights(), at(2))
	if !(s["p0"] > s["p1"]) {
		t.Fatalf("want GPU p0 over pending p1, got %v", s)
	}
}
