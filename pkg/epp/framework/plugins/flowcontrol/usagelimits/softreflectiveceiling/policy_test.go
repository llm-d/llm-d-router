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

package softreflectiveceiling

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

func newTestPolicy() *policy {
	return newPolicy("soft-reflective", logr.Discard())
}

// strictDecoder mirrors the framework's plugin registry: DisallowUnknownFields
// over the raw parameters block.
func strictDecoder(s string) *json.Decoder {
	dec := json.NewDecoder(bytes.NewReader([]byte(s)))
	dec.DisallowUnknownFields()
	return dec
}

func TestFactory(t *testing.T) {
	p, err := Factory("soft-reflective", nil, nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	if p == nil {
		t.Fatal("Factory returned nil plugin")
	}
	if _, ok := p.(flowcontrol.UsageLimitPolicy); !ok {
		t.Fatalf("Factory result %T does not implement UsageLimitPolicy", p)
	}
	tn := p.TypedName()
	if tn.Name != "soft-reflective" {
		t.Errorf("TypedName.Name = %q, want %q", tn.Name, "soft-reflective")
	}
	if tn.Type != PolicyType {
		t.Errorf("TypedName.Type = %q, want %q", tn.Type, PolicyType)
	}
}

func TestFactory_EmptyConfigAccepted(t *testing.T) {
	if _, err := Factory("sr", strictDecoder(`{}`), nil); err != nil {
		t.Fatalf("Factory({}) returned error: %v", err)
	}
}

func TestFactory_UnknownFieldRejected(t *testing.T) {
	_, err := Factory("sr", strictDecoder(`{"threshold": 0.5}`), nil)
	if err == nil {
		t.Fatal("Factory with unknown field should have failed, got nil error")
	}
	if !strings.Contains(err.Error(), "soft-reflective-ceiling-policy takes no parameters") {
		t.Errorf("error message = %q, want it to mention the policy takes no parameters", err.Error())
	}
}

func TestFactory_MalformedJSONRejected(t *testing.T) {
	_, err := Factory("sr", strictDecoder(`{not-json`), nil)
	if err == nil {
		t.Fatal("Factory with malformed JSON should have failed, got nil error")
	}
}

func TestPolicyType(t *testing.T) {
	if PolicyType != "soft-reflective-ceiling-policy" {
		t.Errorf("PolicyType = %q, want %q", PolicyType, "soft-reflective-ceiling-policy")
	}
}

func TestComputeLimit_NoBands(t *testing.T) {
	p := newTestPolicy()
	got := p.ComputeLimit(context.Background(), 0.9, nil)
	if len(got) != 0 {
		t.Errorf("expected empty ceilings, got %v", got)
	}
}

func TestComputeLimit_SingleBand(t *testing.T) {
	p := newTestPolicy()
	for _, sat := range []float64{0.0, 0.5, 0.99, 1.0} {
		got := p.ComputeLimit(context.Background(), sat, []int{100})
		if len(got) != 1 || got[0] != 1.0 {
			t.Errorf("saturation=%.2f single-band ceilings=%v, want [1.0]", sat, got)
		}
	}
}

func TestComputeLimit_CriticalBandNeverGated(t *testing.T) {
	p := newTestPolicy()
	priorities := []int{100, 0, -50}
	for _, sat := range []float64{0.0, 0.3, 0.5, 0.7, 0.99, 1.0, 1.5} {
		got := p.ComputeLimit(context.Background(), sat, priorities)
		if got[0] != 1.0 {
			t.Errorf("saturation=%.2f: ceilings[0]=%v, want 1.0", sat, got[0])
		}
	}
}

func TestComputeLimit_BelowCeilingAllOpen(t *testing.T) {
	// At saturation=0.3 with N=3, reflective ceilings are
	//   [1.0, 1-0.15=0.85, 1-0.30=0.70].
	// Saturation is below all ceilings, so every band returns 1.0.
	p := newTestPolicy()
	got := p.ComputeLimit(context.Background(), 0.3, []int{100, 0, -50})
	for i, c := range got {
		if c != 1.0 {
			t.Errorf("band %d: got %v, want 1.0 (below ceiling)", i, c)
		}
	}
}

func TestComputeLimit_FullySaturated(t *testing.T) {
	// At saturation>=1.0, non-critical bands hard-block (ceiling=0.0).
	p := newTestPolicy()
	got := p.ComputeLimit(context.Background(), 1.0, []int{100, 0, -50})
	if got[0] != 1.0 {
		t.Errorf("ceilings[0]=%v, want 1.0 (critical band)", got[0])
	}
	for i := 1; i < len(got); i++ {
		if got[i] != 0.0 {
			t.Errorf("ceilings[%d]=%v, want 0.0 (fully saturated)", i, got[i])
		}
	}
}

func TestComputeLimit_ReflectiveFormula(t *testing.T) {
	// Discriminating saturation: with N=4 and saturation=0.7, reflective
	// ceilings are
	//   [1.0, 1-0.7/3=0.7667, 1-1.4/3=0.5333, 1-2.1/3=0.3].
	// Band 1 is below its ceiling (open), bands 2 and 3 are at/over their
	// ceilings (gated). This exercises both branches of the formula.
	p := newTestPolicy()
	got := p.ComputeLimit(context.Background(), 0.7, []int{100, 50, 0, -50})

	if got[0] != 1.0 {
		t.Errorf("band 0: got %v, want 1.0 (critical)", got[0])
	}
	if got[1] != 1.0 {
		t.Errorf("band 1: got %v, want 1.0 (below reflective ceiling 0.7667)", got[1])
	}
	for i := 2; i <= 3; i++ {
		if got[i] != 0.0 && got[i] != 1.0 {
			t.Errorf("band %d: got %v, want 0.0 or 1.0 (gated)", i, got[i])
		}
	}
}

func TestComputeLimit_Alternation(t *testing.T) {
	// At N=2, saturation=0.7: reflective ceiling[1] = 0.3, so band 1 is
	// gated. period = round(0.7/0.3) = 2. The shared tick increments
	// 1,2,3,4,...; tick%2==0 is open. Expected pattern: closed, open,
	// closed, open, ...
	p := newTestPolicy()
	want := []float64{0.0, 1.0, 0.0, 1.0, 0.0, 1.0}
	for i, w := range want {
		got := p.ComputeLimit(context.Background(), 0.7, []int{100, -50})
		if got[0] != 1.0 {
			t.Errorf("call %d: critical band gated unexpectedly: %v", i, got[0])
		}
		if got[1] != w {
			t.Errorf("call %d: ceilings[1]=%v, want %v", i, got[1], w)
		}
	}
}

func TestComputeLimit_AlternationLongerPeriod(t *testing.T) {
	// At N=2, saturation=0.75: reflective ceiling[1] = 0.25, gated;
	// period = round(0.75/0.25) = 3. Pattern: two closed then open, repeating.
	p := newTestPolicy()
	want := []float64{0.0, 0.0, 1.0, 0.0, 0.0, 1.0}
	for i, w := range want {
		got := p.ComputeLimit(context.Background(), 0.75, []int{100, -50})
		if got[1] != w {
			t.Errorf("call %d: ceilings[1]=%v, want %v (period=3)", i, got[1], w)
		}
	}
}

func TestComputeLimit_GrowingPriorities(t *testing.T) {
	// The active priority domain can expand mid-flight without panic and
	// band 0 stays ungated.
	p := newTestPolicy()
	_ = p.ComputeLimit(context.Background(), 0.7, []int{100, -50})
	got := p.ComputeLimit(context.Background(), 0.7, []int{100, 50, 0, -50})
	if len(got) != 4 {
		t.Fatalf("len(ceilings)=%d, want 4", len(got))
	}
	if got[0] != 1.0 {
		t.Errorf("ceilings[0]=%v, want 1.0", got[0])
	}
}

func TestComputeLimit_NoStarvationUnderRisingSaturation(t *testing.T) {
	// Regression test for a starvation bug where per-band alternation
	// counters could desynchronize under a rising-saturation ramp, letting
	// a lower gated band's open ticks land on a higher gated band's closed
	// ticks. Because the dispatch loop aborts at the first gated band, the
	// lower band would then receive zero dispatch even while claiming
	// ceiling=1.0 on some calls.
	//
	// Scenario: priorities=[100, 0, -50]. Warm up at sat=0.6 where only
	// band 2 is gated (period 2), then run at sat=0.7 where bands 1 and 2
	// are both gated (period 2). With the fix (single shared tick), every
	// call must produce monotone ceilings across ranks -- either all gated
	// bands open together or all closed together.
	p := newTestPolicy()
	priorities := []int{100, 0, -50}

	for i := 0; i < 3; i++ {
		p.ComputeLimit(context.Background(), 0.6, priorities)
	}

	band1Open := 0
	band2Open := 0
	const iterations = 100
	for i := 0; i < iterations; i++ {
		got := p.ComputeLimit(context.Background(), 0.7, priorities)
		if got[1] == 1.0 {
			band1Open++
		}
		if got[2] == 1.0 {
			band2Open++
		}
		// Monotonicity: whenever band 2 is open, band 1 must also be open;
		// otherwise the dispatch loop aborts at band 1 and band 2 is
		// unreachable regardless of its ceiling.
		if got[2] == 1.0 && got[1] != 1.0 {
			t.Errorf("call %d: band 2 open (1.0) while band 1 closed (%v); dispatch loop would abort at band 1 and starve band 2",
				i, got[1])
		}
	}

	if band1Open != band2Open {
		t.Errorf("band1Open=%d, band2Open=%d; gated bands must open on the same calls", band1Open, band2Open)
	}
	if band2Open == 0 {
		t.Errorf("band 2 opened %d times over %d calls; expected non-zero (starvation regression)", band2Open, iterations)
	}
}
