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
	// gated. period = round(0.7/0.3) = 2. The internal counter increments
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
	// Counters are allocated per-priority on first sight; the active priority
	// domain can expand mid-flight without panic and band 0 stays ungated.
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

func TestComputeLimit_StatePersistsAcrossPrioritySetChanges(t *testing.T) {
	// A counter belongs to a priority, not to its rank. When a new priority
	// is added between existing ones (e.g. via dynamic InferenceObjective
	// provisioning in the flow-control registry), an existing band's
	// alternation state must continue from where it left off rather than be
	// reset by rank reassignment.
	//
	// Start with priorities=[100, -50], saturation=0.75.
	// N=2: ceiling[1] = 1 - 0.75 = 0.25 (gated), period = round(0.75/0.25) = 3.
	// After 5 calls, priority -50's tick counter is at 5.
	p := newTestPolicy()
	for i := 0; i < 5; i++ {
		p.ComputeLimit(context.Background(), 0.75, []int{100, -50})
	}

	// Insert priority 0 between the existing two. With rank-indexed counters
	// this would move -50's tick state onto the fresh priority 0.
	got := p.ComputeLimit(context.Background(), 0.75, []int{100, 0, -50})

	// N=3 at saturation=0.75:
	//   ceiling[1] = 1 - 0.75/2 = 0.625 (gated), period = round(0.75/0.25) = 3.
	//   ceiling[2] = 1 - 1.5/2  = 0.25  (gated), period = 3.
	// Priority -50 (rank 2) continues from tick 5: increments to 6, 6%3 == 0
	// so it opens on this call.
	if got[2] != 1.0 {
		t.Errorf("ceilings[2] (priority -50 after inserting priority 0) = %v, want 1.0; state must persist across priority-set changes",
			got[2])
	}
	// Priority 0 (rank 1, first sight) starts from tick 0: increments to 1,
	// 1%3 != 0, so it stays closed.
	if got[1] != 0.0 {
		t.Errorf("ceilings[1] (new priority 0) = %v, want 0.0; new priority starts fresh at tick=1", got[1])
	}
}
