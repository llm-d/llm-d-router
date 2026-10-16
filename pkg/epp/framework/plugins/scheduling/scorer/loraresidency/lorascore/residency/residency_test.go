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

package residency

import (
	"reflect"
	"testing"
	"time"
)

var t0 = time.Unix(1_000_000, 0)

func at(s int) time.Time { return t0.Add(time.Duration(s) * time.Second) }

// served runs one request to completion, leaving the adapter idle at time s.
func served(p *Pod, adapter string, s int) {
	p.Routed(adapter)
	p.Started(adapter, at(s))
	p.Finished(adapter, at(s), true)
}

// running leaves one request for adapter executing since time s.
func running(p *Pod, adapter string, s int) {
	p.Routed(adapter)
	p.Started(adapter, at(s))
}

func TestUnknownAdapterIsAbsent(t *testing.T) {
	p := NewPod(2, 4)
	if got := p.Tier("a"); got != Absent {
		t.Fatalf("Tier(a) = %v, want Absent", got)
	}
}

func TestMostRecentFillGPUThenCPU(t *testing.T) {
	p := NewPod(2, 3)
	served(p, "a", 1)
	served(p, "b", 2)
	served(p, "c", 3)
	if got, want := p.Order(), []string{"c", "b", "a"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Order = %v, want %v", got, want)
	}
	for adapter, want := range map[string]Tier{"c": GPU, "b": GPU, "a": CPU} {
		if got := p.Tier(adapter); got != want {
			t.Errorf("Tier(%s) = %v, want %v", adapter, got, want)
		}
	}
}

func TestBeyondCPUCapacityIsForgotten(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	served(p, "b", 2)
	served(p, "c", 3)
	if got := p.Tier("a"); got != Absent {
		t.Fatalf("Tier(a) = %v, want Absent once pushed past the CPU cache", got)
	}
	if got := len(p.Order()); got != 2 {
		t.Fatalf("len(Order) = %d, want 2 (CPU capacity)", got)
	}
}

// vLLM schedules RUNNING requests first and skips a WAITING
// request whose adapter would exceed max_loras, so a merely routed request
// must not displace a running adapter.
func TestRoutedDoesNotDisplaceRunning(t *testing.T) {
	p := NewPod(1, 2)
	running(p, "a", 1)
	p.Routed("b")
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (running)", got)
	}
	if got := p.Tier("b"); got != Absent {
		t.Fatalf("Tier(b) = %v, want Absent (routed, not executed)", got)
	}
}

// A request aborted before it reached the engine must leave
// no phantom resident adapter.
func TestAbortBeforeStartLeavesNoTrace(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	p.Routed("b")
	p.Finished("b", at(2), false)
	if got := p.Tier("b"); got != Absent {
		t.Fatalf("Tier(b) = %v, want Absent after an abort before start", got)
	}
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (untouched by the aborted request)", got)
	}
	if got := p.Pending("b"); got != 0 {
		t.Fatalf("Pending(b) = %d, want 0 after the abort", got)
	}
}

// vLLM touches a running adapter every engine step, so while it executes it
// outranks every idle adapter, however recently they were used.
func TestRunningOutranksIdle(t *testing.T) {
	p := NewPod(1, 3)
	running(p, "a", 1)
	served(p, "b", 5)
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU while running", got)
	}
	p.Finished("a", at(3), true) // a's last step precedes b's
	if got := p.Tier("b"); got != GPU {
		t.Fatalf("after a finishes, Tier(b) = %v, want GPU", got)
	}
}

// A finish with no matching start (e.g. after an EPP failover) must not drive
// the running count negative and so make a later request look idle.
func TestUnmatchedFinishDoesNotUnderflow(t *testing.T) {
	p := NewPod(1, 2)
	p.Finished("a", at(1), true)
	running(p, "a", 3)
	served(p, "b", 4)
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU while its request runs", got)
	}
}

func TestUnmatchedStartDoesNotUnderflowPending(t *testing.T) {
	p := NewPod(1, 2)
	p.Started("a", at(1)) // no Routed seen
	p.Routed("a")
	if got := p.Pending("a"); got != 1 {
		t.Fatalf("Pending(a) = %d, want 1 (the later route)", got)
	}
}

func TestRunningKeptPastCPUCapacity(t *testing.T) {
	p := NewPod(1, 1)
	running(p, "a", 1)
	running(p, "b", 2)
	if got := len(p.Order()); got != 2 {
		t.Fatalf("len(Order) = %d, want 2 while both run", got)
	}
}

// A stale observation must not move an adapter's recency backwards.
func TestStaleTouchDoesNotRewindRecency(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 5)
	served(p, "b", 4)
	p.ObserveRunning([]string{"a"}, at(3))
	p.ObserveRunning(nil, at(3))
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (last use 5 beats b's 4)", got)
	}
}

func TestObserveRunningTouchesAndPins(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 1)
	served(p, "b", 2)
	p.ObserveRunning([]string{"a"}, at(3)) // traffic this EPP did not route
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU while reported running", got)
	}
	if got := p.Admit("c"); !got.Blocked {
		t.Fatalf("Admit(c) = %+v, want Blocked: a is reported running", got)
	}
	p.ObserveRunning(nil, at(1)) // a later snapshot without a
	if got := p.Admit("c"); got.Blocked {
		t.Fatalf("Admit(c) = %+v, want not Blocked once a is no longer reported", got)
	}
}

// An adapter the pod reports active (running or queued) must
// not be chosen as a victim, and must count toward Blocked.
func TestObserveActiveProtectsResidentAdapter(t *testing.T) {
	p := NewPod(2, 4)
	served(p, "a", 1)
	served(p, "b", 2) // GPU = {b, a}; a is the LRU
	p.ObserveActive([]string{"a"})
	if got := p.Admit("c"); got.GPUVictim != "b" {
		t.Fatalf("Admit(c) = %+v, want GPUVictim b (a reported active)", got)
	}
	p.ObserveActive([]string{"a", "b"})
	if got := p.Admit("c"); !got.Blocked {
		t.Fatalf("Admit(c) = %+v, want Blocked with both GPU adapters active", got)
	}
}

// ActiveModels includes queued adapters, so it is not evidence of execution:
// it must not touch recency or create residency.
func TestObserveActiveDoesNotCreateResidency(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 1)
	p.ObserveActive([]string{"x"})
	if got := p.Tier("x"); got != Absent {
		t.Fatalf("Tier(x) = %v, want Absent (active may mean queued)", got)
	}
	served(p, "b", 2)
	p.ObserveActive([]string{"a"}) // a is in CPU, not GPU
	if got := p.Tier("b"); got != GPU {
		t.Fatalf("Tier(b) = %v, want GPU (ObserveActive must not touch a)", got)
	}
}

func TestObserveActiveSnapshotReplaces(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 1)
	p.ObserveActive([]string{"a"})
	p.ObserveActive(nil)
	if got := p.Admit("c"); got.GPUVictim != "a" || got.Blocked {
		t.Fatalf("Admit(c) = %+v, want GPUVictim a after the snapshot clears", got)
	}
}

func TestAdmitResidentNeedsNoVictim(t *testing.T) {
	p := NewPod(2, 3)
	served(p, "a", 1)
	if got, want := p.Admit("a"), (Admission{Tier: GPU}); got != want {
		t.Fatalf("Admit(a) = %+v, want %+v", got, want)
	}
}

func TestAdmitWithFreeSlotNeedsNoVictim(t *testing.T) {
	p := NewPod(2, 3)
	served(p, "a", 1)
	if got, want := p.Admit("b"), (Admission{Tier: Absent}); got != want {
		t.Fatalf("Admit(b) = %+v, want %+v", got, want)
	}
}

// A CPU-tier adapter is activated without touching the CPU cache, so the GPU
// victim is demoted to CPU.
func TestAdmitFromCPUDemotesGPUVictim(t *testing.T) {
	p := NewPod(2, 4)
	served(p, "a", 1)
	served(p, "b", 2)
	served(p, "c", 3) // GPU = {c, b}, CPU adds a
	if got, want := p.Admit("a"), (Admission{Tier: CPU, GPUVictim: "b"}); got != want {
		t.Fatalf("Admit(a) = %+v, want %+v", got, want)
	}
}

// Default configuration: with max_cpu_loras == max_loras the
// oldest CPU entry is the GPU victim, and it is dropped to storage, not kept
// in CPU.
func TestAdmitAbsentDefaultConfigDropsVictimToStorage(t *testing.T) {
	p := NewPod(2, 2)
	served(p, "a", 1)
	served(p, "b", 2)
	if got, want := p.Admit("c"), (Admission{Tier: Absent, CPUVictim: "a"}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

// Larger CPU cache: an absent admission drops the oldest
// CPU-only entry to storage and also demotes the GPU LRU to CPU.
func TestAdmitAbsentFullCPUEvictsBothTiers(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	served(p, "b", 2) // GPU = {b}, CPU = {b, a}
	if got, want := p.Admit("c"), (Admission{Tier: Absent, GPUVictim: "b", CPUVictim: "a"}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

// An absent admission with CPU room only demotes the GPU victim.
func TestAdmitAbsentCPURoomDemotesOnly(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 1)
	served(p, "b", 2)
	if got, want := p.Admit("c"), (Admission{Tier: Absent, GPUVictim: "b"}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

func TestAdmitSkipsRunningVictim(t *testing.T) {
	p := NewPod(2, 4)
	running(p, "b", 1) // older but running
	served(p, "c", 2)
	if got, want := p.Admit("d"), (Admission{Tier: Absent, GPUVictim: "c"}); got != want {
		t.Fatalf("Admit(d) = %+v, want %+v", got, want)
	}
}

// With every GPU slot running, vLLM skips the request rather than evicting.
func TestAdmitBlockedWhenAllGPUSlotsRunning(t *testing.T) {
	p := NewPod(2, 4)
	running(p, "a", 1)
	running(p, "b", 2)
	if got, want := p.Admit("c"), (Admission{Tier: Absent, Blocked: true}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

func TestAdmitDoesNotMutate(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	before := p.Order()
	p.Admit("b")
	if after := p.Order(); !reflect.DeepEqual(before, after) {
		t.Fatalf("Admit mutated the order: %v -> %v", before, after)
	}
}

// vLLM forces max_cpu_loras >= max_loras; a smaller value must not shrink the GPU tier.
func TestCPUCapacityNeverBelowGPU(t *testing.T) {
	p := NewPod(3, 1)
	served(p, "a", 1)
	served(p, "b", 2)
	served(p, "c", 3)
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (cpu clamped up to gpu=3)", got)
	}
}

func TestSetCapacityShrinkForgetsOverflow(t *testing.T) {
	p := NewPod(2, 3)
	served(p, "a", 1)
	served(p, "b", 2)
	served(p, "c", 3)
	p.SetCapacity(1, 1)
	if got, want := p.Order(), []string{"c"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Order = %v, want %v", got, want)
	}
}

func modelServed(m *Model, pod, adapter string, s int) {
	m.Routed(pod, adapter)
	m.Started(pod, adapter, at(s))
	m.Finished(pod, adapter, at(s), true)
}

func TestModelKeepsPodsSeparateAndDeletes(t *testing.T) {
	m := NewModel(2)
	modelServed(m, "pod-0", "a", 1)
	modelServed(m, "pod-1", "b", 2)
	if got := m.Tier("pod-1", "a"); got != Absent {
		t.Fatalf("pod-1 Tier(a) = %v, want Absent", got)
	}
	if got := m.Tier("pod-0", "b"); got != Absent {
		t.Fatalf("pod-0 Tier(b) = %v, want Absent", got)
	}
	if got := m.Tier("pod-0", "a"); got != GPU {
		t.Fatalf("pod-0 Tier(a) = %v, want GPU", got)
	}
	m.Delete("pod-0")
	if got := m.Tier("pod-0", "a"); got != Absent {
		t.Fatalf("after Delete, pod-0 Tier(a) = %v, want Absent", got)
	}
}

func TestModelObserveCapacity(t *testing.T) {
	m := NewModel(2) // max_cpu_loras = 2
	m.ObserveCapacity("pod-0", 1)
	modelServed(m, "pod-0", "a", 1)
	modelServed(m, "pod-0", "b", 2)
	if got := m.Tier("pod-0", "a"); got != CPU {
		t.Fatalf("Tier(a) = %v, want CPU with gpu=1, cpu=2", got)
	}
	m.ObserveCapacity("pod-0", 2)
	if got := m.Tier("pod-0", "a"); got != GPU {
		t.Fatalf("after capacity 2, Tier(a) = %v, want GPU", got)
	}
}

// MaxActiveModels reads 0 when the max_lora label is absent (e.g. behind
// vLLM's Rust frontend); that must not overwrite a real capacity.
func TestModelIgnoresZeroCapacity(t *testing.T) {
	m := NewModel(0)
	m.ObserveCapacity("pod-0", 2)
	m.ObserveCapacity("pod-0", 0)
	modelServed(m, "pod-0", "a", 1)
	modelServed(m, "pod-0", "b", 2)
	if got := m.Tier("pod-0", "a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (capacity 0 ignored, 2 kept)", got)
	}
}

func TestModelObservations(t *testing.T) {
	m := NewModel(0)
	m.ObserveCapacity("pod-0", 1)
	modelServed(m, "pod-0", "a", 1)
	m.ObserveActive("pod-0", []string{"a"})
	if got := m.Admit("pod-0", "b"); !got.Blocked {
		t.Fatalf("Admit(b) = %+v, want Blocked (a reported active)", got)
	}
	m.ObserveActive("pod-0", nil)
	m.ObserveRunning("pod-0", []string{"a"}, at(2))
	if got := m.Admit("pod-0", "b"); !got.Blocked {
		t.Fatalf("Admit(b) = %+v, want Blocked (a reported running)", got)
	}
	if got := m.Admit("pod-9", "b"); got != (Admission{Tier: Absent}) {
		t.Fatalf("unknown pod Admit = %+v, want Absent with no victim", got)
	}
}

func TestStartedClearsPending(t *testing.T) {
	p := NewPod(1, 2)
	p.Routed("a")
	p.Started("a", at(1))
	if got := p.Pending("a"); got != 0 {
		t.Fatalf("Pending(a) = %d, want 0 once started", got)
	}
}

// A running report is evidence of execution, so its recency outlives the
// snapshot that carried it.
func TestObserveRunningRecencyOutlivesSnapshot(t *testing.T) {
	p := NewPod(1, 3)
	served(p, "a", 1)
	served(p, "b", 2)
	p.ObserveRunning([]string{"a"}, at(3))
	p.ObserveRunning(nil, at(4))
	if got := p.Tier("a"); got != GPU {
		t.Fatalf("Tier(a) = %v, want GPU (seen running at 3, after b at 2)", got)
	}
}

// Activating a CPU-tier adapter does not load anything into the CPU cache, so
// a full CPU cache drops nothing to storage.
func TestAdmitFromFullCPUDropsNothing(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	served(p, "b", 2) // GPU = {b}, CPU = {b, a}
	if got, want := p.Admit("a"), (Admission{Tier: CPU, GPUVictim: "b"}); got != want {
		t.Fatalf("Admit(a) = %+v, want %+v", got, want)
	}
}

// vLLM's CPU LRU ignores queue state, so an adapter
// reported active while only in the CPU cache is still the CPU victim, and
// the GPU LRU is demoted as well.
func TestActiveCPUOnlyAdapterIsStillCPUVictim(t *testing.T) {
	p := NewPod(1, 2)
	served(p, "a", 1)
	served(p, "b", 2) // GPU = {b}, CPU = {b, a}
	p.ObserveActive([]string{"a"})
	if got, want := p.Admit("c"), (Admission{Tier: Absent, GPUVictim: "b", CPUVictim: "a"}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

// An active adapter estimated on GPU is plausibly running (vLLM touches it each
// step), so it is protected in the CPU-victim choice too, not only the GPU one.
func TestActiveGPUAdapterProtectedInBothTiers(t *testing.T) {
	p := NewPod(2, 2) // default max_cpu_loras == max_loras
	served(p, "a", 1)
	served(p, "b", 2) // GPU = CPU = {b, a}
	p.ObserveActive([]string{"a"})
	if got, want := p.Admit("c"), (Admission{Tier: Absent, CPUVictim: "b"}); got != want {
		t.Fatalf("Admit(c) = %+v, want %+v", got, want)
	}
}

// A pod already loading the adapter for an earlier
// request is a pending tier, and the eviction it causes is not charged twice.
func TestPendingAdmission(t *testing.T) {
	p := NewPod(1, 1)
	served(p, "a", 1)
	p.Routed("b")
	if got, want := p.Admit("b"), (Admission{Tier: Absent, Pending: true}); got != want {
		t.Fatalf("Admit(b) = %+v, want %+v (pending, victim not repriced)", got, want)
	}
	p.Started("b", at(2))
	if got := p.Admit("b"); got.Pending || got.Tier != GPU {
		t.Fatalf("Admit(b) = %+v, want GPU and not pending once started", got)
	}
}

func TestPendingStillBlocked(t *testing.T) {
	p := NewPod(1, 2)
	running(p, "a", 1)
	p.Routed("b")
	if got, want := p.Admit("b"), (Admission{Tier: Absent, Pending: true, Blocked: true}); got != want {
		t.Fatalf("Admit(b) = %+v, want %+v", got, want)
	}
}

func TestModelHas(t *testing.T) {
	m := NewModel(0)
	if m.Has("pod-0") {
		t.Fatal("Has(pod-0) before any event")
	}
	m.ObserveCapacity("pod-0", 1)
	if !m.Has("pod-0") {
		t.Fatal("Has(pod-0) = false after an observation")
	}
	m.Delete("pod-0")
	if m.Has("pod-0") {
		t.Fatal("Has(pod-0) after Delete")
	}
}
