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
	"testing"
)

// TestBenchDetector_NoLeakOnNonDispatchingCycles verifies the detector self-heals empty-queue idle
// grants: the processor's dispatchCycle evaluates Saturation on a 1ms ticker even when the queue is
// empty, so those grants never dispatch and are never Released. The detector may report saturated
// briefly but must never latch -- grants keep recurring and no saturated run exceeds the stuck
// threshold.
func TestBenchDetector_NoLeakOnNonDispatchingCycles(t *testing.T) {
	d := &benchDetector{}
	d.concurrencyLimit.Store(1)

	grants, run, maxRun := 0, 0, 0
	for i := 0; i < 100; i++ {
		if d.Saturation(context.Background(), nil) < 1.0 {
			grants++
			run = 0
		} else {
			run++
			if run > maxRun {
				maxRun = run
			}
		}
	}
	if grants == 0 {
		t.Fatal("detector latched saturated across all idle cycles (permanent leak)")
	}
	if maxRun > stuckReadThreshold {
		t.Fatalf("idle saturated run = %d, want <= %d (self-heal should reclaim)", maxRun, stuckReadThreshold)
	}
}

// TestBenchDetector_SaturatesAtLimit asserts the W>L backpressure the PerformanceMatrix sweep exists
// to measure: with the limit reached by in-flight (un-Released) grants, the next Saturation must
// report saturated.
func TestBenchDetector_SaturatesAtLimit(t *testing.T) {
	d := &benchDetector{}
	d.concurrencyLimit.Store(2)

	// Two grants whose requests are still in flight (not yet Released) occupy both slots.
	for i := 0; i < 2; i++ {
		if got := d.Saturation(context.Background(), nil); got >= 1.0 {
			t.Fatalf("grant %d within limit must not be saturated: got %v", i, got)
		}
	}
	if got := d.Saturation(context.Background(), nil); got < 1.0 {
		t.Fatalf("over-limit Saturation must report saturated: got %v, want >= 1.0", got)
	}
}

// TestBenchDetector_GrantReleaseBalanced verifies the common path: a cycle grants
// a permit, the dispatched request releases it, and in-flight returns to zero.
func TestBenchDetector_GrantReleaseBalanced(t *testing.T) {
	d := &benchDetector{}
	d.concurrencyLimit.Store(1)

	if got := d.Saturation(context.Background(), nil); got >= 1.0 {
		t.Fatalf("first grant must not be saturated: got %v", got)
	}
	d.Release()
	if got := d.inFlight.Load(); got != 0 {
		t.Fatalf("inFlight = %d after release, want 0", got)
	}
	// The freed permit must be grantable again.
	if got := d.Saturation(context.Background(), nil); got >= 1.0 {
		t.Fatalf("grant after release must not be saturated: got %v", got)
	}
}

// TestBenchDetector_ReleaseRecoversCapacity verifies the steady-state load path: grants fill the
// limit and report saturated, Releases free the slots back to zero, and freed capacity is grantable
// again -- the conserve-count property under sustained dispatch.
func TestBenchDetector_ReleaseRecoversCapacity(t *testing.T) {
	d := &benchDetector{}
	d.concurrencyLimit.Store(2)

	d.Saturation(context.Background(), nil) // grant 1 (in flight)
	d.Saturation(context.Background(), nil) // grant 2 (in flight) -> at limit
	if got := d.Saturation(context.Background(), nil); got < 1.0 {
		t.Fatalf("at-limit Saturation must be saturated: got %v, want >= 1.0", got)
	}

	d.Release() // both in-flight requests complete
	d.Release()
	if got := d.inFlight.Load(); got != 0 {
		t.Fatalf("inFlight = %d after releasing all grants, want 0", got)
	}

	if got := d.Saturation(context.Background(), nil); got >= 1.0 {
		t.Fatalf("grant after releases must not be saturated: got %v", got)
	}
}

// TestBenchDetector_FreeFlow confirms that a non-positive limit is pure free-flow
// with no bookkeeping side effects.
func TestBenchDetector_FreeFlow(t *testing.T) {
	d := &benchDetector{} // limit defaults to 0
	if got := d.Saturation(context.Background(), nil); got != 0.0 {
		t.Fatalf("free-flow Saturation = %v, want 0.0", got)
	}
	d.Release() // must be a no-op
	if got := d.inFlight.Load(); got != 0 {
		t.Fatalf("inFlight = %d under free-flow, want 0", got)
	}
}
