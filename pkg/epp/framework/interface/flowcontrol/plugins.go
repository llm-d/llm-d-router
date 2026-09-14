/*
Copyright 2025 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"errors"
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	// ErrIncompatiblePriorityType indicates that a FairnessPolicy attempted to compare items from two different flow
	// queues whose ItemComparators have different ScoreType values, making a meaningful comparison impossible.
	ErrIncompatiblePriorityType = errors.New("incompatible priority score type for comparison")
)

// FairnessPolicy governs the distribution of dispatch opportunities among competing Flows within the same Priority
// Band.
//
// In simple terms, this policy answers the question: "Which flow gets to dispatch a request next?"
//
// While "Priority" determines strictly which group of flows is serviced first, "Fairness" determines how resources are
// shared when multiple flows in that same group are fighting for capacity.
//
// Architecture (Flyweight Pattern):
// Fairness plugins are Singletons. A single instance of a FairnessPolicy handles the logic for potentially many
// different Priority Bands. To support this, the plugin must be purely functional, separating its Logic (methods) from
// its State (data).
//
//   - Logic: Defined here in the FairnessPolicy interface.
//   - State: Created via NewState() and stored on the PriorityBandAccessor.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type FairnessPolicy interface {
	plugin.Plugin

	// NewState creates the scoped, mutable storage required by this policy for a single Priority Band.
	//
	// Because the plugin instance itself is shared globally, it cannot hold state like "current round-robin index" or
	// "accumulated deficits" inside struct fields. Instead, it creates this state object once per Band.
	//
	// The Flow Registry manages the lifecycle of this object, storing it on the Priority Band and passing it back to the
	// plugin via the PriorityBandAccessor during Pick.
	//
	// Returns:
	//   - any: The opaque state object (e.g., &roundRobinCursor{index: 0}).
	NewState(ctx context.Context) any

	// Pick inspects the active flows in the provided Flow Group (Priority Band) and selects the "winner" for the next
	// dispatch attempt.
	//
	// This is the core logic loop. The implementation should:
	//  1. Retrieve its scoped state from band.GetPolicyState().
	//  2. Cast the state to its concrete type (e.g., *roundRobinCursor).
	//  3. Apply its algorithm to select a FlowQueueAccessor.
	//  4. Update the state (e.g., increment the cursor) if necessary.
	//
	// State may also be updated out-of-band (e.g., from monitoring a metrics server, from integrating with request
	// lifecycle hooks, etc.).
	//
	// Returns:
	//   - flow: The Flow to service next. Returns nil if no valid candidate is found (e.g., all queues empty).
	//   - err: Only returned for unrecoverable internal errors. Policies should generally return (nil, nil) if simply
	//     nothing is eligible.
	Pick(ctx context.Context, flowGroup PriorityBandAccessor) (flow FlowQueueAccessor, err error)
}

// OrderingPolicy governs the strict sequence of service within a single Flow.
//
// In simple terms, this policy answers the question: "Which request in this specific queue should be processed next?"
//
// While "Fairness" governs the competition between flows, "Ordering" dictates the internal discipline of a single
// flow. This allows different flows to have different internal service objectives (e.g., FCFS vs. EDF).
//
// Architecture (Flyweight Pattern):
// Ordering policies are Singletons. A single instance handles the logic for all queues in a Priority Band.
// The comparator is a pure function of the items it is given.
//
//   - Logic: Defined here as a Comparator-centric interface.
//   - State: Ordering policies are generally stateless, operating on the intrinsic properties of the items.
//     A policy whose ordering key is revised while an item waits in the queue MUST NOT read that key from
//     Less; it implements ScoringOrderingPolicy instead, and the queue caches and refreshes on its behalf.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe.
type OrderingPolicy interface {
	plugin.Plugin

	// Less reports whether item 'a' should be dispatched before item 'b'.
	// This makes the policy act as a sort.Interface for the queue, determining the dispatch order
	// (the queue's head is the highest-priority item per this comparator).
	//
	// Invariant: returning true means 'a' has higher priority than 'b'.
	Less(a, b QueueItemAccessor) bool
}

// ScoringOrderingPolicy is an optional extension of OrderingPolicy for policies whose ordering key is
// revised in place while an item waits -- for example one keyed on live backend state that shifts as
// requests are served. Reading such a key from Less breaks the heap: the queue re-sorts only on Add and
// Remove so it never reflects the new key, and two comparisons within one sift can observe different
// states, so Less is not transitive.
//
// A policy implementing this interface delegates both problems to the queue, which caches each item's
// Score, compares only cached values, and rebuilds when Generation changes. The queue detects the
// interface via a checked type assertion at construction; a policy that does not implement it is driven
// through Less alone and pays no overhead.
//
// Conformance: implementations MUST satisfy all of the following.
//
//   - All methods MUST be goroutine-safe. The instance is a singleton shared by every queue in every band
//     that references it.
//   - Less MUST be implemented as CompareByScore(p, a, b). The fairness tier compares heads across queues
//     through Less with no access to the queue's cache; any other implementation makes that cross-queue
//     comparison silently inconsistent with the queue's own ordering.
//   - Score MUST NOT return NaN, which compares false against everything and breaks the heap's strict weak
//     ordering. NormalizeScore is a backstop, not a license to rely on it.
//   - Score MUST be cheap and MUST NOT call back into the queue. It runs once per item per rebuild under
//     the queue's write lock: a callback deadlocks, and an expensive Score blocks the flow's mutation path.
//   - Generation MUST be monotonically non-decreasing and throttled. Every observed change triggers an
//     O(n) rebuild in each queue ordered by this instance, so a generation tracking an underlying revision
//     one-to-one rebuilds on nearly every Peek. Latch a value and re-latch at most once per interval.
//   - A decorator wrapping an OrderingPolicy MUST forward Score and Generation, or it hides this interface
//     from the type assertion and the wrapped policy degrades silently to static ordering on Less alone.
//
// Ordering is approximate: an item added after the current generation's rebuild is scored against fresher
// state than its neighbors and is reconciled only at the next generation's rebuild, so worst-case staleness
// spans up to two refresh intervals. Because Add scores against a warmer key than the incumbents' cached
// scores, this consistently favors newly-arriving requests over waiting ones within an interval -- a
// directional bias, not symmetric noise.
type ScoringOrderingPolicy interface {
	OrderingPolicy

	// Score returns the item's current priority key. Higher scores dispatch first.
	Score(item QueueItemAccessor) float64

	// Generation identifies the epoch of the state underlying Score. A change tells the queue that any
	// cached score may be stale and that it must recompute before its next read.
	Generation() uint64
}

// CompareByScore reports whether a dispatches before b under p: higher Score first, ties broken by earlier
// EnqueueTime, NaN normalized to negative infinity.
//
// A scoring policy MUST implement Less by calling this helper. The fairness tier compares heads across
// queues through OrderingPolicy.Less and has no access to any queue's cache, so this is what makes the
// band-level comparison apply the same rule as the queue's cached ordering. The two differ only in
// freshness: this helper reads Score live, the queue reads its cache.
//
// nil handling matches the static ordering policies: a nil item is lowest priority, and two nils compare
// equal.
func CompareByScore(p ScoringOrderingPolicy, a, b QueueItemAccessor) bool {
	if a == nil {
		return false
	}
	if b == nil {
		return true
	}
	scoreA, scoreB := NormalizeScore(p.Score(a)), NormalizeScore(p.Score(b))
	if scoreA != scoreB {
		return scoreA > scoreB
	}
	return a.EnqueueTime().Before(b.EnqueueTime())
}

// NormalizeScore maps NaN to negative infinity, leaving every other float64 -- including both infinities --
// unchanged. A NaN score compares false against everything, which breaks the strict weak ordering a heap
// requires.
//
// ScoringOrderingPolicy forbids returning NaN. This is the backstop for a non-conforming policy, exported so
// that its two consumers share one definition of the rule: CompareByScore when comparing live scores, and
// the queue when caching them.
func NormalizeScore(score float64) float64 {
	if math.IsNaN(score) {
		return math.Inf(-1)
	}
	return score
}

// SaturationDetector provides real-time load signals.
//
// Plugins implementing this interface provide a continuous saturation gradient [0.0, 1.0+] based on
// the observed state of the endpoints.
type SaturationDetector interface {
	plugin.Plugin

	// Saturation returns the aggregate saturation level of the candidate pool.
	//
	//   - A value >= 1.0 indicates that the system is fully saturated. Values strictly > 1.0
	//     represent the depth of overload, scaling proportionally with the excess load.
	//   - A value < 1.0 indicates the ratio of used capacity to total available capacity.
	//
	// The FlowController consumes this signal to make dispatch decisions:
	//   - If Saturation() >= 1.0: Stop dispatching and apply backpressure (buffer requests).
	//   - If Saturation() < 1.0: Continue dispatching traffic to the pool.
	Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64
}

// UsageLimitPolicy computes the usage limit of a priority band dynamically.
//
// The goal of this policy is to enable adaptive capacity management by gating lower-priority traffic
// as the pool approaches saturation, reserving headroom for future higher-priority requests.
//
// Saturation represents resource usage as a fraction of total capacity (0.0 = idle, 1.0 = fully saturated)
// as described in [/pkg/epp/flowcontrol/contracts.SaturationDetector]
//
// Architecture (Mostly Stateless Singleton):
// UsageLimitPolicy plugins are Singletons. A single instance handles limit computation for all priority bands.
// The plugin SHOULD be stateless -- a pure function mapping the current saturation and active priority
// domain to a set of ceilings. Small bounded dispatch-spreading state (e.g. a tick counter used to fold
// successive calls into a proportional duty cycle) is permitted; signal conditioning (trend detection,
// smoothing) is not, and belongs in the SaturationDetector layer.
//
// Integration:
// This policy is called during dispatch decision-making, before a request is allowed to proceed. For each
// priority band, the computed ceiling is compared against current saturation. If saturation exceeds the
// ceiling for a given priority, requests at that priority are gated (not dispatched). The dispatch loop
// visits bands from highest to lowest priority and stops at the first gated band; lower bands are not
// considered on that call.
//
// The framework calls ComputeLimit exactly once per dispatch cycle. This is a contract term, not an
// implementation detail: dispatch-spreading policies use the call itself as their time base (one tick
// per cycle), and computing all ceilings in one call is what lets every gated band observe the same
// open/closed decision within a cycle. The batch shape exists to preserve both properties.
//
// Conformance: Implementations MUST ensure all methods are goroutine-safe. Computed ceilings MUST be
// monotonically non-increasing in the given priority order (highest priority first): because the
// dispatch loop stops at the first gated band, a lower band whose ceiling exceeds that of a higher band
// can be marked open on calls where it is unreachable, starving it.
type UsageLimitPolicy interface {
	plugin.Plugin

	// ComputeLimit calculates usage ceilings for all currently active priority levels based on current
	// saturation, writing the ceiling for the n-th priority into the n-th element of the
	// caller-provided ceilings buffer. The plugin observes the active priority domain (which changes
	// dynamically as workloads come and go) and computes relative ceilings from scratch on each call.
	//
	// The framework guarantees len(ceilings) == len(priorities) and pre-fills every element with 1.0
	// (no gating), so an entry the plugin does not write fails open. Writing into the framework-owned
	// buffer means a result of the wrong size cannot exist; there is no return value to validate.
	//
	// Parameters:
	//   - ctx: Request context for logging, tracing, etc.
	//   - saturation: Current pool-wide resource saturation as a fraction [0.0, 1.0]
	//   - priorities: Ordered list of currently active priority levels (highest first).
	//     The slice is a shared snapshot owned by the framework: read-only, and it MUST NOT be
	//     retained after the call returns.
	//   - ceilings: Output buffer owned by the framework, valid only for the duration of the call.
	//     Ceiling semantics per element:
	//     - 0.0 = fully gated (cannot dispatch regardless of current saturation)
	//     - 1.0 = no gating (can dispatch until fully saturated)
	//     - Values between 0.0 and 1.0 reserve capacity headroom
	ComputeLimit(ctx context.Context, saturation float64, priorities []int, ceilings []float64)
}
