package disaggregation

import (
	"context"
	"math/rand/v2"
	"sort"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// modeSelectorsFilter is a thin Filter wrapper that visits only the
// controller's selectors matching one mode. Registered at head for strict,
// tail for prefer, so downstream filters never see wrong-revision
// candidates and prefer's fallback captures the fully-narrowed pool.
type modeSelectorsFilter struct {
	controller *Controller
	mode       SelectorMode
	typedName  fwkplugin.TypedName
	keepFn     func(Selector) bool
}

var (
	_ fwkplugin.Plugin = (*modeSelectorsFilter)(nil)
	_ fwksched.Filter  = (*modeSelectorsFilter)(nil)
)

const (
	strictFilterType = "disaggregation-strict-filter"
	preferFilterType = "disaggregation-prefer-filter"
	gatingFilterType = "disaggregation-gating-filter"
)

func newModeSelectorsFilter(controller *Controller, mode SelectorMode, typeName string) *modeSelectorsFilter {
	return &modeSelectorsFilter{
		controller: controller,
		mode:       mode,
		typedName:  fwkplugin.TypedName{Type: typeName, Name: typeName},
		keepFn: func(selector Selector) bool {
			return selector.Mode == mode
		},
	}
}

func (f *modeSelectorsFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *modeSelectorsFilter) Filter(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	return f.controller.filterSelectors(ctx, request, pods, f.keepFn)
}

// gatingFilter is where the disaggregation controller decides the revision
// axis when no client header has done so. Runs at the HEAD of the filter
// chain. Two things happen per Filter call:
//
//  1. Drop any candidate whose revision fails the coverage check —
//     revisions missing pods on any listed role (rollout drift) are
//     removed from the survivor set. This is the safety gate that keeps
//     prefill from tagging requests for revisions decode can't serve.
//
//  2. Among surviving revisions, weighted-random-pick ONE revision per
//     request. Weight per revision = Σ Ready pod count across the roles
//     listed in gating.requireRoles.values. The picker downstream sees
//     only pods of the chosen revision and picks uniformly among them.
//
// This gives traffic ∝ cross-role pod count per revision, independent of
// the picker choice in the operator YAML: prefill traffic to revision R
// converges on (crossRolePods(R) / Σ crossRolePods) even when prefill and
// decode replica counts differ per revision (2p+18d vs. 1p+2d etc.).
//
// The filter short-circuits (passthrough, no cache read) when:
//   - The request carries the revision-axis strict header — the strict
//     filter downstream will pin the revision, we shouldn't stochastic-
//     pick something that would then get dropped as a mismatch (503).
//   - The candidate pool contains at most one unique revision — nothing
//     to shape, either because the fleet has one revision or because an
//     upstream filter already narrowed things.
type gatingFilter struct {
	controller *Controller
	typedName  fwkplugin.TypedName
	// rand01 returns a float in [0, 1). Injected so tests can drive the
	// weighted pick deterministically.
	rand01 func() float64
}

var (
	_ fwkplugin.Plugin = (*gatingFilter)(nil)
	_ fwksched.Filter  = (*gatingFilter)(nil)
)

func newGatingFilter(controller *Controller) *gatingFilter {
	return &gatingFilter{
		controller: controller,
		typedName:  fwkplugin.TypedName{Type: gatingFilterType, Name: gatingFilterType},
		// math/rand/v2 top-level RNG is process-global and thread-safe.
		rand01: rand.Float64,
	}
}

func (f *gatingFilter) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *gatingFilter) Filter(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	gating := f.controller.config.Gating
	revisionLabelKey := f.controller.revisionLabelKey
	if !gating.Active() || revisionLabelKey == "" {
		return append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	}

	// Fast path: header-pinned. Strict downstream will do the narrowing.
	if f.hasRevisionHeader(request) {
		return append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	}

	// Fast path: pool already has ≤1 unique revision. Nothing to shape.
	seenRevisions := uniqueRevisions(pods, revisionLabelKey)
	if len(seenRevisions) <= 1 {
		return append(make([]fwksched.Endpoint, 0, len(pods)), pods...)
	}

	requiredRoles := gating.RequireRoles.Values

	_, roleCounts, err := f.controller.readyPodsByRevisionRole(ctx)
	if err != nil {
		// Cache-backed List failing is unexpected; be pessimistic and drop
		// everything rather than route to potentially-broken revisions.
		return nil
	}

	// Weight per revision in the pool (0 if any required role has no Ready
	// pod → gated).
	weights := make(map[string]int, len(seenRevisions))
	for revision := range seenRevisions {
		weights[revision] = crossRoleWeight(roleCounts[revision], requiredRoles)
	}

	// Emit gating-drop metrics for revisions that flunked coverage,
	// before we discard them.
	for revision, weight := range weights {
		if weight == 0 {
			recordGatingDropped(revision)
			delete(weights, revision)
		}
	}

	if len(weights) == 0 {
		return nil
	}

	chosen := f.pickWeightedRevision(weights)

	survivors := make([]fwksched.Endpoint, 0, len(pods))
	for _, endpoint := range pods {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		if endpoint.GetMetadata().Labels[revisionLabelKey] == chosen {
			survivors = append(survivors, endpoint)
		}
	}
	return survivors
}

// crossRoleWeight returns the sum of Ready pod counts across every listed
// role for a revision, or 0 if any listed role has no Ready pod (gated).
func crossRoleWeight(perRole map[string]int, required []string) int {
	total := 0
	for _, role := range required {
		count := perRole[role]
		if count == 0 {
			return 0
		}
		total += count
	}
	return total
}

// hasRevisionHeader reports whether the request carries the header that
// pins the revision axis. Convention: the revision axis is the first
// entry in Selectors — same source as controller.revisionLabelKey — and
// only counts when it's a strict-mode selector (prefer mode is a hint,
// not a pin).
func (f *gatingFilter) hasRevisionHeader(request *fwksched.InferenceRequest) bool {
	if request == nil {
		return false
	}
	selectors := f.controller.config.Selectors
	if len(selectors) == 0 {
		return false
	}
	revSelector := selectors[0]
	if revSelector.Mode != ModeStrict {
		return false
	}
	return request.Headers[revSelector.HeaderName] != ""
}

// uniqueRevisions returns the set of distinct revision-label values in
// the candidate pool. Used only by the gating fast path.
func uniqueRevisions(pods []fwksched.Endpoint, revisionLabelKey string) map[string]struct{} {
	seen := make(map[string]struct{})
	for _, endpoint := range pods {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		if revision := endpoint.GetMetadata().Labels[revisionLabelKey]; revision != "" {
			seen[revision] = struct{}{}
		}
	}
	return seen
}

// pickWeightedRevision returns one revision from weights, chosen with
// probability weights[r] / Σ weights. Iteration order is stabilised via
// sort so a fixed rand01 gives repeatable output in tests.
func (f *gatingFilter) pickWeightedRevision(weights map[string]int) string {
	revs := make([]string, 0, len(weights))
	total := 0
	for revision, weight := range weights {
		revs = append(revs, revision)
		total += weight
	}
	if total == 0 {
		return ""
	}
	sort.Strings(revs)
	// Handle a single-revision fast path so we don't call rand01 needlessly.
	if len(revs) == 1 {
		return revs[0]
	}
	x := f.rand01() * float64(total)
	cumulative := 0.0
	for _, revision := range revs {
		cumulative += float64(weights[revision])
		if x < cumulative {
			return revision
		}
	}
	// Floating-point tail: fall through to the last revision.
	return revs[len(revs)-1]
}
