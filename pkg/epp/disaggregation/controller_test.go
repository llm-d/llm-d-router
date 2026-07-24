package disaggregation

import (
	"context"
	"strconv"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	testRevLabel  = "disaggregatedset.x-k8s.io/revision"
	testRoleLabel = "disaggregatedset.x-k8s.io/role"
	testNS        = "default"
	testSelector  = "disaggregatedset.x-k8s.io/name=my-set"
)

func readyPod(name, revision, role string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNS,
			Labels: map[string]string{
				testRevLabel:                     revision,
				testRoleLabel:                    role,
				"disaggregatedset.x-k8s.io/name": "my-set",
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
		},
	}
}

// fakeReader builds a controller-runtime fake client seeded with the given
// pods. Used everywhere a test needs a controller with a functional reader.
func fakeReader(pods ...*corev1.Pod) client.Reader {
	objs := make([]client.Object, 0, len(pods))
	for _, p := range pods {
		objs = append(objs, p)
	}
	return fake.NewClientBuilder().WithScheme(scheme.Scheme).WithObjects(objs...).Build()
}

func newTestController(cfg Config, pods ...*corev1.Pod) *Controller {
	scope, _ := labels.Parse(testSelector)
	return newController(cfg, fakeReader(pods...), testNS, scope)
}

func endpoint(name string, labels map[string]string) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Labels:         labels,
	}
	return fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, nil)
}

func revLabels(revision string) map[string]string {
	return map[string]string{
		testRevLabel:  revision,
		testRoleLabel: "prefill",
	}
}

// --- Filter tests -----------------------------------------------------------

func TestFilter_StrictWithMatch(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v1"}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p1" {
		t.Fatalf("expected only p1, got %v", got)
	}
}

func TestFilter_StrictNoMatchReturnsEmpty(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 0 {
		t.Fatalf("strict mode should return empty on no match, got %d", len(got))
	}
}

func TestFilter_PreferFallsBack(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].Mode = ModePrefer
	c := newTestController(cfg)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 2 {
		t.Fatalf("prefer mode should fall back to full set, got %d", len(got))
	}
}

func TestFilter_HeaderAbsentIsNoop(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 2 {
		t.Fatalf("absent header should be no-op, got %d", len(got))
	}
}

func TestFilter_MultipleSelectorsAppliedInOrder(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors = append(cfg.Selectors, Selector{
		Name:       "slice",
		HeaderName: "x-disagg-slice",
		LabelKey:   "mistral.ai/slice",
		Mode:       ModeStrict,
	})
	c := newTestController(cfg)
	makeLabels := func(rev, slice string) map[string]string {
		m := revLabels(rev)
		m["mistral.ai/slice"] = slice
		return m
	}
	pods := []fwksched.Endpoint{
		endpoint("p1", makeLabels("v1", "s1")),
		endpoint("p2", makeLabels("v1", "s2")),
		endpoint("p3", makeLabels("v2", "s1")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{
		"x-disagg-revision": "v1",
		"x-disagg-slice":    "s2",
	}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p2" {
		t.Fatalf("expected only p2 (v1+s2), got %v", got)
	}
}

func TestFilter_NilRequestIsNoop(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	got := c.filter(context.Background(), nil, pods)
	if len(got) != 1 {
		t.Fatalf("nil request should be no-op, got %d", len(got))
	}
}

func TestFilter_PreferDoesNotRescueAfterStrictEmpties(t *testing.T) {
	config := validConfig()
	config.Selectors = append(config.Selectors, Selector{
		Name:       "slice",
		HeaderName: "x-disagg-slice",
		LabelKey:   "mistral.ai/slice",
		Mode:       ModePrefer,
	})
	c := newTestController(config)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{
		"x-disagg-revision": "v99",
		"x-disagg-slice":    "any",
	}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 0 {
		t.Fatalf("prefer must not restore after strict emptied; got %d pods", len(got))
	}
}

func TestFilter_ReturnsFreshSlice(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{}}
	got := c.filter(context.Background(), req, pods)
	if len(got) != 2 || len(pods) != 2 {
		t.Fatalf("setup wrong: got %d, pods %d", len(got), len(pods))
	}
	got[0] = endpoint("mutated", revLabels("mutated"))
	if pods[0].GetMetadata().PodName != "p1" {
		t.Fatalf("Filter leaked aliased slice: mutation of returned slice changed caller's input")
	}
}

// --- gatingFilter tests ----------------------------------------------------

// gatingFixture builds a controller wired to a fake reader with the given
// pod counts. counts[revision][role] = ready pod count.
func gatingFixture(t *testing.T, cfg Config, counts map[string]map[string]int) *Controller {
	t.Helper()
	var pods []*corev1.Pod
	i := 0
	for rev, roles := range counts {
		for role, n := range roles {
			for k := 0; k < n; k++ {
				i++
				pods = append(pods, readyPod("pod-"+rev+"-"+role+"-"+strconv.Itoa(i), rev, role))
			}
		}
	}
	return newTestController(cfg, pods...)
}

func TestGatingFilter_KeepsOneRevisionPerCall(t *testing.T) {
	// Both revisions are alive → gating picks ONE per request, weighted by
	// cross-role pod count. The picker downstream then sees only that
	// revision's pods. Which one wins depends on rand01; here we fix it
	// so the assertion is deterministic.
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3, "decode": 3},
		"v2": {"prefill": 1, "decode": 1},
	})
	f := newGatingFilter(c)
	f.rand01 = constRand(0.0) // pick the first revision in sorted order (v1)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p1" {
		t.Fatalf("rand01=0 with sorted revs [v1,v2] must pick v1's p1; got %v", got)
	}
}

// TestGatingFilter_WeightedPickMatchesLiveSlowTransitionShapes covers the
// three shapes exercised by scripts/verify-slow-transition.sh
// (10p10d, 2p20d, 20p2d). Each case picks the middle-rollout step where
// the pod counts are unbalanced enough that the weighted-pick math is
// observable, and pins the expected v2 share to (crossRoleWeight(v2) /
// Σ crossRoleWeight). 10k iterations per case; ±3pp tolerance is
// comfortable at that sample size (3σ for p=0.13, N=10k is ~1pp).
//
// The 2p20d / 20p2d "step2" rows are the specific regression case:
// under the pre-weighted-pick behaviour they would land near 33%
// (prefill count ratio), not 13% (cross-role weight ratio).
func TestGatingFilter_WeightedPickMatchesLiveSlowTransitionShapes(t *testing.T) {
	tests := []struct {
		name        string
		podCounts   map[string]map[string]int // Ready pods per revision per role in scope
		candidates  []fwksched.Endpoint       // prefill-side candidates the picker sees
		wantV2Share float64                   // expected v2 fraction of the weighted pick
	}{
		{
			name: "10p10d step2 (7+3 p / 7+3 d) — balanced, expect v2=30%",
			podCounts: map[string]map[string]int{
				"v1": {"prefill": 7, "decode": 7},
				"v2": {"prefill": 3, "decode": 3},
			},
			candidates:  candidatePool(7, 3),
			wantV2Share: 6.0 / 20.0, // = 30%
		},
		{
			name: "2p20d step2 (2+1 p / 18+2 d) — decode-heavy, expect v2=13% (would be 33% under prefill-only)",
			podCounts: map[string]map[string]int{
				"v1": {"prefill": 2, "decode": 18},
				"v2": {"prefill": 1, "decode": 2},
			},
			candidates:  candidatePool(2, 1),
			wantV2Share: 3.0 / 23.0, // ≈ 13%
		},
		{
			name: "2p20d step4 (1+2 p / 2+18 d) — mirror of step2, expect v2=87%",
			podCounts: map[string]map[string]int{
				"v1": {"prefill": 1, "decode": 2},
				"v2": {"prefill": 2, "decode": 18},
			},
			candidates:  candidatePool(1, 2),
			wantV2Share: 20.0 / 23.0, // ≈ 87%
		},
		{
			name: "20p2d step2 (18+2 p / 2+1 d) — prefill-heavy, expect v2=13%",
			podCounts: map[string]map[string]int{
				"v1": {"prefill": 18, "decode": 2},
				"v2": {"prefill": 2, "decode": 1},
			},
			candidates:  candidatePool(18, 2),
			wantV2Share: 3.0 / 23.0, // ≈ 13%
		},
		{
			name: "20p2d step4 (2+18 p / 1+2 d) — mirror of step2, expect v2=87%",
			podCounts: map[string]map[string]int{
				"v1": {"prefill": 2, "decode": 1},
				"v2": {"prefill": 18, "decode": 2},
			},
			candidates:  candidatePool(2, 18),
			wantV2Share: 20.0 / 23.0, // ≈ 87%
		},
	}

	const iterations = 10000
	const tolerance = 0.03 // ±3pp — comfortable at N=10k

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := gatingFixture(t, validConfig(), tt.podCounts)
			f := newGatingFilter(c)

			v2Wins := 0
			for i := 0; i < iterations; i++ {
				got := f.Filter(context.Background(), nil, tt.candidates)
				if len(got) == 0 {
					t.Fatalf("iter %d: empty survivor set", i)
				}
				// All survivors are the same revision (gating collapsed
				// candidates to one revision). Sample the first.
				switch got[0].GetMetadata().Labels[testRevLabel] {
				case "v2":
					v2Wins++
				case "v1":
					// counted implicitly
				default:
					t.Fatalf("iter %d: unexpected revision label %q", i, got[0].GetMetadata().Labels[testRevLabel])
				}
			}

			gotV2Share := float64(v2Wins) / float64(iterations)
			diff := gotV2Share - tt.wantV2Share
			if diff < -tolerance || diff > tolerance {
				t.Fatalf("v2 share: want %.3f (±%.3f), got %.3f (%d/%d) — diff %.3fpp",
					tt.wantV2Share, tolerance, gotV2Share, v2Wins, iterations, diff*100)
			}
		})
	}
}

// candidatePool builds a candidate list with n1 v1-labelled endpoints and
// n2 v2-labelled endpoints — mirrors what a prefill EPP would receive from
// endpoint discovery when the pool has (n1, n2) prefill Ready pods.
func candidatePool(n1, n2 int) []fwksched.Endpoint {
	pods := make([]fwksched.Endpoint, 0, n1+n2)
	for i := 0; i < n1; i++ {
		pods = append(pods, endpoint("v1-"+strconv.Itoa(i), revLabels("v1")))
	}
	for i := 0; i < n2; i++ {
		pods = append(pods, endpoint("v2-"+strconv.Itoa(i), revLabels("v2")))
	}
	return pods
}

func TestGatingFilter_SingleRevisionDoesNotConsumeRand(t *testing.T) {
	// When only one revision is present, the pick is deterministic and
	// rand01 must not be called (avoiding needless RNG contention). A
	// panicking rand asserts this.
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3, "decode": 3},
	})
	f := newGatingFilter(c)
	f.rand01 = func() float64 { panic("rand01 should not be called with a single revision") }
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 1 {
		t.Fatalf("single-revision passthrough failed: %v", got)
	}
}

// constRand returns a rand01 stub that always yields x. Used in tests to
// pin the weighted pick to a known outcome.
func constRand(x float64) func() float64 {
	return func() float64 { return x }
}

func TestGatingFilter_DropsCandidatesWithMissingRole(t *testing.T) {
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3, "decode": 0},
		"v2": {"prefill": 1, "decode": 4},
	})
	f := newGatingFilter(c)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v1")),
		endpoint("p3", revLabels("v1")),
		endpoint("p4", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p4" {
		t.Fatalf("v1 must be dropped, only p4 survives; got %v", got)
	}
}

func TestGatingFilter_AllRevisionsDeadReturnsEmpty(t *testing.T) {
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3},
		"v2": {"prefill": 1},
	})
	f := newGatingFilter(c)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 0 {
		t.Fatalf("all dropped; got %v", got)
	}
}

func TestGatingFilter_NoGatingConfigIsNoop(t *testing.T) {
	cfg := validConfig()
	cfg.Gating = nil
	c := newTestController(cfg)
	f := newGatingFilter(c)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 2 {
		t.Fatalf("nil Gating → passthrough; got %d", len(got))
	}
}

func TestGatingFilter_SkipsWhenRevisionHeaderPresent(t *testing.T) {
	// A request with the revision-axis strict header means strict downstream
	// will pin the revision; gating firing here would either waste work or,
	// worse, weight-pick a revision that strict then rejects (503). The
	// filter must passthrough without reading the cache — assert by using a
	// panicking rand01 as a canary.
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 2, "decode": 18},
		"v2": {"prefill": 1, "decode": 2},
	})
	f := newGatingFilter(c)
	f.rand01 = func() float64 { panic("rand01 must not be called when the revision header is set") }
	pods := []fwksched.Endpoint{
		endpoint("p1a", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v1"}}
	got := f.Filter(context.Background(), req, pods)
	if len(got) != len(pods) {
		t.Fatalf("expected passthrough of all %d pods, got %d", len(pods), len(got))
	}
}

func TestGatingFilter_SkipsWhenSingleRevisionInPool(t *testing.T) {
	// If the candidate pool already has only one unique revision — e.g.
	// because the fleet has one revision, or an upstream filter narrowed —
	// gating has nothing to shape and must skip. Panicking rand01 asserts
	// the fast path fires.
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3, "decode": 3},
	})
	f := newGatingFilter(c)
	f.rand01 = func() float64 { panic("rand01 must not be called with a single-revision pool") }
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v1")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != len(pods) {
		t.Fatalf("expected passthrough of all %d pods, got %d", len(pods), len(got))
	}
}

func TestGatingFilter_DisabledModeIsNoop(t *testing.T) {
	cfg := validConfig()
	cfg.Gating.Mode = GatingModeDisabled
	c := newTestController(cfg)
	f := newGatingFilter(c)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 2 {
		t.Fatalf("mode=disabled → passthrough; got %d", len(got))
	}
}

// --- ResponseHeader tests ---------------------------------------------------

func TestResponseHeader_StampsSelectorHeader(t *testing.T) {
	c := newTestController(validConfig())
	resp := &fwkrc.Response{Headers: map[string]string{}}
	ep := &fwkdl.EndpointMetadata{Labels: revLabels("v1")}
	c.ResponseHeader(context.Background(), nil, resp, ep)
	if resp.Headers["x-disagg-revision"] != "v1" {
		t.Fatalf("header not stamped: %v", resp.Headers)
	}
}

func TestResponseHeader_MultipleSelectorsStampAll(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors = append(cfg.Selectors, Selector{
		Name: "slice", HeaderName: "x-disagg-slice",
		LabelKey: "mistral.ai/slice", Mode: ModeStrict,
	})
	c := newTestController(cfg)
	resp := &fwkrc.Response{Headers: map[string]string{}}
	labels := revLabels("v1")
	labels["mistral.ai/slice"] = "s1"
	c.ResponseHeader(context.Background(), nil, resp, &fwkdl.EndpointMetadata{Labels: labels})
	if resp.Headers["x-disagg-revision"] != "v1" || resp.Headers["x-disagg-slice"] != "s1" {
		t.Fatalf("multi-header stamp: %v", resp.Headers)
	}
}

func TestResponseHeader_MissingLabelSkipsSilently(t *testing.T) {
	c := newTestController(validConfig())
	resp := &fwkrc.Response{Headers: map[string]string{}}
	c.ResponseHeader(context.Background(), nil, resp, &fwkdl.EndpointMetadata{Labels: map[string]string{}})
	if _, ok := resp.Headers["x-disagg-revision"]; ok {
		t.Fatalf("no-label pod should not stamp: %v", resp.Headers)
	}
}

func TestResponseHeader_NilEndpointIsNoop(t *testing.T) {
	c := newTestController(validConfig())
	resp := &fwkrc.Response{Headers: map[string]string{}}
	c.ResponseHeader(context.Background(), nil, resp, nil)
	if len(resp.Headers) != 0 {
		t.Fatalf("nil endpoint must not modify headers: %v", resp.Headers)
	}
}

func TestResponseHeader_NilResponseIsNoop(t *testing.T) {
	c := newTestController(validConfig())
	c.ResponseHeader(context.Background(), nil, nil, &fwkdl.EndpointMetadata{Labels: revLabels("v1")})
}
