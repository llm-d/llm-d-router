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
	return NewController(cfg, fakeReader(pods...), testNS, scope)
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
	got := c.Filter(context.Background(), req, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p1" {
		t.Fatalf("expected only p1, got %v", got)
	}
}

func TestFilter_StrictNoMatchReturnsEmpty(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}}
	got := c.Filter(context.Background(), req, pods)
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
	got := c.Filter(context.Background(), req, pods)
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
	got := c.Filter(context.Background(), req, pods)
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
	got := c.Filter(context.Background(), req, pods)
	if len(got) != 1 || got[0].GetMetadata().PodName != "p2" {
		t.Fatalf("expected only p2 (v1+s2), got %v", got)
	}
}

func TestFilter_NilRequestIsNoop(t *testing.T) {
	c := newTestController(validConfig())
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	got := c.Filter(context.Background(), nil, pods)
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
	got := c.Filter(context.Background(), req, pods)
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
	got := c.Filter(context.Background(), req, pods)
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

func TestGatingFilter_WeightedPickAcrossRolesLikeUsersExample(t *testing.T) {
	// 2p+18d (v1) / 1p+2d (v2) — weight v1=20, v2=3, so v2's share of the
	// weighted pick is 3/23 ≈ 13% over many iterations. Runs 10k
	// iterations for a tight-enough Monte Carlo bound.
	c := gatingFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 2, "decode": 18},
		"v2": {"prefill": 1, "decode": 2},
	})
	f := newGatingFilter(c)
	// The prefill EPP only sees prefill pods as candidates.
	pods := []fwksched.Endpoint{
		endpoint("p1a", revLabels("v1")),
		endpoint("p1b", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	const iterations = 10000
	v1Wins := 0
	for i := 0; i < iterations; i++ {
		got := f.Filter(context.Background(), nil, pods)
		if len(got) == 0 {
			t.Fatalf("iter %d: empty survivor set", i)
		}
		switch got[0].GetMetadata().Labels[testRevLabel] {
		case "v1":
			v1Wins++
		case "v2":
			// counted implicitly
		default:
			t.Fatalf("iter %d: unexpected revision label", i)
		}
	}
	v2Share := float64(iterations-v1Wins) / float64(iterations)
	// Expected 3/23 ≈ 0.1304 — tolerance loose enough for 10k iterations
	// with the process-global RNG.
	if v2Share < 0.11 || v2Share > 0.16 {
		t.Fatalf("v2 share (2p+18d vs 1p+2d): want ≈13%%, got %.2f%% over %d iterations", v2Share*100, iterations)
	}
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
