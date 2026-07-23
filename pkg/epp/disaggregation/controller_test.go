package disaggregation

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

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
		"disaggregatedset.x-k8s.io/revision": revision,
		"disaggregatedset.x-k8s.io/role":     "prefill",
	}
}

// --- Filter tests -----------------------------------------------------------

func TestFilter_StrictWithMatch(t *testing.T) {
	c := NewController(validConfig(), nil)
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
	c := NewController(validConfig(), nil)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
	}
	req := &fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}}
	got := c.Filter(context.Background(), req, pods)
	if len(got) != 0 {
		t.Fatalf("strict mode should return empty on no match, got %d", len(got))
	}
}

func TestFilter_PreferFallsBack(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].Mode = ModePrefer
	c := NewController(cfg, nil)
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
	c := NewController(validConfig(), nil)
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
	c := NewController(cfg, nil)
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
	c := NewController(validConfig(), nil)
	pods := []fwksched.Endpoint{endpoint("p1", revLabels("v1"))}
	got := c.Filter(context.Background(), nil, pods)
	if len(got) != 1 {
		t.Fatalf("nil request should be no-op, got %d", len(got))
	}
}

// TestFilter_PreferDoesNotRescueAfterStrictEmpties verifies that when a
// strict selector has already emptied the current pool, a subsequent prefer
// selector does NOT restore anything. Prefer's fallback is "keep what you
// have," and after strict zeroed it, "what you have" is nothing.
func TestFilter_PreferDoesNotRescueAfterStrictEmpties(t *testing.T) {
	config := validConfig()
	// Second selector: prefer mode on a different label.
	config.Selectors = append(config.Selectors, Selector{
		Name:       "slice",
		HeaderName: "x-disagg-slice",
		LabelKey:   "mistral.ai/slice",
		Mode:       ModePrefer,
	})
	c := NewController(config, nil)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	// Strict revision header will match zero pods → current becomes empty.
	// Prefer slice header with a match SHOULD NOT resurrect anything.
	req := &fwksched.InferenceRequest{Headers: map[string]string{
		"x-disagg-revision": "v99",
		"x-disagg-slice":    "any",
	}}
	got := c.Filter(context.Background(), req, pods)
	if len(got) != 0 {
		t.Fatalf("prefer must not restore after strict emptied; got %d pods", len(got))
	}
}

// TestFilter_ReturnsFreshSlice locks in the convention that Filter never
// hands back the caller's slice. Callers should be free to mutate the input
// after Filter returns without seeing our result change.
func TestFilter_ReturnsFreshSlice(t *testing.T) {
	c := NewController(validConfig(), nil)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	// No header → filter is a no-op, but must still return a copy.
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

// --- gatingFilter fixture --------------------------------------------------

// scoreFixture builds a controller wired to a cache with the given pod counts.
// counts[revision][role] = ready pod count.
func scoreFixture(t *testing.T, cfg Config, counts map[string]map[string]int) *Controller {
	t.Helper()
	var pods []*corev1.Pod
	i := 0
	for rev, roles := range counts {
		for role, n := range roles {
			for k := 0; k < n; k++ {
				i++
				name := "pod-" + rev + "-" + role + "-" + itoa(i)
				pods = append(pods, readyPod(name, rev, role))
			}
		}
	}
	return NewController(cfg, seedCache(t, pods...))
}

// itoa avoids importing strconv just for tests.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := "0123456789"
	buf := make([]byte, 0, 6)
	for n > 0 {
		buf = append([]byte{digits[n%10]}, buf...)
		n /= 10
	}
	return string(buf)
}

// --- gatingFilter tests ----------------------------------------------------

func TestGatingFilter_KeepsCandidatesWithLiveCrossRoles(t *testing.T) {
	c := scoreFixture(t, validConfig(), map[string]map[string]int{
		"v1": {"prefill": 3, "decode": 3},
		"v2": {"prefill": 1, "decode": 1},
	})
	f := newGatingFilter(c)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v2")),
	}
	got := f.Filter(context.Background(), nil, pods)
	if len(got) != 2 {
		t.Fatalf("both revisions alive → both survive; got %d", len(got))
	}
}

func TestGatingFilter_DropsCandidatesWithMissingRole(t *testing.T) {
	// decode-v1 has zero Ready pods → v1 candidates must be dropped.
	c := scoreFixture(t, validConfig(), map[string]map[string]int{
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
	// no decode pods anywhere → every revision fails the gate.
	c := scoreFixture(t, validConfig(), map[string]map[string]int{
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
	// Fresh controller without a pod cache (nil is safe when gating is off).
	c := NewController(cfg, nil)
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
	// mode=disabled with a filled sub-block should still pass through: the
	// user asked us not to gate.
	cfg := validConfig()
	cfg.Gating.Mode = GatingModeDisabled
	c := NewController(cfg, nil)
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
	c := NewController(validConfig(), nil)
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
	c := NewController(cfg, nil)
	resp := &fwkrc.Response{Headers: map[string]string{}}
	labels := revLabels("v1")
	labels["mistral.ai/slice"] = "s1"
	c.ResponseHeader(context.Background(), nil, resp, &fwkdl.EndpointMetadata{Labels: labels})
	if resp.Headers["x-disagg-revision"] != "v1" || resp.Headers["x-disagg-slice"] != "s1" {
		t.Fatalf("multi-header stamp: %v", resp.Headers)
	}
}

func TestResponseHeader_MissingLabelSkipsSilently(t *testing.T) {
	c := NewController(validConfig(), nil)
	resp := &fwkrc.Response{Headers: map[string]string{}}
	// no revision label on this pod
	c.ResponseHeader(context.Background(), nil, resp, &fwkdl.EndpointMetadata{Labels: map[string]string{}})
	if _, ok := resp.Headers["x-disagg-revision"]; ok {
		t.Fatalf("no-label pod should not stamp: %v", resp.Headers)
	}
}

func TestResponseHeader_NilEndpointIsNoop(t *testing.T) {
	c := NewController(validConfig(), nil)
	resp := &fwkrc.Response{Headers: map[string]string{}}
	c.ResponseHeader(context.Background(), nil, resp, nil)
	if len(resp.Headers) != 0 {
		t.Fatalf("nil endpoint must not modify headers: %v", resp.Headers)
	}
}

func TestResponseHeader_NilResponseIsNoop(t *testing.T) {
	c := NewController(validConfig(), nil)
	// must not panic
	c.ResponseHeader(context.Background(), nil, nil, &fwkdl.EndpointMetadata{Labels: revLabels("v1")})
}
