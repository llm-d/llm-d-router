// Metric tests share process-global Prometheus collectors registered once
// via sync.Once. Do NOT call t.Parallel() in this file — the shared
// registry would race.
package disaggregation

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// resetMetrics zeros the underlying collectors between tests. Registration is
// process-global via sync.Once so we can't re-register — but Reset() clears
// the observed samples.
func resetMetrics(t *testing.T) {
	t.Helper()
	registerMetrics()
	headerStampedTotal.Reset()
	filterOutcomeTotal.Reset()
	gatingDroppedTotal.Reset()
	cacheReadyPods.Reset()
}

// --- Header stamped --------------------------------------------------------

func TestMetric_HeaderStamped_IncrementsPerSelector(t *testing.T) {
	resetMetrics(t)
	controller := NewController(validConfig(), nil)
	controller.ResponseHeader(context.Background(), nil,
		&fwkrc.Response{Headers: map[string]string{}},
		&fwkdl.EndpointMetadata{Labels: revLabels("v1")},
	)
	if got := testutil.ToFloat64(headerStampedTotal.WithLabelValues("revision")); got != 1 {
		t.Fatalf("want 1, got %v", got)
	}
}

func TestMetric_HeaderStamped_SkipsMissingLabel(t *testing.T) {
	resetMetrics(t)
	controller := NewController(validConfig(), nil)
	controller.ResponseHeader(context.Background(), nil,
		&fwkrc.Response{Headers: map[string]string{}},
		&fwkdl.EndpointMetadata{Labels: map[string]string{}},
	)
	if got := testutil.ToFloat64(headerStampedTotal.WithLabelValues("revision")); got != 0 {
		t.Fatalf("want 0 (no label), got %v", got)
	}
}

// --- Filter outcomes -------------------------------------------------------

func TestMetric_FilterOutcome_Matched(t *testing.T) {
	resetMetrics(t)
	controller := NewController(validConfig(), nil)
	controller.Filter(context.Background(),
		&fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v1"}},
		[]fwksched.Endpoint{endpoint("p1", revLabels("v1"))},
	)
	got := testutil.ToFloat64(filterOutcomeTotal.WithLabelValues("revision", string(ModeStrict), filterOutcomeMatched))
	if got != 1 {
		t.Fatalf("matched: want 1, got %v", got)
	}
}

func TestMetric_FilterOutcome_NoMatchStrict(t *testing.T) {
	resetMetrics(t)
	controller := NewController(validConfig(), nil)
	controller.Filter(context.Background(),
		&fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}},
		[]fwksched.Endpoint{endpoint("p1", revLabels("v1"))},
	)
	got := testutil.ToFloat64(filterOutcomeTotal.WithLabelValues("revision", string(ModeStrict), filterOutcomeNoMatchStrict))
	if got != 1 {
		t.Fatalf("no_match_strict: want 1, got %v", got)
	}
}

func TestMetric_FilterOutcome_NoMatchPreferFallback(t *testing.T) {
	resetMetrics(t)
	config := validConfig()
	config.Selectors[0].Mode = ModePrefer
	controller := NewController(config, nil)
	controller.Filter(context.Background(),
		&fwksched.InferenceRequest{Headers: map[string]string{"x-disagg-revision": "v99"}},
		[]fwksched.Endpoint{endpoint("p1", revLabels("v1"))},
	)
	got := testutil.ToFloat64(filterOutcomeTotal.WithLabelValues("revision", string(ModePrefer), filterOutcomeNoMatchPreferFallback))
	if got != 1 {
		t.Fatalf("prefer_fallback: want 1, got %v", got)
	}
}

// --- Gating dropped -------------------------------------------------------

func TestMetric_GatingDropped_OncePerRevisionPerCall(t *testing.T) {
	resetMetrics(t)
	client := newSeededClient(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("p2", "v1", "prefill"),
		readyPod("p3", "v1", "prefill"),
		readyPod("p4", "v2", "prefill"),
		readyPod("d4", "v2", "decode"),
	)
	podCache, err := NewPodCache(client, testNS, "disaggregatedset.x-k8s.io/name=my-set", testRevLabel, testRoleLabel)
	if err != nil {
		t.Fatalf("NewPodCache: %v", err)
	}
	ctx, cancel := registerCtx(t)
	defer cancel()
	podCache.Start(ctx)
	if !podCache.WaitForCacheSync(ctx) {
		t.Fatalf("cache did not sync")
	}
	controller := NewController(validConfig(), podCache)
	filter := newGatingFilter(controller)
	pods := []fwksched.Endpoint{
		endpoint("p1", revLabels("v1")),
		endpoint("p2", revLabels("v1")),
		endpoint("p3", revLabels("v1")),
		endpoint("p4", revLabels("v2")),
	}

	filter.Filter(context.Background(), nil, pods)

	// Three v1 endpoints hit the gate in one call — counter should read 1,
	// not 3.
	if got := testutil.ToFloat64(gatingDroppedTotal.WithLabelValues("v1")); got != 1 {
		t.Fatalf("v1 dropped once per call: want 1, got %v", got)
	}
	if got := testutil.ToFloat64(gatingDroppedTotal.WithLabelValues("v2")); got != 0 {
		t.Fatalf("v2 satisfied gate: want 0, got %v", got)
	}
}

// --- Cache gauge -----------------------------------------------------------

func TestMetric_CacheReadyPods_TracksInformerState(t *testing.T) {
	resetMetrics(t)
	client := newSeededClient(t,
		readyPod("p1", "v1", "prefill"),
		readyPod("p2", "v1", "prefill"),
		readyPod("d1", "v1", "decode"),
	)
	podCache, err := NewPodCache(client, testNS, "disaggregatedset.x-k8s.io/name=my-set", testRevLabel, testRoleLabel)
	if err != nil {
		t.Fatalf("NewPodCache: %v", err)
	}
	ctx, cancel := registerCtx(t)
	defer cancel()
	podCache.Start(ctx)
	if !podCache.WaitForCacheSync(ctx) {
		t.Fatalf("cache did not sync")
	}

	if got := testutil.ToFloat64(cacheReadyPods.WithLabelValues("prefill", "v1")); got != 2 {
		t.Errorf("prefill v1: want 2, got %v", got)
	}
	if got := testutil.ToFloat64(cacheReadyPods.WithLabelValues("decode", "v1")); got != 1 {
		t.Errorf("decode v1: want 1, got %v", got)
	}
}
