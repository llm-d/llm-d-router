package groupsched

// O: THE DISPATCH-TIME OBSERVABILITY — the queue wait and the PREDICTED cache.
//
// Both are read-only: nothing in a decision reads either. The queue wait is dispatch − the head's
// arrival; the predicted cache is the engine's resident prefix for the turn's chains, read BEFORE
// the charge, because after the charge the charge's own blocks are resident and the prediction
// would be "everything" every time — a 100% hit rate that measures nothing. That is mutant M32,
// and TestPredictedCacheIsReadBeforeTheCharge is the test that catches it.

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// obsFixture is a scheduler with a registry and one measured engine, for the observability tests.
type obsFixture struct {
	s   *Scheduler
	reg *prometheus.Registry
}

func newObsFixture(t *testing.T) *obsFixture {
	t.Helper()
	f := newMetricFixture(t, "e0")
	return &obsFixture{s: f.s, reg: f.reg}
}

// seat seats a thread through the public path and brings it to its pod.
func (f *obsFixture) seat(t *testing.T, id string, blocks int, now time.Time) {
	t.Helper()
	if i := f.s.Pass([]Head{head(id, time.Second, blocks)}, now); i != 0 {
		t.Fatalf("%s must seat", id)
	}
	f.s.AtPod("r-"+id, "e0", now)
}

// observables reads one head's dispatch observables under the scheduler's lock (as the pass does).
func (f *obsFixture) observables(t *testing.T, h Head, now time.Time) dispatchObservables {
	t.Helper()
	f.s.mu.Lock()
	defer f.s.mu.Unlock()
	g := f.s.groups[f.s.bindings[h.Thread]]
	require.NotNil(t, g, "the head's thread must be seated")
	return f.s.observablesFor(g, h, now)
}

// TestPredictedCacheCountsTheResidentPrefix: a prompt whose first blocks the engine already holds
// predicts exactly those blocks' tokens; a cold prompt predicts nothing.
func TestPredictedCacheCountsTheResidentPrefix(t *testing.T) {
	f := newObsFixture(t)
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "warm", 400, now)
	f.s.Settle("r-warm", "", now)

	// a NEW thread whose prompt shares the warm thread's first two blocks: the prediction must see
	// them through the HEAD's own chain (the head is not seated, so its engine is the warm one's)
	f.s.mu.Lock()
	g := f.s.groups["e0"]
	f.s.mu.Unlock()
	h := headWithChain("next", time.Second, []Chain{{
		Keys:   append(keys(2, seedFor("warm")), keys(4, seedFor("next"))...),
		Tokens: 6 * 64,
	}})
	f.s.mu.Lock()
	obs := f.s.observablesFor(g, h, now)
	f.s.mu.Unlock()
	require.Equal(t, int64(2*64), obs.predictedCached,
		"only the resident prefix counts: two shared blocks of 64 tokens")
	require.Equal(t, int64(6*64), obs.promptTokens)
}

// TestPredictedCacheCapsAtThePromptTokens: a PARTIAL trailing block is not resident, so a prompt
// whose tokens do not fill its last block cannot predict more tokens than it has.
func TestPredictedCacheCapsAtThePromptTokens(t *testing.T) {
	f := newObsFixture(t)
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "warm", 400, now)
	f.s.Settle("r-warm", "", now)

	h := headWithChain("short", time.Second, []Chain{{Keys: keys(3, seedFor("warm")), Tokens: 150}})
	f.s.mu.Lock()
	obs := f.s.observablesFor(f.s.groups["e0"], h, now)
	f.s.mu.Unlock()
	require.Equal(t, int64(150), obs.predictedCached,
		"the prediction is bounded by the prompt's own tokens: a partial block is not resident")
}

// TestPredictedCacheSumsOverPrompts: a multi-prompt request (a corpus sample that carries several
// prompts) sums the prediction and the prompt tokens over every prompt, not just the first.
func TestPredictedCacheSumsOverPrompts(t *testing.T) {
	f := newObsFixture(t)
	now := time.Unix(1793000000, 0).UTC()
	f.seat(t, "warm", 400, now)
	f.s.Settle("r-warm", "", now)

	h := headWithChain("multi", time.Second, []Chain{
		{Keys: keys(2, seedFor("warm")), Tokens: 2 * 64},  // fully resident
		{Keys: keys(3, seedFor("other")), Tokens: 3 * 64}, // cold
		{Keys: keys(1, seedFor("third")), Tokens: 40},     // cold, partial
	})
	f.s.mu.Lock()
	obs := f.s.observablesFor(f.s.groups["e0"], h, now)
	f.s.mu.Unlock()
	require.Equal(t, int64(2*64), obs.predictedCached, "one resident prompt, two cold ones")
	require.Equal(t, int64(2*64+3*64+40), obs.promptTokens, "every prompt's tokens count")
}

// TestPredictedCacheOnColdSeatEntry: a newcomer whose prompt shares nothing predicts zero — the
// seat's own reservation is not cache.
func TestPredictedCacheOnColdSeatEntry(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	if i := f.s.Pass([]Head{obsHead("cold", now.Add(-time.Second), 2)}, now); i != 0 {
		t.Fatal("the head must dispatch")
	}
	require.InDelta(t, 0.0, gatherOrZero(t, f.reg, subsystem+"_predicted_cached_tokens_total",
		map[string]string{"engine": "e0"}), 0.001,
		"nothing was resident before the seat: the reservation it just made is not a cache hit")
	require.InDelta(t, 128.0, gatherOrZero(t, f.reg, subsystem+"_dispatched_prompt_tokens_total",
		map[string]string{"engine": "e0"}), 0.001)
}

// TestPredictedCacheOnSeatEntryCountsTheSharedPrefix: a newcomer that shares a prefix with an
// already seated thread predicts exactly that prefix — not its whole prompt.
func TestPredictedCacheOnSeatEntryCountsTheSharedPrefix(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	if i := f.s.Pass([]Head{obsHead("warm", now.Add(-time.Second), 4)}, now); i != 0 {
		t.Fatal("the warm thread must seat")
	}

	// a newcomer whose first two blocks are the warm thread's: 2 of its 4 blocks are resident
	shared := append(keys(2, seedFor("warm")), keys(2, seedFor("newcomer"))...)
	h := headWithChain("newcomer", time.Second, []Chain{{Keys: shared, Tokens: 4 * 64}})
	h.Arrival = now.Add(-time.Second)
	cached := 0.0
	{
		f.s.mu.Lock()
		cached = float64(f.s.groups["e0"].index.CachedTokens(h.Chains))
		f.s.mu.Unlock()
	}
	if i := f.s.Pass([]Head{h}, now); i != 0 {
		t.Fatal("the newcomer must seat")
	}
	require.InDelta(t, 2.0*64, cached, 0.001, "the shared prefix is two blocks")
	require.InDelta(t, cached, gatherOrZero(t, f.reg, subsystem+"_predicted_cached_tokens_total",
		map[string]string{"engine": "e0"}), 0.001,
		"the seat-entry prediction is the shared prefix that was resident BEFORE the seat reserved the rest")
}

// TestPredictedCacheOnAReenteringReturner: a thread that exited and comes back predicts what the
// OTHER seated threads hold — its own old blocks were released when it left, and its new seat's
// reservation is not cache either.
func TestPredictedCacheOnAReenteringReturner(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()

	// one thread keeps its blocks: its prefix is what the returning thread can still hit
	if i := f.s.Pass([]Head{obsHead("keeper", now.Add(-time.Second), 2)}, now); i != 0 {
		t.Fatal("the keeper must seat")
	}
	// a second thread seats and then leaves (silence past the idle window)
	if i := f.s.Pass([]Head{obsHead("gone", now.Add(-time.Second), 3)}, now); i != 0 {
		t.Fatal("the leaving thread must seat")
	}
	f.s.Settle("r-gone", "", now)
	later := now.Add(2 * time.Minute)
	// the thread leaves: the scheduler's own exit path, which releases its blocks
	f.s.mu.Lock()
	f.s.exit(f.s.groups["e0"], f.s.groups["e0"].threads["gone"], "test-exit", later)
	seated := f.s.bindings["gone"] != ""
	f.s.mu.Unlock()
	require.False(t, seated, "the silenced thread must have exited, releasing its blocks")

	// it comes back with a chain that starts with the KEEPER's blocks
	back := append(keys(2, seedFor("keeper")), keys(3, seedFor("gone"))...)
	before := gatherOrZero(t, f.reg, subsystem+"_predicted_cached_tokens_total", map[string]string{"engine": "e0"})
	h := headWithChain("gone", time.Second, []Chain{{Keys: back, Tokens: 5 * 64}})
	h.Arrival = later.Add(-time.Second)
	h.ReqID = "r-gone-2"
	h.Returner = true
	if i := f.s.Pass([]Head{h}, later); i != 0 {
		t.Fatal("the returning thread must re-seat")
	}
	after := gatherOrZero(t, f.reg, subsystem+"_predicted_cached_tokens_total", map[string]string{"engine": "e0"})
	require.InDelta(t, 2.0*64, after-before, 0.001,
		"the returning thread predicts the keeper's two blocks: its own were released on exit, and its new reservation is not cache")
}

// TestPredictedCacheIsReadBeforeTheCharge is M32's test. A SEATED returner's turn is the case that
// can tell the two moments apart: its seat holds the PREVIOUS chain, so before the charge the
// prediction is the old prefix, and after it the new tail would be resident too and the prediction
// would be the whole prompt — a 100% hit rate that measures nothing.
func TestPredictedCacheIsReadBeforeTheCharge(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()

	// turn one: 2 blocks, seated (its seat-entry prediction is 0: nothing was resident)
	if i := f.s.Pass([]Head{obsHead("m", now.Add(-time.Second), 2)}, now); i != 0 {
		t.Fatal("the first turn must dispatch")
	}
	// turn two GROWS the chain to 4 blocks: 2 resident (the seat), 2 new
	grown := obsHead("m", now.Add(-time.Second), 4)
	if i := f.s.Pass([]Head{grown}, now); i != 0 {
		t.Fatal("the second turn must dispatch")
	}

	// total predicted = 0 (seat entry) + 128 (turn two's resident 2 blocks)
	require.InDelta(t, 128.0, gatherOrZero(t, f.reg, subsystem+"_predicted_cached_tokens_total",
		map[string]string{"engine": "e0"}), 0.001,
		"the prediction is read BEFORE this turn's charge — after it, its own tail would be resident too")
	require.InDelta(t, 2.0*64+4.0*64, gatherOrZero(t, f.reg, subsystem+"_dispatched_prompt_tokens_total",
		map[string]string{"engine": "e0"}), 0.001, "both turns' prompt tokens are the denominator")
}

// TestQueueWaitIsMeasuredPerDispatchedTurn: the histogram sees every dispatched turn, by kind.
func TestQueueWaitIsMeasuredPerDispatchedTurn(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()

	if i := f.s.Pass([]Head{obsHead("m", now.Add(-time.Second), 100)}, now); i != 0 {
		t.Fatal("the first turn must dispatch")
	}
	// a SECOND turn for the same thread, waiting 3 s in the queue: kind returner
	second := obsHead("m", now.Add(-3*time.Second), 100)
	second.ReqID = "r-m-2"
	second.Returner = true
	if i := f.s.Pass([]Head{second}, now); i != 0 {
		t.Fatal("the second turn must dispatch")
	}

	require.Equal(t, uint64(1), histCount(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "new"}))
	require.Equal(t, uint64(1), histCount(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "returner"}))
	require.InDelta(t, 3.0, histSum(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "returner"}), 0.001,
		"the returner waited its three seconds")
	require.InDelta(t, 1.0, histSum(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "new"}), 0.001)
}

// TestQueueWaitCountsThePassThroughAsUntracked: a head the pass served without seating is still a
// request that waited, and it is labelled untracked (it has no engine and no thread seat).
func TestQueueWaitCountsThePassThroughAsUntracked(t *testing.T) {
	f := newMetricFixture(t, "e0")
	now := time.Unix(1793000000, 0).UTC()
	// no Chains: the pass-through serves it
	if i := f.s.Pass([]Head{{Thread: "n", ReqID: "r-n", Arrival: now.Add(-5 * time.Second)}}, now); i != 0 {
		t.Fatal("the tokenless head must be served")
	}
	require.Equal(t, uint64(1), histCount(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "untracked"}))
	require.InDelta(t, 5.0, histSum(t, f.reg, "queue_wait_seconds", map[string]string{"kind": "untracked"}), 0.001)
}

// TestPreRequestPutsTheObservablesOnTheSpan: the span carries the four attributes when — and only
// when — the ctx holds a RECORDING span, and it never panics without one.
func TestPreRequestPutsTheObservablesOnTheSpan(t *testing.T) {
	build := func(t *testing.T) (*obsFixture, time.Time, Head) {
		t.Helper()
		f := newObsFixture(t)
		now := time.Unix(1793000000, 0).UTC()
		if i := f.s.Pass([]Head{obsHead("span", now.Add(-time.Second), 3)}, now); i != 0 {
			t.Fatal("the thread must seat")
		}
		second := obsHead("span", now.Add(-2*time.Second), 3)
		second.ReqID = "r-span-2"
		if i := f.s.Pass([]Head{second}, now); i != 0 {
			t.Fatal("the second turn must dispatch")
		}
		return f, now, second
	}
	target := func() *fwksched.SchedulingResult {
		return &fwksched.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{schedulerEndpoint("e0")}},
			},
		}
	}

	t.Run("a recording span gets the attributes", func(t *testing.T) {
		f, _, second := build(t)
		recorder := tracetest.NewSpanRecorder()
		tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
		ctx, span := tp.Tracer("test").Start(context.Background(), "request_orchestration")

		pluginPreRequest(t, f.s, ctx, inferenceRequestFor(second), target())
		span.End()

		got := map[attribute.Key]attribute.Value{}
		for _, kv := range recorder.Ended()[0].Attributes() {
			got[kv.Key] = kv.Value
		}
		require.Contains(t, got, attribute.Key(spanQueueWaitMs))
		require.InDelta(t, 2000.0, got[attribute.Key(spanQueueWaitMs)].AsFloat64(), 200.0,
			"the span's queue wait is the returner's ~2 s")
		require.Equal(t, int64(3*64), got[attribute.Key(spanPromptTokens)].AsInt64())
		require.Contains(t, got[attribute.Key(spanSeatEngine)].AsString(), "e0", "the span names the seat engine")
	})

	t.Run("tracing off writes nothing and does not panic", func(t *testing.T) {
		f, _, second := build(t)
		pluginPreRequest(t, f.s, context.Background(), inferenceRequestFor(second), target())
	})
}

// pluginPreRequest drives the hook the framework calls: the scheduler's AtPod plus the span
// annotation, exactly as GroupScheduler.PreRequest does.
func pluginPreRequest(t *testing.T, s *Scheduler, ctx context.Context, req *fwksched.InferenceRequest,
	result *fwksched.SchedulingResult) {
	t.Helper()
	engine := ""
	if ep := pickedEndpoint(result); ep != nil {
		engine = endpointKey(ep.GetMetadata())
	}
	obs := s.AtPod(req.RequestID, engine, s.now)
	annotateSpan(ctx, engine, obs)
}

// obsHead is head() with the arrival on the fixture's CLOCK: the queue wait is now − arrival, so a
// fixture whose heads carry time.Now() measures the distance to 2026 instead of its own wait.
func obsHead(id string, arrivedAt time.Time, blocks int) Head {
	h := head(id, 0, blocks)
	h.Arrival = arrivedAt
	return h
}

// inferenceRequestFor builds the framework request a PreRequest hook receives for one head: the
// only field these tests need is the request id the record is keyed by.
func inferenceRequestFor(h Head) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: h.ReqID}
}

// gatherOrZero reads one sample, 0 when the series does not exist.
func gatherOrZero(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	v, _ := gatherValue(t, reg, name, labels)
	return v
}

// histCount is the sample count of one histogram series.
func histCount(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) uint64 {
	t.Helper()
	return histOf(t, reg, name, labels).GetSampleCount()
}

// histSum is the sample sum of one histogram series.
func histSum(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) float64 {
	t.Helper()
	return histOf(t, reg, name, labels).GetSampleSum()
}

// histOf finds one histogram series by its labels.
func histOf(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) *dto.Histogram {
	t.Helper()
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, mf := range mfs {
		if mf.GetName() != subsystem+"_"+name {
			continue
		}
		for _, m := range mf.GetMetric() {
			match := true
			for _, lp := range m.GetLabel() {
				if want, ok := labels[lp.GetName()]; ok && want != lp.GetValue() {
					match = false
				}
			}
			if match && m.GetHistogram() != nil {
				return m.GetHistogram()
			}
		}
	}
	t.Fatalf("no histogram series %s%v", name, labels)
	return nil
}
