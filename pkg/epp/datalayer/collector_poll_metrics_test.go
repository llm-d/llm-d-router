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

package datalayer

import (
	"context"
	"errors"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer/mocks"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	pollsTotalName   = "llm_d_epp_datalayer_polls_total"
	pollDurationName = "llm_d_epp_datalayer_poll_duration_seconds"

	// pollDeadlineWait bounds waits on a poll that runs until the collection
	// timeout expires.
	pollDeadlineWait = defaultCollectionTimeout + 2*time.Second
	// metricSettleWait bounds waits for a metric write that follows a Dispatch
	// return the test has already observed.
	metricSettleWait = time.Second
)

var pollOutcomes = []string{metrics.DataLayerPollOutcomeSuccess, metrics.DataLayerPollOutcomeError, metrics.DataLayerPollOutcomeTimeout}

// pollSource is a PollingDispatcher whose Dispatch counts the call and then
// delegates to poll. Type and name are set per test so metric labels and
// PollStatus keys can be asserted independently.
type pollSource struct {
	sourceType, name string
	poll             func(ctx context.Context) error
	CallCount        int64
}

func (s *pollSource) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: s.sourceType, Name: s.name}
}
func (s *pollSource) Dispatch(ctx context.Context, _ fwkdl.Endpoint) error {
	atomic.AddInt64(&s.CallCount, 1)
	return s.poll(ctx)
}
func (s *pollSource) AppendExtractor(_ fwkplugin.Plugin) error { return nil }
func (s *pollSource) Interval() time.Duration                  { return 0 }

func succeedingPoll(_ context.Context) error { return nil }

// pollAfterDeadline returns a poll that waits for its context to end and then
// returns result; with useCtxErr it returns the context's error instead.
func pollAfterDeadline(result error, useCtxErr bool) func(ctx context.Context) error {
	return func(ctx context.Context) error {
		<-ctx.Done()
		if useCtxErr {
			return ctx.Err()
		}
		return result
	}
}

// resetPollMetrics registers the EPP metrics once and clears them before and
// after the test so poll series start from zero.
func resetPollMetrics(t *testing.T) {
	t.Helper()
	metrics.Register()
	metrics.Reset()
	t.Cleanup(metrics.Reset)
}

// gatherFamily returns the named metric family from the registry, or nil when
// the registry exposes no series under that name.
func gatherFamily(t *testing.T, name string) *dto.MetricFamily {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() == name {
			return family
		}
	}
	return nil
}

// familySeries returns the series of the named family whose source_type label
// matches sourceType.
func familySeries(t *testing.T, name, sourceType string) []*dto.Metric {
	t.Helper()
	family := gatherFamily(t, name)
	if family == nil {
		return nil
	}
	var out []*dto.Metric
	for _, m := range family.GetMetric() {
		for _, l := range m.GetLabel() {
			if l.GetName() == "source_type" && l.GetValue() == sourceType {
				out = append(out, m)
			}
		}
	}
	return out
}

// pollsTotal returns the polls counter for a source type and outcome, or 0
// when no such series exists.
func pollsTotal(t *testing.T, sourceType, outcome string) float64 {
	t.Helper()
	for _, m := range familySeries(t, pollsTotalName, sourceType) {
		for _, l := range m.GetLabel() {
			if l.GetName() == "outcome" && l.GetValue() == outcome {
				return m.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// pollDuration returns the duration histogram for a source type, or nil when
// no such series exists.
func pollDuration(t *testing.T, sourceType string) *dto.Histogram {
	t.Helper()
	series := familySeries(t, pollDurationName, sourceType)
	if len(series) == 0 {
		return nil
	}
	require.Len(t, series, 1, "one duration series per source type")
	return series[0].GetHistogram()
}

func pollDurationCount(t *testing.T, sourceType string) uint64 {
	t.Helper()
	return pollDuration(t, sourceType).GetSampleCount()
}

func labelNames(m *dto.Metric) []string {
	names := make([]string, 0, len(m.GetLabel()))
	for _, l := range m.GetLabel() {
		names = append(names, l.GetName())
	}
	sort.Strings(names)
	return names
}

func readPollStatusFor(ep fwkdl.Endpoint, name string) (*fwkdl.PollStatus, bool) {
	return fwkdl.ReadAttribute[*fwkdl.PollStatus](ep.GetAttributes(), fwkdl.PollStatusDataKey(name))
}

// pollsRecorded reports whether n polls of the source type are visible in both
// the outcome counter and the duration histogram. The collector writes these
// separately, so a test must not treat one of them as proof of the other.
func pollsRecorded(t *testing.T, sourceType, outcome string, n uint64) bool {
	t.Helper()
	return pollsTotal(t, sourceType, outcome) == float64(n) && pollDurationCount(t, sourceType) == n
}

// waitForPolls blocks until n polls of the source type are recorded in both
// families, then stops and joins the collector so no write is still in flight
// when the caller asserts.
func waitForPolls(t *testing.T, c *Collector, sourceType, outcome string, n uint64, wait time.Duration) {
	t.Helper()
	require.Eventually(t, func() bool {
		return pollsRecorded(t, sourceType, outcome, n)
	}, wait, 2*time.Millisecond, "expected %d %s polls in counter and histogram", n, outcome)
	c.Stop()
}

// TestCollectorPollMetricsOutcomes confirms each poll is counted once under
// one outcome and observed once in the duration histogram, and that the
// existing poll error counter and PollStatus keep their contract.
func TestCollectorPollMetricsOutcomes(t *testing.T) {
	pollErr := errors.New("poll boom")

	tests := []struct {
		name           string
		poll           func(ctx context.Context) error
		wait           time.Duration
		wantOutcome    string
		wantPollErrors float64
		wantFailures   int
		wantMinSeconds float64
	}{
		{
			name:        "nil return is success",
			poll:        succeedingPoll,
			wait:        metricSettleWait,
			wantOutcome: metrics.DataLayerPollOutcomeSuccess,
		},
		{
			name:           "returned error is error",
			poll:           func(context.Context) error { return pollErr },
			wait:           metricSettleWait,
			wantOutcome:    metrics.DataLayerPollOutcomeError,
			wantPollErrors: 1,
			wantFailures:   1,
		},
		{
			name:           "context.Canceled with a live parent is error",
			poll:           func(context.Context) error { return context.Canceled },
			wait:           metricSettleWait,
			wantOutcome:    metrics.DataLayerPollOutcomeError,
			wantPollErrors: 1,
			wantFailures:   1,
		},
		{
			name:           "context error after deadline is timeout",
			poll:           pollAfterDeadline(nil, true),
			wait:           pollDeadlineWait,
			wantOutcome:    metrics.DataLayerPollOutcomeTimeout,
			wantPollErrors: 1,
			wantFailures:   1,
			wantMinSeconds: defaultCollectionTimeout.Seconds(),
		},
		{
			name:           "nil return after deadline is timeout and not a poll error",
			poll:           pollAfterDeadline(nil, false),
			wait:           pollDeadlineWait,
			wantOutcome:    metrics.DataLayerPollOutcomeTimeout,
			wantMinSeconds: defaultCollectionTimeout.Seconds(),
		},
		{
			name:           "other error after deadline is timeout",
			poll:           pollAfterDeadline(pollErr, false),
			wait:           pollDeadlineWait,
			wantOutcome:    metrics.DataLayerPollOutcomeTimeout,
			wantPollErrors: 1,
			wantFailures:   1,
			wantMinSeconds: defaultCollectionTimeout.Seconds(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetPollMetrics(t)
			src := &pollSource{sourceType: "outcome-type", name: "outcome", poll: tt.poll}
			ep := defaultEndpoint()
			c := NewCollector()
			ticker := mocks.NewTicker()
			require.NoError(t, c.Start(context.Background(), ticker, ep, []fwkdl.PollingDispatcher{src}))
			defer c.Stop()

			ticker.Tick()
			// Wait for every write the poll produces, then stop and join the
			// collector before asserting. CallCount advances when Dispatch is
			// entered, so the wait must itself span the poll deadline.
			statusSettled := func() bool {
				status, ok := readPollStatusFor(ep, src.name)
				if tt.wantFailures == 0 {
					return !ok
				}
				return ok && status.ConsecutiveFailures == tt.wantFailures
			}
			require.Eventually(t, func() bool {
				return pollsRecorded(t, src.sourceType, tt.wantOutcome, 1) &&
					testutil.ToFloat64(metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(src.sourceType)) == tt.wantPollErrors &&
					statusSettled()
			}, tt.wait, 2*time.Millisecond, "expected one %s poll with its error counter and PollStatus settled", tt.wantOutcome)
			c.Stop()
			assert.Equal(t, int64(1), atomic.LoadInt64(&src.CallCount))

			for _, outcome := range pollOutcomes {
				if outcome != tt.wantOutcome {
					assert.Zero(t, pollsTotal(t, src.sourceType, outcome), "unexpected %s poll", outcome)
				}
			}
			hist := pollDuration(t, src.sourceType)
			require.NotNil(t, hist, "expected a duration observation")
			assert.Equal(t, uint64(1), hist.GetSampleCount())
			// The deadline cases return only after the context ends, so the
			// observed duration covers the whole wait. A small tolerance absorbs
			// the gap between creating the context and starting the clock.
			assert.GreaterOrEqual(t, hist.GetSampleSum(), tt.wantMinSeconds*0.9)

			assert.Equal(t, tt.wantPollErrors, testutil.ToFloat64(metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(src.sourceType)))
			status, ok := readPollStatusFor(ep, src.name)
			if tt.wantFailures == 0 {
				assert.False(t, ok, "PollStatus must stay absent without a returned error")
			} else {
				require.True(t, ok)
				assert.Equal(t, tt.wantFailures, status.ConsecutiveFailures)
			}
		})
	}
}

// assertNoPollRecorded confirms no poll was counted or timed for the source
// type, the existing poll error counter is untouched and no PollStatus was
// written for the source name.
func assertNoPollRecorded(t *testing.T, ep fwkdl.Endpoint, srcType, srcName string) {
	t.Helper()
	for _, outcome := range pollOutcomes {
		assert.Zero(t, pollsTotal(t, srcType, outcome), "unexpected %s poll", outcome)
	}
	// Start pre-creates the histogram series; what must not exist is an observation.
	assert.Zero(t, pollDuration(t, srcType).GetSampleCount(), "unexpected duration observation")
	assert.Zero(t, testutil.ToFloat64(metrics.LlmdDataLayerPollErrorsTotal.WithLabelValues(srcType)))
	_, ok := readPollStatusFor(ep, srcName)
	assert.False(t, ok, "unexpected PollStatus")
}

// TestCollectorPollMetricsStopDuringPoll confirms Stop cancels an in-flight
// poll, stays blocked until the dispatcher returns, and records nothing for
// it. The dispatcher reports entry and the cancellation it observes, and does
// not return until the test releases it.
func TestCollectorPollMetricsStopDuringPoll(t *testing.T) {
	resetPollMetrics(t)
	entered := make(chan struct{})
	cancelObserved := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseDispatch := func() { releaseOnce.Do(func() { close(release) }) }

	var gotErr error
	src := &pollSource{sourceType: "stopped-type", name: "stopped", poll: func(ctx context.Context) error {
		close(entered)
		<-ctx.Done()
		gotErr = ctx.Err()
		close(cancelObserved)
		<-release
		return gotErr
	}}
	ep := defaultEndpoint()
	c := NewCollector()
	// Failure paths must free the dispatcher before joining, or Stop would
	// block forever.
	defer func() {
		releaseDispatch()
		c.Stop()
	}()
	ticker := mocks.NewTicker()
	require.NoError(t, c.Start(context.Background(), ticker, ep, []fwkdl.PollingDispatcher{src}))

	ticker.Tick()
	select {
	case <-entered:
	case <-time.After(metricSettleWait):
		t.Fatal("poll did not start")
	}

	stopDone := make(chan struct{})
	go func() {
		c.Stop()
		close(stopDone)
	}()

	// Stop cancels the poll context.
	select {
	case <-cancelObserved:
	case <-time.After(metricSettleWait):
		t.Fatal("Stop did not cancel the in-flight poll")
	}
	// Stop keeps blocking while the dispatcher has not returned.
	select {
	case <-stopDone:
		t.Fatal("Stop returned while the dispatcher was still running")
	case <-time.After(50 * time.Millisecond):
	}

	releaseDispatch()
	select {
	case <-stopDone:
	case <-time.After(metricSettleWait):
		t.Fatal("Stop did not return after the dispatcher was released")
	}

	// stopDone is closed after Stop joined the collection goroutine, which
	// returned after the dispatcher wrote gotErr.
	assert.ErrorIs(t, gotErr, context.Canceled)
	assert.Equal(t, int64(1), atomic.LoadInt64(&src.CallCount))
	assertNoPollRecorded(t, ep, src.sourceType, src.name)
}

// TestCollectorPollMetricsCancelledParent confirms a collector started with
// a cancelled context refuses to start, never dispatches and records nothing.
func TestCollectorPollMetricsCancelledParent(t *testing.T) {
	resetPollMetrics(t)
	src := &pollSource{sourceType: "cancelled-type", name: "cancelled", poll: succeedingPoll}
	ep := defaultEndpoint()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := NewCollector()
	ticker := mocks.NewTicker()
	require.ErrorIs(t, c.Start(ctx, ticker, ep, []fwkdl.PollingDispatcher{src}), context.Canceled)
	ticker.Tick()
	c.Stop()

	assert.Zero(t, atomic.LoadInt64(&src.CallCount), "no dispatch after a refused Start")
	assertNoPollRecorded(t, ep, src.sourceType, src.name)
}

// TestCollectorPollMetricsIntervalDispatcher confirms the ticks an interval
// source skips record nothing and each due tick records exactly one poll.
func TestCollectorPollMetricsIntervalDispatcher(t *testing.T) {
	resetPollMetrics(t)
	src := &pollSource{sourceType: "interval-type", name: "interval", poll: succeedingPoll}
	c := NewCollector()
	ticker := mocks.NewTicker()
	disp := newIntervalDispatcher(src, 3)
	require.IsType(t, &intervalDispatcher{}, disp)
	require.NoError(t, c.Start(context.Background(), ticker, defaultEndpoint(), []fwkdl.PollingDispatcher{disp}))
	defer c.Stop()

	// Ticks 1, 4 and 7 poll; the other four are skipped.
	for range 7 {
		ticker.Tick()
	}
	require.Eventually(t, func() bool {
		return len(ticker.Channel()) == 0
	}, metricSettleWait, 2*time.Millisecond, "expected all ticks consumed")
	waitForPolls(t, c, src.sourceType, metrics.DataLayerPollOutcomeSuccess, 3, metricSettleWait)

	assert.Equal(t, int64(3), atomic.LoadInt64(&src.CallCount), "due must advance once per tick")
	for _, outcome := range []string{metrics.DataLayerPollOutcomeError, metrics.DataLayerPollOutcomeTimeout} {
		assert.Zero(t, pollsTotal(t, src.sourceType, outcome), "unexpected %s poll", outcome)
	}
}

// TestCollectorPollMetricsPerDispatcher confirms an unwrapped dispatcher and
// an interval dispatcher on one collector are attributed to their own source
// types at their own cadences.
func TestCollectorPollMetricsPerDispatcher(t *testing.T) {
	resetPollMetrics(t)
	every := &pollSource{sourceType: "every-tick-type", name: "every", poll: succeedingPoll}
	third := &pollSource{sourceType: "third-tick-type", name: "third", poll: succeedingPoll}
	everyDisp := newIntervalDispatcher(every, 1)
	require.Same(t, every, everyDisp, "period 1 must stay unwrapped")
	c := NewCollector()
	ticker := mocks.NewTicker()
	require.NoError(t, c.Start(context.Background(), ticker, defaultEndpoint(),
		[]fwkdl.PollingDispatcher{everyDisp, newIntervalDispatcher(third, 3)}))
	defer c.Stop()

	for range 6 {
		ticker.Tick()
	}
	require.Eventually(t, func() bool {
		return pollsRecorded(t, every.sourceType, metrics.DataLayerPollOutcomeSuccess, 6) &&
			pollsRecorded(t, third.sourceType, metrics.DataLayerPollOutcomeSuccess, 2)
	}, metricSettleWait, 2*time.Millisecond, "expected per-type counts in counter and histogram")
	c.Stop()

	assert.Equal(t, int64(6), atomic.LoadInt64(&every.CallCount))
	assert.Equal(t, int64(2), atomic.LoadInt64(&third.CallCount))
	assert.Equal(t, uint64(6), pollDurationCount(t, every.sourceType))
	assert.Equal(t, uint64(2), pollDurationCount(t, third.sourceType))
}

// TestCollectorPollMetricsFirstFailureVisible confirms Start exposes all three
// outcome series at zero before any poll, so a first failure after a healthy
// run is a 0 -> 1 increment on an existing series rather than a new series.
func TestCollectorPollMetricsFirstFailureVisible(t *testing.T) {
	resetPollMetrics(t)
	const srcType = "first-failure-type"
	var fail atomic.Bool
	pollErr := errors.New("poll boom")
	src := &pollSource{sourceType: srcType, name: "flaky", poll: func(context.Context) error {
		if fail.Load() {
			return pollErr
		}
		return nil
	}}
	c := NewCollector()
	ticker := mocks.NewTicker()
	require.NoError(t, c.Start(context.Background(), ticker, defaultEndpoint(), []fwkdl.PollingDispatcher{src}))
	defer c.Stop()

	// Scrape 1: nothing polled yet, every series present at zero.
	series := familySeries(t, pollsTotalName, srcType)
	require.Len(t, series, len(pollOutcomes), "all outcome series exist before the first poll")
	for _, outcome := range pollOutcomes {
		assert.Zero(t, pollsTotal(t, srcType, outcome))
	}
	assert.Zero(t, pollDuration(t, srcType).GetSampleCount())

	// Scrape 2: healthy polls move only the success series.
	for range 3 {
		ticker.Tick()
	}
	require.Eventually(t, func() bool {
		return pollsRecorded(t, srcType, metrics.DataLayerPollOutcomeSuccess, 3)
	}, metricSettleWait, 2*time.Millisecond)
	assert.Zero(t, pollsTotal(t, srcType, metrics.DataLayerPollOutcomeError))
	assert.Zero(t, pollsTotal(t, srcType, metrics.DataLayerPollOutcomeTimeout))

	// Scrape 3: the first failure is an increment of the existing error series.
	fail.Store(true)
	ticker.Tick()
	require.Eventually(t, func() bool {
		return pollsTotal(t, srcType, metrics.DataLayerPollOutcomeError) == 1
	}, metricSettleWait, 2*time.Millisecond)
	c.Stop()
	assert.Len(t, familySeries(t, pollsTotalName, srcType), len(pollOutcomes), "no series created by the failure")
	assert.Equal(t, uint64(4), pollDurationCount(t, srcType))
}

// TestCollectorPollMetricsBoundedLabels confirms dispatchers with different
// names on different endpoints share one series per source type, and that the
// series carry only the source_type and outcome labels.
func TestCollectorPollMetricsBoundedLabels(t *testing.T) {
	resetPollMetrics(t)
	const srcType = "shared-type"
	other := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "other-pod", Namespace: "default"},
		Address: "5.6.7.8:5678",
	}, nil)

	endpoints := []fwkdl.Endpoint{defaultEndpoint(), other}
	srcs := make([]*pollSource, 0, 2*len(endpoints))
	collectors := make([]*Collector, 0, len(endpoints))
	for _, ep := range endpoints {
		a := &pollSource{sourceType: srcType, name: "shared-a-" + ep.GetMetadata().ID.Name, poll: succeedingPoll}
		b := &pollSource{sourceType: srcType, name: "shared-b-" + ep.GetMetadata().ID.Name, poll: succeedingPoll}
		srcs = append(srcs, a, b)
		c := NewCollector()
		collectors = append(collectors, c)
		ticker := mocks.NewTicker()
		require.NoError(t, c.Start(context.Background(), ticker, ep, []fwkdl.PollingDispatcher{a, b}))
		defer c.Stop()
		ticker.Tick()
		ticker.Tick()
	}
	// 2 collectors x 2 dispatchers x 2 ticks, all under one source type.
	require.Eventually(t, func() bool {
		return pollsRecorded(t, srcType, metrics.DataLayerPollOutcomeSuccess, 8)
	}, metricSettleWait, 2*time.Millisecond, "expected all polls under one series in counter and histogram")
	for _, c := range collectors {
		c.Stop()
	}
	for _, src := range srcs {
		assert.Equal(t, int64(2), atomic.LoadInt64(&src.CallCount), "dispatcher %s", src.name)
	}

	// One series per outcome for the shared type (all pre-created by Start), no
	// matter how many endpoints or dispatcher names feed it.
	counters := familySeries(t, pollsTotalName, srcType)
	require.Len(t, counters, len(pollOutcomes), "one counter series per source type and outcome")
	for _, counter := range counters {
		assert.Equal(t, []string{"outcome", "source_type"}, labelNames(counter))
	}
	assert.Equal(t, float64(8), pollsTotal(t, srcType, metrics.DataLayerPollOutcomeSuccess))

	durations := familySeries(t, pollDurationName, srcType)
	require.Len(t, durations, 1, "one duration series per source type")
	assert.Equal(t, []string{"source_type"}, labelNames(durations[0]))
	assert.Equal(t, uint64(8), durations[0].GetHistogram().GetSampleCount())
}
