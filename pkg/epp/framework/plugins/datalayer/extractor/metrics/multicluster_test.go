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

package metrics

import (
	"context"
	"net/url"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// TestMultiClusterMetricsPipeline is the end-to-end path: a peer cluster's pool
// aggregates, served over httptest, flow through the multicluster source and
// extractor into the pool attributes the multicluster scorers read. Turning those
// attributes into scores is covered by the scorers' own tests.
func TestMultiClusterMetricsPipeline(t *testing.T) {
	srv := createMockServer([]MetricMock{
		{Name: defaultKVCacheUtilizationMetric, Value: 0.62},
		{Name: defaultQueueSizeMetric, Value: 4},
	})
	defer srv.Close()

	u, err := url.Parse(srv.URL)
	require.NoError(t, err)
	source, err := sourcemetrics.NewHTTPMultiClusterMetricsDataSource(u.Scheme, u.Path, "mc-source")
	require.NoError(t, err)
	ext, err := NewMultiClusterMetricsExtractor("mc-extractor", nil)
	require.NoError(t, err)
	require.NoError(t, source.AppendExtractor(ext))

	ep := newEndpointAt(mustHost(t, srv.URL), nil)
	require.NoError(t, source.Dispatch(context.Background(), ep))

	kv, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterKVCacheUtilizationDataKey)
	require.True(t, ok, "pool KV-cache utilization attribute")
	require.InDelta(t, 0.62, float64(kv), 1e-9)
	q, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterQueueSizeDataKey)
	require.True(t, ok, "pool queue size attribute")
	require.InDelta(t, 4, float64(q), 1e-9)
}

func gauge(v float64) *dto.MetricFamily {
	return &dto.MetricFamily{
		Type:   dto.MetricType_GAUGE.Enum(),
		Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(v)}}},
	}
}

func TestMultiClusterMetricsExtractor(t *testing.T) {
	tests := []struct {
		name    string
		data    sourcemetrics.PrometheusMetricMap
		wantErr bool
		wantKV  *float64 // nil = attribute expected absent
		wantQ   *float64
		// wantStamp: the scrape time is written when any aggregate was read.
		wantStamp *float64
	}{
		{
			name: "both aggregates present",
			data: sourcemetrics.PrometheusMetricMap{
				defaultKVCacheUtilizationMetric: gauge(0.42),
				defaultQueueSizeMetric:          gauge(7),
			},
			wantKV:    ptr.To(0.42),
			wantQ:     ptr.To(7.0),
			wantStamp: ptr.To(1_800_000_000.0),
		},
		{
			name: "missing queue still writes kv, the stamp, and errors",
			data: sourcemetrics.PrometheusMetricMap{
				defaultKVCacheUtilizationMetric: gauge(0.9),
			},
			wantErr:   true,
			wantKV:    ptr.To(0.9),
			wantStamp: ptr.To(1_800_000_000.0),
		},
		{
			name:    "empty map errors and writes nothing",
			data:    sourcemetrics.PrometheusMetricMap{},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ext, err := NewMultiClusterMetricsExtractor("multicluster-metrics", nil)
			require.NoError(t, err)
			require.Equal(t, MultiClusterMetricsExtractorType, ext.TypedName().Type)
			ext.now = func() time.Time { return time.Unix(1_800_000_000, 0) }

			ep := fwkdl.NewEndpoint(nil, nil)
			err = ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: tc.data, Endpoint: ep})
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			assertAttr(t, ep, attrmetrics.MultiClusterKVCacheUtilizationDataKey, tc.wantKV)
			assertAttr(t, ep, attrmetrics.MultiClusterQueueSizeDataKey, tc.wantQ)
			assertAttr(t, ep, attrmetrics.MultiClusterMetricsUpdatedAtDataKey, tc.wantStamp)
			if _, ok := fwkdl.ReadAttribute[attrmetrics.MultiClusterFlowControl](ep.GetAttributes(), attrmetrics.MultiClusterFlowControlDataKey); ok {
				t.Fatal("flow-control bands written for a pool without flow control")
			}
		})
	}
}

func assertAttr(t *testing.T, ep fwkdl.Endpoint, key fwkplugin.DataKey, want *float64) {
	t.Helper()
	got, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)
	if want == nil {
		require.False(t, ok, "expected %s absent", key)
		return
	}
	require.True(t, ok, "expected %s present", key)
	require.InDelta(t, *want, float64(got), 1e-9)
}

// A custom metric name resolves to the same attribute key.
func TestMultiClusterMetricsExtractorCustomNames(t *testing.T) {
	ext, err := NewMultiClusterMetricsExtractor("multicluster-metrics", &multiClusterMetricsExtractorParams{
		KVCacheUtilizationMetric: "custom_kv",
		QueueSizeMetric:          "custom_queue",
	})
	require.NoError(t, err)

	ep := fwkdl.NewEndpoint(nil, nil)
	data := sourcemetrics.PrometheusMetricMap{"custom_kv": gauge(0.3), "custom_queue": gauge(2)}
	require.NoError(t, ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}))

	kv, ok := attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), attrmetrics.MultiClusterKVCacheUtilizationDataKey)
	require.True(t, ok)
	require.InDelta(t, 0.3, float64(kv), 1e-9)
}

func TestMultiClusterMetricsExtractorInvalidMetricName(t *testing.T) {
	_, err := NewMultiClusterMetricsExtractor("multicluster-metrics", &multiClusterMetricsExtractorParams{KVCacheUtilizationMetric: "1_bad"})
	require.Error(t, err)
}

func labeled(priority string, labels ...string) []*dto.LabelPair {
	pairs := []*dto.LabelPair{{Name: ptr.To("priority"), Value: ptr.To(priority)}}
	for i := 0; i+1 < len(labels); i += 2 {
		pairs = append(pairs, &dto.LabelPair{Name: ptr.To(labels[i]), Value: ptr.To(labels[i+1])})
	}
	return pairs
}

func flowPayload(held map[string]float64, waits map[string][2]float64) sourcemetrics.PrometheusMetricMap {
	heldFamily := &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum()}
	for p, v := range held {
		// Two tenants per priority, summed by the extractor.
		heldFamily.Metric = append(heldFamily.Metric,
			&dto.Metric{Label: labeled(p, "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(v / 2)}},
			&dto.Metric{Label: labeled(p, "fairness_id", "y"), Gauge: &dto.Gauge{Value: ptr.To(v / 2)}})
	}
	waitFamily := &dto.MetricFamily{Type: dto.MetricType_HISTOGRAM.Enum()}
	for p, sc := range waits {
		waitFamily.Metric = append(waitFamily.Metric, &dto.Metric{Label: labeled(p, "outcome", "Dispatched"),
			Histogram: &dto.Histogram{SampleSum: ptr.To(sc[0]), SampleCount: ptr.To(uint64(sc[1]))}})
	}
	return sourcemetrics.PrometheusMetricMap{
		defaultKVCacheUtilizationMetric: gauge(0.5),
		defaultQueueSizeMetric:          gauge(1),
		defaultFlowControlQueueMetric:   heldFamily,
		defaultFlowControlWaitMetric:    waitFamily,
	}
}

func waitSeries(priority, tenant string, sum, count float64) *dto.Metric {
	return &dto.Metric{Label: labeled(priority, "fairness_id", tenant, "outcome", "Dispatched"),
		Histogram: &dto.Histogram{SampleSum: ptr.To(sum), SampleCount: ptr.To(uint64(count))}}
}

// flowScraper scrapes one endpoint with a settable clock.
type flowScraper struct {
	t     *testing.T
	ext   *MultiClusterMetricsExtractor
	ep    fwkdl.Endpoint
	clock time.Time
}

func newFlowScraper(t *testing.T) *flowScraper {
	ext, err := NewMultiClusterMetricsExtractor("multicluster-metrics", nil)
	require.NoError(t, err)
	s := &flowScraper{t: t, ext: ext, ep: fwkdl.NewEndpoint(nil, nil), clock: time.Unix(1_800_000_000, 0)}
	ext.now = func() time.Time { return s.clock }
	return s
}

// scrape advances the clock and scrapes; nil families are left out of the payload.
func (s *flowScraper) scrape(after time.Duration, held []*dto.Metric, waits []*dto.Metric) (attrmetrics.MultiClusterFlowControl, bool) {
	payload := sourcemetrics.PrometheusMetricMap{defaultKVCacheUtilizationMetric: gauge(0.5), defaultQueueSizeMetric: gauge(1)}
	if held != nil {
		payload[defaultFlowControlQueueMetric] = &dto.MetricFamily{Type: dto.MetricType_GAUGE.Enum(), Metric: held}
	}
	if waits != nil {
		payload[defaultFlowControlWaitMetric] = &dto.MetricFamily{Type: dto.MetricType_HISTOGRAM.Enum(), Metric: waits}
	}
	return s.extract(after, payload)
}

// extract advances the clock, extracts payload and returns the bands.
func (s *flowScraper) extract(after time.Duration, payload sourcemetrics.PrometheusMetricMap) (attrmetrics.MultiClusterFlowControl, bool) {
	s.clock = s.clock.Add(after)
	require.NoError(s.t, s.ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: payload, Endpoint: s.ep}))
	return fwkdl.ReadAttribute[attrmetrics.MultiClusterFlowControl](s.ep.GetAttributes(), attrmetrics.MultiClusterFlowControlDataKey)
}

// TestFlowControlIdleTenantIsNotAReset: when one tenant's series is deleted, the
// priority's summed count drops. Read as a reset, RecentWait would turn into the
// other tenant's lifetime mean (10 s) instead of the recent 1 s.
func TestFlowControlIdleTenantIsNotAReset(t *testing.T) {
	s := newFlowScraper(t)
	s.scrape(0, nil, []*dto.Metric{waitSeries("40", "x", 1000, 100), waitSeries("40", "y", 1000, 100)})
	b, _ := s.scrape(10*time.Second, nil, []*dto.Metric{waitSeries("40", "x", 1001, 101), waitSeries("40", "y", 1001, 101)})
	require.InDelta(t, 1.0, b[40].RecentWait, 1e-9)
	b, _ = s.scrape(10*time.Second, nil, []*dto.Metric{waitSeries("40", "y", 1002, 102)})
	require.InDelta(t, 1.0, b[40].RecentWait, 1e-9, "x went idle; the window still reads 1 s")
	require.InDelta(t, float64(s.clock.Unix()), b[40].LastProgress, 1e-6)
}

// TestFlowControlBandsClearedWhenFamiliesVanish: once every flow is deleted the
// pool reports no families; old bands must not stay on the endpoint.
func TestFlowControlBandsClearedWhenFamiliesVanish(t *testing.T) {
	s := newFlowScraper(t)
	s.scrape(0, nil, []*dto.Metric{waitSeries("60", "x", 0, 0)})
	b, _ := s.scrape(10*time.Second, nil, []*dto.Metric{waitSeries("60", "x", 50, 2)})
	require.InDelta(t, 25, b[60].RecentWait, 1e-9, "a spike")
	b, ok := s.scrape(10*time.Second, nil, nil)
	require.True(t, ok)
	require.Empty(t, b, "no flows left: no bands")
}

// TestFlowControlIdleBandStartsFresh: a priority whose series were deleted is
// forgotten, so a new request that is only held (no histogram yet) does not
// inherit the last progress of minutes ago and look stalled.
func TestFlowControlIdleBandStartsFresh(t *testing.T) {
	s := newFlowScraper(t)
	held := func(n float64) []*dto.Metric {
		return []*dto.Metric{{Label: labeled("60", "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(n)}}}
	}
	s.scrape(0, held(0), []*dto.Metric{waitSeries("60", "x", 10, 5)})
	s.scrape(10*time.Second, held(0), []*dto.Metric{waitSeries("60", "x", 12, 6)})
	s.scrape(5*time.Minute, held(0), []*dto.Metric{}) // series deleted while idle
	b, _ := s.scrape(10*time.Second, held(1), []*dto.Metric{})
	require.InDelta(t, float64(s.clock.Unix()), b[60].LastProgress, 1e-6, "first sight of the new request")
	require.Zero(t, b[60].RecentWait)

	// A held band that never progresses still ages, so a real stall shows.
	b, _ = s.scrape(20*time.Second, held(1), []*dto.Metric{})
	require.InDelta(t, float64(s.clock.Unix()-20), b[60].LastProgress, 1e-6)
}

// TestFlowControlStallClockStartsWhenHeld: a band whose series outlives an idle
// spell keeps its last departure time; when it starts holding again, the stall
// clock starts then, or its first held request reads as a long stall.
func TestFlowControlStallClockStartsWhenHeld(t *testing.T) {
	s := newFlowScraper(t)
	held := func(n float64) []*dto.Metric {
		return []*dto.Metric{{Label: labeled("40", "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(n)}}}
	}
	s.scrape(0, held(0), []*dto.Metric{waitSeries("40", "x", 10, 5)})
	s.scrape(10*time.Second, held(0), []*dto.Metric{waitSeries("40", "x", 12, 6)}) // last departure
	s.scrape(time.Minute, held(0), []*dto.Metric{waitSeries("40", "x", 12, 6)})    // idle, series kept
	b, _ := s.scrape(10*time.Second, held(3), []*dto.Metric{waitSeries("40", "x", 12, 6)})
	require.InDelta(t, float64(s.clock.Unix()), b[40].LastProgress, 1e-6, "holding again starts the stall clock")
	b, _ = s.scrape(20*time.Second, held(3), []*dto.Metric{waitSeries("40", "x", 12, 6)})
	require.InDelta(t, float64(s.clock.Unix()-20), b[40].LastProgress, 1e-6, "still held, no departure: it ages")
}

// TestFlowControlPartialScrapeKeepsBaseline: a scrape that lost only the wait
// family must not wipe the per-series baseline, or the next scrape would count
// lifetime totals as recent.
func TestFlowControlPartialScrapeKeepsBaseline(t *testing.T) {
	s := newFlowScraper(t)
	held := []*dto.Metric{{Label: labeled("40", "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(0.0)}}}
	s.scrape(0, held, []*dto.Metric{waitSeries("40", "x", 10000, 100)}) // lifetime mean 100 s
	b, _ := s.scrape(10*time.Second, held, []*dto.Metric{waitSeries("40", "x", 10001, 101)})
	require.InDelta(t, 1.0, b[40].RecentWait, 1e-9)
	s.scrape(5*time.Second, held, nil) // wait family missing from this scrape
	b, _ = s.scrape(5*time.Second, held, []*dto.Metric{waitSeries("40", "x", 10002, 102)})
	require.InDelta(t, 1.0, b[40].RecentWait, 1e-9, "the window still reads 1 s, not the lifetime 100 s")
}

// TestEndpointStatePruned: the state of an endpoint not scraped for
// endpointStateTTL (a removed cluster) is dropped.
func TestEndpointStatePruned(t *testing.T) {
	s := newFlowScraper(t)
	s.scrape(0, nil, []*dto.Metric{waitSeries("40", "x", 1, 1)})
	other := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "other"}}, nil)
	s.clock = s.clock.Add(endpointStateTTL + time.Minute)
	require.NoError(t, s.ext.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{
		Payload: sourcemetrics.PrometheusMetricMap{defaultKVCacheUtilizationMetric: gauge(0.5), defaultQueueSizeMetric: gauge(1)}, Endpoint: other}))
	require.Len(t, s.ext.waits, 0, "the first endpoint's history was dropped")
	require.Contains(t, s.ext.seen, "/other")
}

// TestFlowControlPartialScrapeExpires: a pool that keeps reporting the held
// family without the wait family for a whole window has lost its series; its
// bands stop being kept active.
func TestFlowControlPartialScrapeExpires(t *testing.T) {
	s := newFlowScraper(t)
	held := []*dto.Metric{{Label: labeled("40", "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(0.0)}}}
	s.scrape(0, held, []*dto.Metric{waitSeries("40", "x", 10, 5)})
	b, _ := s.scrape(10*time.Second, held, nil)
	require.Contains(t, b, 40, "a brief partial scrape keeps the band")
	s.scrape(flowControlWaitWindow, held, nil)
	b, _ = s.scrape(10*time.Second, held, nil)
	require.NotContains(t, b, 40, "partial for over a window: the band expires")
}

// TestFlowControlCountsDispatchedOnly: fast rejections must not pull a band's
// recent wait down or count as progress.
func TestFlowControlCountsDispatchedOnly(t *testing.T) {
	s := newFlowScraper(t)
	rejected := func(sum, count float64) *dto.Metric {
		return &dto.Metric{Label: labeled("40", "fairness_id", "x", "outcome", "RejectedCapacity"),
			Histogram: &dto.Histogram{SampleSum: ptr.To(sum), SampleCount: ptr.To(uint64(count))}}
	}
	held := []*dto.Metric{{Label: labeled("40", "fairness_id", "x"), Gauge: &dto.Gauge{Value: ptr.To(5.0)}}}
	s.scrape(0, held, []*dto.Metric{waitSeries("40", "x", 0, 0), rejected(0, 0)})
	b, _ := s.scrape(10*time.Second, held, []*dto.Metric{waitSeries("40", "x", 30, 1), rejected(1, 100)})
	require.InDelta(t, 30, b[40].RecentWait, 1e-9, "100 instant rejections do not dilute a 30 s wait")
	first := b[40].LastProgress
	b, _ = s.scrape(20*time.Second, held, []*dto.Metric{waitSeries("40", "x", 30, 1), rejected(2, 200)})
	require.InDelta(t, first, b[40].LastProgress, 1e-6, "rejections are not progress")
}

func TestMultiClusterFlowControlBands(t *testing.T) {
	s := newFlowScraper(t)
	start := float64(s.clock.Unix())

	// First scrape: held per priority, no history yet.
	b, ok := s.extract(0, flowPayload(map[string]float64{"60": 4, "40": 200}, map[string][2]float64{"60": {10, 100}, "40": {500, 50}}))
	require.True(t, ok)
	require.InDelta(t, 4, b[60].Held, 1e-9)
	require.InDelta(t, 200, b[40].Held, 1e-9)
	require.Zero(t, b[60].RecentWait)
	require.InDelta(t, start, b[60].LastProgress, 1e-6)

	// 10 s later: 60 moved 20 requests that waited 1 s each; 40 moved none.
	b, _ = s.extract(10*time.Second, flowPayload(map[string]float64{"60": 2, "40": 230}, map[string][2]float64{"60": {30, 120}, "40": {500, 50}}))
	require.InDelta(t, 1.0, b[60].RecentWait, 1e-9)
	require.InDelta(t, start+10, b[60].LastProgress, 1e-6)
	require.Zero(t, b[40].RecentWait)
	require.InDelta(t, start, b[40].LastProgress, 1e-6, "a band with no departures keeps its last progress")

	// Counter reset (pool EPP restarted): the new totals are growth, not
	// negative waits. The window still holds the 20 requests of 10 s ago.
	b, _ = s.extract(10*time.Second, flowPayload(map[string]float64{"60": 1}, map[string][2]float64{"60": {1, 2}}))
	require.InDelta(t, 21.0/22, b[60].RecentWait, 1e-9, "20 requests waited 1 s, then 2 waited 1 s in total")
	require.InDelta(t, start+20, b[60].LastProgress, 1e-6)
}

// TestFlowControlWaitBetweenDepartures reproduces the rig failure: scrapes are
// seconds apart and a backed-up band lets a request go only every few scrapes.
// A per-scrape mean read 0 between departures, so the hub saw room and sent a
// request into a backlog it then waited 900 s in.
func TestFlowControlWaitBetweenDepartures(t *testing.T) {
	s := newFlowScraper(t)
	start := float64(s.clock.Unix())
	wait := func(after time.Duration, sum, count float64) attrmetrics.FlowControlBand {
		b, ok := s.extract(after, flowPayload(map[string]float64{"40": 18}, map[string][2]float64{"40": {sum, count}}))
		require.True(t, ok)
		return b[40]
	}
	wait(0, 0, 0)
	require.InDelta(t, 15, wait(2*time.Second, 15, 1).RecentWait, 1e-9, "a request left after 15 s")
	require.InDelta(t, 15, wait(2*time.Second, 15, 1).RecentWait, 1e-9, "no departure this scrape: still 15 s")
	require.InDelta(t, 15, wait(2*time.Second, 15, 1).RecentWait, 1e-9)
	require.InDelta(t, 15, wait(2*time.Second, 30, 2).RecentWait, 1e-9, "a second one, also 15 s")
	// Past the window with no departure: the wait ages out, and the stall rule
	// takes over from the last progress at 8 s.
	b := wait(52*time.Second, 30, 2)
	require.Zero(t, b.RecentWait)
	require.InDelta(t, start+8, b.LastProgress, 1e-6)
}

// TestFlowControlNewBandCountsFirstWaits: a priority's wait series appears only
// when its first request leaves the queue. On a pool already tracked its first
// waits must count, or a high-priority band stuck behind full slots stays
// invisible until a second request leaves. On the very first scrape of a pool
// nothing is known about the past: no wait.
func TestFlowControlNewBandCountsFirstWaits(t *testing.T) {
	s := newFlowScraper(t)
	s.extract(0, flowPayload(map[string]float64{"40": 18}, map[string][2]float64{"40": {100, 10}})) // no priority-60 series yet
	b, _ := s.extract(2*time.Second, flowPayload(map[string]float64{"40": 18}, map[string][2]float64{"40": {100, 10}, "60": {14, 1}}))
	require.InDelta(t, 14, b[60].RecentWait, 1e-9, "the new band's first request waited 14 s")

	b, _ = newFlowScraper(t).extract(0, flowPayload(nil, map[string][2]float64{"60": {500, 10}}))
	require.Zero(t, b[60].RecentWait)
}

// TestReadyEndpoints: the cluster weight is its ready pods. A new count applies
// once it held for readyEndpointsHold (readyFromZeroHold from 0), so a pod
// restart moves nothing; a drop to 0 applies at once; a count that stops being
// reported is kept for the hold, then becomes readyNotReported.
func TestReadyEndpoints(t *testing.T) {
	const absent = -100.0 // no attribute
	n := func(v float64) *float64 { return &v }
	type step struct {
		after time.Duration
		ready *float64 // nil: not in the payload
		want  float64
		why   string
	}
	for name, steps := range map[string][]step{
		"hold": {
			{0, nil, absent, "never reported: no attribute"},
			{0, n(5), 5, "first count applies at once"},
			{10 * time.Second, n(4), 5, "a pod restarting: held"},
			{10 * time.Second, n(5), 5, "back before the hold: never applied"},
			{10 * time.Second, n(4), 5, "down again: the hold restarts"},
			{4 * time.Minute, n(4), 5, "still held"},
			{time.Minute, n(4), 4, "held 5 min: applies"},
			{10 * time.Second, n(0), 0, "no pods: at once"},
			{10 * time.Second, n(4), 0, "back up: held"},
			{5 * time.Minute, n(4), 4, "applies"},
		},
		"from zero": {
			{0, n(4), 4, "first count"},
			{time.Second, n(0), 0, "drop to 0 at once"},
			{time.Second, n(2), 0, "back up: held"},
			{30 * time.Second, n(2), 2, "from 0 the short hold (30 s) applies"},
		},
		"stops reported": {
			{0, n(8), 8, "first count"},
			{0, nil, 8, "one missing scrape keeps the count"},
			{readyEndpointsHold, nil, readyNotReported, "missing for the hold: no stale count"},
			{0, n(6), 6, "reported again: first count applies at once"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			s := newFlowScraper(t)
			for i, st := range steps {
				payload := sourcemetrics.PrometheusMetricMap{defaultKVCacheUtilizationMetric: gauge(0.5), defaultQueueSizeMetric: gauge(1)}
				if st.ready != nil {
					payload[defaultReadyEndpointsMetric] = gauge(*st.ready)
				}
				s.extract(st.after, payload)
				got := absent
				if v, ok := attrmetrics.ReadScalarMetricValue(s.ep.GetAttributes(), attrmetrics.MultiClusterReadyEndpointsDataKey); ok {
					got = float64(v)
				}
				require.InDelta(t, st.want, got, 0, "step %d: %s", i, st.why)
			}
		})
	}
}
