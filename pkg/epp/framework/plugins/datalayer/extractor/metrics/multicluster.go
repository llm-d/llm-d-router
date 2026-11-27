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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const (
	// MultiClusterMetricsExtractorType reads only a pool's aggregate metrics, not engine metrics.
	MultiClusterMetricsExtractorType = "multicluster-metrics-extractor"

	defaultKVCacheUtilizationMetric = "llm_d_epp_average_kv_cache_utilization"
	defaultQueueSizeMetric          = "llm_d_epp_average_queue_size"
	defaultFlowControlQueueMetric   = "llm_d_epp_flow_control_queue_size"
	defaultFlowControlWaitMetric    = "llm_d_epp_flow_control_request_queue_duration_seconds"
	defaultReadyEndpointsMetric     = "llm_d_epp_ready_endpoints"

	// flowControlWaitWindow is how far back RecentWait looks. Scrapes are
	// seconds apart and most see no request leave a busy band, so a per-scrape
	// mean would read 0 between departures and hide the backlog.
	flowControlWaitWindow = 30 * time.Second

	// readyEndpointsHold is how long a new ready-pod count must hold before it
	// applies. A drop to 0 applies at once: the cluster has nothing to serve.
	// Coming back from 0 only needs readyFromZeroHold, so a recovered cluster
	// is not left out for minutes. A count that stops being reported keeps its
	// value for readyEndpointsHold, so one bad scrape moves nothing.
	readyEndpointsHold = 5 * time.Minute
	readyFromZeroHold  = 30 * time.Second

	// endpointStateTTL drops the history of an endpoint not scraped this long.
	endpointStateTTL = 10 * time.Minute

	// readyNotReported is written when a pool stops reporting ready pods, so a
	// stale count does not keep weighting it; the hub then weighs it as one.
	readyNotReported = -1.0
)

// dispatchedOutcome is the flow-control outcome counted toward a band's wait
// and progress: fast rejections would otherwise hide a saturated band.
var dispatchedOutcome = types.QueueOutcomeDispatched.String()

type multiClusterMetricsExtractorParams struct {
	KVCacheUtilizationMetric string `json:"kvCacheUtilizationMetric"`
	QueueSizeMetric          string `json:"queueSizeMetric"`
	// FlowControlQueueMetric (gauge) and FlowControlWaitMetric (histogram) are
	// the pool's flow-control families, read per priority. Optional in the payload.
	FlowControlQueueMetric string `json:"flowControlQueueMetric"`
	FlowControlWaitMetric  string `json:"flowControlWaitMetric"`
	// ReadyEndpointsMetric (gauge) is the pool's ready model-server pods.
	// Optional in the payload.
	ReadyEndpointsMetric string `json:"readyEndpointsMetric"`
}

type poolMetric struct {
	spec *Spec
	key  fwkplugin.DataKey
}

// MultiClusterMetricsExtractor writes a pool's aggregate KV-cache utilization and queue
// size to the attributes the multicluster scorers read. The stock per-pod fields are empty
// on a cluster-endpoint.
type MultiClusterMetricsExtractor struct {
	typedName              fwkplugin.TypedName
	metrics                []poolMetric
	flowControlQueueFamily string
	flowControlWaitFamily  string
	readySpec              *Spec
	now                    func() time.Time

	mu sync.Mutex
	// waits keeps, per endpoint, the wait histogram's series and per-priority
	// history, to turn cumulative counters into recent waits.
	waits map[string]*poolWaits
	// ready keeps, per endpoint, the applied and pending ready-pod counts.
	ready map[string]*readyCount
	// seen is when each endpoint was last scraped; state of an endpoint not
	// scraped for endpointStateTTL (a removed cluster) is dropped.
	seen      map[string]float64
	lastPrune float64
}

// readyCount is a pool's ready-pod count as applied, and a different count
// seen since pendingSince that applies once it held for readyEndpointsHold.
type readyCount struct {
	applied, pending, pendingSince float64
	// missingSince is when the count stopped being reported; 0 while reported.
	missingSince float64
}

// poolWaits is one pool's flow-control wait state.
type poolWaits struct {
	// series holds each wait series' last totals, keyed by its full label set,
	// so a series that disappears (an idle flow deleted) drops out instead of
	// reading as a counter reset of its priority.
	series map[string]waitSample
	// bands holds, per priority, running totals built from per-series growth.
	bands map[int]*waitHistory
	// partialSince is when scrapes started reporting the held family without
	// the wait family; 0 while both or neither are reported.
	partialSince float64
}

// waitHistory holds a band's running totals at recent scrapes, oldest first,
// and when a request of the band last left the queue.
type waitHistory struct {
	samples      []waitSample
	lastProgress float64
	// held is the band's held count at the previous scrape: a band that starts
	// holding again starts its stall clock then, not at its last departure.
	held float64
}

type waitSample struct {
	at, sum, count float64
}

func NewMultiClusterMetricsExtractor(name string, params *multiClusterMetricsExtractorParams) (*MultiClusterMetricsExtractor, error) {
	if params == nil {
		params = &multiClusterMetricsExtractorParams{}
	}
	kvSpec, err := parseStringToSpec(cmp.Or(params.KVCacheUtilizationMetric, defaultKVCacheUtilizationMetric))
	if err != nil {
		return nil, fmt.Errorf("kv-cache utilization metric: %w", err)
	}
	queueSpec, err := parseStringToSpec(cmp.Or(params.QueueSizeMetric, defaultQueueSizeMetric))
	if err != nil {
		return nil, fmt.Errorf("queue size metric: %w", err)
	}
	readySpec, err := parseStringToSpec(cmp.Or(params.ReadyEndpointsMetric, defaultReadyEndpointsMetric))
	if err != nil {
		return nil, fmt.Errorf("ready endpoints metric: %w", err)
	}
	return &MultiClusterMetricsExtractor{
		typedName: fwkplugin.TypedName{Type: MultiClusterMetricsExtractorType, Name: name},
		metrics: []poolMetric{
			{spec: kvSpec, key: attrmetrics.MultiClusterKVCacheUtilizationDataKey},
			{spec: queueSpec, key: attrmetrics.MultiClusterQueueSizeDataKey},
		},
		flowControlQueueFamily: cmp.Or(params.FlowControlQueueMetric, defaultFlowControlQueueMetric),
		flowControlWaitFamily:  cmp.Or(params.FlowControlWaitMetric, defaultFlowControlWaitMetric),
		readySpec:              readySpec,
		now:                    time.Now,
		waits:                  map[string]*poolWaits{},
		ready:                  map[string]*readyCount{},
		seen:                   map[string]float64{},
	}, nil
}

func (e *MultiClusterMetricsExtractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// Extract writes each pool aggregate to its attribute. A missing metric is reported
// but does not stop the others.
//
// Writes through the bare attribute key, not a *Slot, because the DataKey
// form here (NewDataKey(key, "")) would String() to "key/" with a trailing
// slash, and downstream readers (the multicluster scorers, the tests) look
// up the bare key. The slot's typed Put would silently miss. Leaving as a
// string-keyed write until the DataKey convention is unified.
func (e *MultiClusterMetricsExtractor) Extract(_ context.Context, in fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]) error {
	ep := in.Endpoint
	e.prune(endpointID(ep))
	var errs []error
	for _, m := range e.metrics {
		metric, err := m.spec.getLatestMetric(in.Payload)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", m.key, err))
			continue
		}
		ep.GetAttributes().Put(m.key, attrmetrics.ScalarMetricValue(extractValue(metric)))
	}
	// Flow control is optional on a pool: missing families are not an error.
	// The attribute stays unset on a pool that never reported them, and is
	// emptied on one whose flows were all deleted, so old bands do not linger.
	if bands := e.flowControl(ep, in.Payload); bands != nil {
		ep.GetAttributes().Put(attrmetrics.MultiClusterFlowControlDataKey, bands)
	}
	// Ready pods are optional too: a pool that never reported them gets no
	// attribute, one that stopped gets readyNotReported; the hub weighs both as one.
	if ready, ok := e.readyEndpoints(ep, in.Payload); ok {
		ep.GetAttributes().Put(attrmetrics.MultiClusterReadyEndpointsDataKey, attrmetrics.ScalarMetricValue(ready))
	}
	// One missing aggregate must not mark a cluster down for good; an empty scrape does.
	if len(errs) < len(e.metrics) {
		ep.GetAttributes().Put(attrmetrics.MultiClusterMetricsUpdatedAtDataKey,
			attrmetrics.ScalarMetricValue(float64(e.now().UnixNano())/1e9))
	}
	return errors.Join(errs...)
}

// flowControl builds the pool's per-priority flow-control bands: requests held
// now, and from the wait histogram's growth over the last flowControlWaitWindow,
// the mean wait of the requests that left and when a request of the band last
// left. It returns nil when the pool reports neither family and never did.
func (e *MultiClusterMetricsExtractor) flowControl(ep fwkdl.Endpoint, payload sourcemetrics.PrometheusMetricMap) attrmetrics.MultiClusterFlowControl {
	heldFamily, waitFamily := payload[e.flowControlQueueFamily], payload[e.flowControlWaitFamily]
	id := endpointID(ep)
	e.mu.Lock()
	defer e.mu.Unlock()
	pool := e.waits[id]
	if heldFamily == nil && waitFamily == nil && pool == nil {
		return nil
	}
	// On the first scrape of a pool nothing is known about the past: existing
	// series are a baseline. Later, a new series started from zero, so all of
	// its totals are growth.
	tracked := pool != nil
	if pool == nil {
		pool = &poolWaits{bands: map[int]*waitHistory{}}
		e.waits[id] = pool
	}

	held := map[int]float64{}
	for _, metric := range heldFamily.GetMetric() {
		if p, ok := priorityOf(metric); ok {
			held[p] += extractValue(metric)
		}
	}
	// Growth per priority, summed over series diffed one by one.
	growth := map[int]waitSample{}
	series := map[string]waitSample{}
	for _, metric := range waitFamily.GetMetric() {
		p, ok := priorityOf(metric)
		if !ok || metric.GetHistogram() == nil {
			continue
		}
		if outcome, labeled := labelOf(metric, "outcome"); labeled && outcome != dispatchedOutcome {
			continue
		}
		key := seriesKey(metric)
		cur := waitSample{sum: metric.GetHistogram().GetSampleSum(), count: float64(metric.GetHistogram().GetSampleCount())}
		series[key] = cur
		g := growth[p]
		prev, seen := pool.series[key]
		switch {
		case !seen && !tracked:
		case !seen || cur.count < prev.count: // new series, or reset (pool EPP restarted)
			g.sum += cur.sum
			g.count += cur.count
		default:
			g.sum += cur.sum - prev.sum
			g.count += cur.count - prev.count
		}
		growth[p] = g
	}
	// A scrape that lost only the wait family says nothing about its series:
	// keep their baseline, or their lifetime totals would read as recent. If
	// that lasts a whole window, the series are gone and their bands expire.
	nowSec := float64(e.now().UnixNano()) / 1e9
	partial := waitFamily == nil && heldFamily != nil
	switch {
	case !partial:
		pool.partialSince = 0
	case pool.partialSince == 0:
		pool.partialSince = nowSec
	case nowSec-pool.partialSince > flowControlWaitWindow.Seconds():
		partial = false
	}
	// The baseline is replaced only when the wait family is reported. Without
	// it (alone or with the held family), a series that comes back with its
	// counters intact is diffed against its last totals, not read as new.
	if waitFamily != nil {
		pool.series = series
	}

	now := float64(e.now().UnixNano()) / 1e9
	window := flowControlWaitWindow.Seconds()
	// A priority is active while it has a wait series or holds requests. An
	// idle one is forgotten, so its next request does not inherit an old last
	// progress and look stalled.
	active := map[int]bool{}
	for p := range growth {
		active[p] = true
	}
	for p, n := range held {
		if n > 0 {
			active[p] = true
		}
	}
	if partial {
		for p := range pool.bands {
			active[p] = true
		}
	}
	for p := range pool.bands {
		if !active[p] {
			delete(pool.bands, p)
		}
	}
	bands := attrmetrics.MultiClusterFlowControl{}
	for p := range active {
		h := pool.bands[p]
		if h == nil {
			// A new band starts from zero with progress now: a held-only band
			// has had no departure yet, and a new series' totals are growth.
			h = &waitHistory{samples: []waitSample{{at: now}}, lastProgress: now}
			pool.bands[p] = h
		}
		bands[p] = h.add(growth[p], held[p], now, window)
	}
	return bands
}

// add records a scrape's growth in the history and returns the band.
func (h *waitHistory) add(g waitSample, held, now, window float64) attrmetrics.FlowControlBand {
	last := h.samples[len(h.samples)-1]
	if g.count > 0 || (held > 0 && h.held == 0) {
		h.lastProgress = now
	}
	h.held = held
	cur := waitSample{at: now, sum: last.sum + g.sum, count: last.count + g.count}
	h.samples = append(h.samples, cur)
	// Keep one sample at or before the window start as the baseline.
	for len(h.samples) > 2 && h.samples[1].at <= now-window {
		h.samples = h.samples[1:]
	}
	band := attrmetrics.FlowControlBand{Held: held, LastProgress: h.lastProgress}
	if base := h.samples[0]; cur.count > base.count {
		band.RecentWait = (cur.sum - base.sum) / (cur.count - base.count)
	}
	return band
}

// seriesKey identifies a series by its full label set.
func seriesKey(metric *dto.Metric) string {
	labels := make([]string, 0, len(metric.GetLabel()))
	for _, label := range metric.GetLabel() {
		labels = append(labels, label.GetName()+"="+label.GetValue())
	}
	sort.Strings(labels)
	return strings.Join(labels, "\xff")
}

// readyEndpoints returns the pool's ready-pod count to apply: the first one
// seen, then a new count only once it held for readyEndpointsHold (from 0:
// readyFromZeroHold), except a drop to 0, which applies at once.
func (e *MultiClusterMetricsExtractor) readyEndpoints(ep fwkdl.Endpoint, payload sourcemetrics.PrometheusMetricMap) (float64, bool) {
	id := endpointID(ep)
	metric, err := e.readySpec.getLatestMetric(payload)
	e.mu.Lock()
	defer e.mu.Unlock()
	now := float64(e.now().UnixNano()) / 1e9
	r := e.ready[id]
	if err != nil {
		switch {
		case r == nil:
			return 0, false
		case r.missingSince == 0:
			r.missingSince = now
		case now-r.missingSince >= readyEndpointsHold.Seconds():
			delete(e.ready, id)
			return readyNotReported, true
		}
		return r.applied, true
	}
	seen := extractValue(metric)
	hold := readyEndpointsHold.Seconds()
	switch {
	case r == nil:
		r = &readyCount{applied: seen, pending: seen}
		e.ready[id] = r
	case r.applied == 0:
		hold = readyFromZeroHold.Seconds()
	}
	r.missingSince = 0
	switch {
	case seen == r.applied || seen == 0:
		r.applied, r.pending = seen, seen
	case seen != r.pending:
		r.pending, r.pendingSince = seen, now
	case now-r.pendingSince >= hold:
		r.applied = seen
	}
	return r.applied, true
}

// prune records that id was scraped now and, at most once a minute, drops the
// state of endpoints not scraped for endpointStateTTL.
func (e *MultiClusterMetricsExtractor) prune(id string) {
	now := float64(e.now().UnixNano()) / 1e9
	e.mu.Lock()
	defer e.mu.Unlock()
	e.seen[id] = now
	if now-e.lastPrune < 60 {
		return
	}
	e.lastPrune = now
	for other, at := range e.seen {
		if now-at > endpointStateTTL.Seconds() {
			delete(e.seen, other)
			delete(e.waits, other)
			delete(e.ready, other)
		}
	}
}

func endpointID(ep fwkdl.Endpoint) string {
	if meta := ep.GetMetadata(); meta != nil {
		return meta.ID.String()
	}
	return ""
}

// labelOf reads a series' label.
func labelOf(metric *dto.Metric, name string) (string, bool) {
	for _, label := range metric.GetLabel() {
		if label.GetName() == name {
			return label.GetValue(), true
		}
	}
	return "", false
}

// priorityOf reads a series' priority label.
func priorityOf(metric *dto.Metric) (int, bool) {
	for _, label := range metric.GetLabel() {
		if label.GetName() == "priority" {
			p, err := strconv.Atoi(label.GetValue())
			return p, err == nil
		}
	}
	return 0, false
}

var _ sourcemetrics.FamilyReader = &MultiClusterMetricsExtractor{}

// MetricFamilies lists the pool aggregate families.
func (e *MultiClusterMetricsExtractor) MetricFamilies() []string {
	names := make([]string, 0, len(e.metrics)+3)
	for _, m := range e.metrics {
		names = append(names, m.spec.Name)
	}
	return append(names, e.flowControlQueueFamily, e.flowControlWaitFamily, e.readySpec.Name)
}

var _ fwkplugin.ProducerPlugin = &MultiClusterMetricsExtractor{}

// Produces advertises the pool attributes so the dependency graph can verify a
// scorer's required key has a producer.
func (e *MultiClusterMetricsExtractor) Produces() map[fwkplugin.DataKey]any {
	out := make(map[fwkplugin.DataKey]any, len(e.metrics)+1)
	for _, m := range e.metrics {
		out[m.key] = attrmetrics.ScalarMetricValue(0)
	}
	out[attrmetrics.MultiClusterMetricsUpdatedAtDataKey] = attrmetrics.ScalarMetricValue(0)
	out[attrmetrics.MultiClusterFlowControlDataKey] = attrmetrics.MultiClusterFlowControl{}
	out[attrmetrics.MultiClusterReadyEndpointsDataKey] = attrmetrics.ScalarMetricValue(0)
	return out
}

// MultiClusterMetricsExtractorFactory instantiates the extractor from configuration.
func MultiClusterMetricsExtractorFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := &multiClusterMetricsExtractorParams{}
	if parameters != nil {
		if err := parameters.Decode(params); err != nil {
			return nil, err
		}
	}
	return NewMultiClusterMetricsExtractor(name, params)
}
