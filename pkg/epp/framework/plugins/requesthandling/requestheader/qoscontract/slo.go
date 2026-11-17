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

package qoscontract

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// SLOStatusHeader reports the team's promise status when a request within the contract was
	// admitted: met, violated, insufficient_data or unknown.
	SLOStatusHeader = "x-llm-d-qos-slo-status"

	sloTTFT = "ttft"
	sloTPOT = "tpot"
	sloITL  = "itl"

	// scopeGuaranteed is the team's traffic at its committed class, the traffic the promise
	// covers. scopeAll is all of the team's traffic, at any class.
	scopeGuaranteed = "guaranteed"
	scopeAll        = "all"

	sloStatusMet              = "met"
	sloStatusViolated         = "violated"
	sloStatusInsufficientData = "insufficient_data"
	sloStatusUnknown          = "unknown"

	// A status older than this many intervals reads as unknown, so a stalled validation loop is
	// not reported as a current verdict.
	staleIntervals = 3
)

var sloStatuses = []string{sloStatusMet, sloStatusViolated, sloStatusInsufficientData, sloStatusUnknown}

// sloStatusRank orders statuses by severity. A team's status is the most severe of its objectives'.
var sloStatusRank = map[string]int{sloStatusMet: 0, sloStatusInsufficientData: 1, sloStatusUnknown: 2, sloStatusViolated: 3}

// sloSeries is the EPP histogram each objective is validated against.
var sloSeries = map[string]struct {
	metric        string
	streamingOnly bool
}{
	// A non-streaming TTFT is the end-to-end latency, so only streamed requests count.
	sloTTFT: {eppmetrics.LLMDRouterEndpointPickerSubsystem + "_request_ttft_seconds", true},
	sloTPOT: {eppmetrics.LLMDRouterEndpointPickerSubsystem + "_request_streaming_tpot_seconds", false},
	sloITL:  {eppmetrics.LLMDRouterEndpointPickerSubsystem + "_request_streaming_itl_seconds", false},
}

type namedObjective struct {
	slo string
	LatencyObjective
}

// objectives returns the set objectives in a fixed order.
func (p *PromiseConfig) objectives() []namedObjective {
	var out []namedObjective
	for _, o := range []struct {
		slo string
		obj *LatencyObjective
	}{{sloTTFT, p.TTFT}, {sloTPOT, p.TPOT}, {sloITL, p.ITL}} {
		if o.obj != nil {
			out = append(out, namedObjective{slo: o.slo, LatencyObjective: *o.obj})
		}
	}
	return out
}

// histogram is a cumulative histogram: counts[i] observations are at most bounds[i], and total
// observations in all, the +Inf bucket included. The zero value has no observations.
type histogram struct {
	bounds []float64
	counts []float64
	total  float64
}

// add sums o into h. A histogram with other bucket bounds is skipped.
func (h *histogram) add(o histogram) {
	if h.total == 0 && len(h.bounds) == 0 {
		h.bounds = append([]float64(nil), o.bounds...)
		h.counts = append([]float64(nil), o.counts...)
		h.total = o.total
		return
	}
	if !sameBounds(h.bounds, o.bounds) {
		return
	}
	for i := range h.counts {
		h.counts[i] += o.counts[i]
	}
	h.total += o.total
}

// sub returns the observations made after base. When base cannot precede h (other buckets, or
// a count that went down because a series was removed), it returns h.
func (h histogram) sub(base histogram) histogram {
	if base.total == 0 {
		return h
	}
	if !sameBounds(h.bounds, base.bounds) || h.total < base.total {
		return h
	}
	out := histogram{bounds: h.bounds, counts: make([]float64, len(h.counts)), total: h.total - base.total}
	for i := range h.counts {
		if h.counts[i] < base.counts[i] {
			return h
		}
		out.counts[i] = h.counts[i] - base.counts[i]
	}
	return out
}

// quantile estimates the q-quantile as PromQL histogram_quantile does: linear interpolation
// within the bucket holding the rank, starting from 0 in the first bucket. A rank in the +Inf
// bucket returns the largest finite bound. ok is false when there are no observations.
func (h histogram) quantile(q float64) (value float64, ok bool) {
	if h.total == 0 || len(h.bounds) == 0 {
		return math.NaN(), false
	}
	rank := q * h.total
	b := -1
	for i, c := range h.counts {
		if c >= rank {
			b = i
			break
		}
	}
	if b < 0 {
		return h.bounds[len(h.bounds)-1], true
	}
	start, below := 0.0, 0.0
	if b > 0 {
		start, below = h.bounds[b-1], h.counts[b-1]
	} else if h.bounds[0] <= 0 {
		return h.bounds[0], true
	}
	return start + (h.bounds[b]-start)*((rank-below)/(h.counts[b]-below)), true
}

func sameBounds(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// seriesFilter selects the series of one scope of a team's traffic.
type seriesFilter struct {
	// fairnessIDs are matched alone or as the tenant of a composed "<tenant>/<program>" fairness ID.
	fairnessIDs []string
	// exclude selects the series whose fairness ID matches none of fairnessIDs.
	exclude       bool
	priority      string // empty matches any priority
	streamingOnly bool
}

func (f seriesFilter) matches(labels map[string]string) bool {
	if f.priority != "" && labels["priority"] != f.priority {
		return false
	}
	if f.streamingOnly && labels["streaming"] != "true" {
		return false
	}
	id := labels["fairness_id"]
	in := false
	for _, t := range f.fairnessIDs {
		if id == t || strings.HasPrefix(id, t+"/") {
			in = true
			break
		}
	}
	return in != f.exclude
}

// sloSample is one reading of a check's cumulative histograms.
type sloSample struct {
	at                     time.Time
	guaranteed, allTraffic histogram
}

// sloCheck is one objective of one team's promise. Its samples are read and written only by the
// validation loop.
type sloCheck struct {
	slo        string
	metric     string
	quantile   float64
	threshold  float64 // seconds
	guaranteed seriesFilter
	allTraffic seriesFilter
	samples    []sloSample
}

// windowBase returns the reading the window starts from: the newest one at or before at-window,
// so the window covers at least its length. Older readings are dropped. Before the validator
// has run for a window, the base is the zero histogram, and the window covers everything since
// the validator started.
func (c *sloCheck) windowBase(at time.Time, window time.Duration) sloSample {
	start := at.Add(-window)
	i := -1
	for j, s := range c.samples {
		if s.at.After(start) {
			break
		}
		i = j
	}
	if i < 0 {
		return sloSample{}
	}
	c.samples = c.samples[i:]
	return c.samples[0]
}

// sloResult is the last validation of one check.
type sloResult struct {
	status      string
	observed    float64 // NaN when there is no data
	observedAll float64 // NaN when there is no data
	requests    float64
}

// sloSnapshot is the last validation of a team's promise.
type sloSnapshot struct {
	status  string
	results map[string]sloResult
	at      time.Time
}

// teamSLO is a team's promise and its last validation.
type teamSLO struct {
	class      string
	checks     []*sloCheck
	staleAfter time.Duration
	snapshot   atomic.Pointer[sloSnapshot]
}

// current returns the team's promise status at now.
func (s *teamSLO) current(now time.Time) string {
	snap := s.snapshot.Load()
	if snap == nil || now.Sub(snap.at) > s.staleAfter {
		return sloStatusUnknown
	}
	return snap.status
}

// newTeamSLO builds the checks for team. For the default team, whose traffic carries the
// requester's own fairness ID, the series of every listed team are excluded instead.
func newTeamSLO(team *teamState, class ClassConfig, isDefault bool, listedLabels []string, v sloValidation) (*teamSLO, error) {
	fairness := seriesFilter{fairnessIDs: []string{team.cfg.ID}}
	if isDefault {
		fairness = seriesFilter{fairnessIDs: listedLabels, exclude: true}
	}
	s := &teamSLO{class: class.Name, staleAfter: staleIntervals * v.interval}
	for _, o := range team.cfg.Promise.objectives() {
		q, err := parsePercentile(o.Percentile)
		if err != nil {
			return nil, err
		}
		series := sloSeries[o.slo]
		all := fairness
		all.streamingOnly = series.streamingOnly
		guaranteed := all
		guaranteed.priority = strconv.Itoa(class.Priority)
		s.checks = append(s.checks, &sloCheck{
			slo:        o.slo,
			metric:     series.metric,
			quantile:   q,
			threshold:  o.ThresholdMs / 1000,
			guaranteed: guaranteed,
			allTraffic: all,
		})
	}
	return s, nil
}

// sloValidator periodically validates team promises against the EPP's own latency histograms.
type sloValidator struct {
	cfg      sloValidation
	gatherer prometheus.Gatherer
	now      func() time.Time
	teams    []*teamState // teams with a promise
}

// newSLOValidator builds the validator for the teams with a promise, and sets their checks. It
// returns nil when no team has a promise.
func newSLOValidator(params Parameters, listed []*teamState, defaultTeam *teamState, now func() time.Time) (*sloValidator, error) {
	var cfg SLOValidationConfig
	if params.SLOValidation != nil {
		cfg = *params.SLOValidation
	}
	v, err := cfg.resolve()
	if err != nil {
		return nil, fmt.Errorf("sloValidation: %w", err)
	}
	listedLabels := make([]string, len(listed))
	for i, t := range listed {
		listedLabels[i] = t.cfg.ID
	}
	validator := &sloValidator{cfg: v, gatherer: ctrlmetrics.Registry, now: now}
	for _, t := range append(append([]*teamState{}, listed...), defaultTeam) {
		if t.cfg.Promise == nil {
			continue
		}
		s, err := newTeamSLO(t, params.Classes[t.classIndex], t == defaultTeam, listedLabels, v)
		if err != nil {
			return nil, fmt.Errorf("team %q: %w", t.cfg.ID, err)
		}
		t.slo = s
		validator.teams = append(validator.teams, t)
	}
	if len(validator.teams) == 0 {
		return nil, nil
	}
	return validator, nil
}

// run validates at once and then every interval until ctx is done.
func (v *sloValidator) run(ctx context.Context) {
	v.validate(ctx)
	ticker := time.NewTicker(v.cfg.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			v.validate(ctx)
		}
	}
}

// validate reads the latency histograms once and validates every promise against them.
func (v *sloValidator) validate(ctx context.Context) {
	ts := v.now()
	families, err := v.gatherer.Gather()
	if err != nil {
		sloValidationErrors.Inc()
		log.FromContext(ctx).Error(err, "QoS promise validation could not read the latency metrics")
		if len(families) == 0 {
			v.storeUnknown(ts)
			return
		}
	}
	byName := make(map[string]*dto.MetricFamily, len(sloSeries))
	for _, mf := range families {
		byName[mf.GetName()] = mf
	}
	for _, team := range v.teams {
		s := team.slo
		snap := &sloSnapshot{status: sloStatusMet, results: make(map[string]sloResult, len(s.checks)), at: ts}
		for _, c := range s.checks {
			mf := byName[c.metric]
			sample := sloSample{at: ts, guaranteed: aggregate(mf, c.guaranteed), allTraffic: aggregate(mf, c.allTraffic)}
			base := c.windowBase(ts, v.cfg.window)
			c.samples = append(c.samples, sample)
			r := v.evaluate(c, sample.guaranteed.sub(base.guaranteed), sample.allTraffic.sub(base.allTraffic))
			snap.results[c.slo] = r
			if sloStatusRank[r.status] > sloStatusRank[snap.status] {
				snap.status = r.status
			}
			recordSLOResult(team.cfg.ID, s.class, c.slo, r)
		}
		s.snapshot.Store(snap)
	}
}

// storeUnknown records every promise as unknown.
func (v *sloValidator) storeUnknown(ts time.Time) {
	for _, team := range v.teams {
		s := team.slo
		snap := &sloSnapshot{status: sloStatusUnknown, results: make(map[string]sloResult, len(s.checks)), at: ts}
		for _, c := range s.checks {
			r := sloResult{status: sloStatusUnknown, observed: math.NaN(), observedAll: math.NaN()}
			snap.results[c.slo] = r
			recordSLOResult(team.cfg.ID, s.class, c.slo, r)
		}
		s.snapshot.Store(snap)
	}
}

// evaluate validates one check over the window. The promise is met when the latency at the
// promised percentile of the guaranteed traffic is within the threshold.
func (v *sloValidator) evaluate(c *sloCheck, guaranteed, allTraffic histogram) sloResult {
	r := sloResult{requests: guaranteed.total}
	observed, ok := guaranteed.quantile(c.quantile)
	r.observed = observed
	r.observedAll, _ = allTraffic.quantile(c.quantile)
	switch {
	case !ok || guaranteed.total < v.cfg.minRequests:
		r.status = sloStatusInsufficientData
	case observed <= c.threshold:
		r.status = sloStatusMet
	default:
		r.status = sloStatusViolated
	}
	return r
}

// aggregate sums the histogram series of mf that f selects.
func aggregate(mf *dto.MetricFamily, f seriesFilter) histogram {
	var out histogram
	if mf == nil || mf.GetType() != dto.MetricType_HISTOGRAM {
		return out
	}
	labels := map[string]string{}
	for _, m := range mf.GetMetric() {
		clear(labels)
		for _, lp := range m.GetLabel() {
			labels[lp.GetName()] = lp.GetValue()
		}
		if !f.matches(labels) {
			continue
		}
		h := m.GetHistogram()
		series := histogram{total: float64(h.GetSampleCount())}
		for _, b := range h.GetBucket() {
			if math.IsInf(b.GetUpperBound(), 1) {
				continue
			}
			series.bounds = append(series.bounds, b.GetUpperBound())
			series.counts = append(series.counts, float64(b.GetCumulativeCount()))
		}
		out.add(series)
	}
	return out
}

// recordSLOResult exports one check's result. A latency with no data removes its series rather
// than report NaN.
func recordSLOResult(team, class, slo string, r sloResult) {
	for _, s := range sloStatuses {
		v := 0.0
		if s == r.status {
			v = 1
		}
		sloPromiseStatus.WithLabelValues(team, class, slo, s).Set(v)
	}
	for scope, val := range map[string]float64{scopeGuaranteed: r.observed, scopeAll: r.observedAll} {
		if math.IsNaN(val) {
			sloObservedSeconds.DeleteLabelValues(team, class, slo, scope)
			continue
		}
		sloObservedSeconds.WithLabelValues(team, class, slo, scope).Set(val)
	}
	sloWindowRequests.WithLabelValues(team, class, slo).Set(r.requests)
}
