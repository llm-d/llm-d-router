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
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// latencyMetrics stands in for the EPP request latency histograms, in a registry of its own.
type latencyMetrics struct {
	registry        *prometheus.Registry
	ttft, tpot, itl *prometheus.HistogramVec
}

var testBuckets = []float64{0.01, 0.02, 0.05, 0.5, 1, 2, 4}

func newLatencyMetrics(t *testing.T) *latencyMetrics {
	t.Helper()
	m := &latencyMetrics{registry: prometheus.NewRegistry()}
	newVec := func(name string, labels ...string) *prometheus.HistogramVec {
		v := prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: name, Buckets: testBuckets},
			append([]string{"model_name", "fairness_id", "priority"}, labels...))
		m.registry.MustRegister(v)
		return v
	}
	m.ttft = newVec(sloSeries[sloTTFT].metric, "streaming")
	m.tpot = newVec(sloSeries[sloTPOT].metric)
	m.itl = newVec(sloSeries[sloITL].metric)
	return m
}

// observeTTFT records n TTFT observations of seconds each.
func (m *latencyMetrics) observeTTFT(fairnessID, priority, streaming string, seconds float64, n int) {
	for range n {
		m.ttft.WithLabelValues("llama", fairnessID, priority, streaming).Observe(seconds)
	}
}

func sloParams(p *Parameters) {
	p.SLOValidation = &SLOValidationConfig{MinRequests: 10}
	p.Teams[0].Promise = &PromiseConfig{
		TTFT: &LatencyObjective{Percentile: "p95", ThresholdMs: 2000},
		TPOT: &LatencyObjective{Percentile: "p95", ThresholdMs: 33},
	}
}

func newSLOTestPlugin(t *testing.T, mutate func(*Parameters)) (*Plugin, *fakeClock, *latencyMetrics) {
	t.Helper()
	p, clock := newTestPlugin(t, func(params *Parameters) {
		sloParams(params)
		if mutate != nil {
			mutate(params)
		}
	})
	require.NotNil(t, p.validator)
	m := newLatencyMetrics(t)
	p.validator.gatherer = m.registry
	return p, clock, m
}

func result(t *testing.T, st *teamState, slo string) sloResult {
	t.Helper()
	snap := st.slo.snapshot.Load()
	require.NotNil(t, snap, "not validated")
	r, ok := snap.results[slo]
	require.True(t, ok, "no %s result", slo)
	return r
}

func ttftOnly(p *Parameters) { p.Teams[0].Promise.TPOT = nil }

func TestSLOConfig_Invalid(t *testing.T) {
	cases := map[string]func(*Parameters){
		"empty promise":         func(p *Parameters) { p.Teams[0].Promise = &PromiseConfig{} },
		"bad percentile":        func(p *Parameters) { p.Teams[0].Promise.TTFT.Percentile = "95" },
		"p100":                  func(p *Parameters) { p.Teams[0].Promise.TTFT.Percentile = "p100" },
		"p0":                    func(p *Parameters) { p.Teams[0].Promise.TTFT.Percentile = "p0" },
		"zero threshold":        func(p *Parameters) { p.Teams[0].Promise.TPOT.ThresholdMs = 0 },
		"custom team header":    func(p *Parameters) { p.TeamHeader = "x-team" },
		"bad window":            func(p *Parameters) { p.SLOValidation.Window = "5 minutes" },
		"zero interval":         func(p *Parameters) { p.SLOValidation.Interval = "0s" },
		"window below interval": func(p *Parameters) { p.SLOValidation.Interval, p.SLOValidation.Window = "5m", "1m" },
		"negative min requests": func(p *Parameters) { p.SLOValidation.MinRequests = -1 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			params := testParams()
			sloParams(&params)
			mutate(&params)
			_, err := newPlugin("qos", params, time.Now)
			assert.Error(t, err)
		})
	}
}

func TestSLOConfig_DefaultsWithoutSection(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) {
		sloParams(params)
		params.SLOValidation = nil
	})
	require.NotNil(t, p.validator, "a promise is validated with the default settings")
	assert.Equal(t, time.Minute, p.validator.cfg.interval)
	assert.Equal(t, 5*time.Minute, p.validator.cfg.window)
	assert.InDelta(t, defaultMinRequests, p.validator.cfg.minRequests, 0)
}

func TestSLOConfig_NoPromiseNoValidator(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) {
		params.SLOValidation = &SLOValidationConfig{Window: "10m"}
	})
	assert.Nil(t, p.validator, "validation runs only for teams with a promise")
}

func TestSLOConfig_FactoryParsesPromise(t *testing.T) {
	raw := `{"classes":[{"name":"critical","priority":100},{"name":"batch","priority":-1}],
		"defaultTeam":{"class":"batch"},
		"teams":[{"id":"a","class":"critical","maxConcurrency":3,
			"promise":{"ttft":{"percentile":"p95","thresholdMs":2000},"itl":{"percentile":"p99.9","thresholdMs":50}}}],
		"sloValidation":{"window":"1d","interval":"30s"}}`
	got, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	p := got.(*Plugin)
	require.NotNil(t, p.validator)
	assert.Equal(t, 30*time.Second, p.validator.cfg.interval)
	assert.Equal(t, 24*time.Hour, p.validator.cfg.window)
	checks := p.teams["a"].slo.checks
	require.Len(t, checks, 2)
	assert.Equal(t, sloTTFT, checks[0].slo)
	assert.Equal(t, sloITL, checks[1].slo)
	assert.InDelta(t, 0.999, checks[1].quantile, 0)
	assert.InDelta(t, 0.05, checks[1].threshold, 1e-12)
}

func TestSLOConfig_PrometheusConnectionFieldsAreGone(t *testing.T) {
	raw := `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},
		"sloValidation":{"prometheusURL":"http://prometheus:9090"}}`
	_, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	assert.Error(t, err, "the metrics are read in-process")
}

func TestHistogramQuantile(t *testing.T) {
	// 10 at most 1, 20 in (1, 2], 10 in (2, 4], 10 above 4.
	h := histogram{bounds: []float64{1, 2, 4}, counts: []float64{10, 30, 40}, total: 50}
	cases := []struct {
		q, want float64
	}{
		{0.1, 0.5},  // rank 5 in the first bucket, interpolated from 0
		{0.5, 1.75}, // rank 25: 15 of the 20 in (1, 2]
		{0.8, 4},    // rank 40: the top of (2, 4]
		{0.9, 4},    // rank 45 in +Inf: the largest finite bound
	}
	for _, tc := range cases {
		got, ok := h.quantile(tc.q)
		assert.True(t, ok)
		assert.InDelta(t, tc.want, got, 1e-12, "q=%v", tc.q)
	}
	_, ok := histogram{}.quantile(0.5)
	assert.False(t, ok, "no observations")
}

func TestHistogramSub(t *testing.T) {
	now := histogram{bounds: []float64{1, 2}, counts: []float64{5, 8}, total: 10}
	base := histogram{bounds: []float64{1, 2}, counts: []float64{2, 3}, total: 4}
	assert.Equal(t, histogram{bounds: []float64{1, 2}, counts: []float64{3, 5}, total: 6}, now.sub(base))
	assert.Equal(t, now, now.sub(histogram{}), "a zero base keeps everything")
	assert.Equal(t, now, now.sub(histogram{bounds: []float64{1}, counts: []float64{1}, total: 1}), "other buckets cannot be a base")
	assert.Equal(t, base, base.sub(now), "a count that went down is a reset")
}

func TestValidate_SelectsGuaranteedAndAllTraffic(t *testing.T) {
	p, _, m := newSLOTestPlugin(t, ttftOnly)
	m.observeTTFT("team-a", "100", "true", 0.3, 40)           // guaranteed
	m.observeTTFT("team-a/program-1", "100", "true", 0.3, 10) // guaranteed, composed fairness ID
	m.observeTTFT("team-a", "10", "true", 3, 50)              // downgraded: all only
	m.observeTTFT("team-a", "100", "false", 30, 50)           // non-streaming: excluded
	m.observeTTFT("team-ab", "100", "true", 30, 50)           // another team with the same prefix
	m.observeTTFT("team-b", "100", "true", 30, 50)            // another team

	p.validator.validate(context.Background())

	r := result(t, p.teams["team-a"], sloTTFT)
	assert.InDelta(t, 50, r.requests, 0)
	assert.Equal(t, sloStatusMet, r.status)
	// All 50 guaranteed requests are in (0.05, 0.5]: rank 47.5 interpolates to 0.05 + 0.45*47.5/50.
	assert.InDelta(t, 0.4775, r.observed, 1e-9)
	// The 50 downgraded requests are in (2, 4]: rank 95 of 100 is 45 of those 50, 2 + 2*45/50.
	assert.InDelta(t, 3.8, r.observedAll, 1e-9)
	assert.InDelta(t, r.observedAll, promtestutil.ToFloat64(sloObservedSeconds.WithLabelValues("team-a", "critical", sloTTFT, scopeAll)), 1e-12)
}

func TestValidate_Verdicts(t *testing.T) {
	cases := []struct {
		name    string
		seconds float64
		n       int
		want    string
	}{
		{"within threshold", 1.5, 100, sloStatusMet},
		{"above threshold", 3, 100, sloStatusViolated},
		{"too few requests", 3, 9, sloStatusInsufficientData},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _, m := newSLOTestPlugin(t, ttftOnly)
			m.observeTTFT("team-a", "100", "true", tc.seconds, tc.n)

			p.validator.validate(context.Background())

			team := p.teams["team-a"]
			assert.Equal(t, tc.want, result(t, team, sloTTFT).status)
			assert.Equal(t, tc.want, team.slo.snapshot.Load().status)
			assert.InDelta(t, 1, promtestutil.ToFloat64(sloPromiseStatus.WithLabelValues(team.label, "critical", sloTTFT, tc.want)), 0)
		})
	}
}

func TestValidate_NoDataIsInsufficient(t *testing.T) {
	p, _, _ := newSLOTestPlugin(t, nil)
	p.validator.validate(context.Background())
	r := result(t, p.teams["team-a"], sloTPOT)
	assert.Equal(t, sloStatusInsufficientData, r.status)
	assert.True(t, math.IsNaN(r.observed))
}

func TestValidate_WindowDropsOldObservations(t *testing.T) {
	p, clock, m := newSLOTestPlugin(t, ttftOnly) // interval 1m, window 5m
	team := p.teams["team-a"]
	m.observeTTFT("team-a", "100", "true", 3, 50)
	p.validator.validate(context.Background())
	assert.Equal(t, sloStatusViolated, result(t, team, sloTTFT).status, "before a full window, everything since the start counts")

	for range 4 {
		clock.t = clock.t.Add(time.Minute)
		p.validator.validate(context.Background())
		assert.Equal(t, sloStatusViolated, result(t, team, sloTTFT).status)
	}
	clock.t = clock.t.Add(time.Minute)
	p.validator.validate(context.Background())
	r := result(t, team, sloTTFT)
	assert.Equal(t, sloStatusInsufficientData, r.status, "the slow requests are older than the window")
	assert.InDelta(t, 0, r.requests, 0)

	m.observeTTFT("team-a", "100", "true", 0.3, 50)
	clock.t = clock.t.Add(time.Minute)
	p.validator.validate(context.Background())
	r = result(t, team, sloTTFT)
	assert.Equal(t, sloStatusMet, r.status)
	assert.InDelta(t, 50, r.requests, 0)
	assert.LessOrEqual(t, len(team.slo.checks[0].samples), 7, "samples older than the window are dropped")
}

func TestValidate_DefaultTeamExcludesListedTeams(t *testing.T) {
	p, _, m := newSLOTestPlugin(t, func(params *Parameters) {
		params.DefaultTeam.Promise = &PromiseConfig{TTFT: &LatencyObjective{Percentile: "p95", ThresholdMs: 1000}}
	})
	m.observeTTFT("someone", "10", "true", 0.3, 30)
	m.observeTTFT("other", "10", "true", 0.3, 30)
	m.observeTTFT("team-a", "10", "true", 30, 30) // a listed team downgraded to the default class

	p.validator.validate(context.Background())

	r := result(t, p.defaultTeam, sloTTFT)
	assert.InDelta(t, 60, r.requests, 0)
	assert.Equal(t, sloStatusMet, r.status)
}

func TestValidate_TPOTAndITL(t *testing.T) {
	p, _, m := newSLOTestPlugin(t, func(params *Parameters) {
		params.Teams[0].Promise = &PromiseConfig{
			TPOT: &LatencyObjective{Percentile: "p95", ThresholdMs: 33},
			ITL:  &LatencyObjective{Percentile: "p99", ThresholdMs: 20},
		}
	})
	for range 30 {
		m.tpot.WithLabelValues("llama", "team-a", "100").Observe(0.015)
		m.itl.WithLabelValues("llama", "team-a", "100").Observe(0.04)
	}

	p.validator.validate(context.Background())

	team := p.teams["team-a"]
	assert.Equal(t, sloStatusMet, result(t, team, sloTPOT).status)
	assert.Equal(t, sloStatusViolated, result(t, team, sloITL).status)
	assert.Equal(t, sloStatusViolated, team.slo.snapshot.Load().status, "the team takes its most severe objective")
}

func TestValidate_GatherErrorIsUnknown(t *testing.T) {
	p, _, _ := newSLOTestPlugin(t, nil)
	p.validator.gatherer = prometheus.GathererFunc(func() ([]*dto.MetricFamily, error) {
		return nil, errors.New("gather failed")
	})
	before := promtestutil.ToFloat64(sloValidationErrors)

	p.validator.validate(context.Background())

	team := p.teams["team-a"]
	assert.Equal(t, sloStatusUnknown, team.slo.snapshot.Load().status)
	assert.Equal(t, sloStatusUnknown, result(t, team, sloTTFT).status)
	assert.InDelta(t, before+1, promtestutil.ToFloat64(sloValidationErrors), 0)
}

func TestValidateSLO_StatusIsStaleAfterThreeIntervals(t *testing.T) {
	p, clock, m := newSLOTestPlugin(t, ttftOnly)
	team := p.teams["team-a"]
	assert.Equal(t, sloStatusUnknown, team.slo.current(clock.now()), "unknown before the first validation")

	m.observeTTFT("team-a", "100", "true", 0.3, 50)
	p.validator.validate(context.Background())
	assert.Equal(t, sloStatusMet, team.slo.current(clock.now()))

	clock.t = clock.t.Add(3 * time.Minute)
	assert.Equal(t, sloStatusMet, team.slo.current(clock.now()))
	clock.t = clock.t.Add(time.Second)
	assert.Equal(t, sloStatusUnknown, team.slo.current(clock.now()))
}

func TestRequestHeader_ReportsSLOStatusWithinContract(t *testing.T) {
	p, _, m := newSLOTestPlugin(t, ttftOnly)
	team := p.teams["team-a"]
	m.observeTTFT("team-a", "100", "true", 3, 50)
	p.validator.validate(context.Background())
	before := promtestutil.ToFloat64(sloAdmissions.WithLabelValues(team.label, "critical", sloStatusViolated))

	req := newRequest("team-a", 100, 10)
	require.NoError(t, p.RequestHeader(context.Background(), req))
	assert.Equal(t, 100, req.Objectives.Priority, "validation changes no decision")
	resp := &fwkrh.Response{Headers: map[string]string{}}
	p.ResponseHeader(context.Background(), req, resp, nil)
	assert.Equal(t, sloStatusViolated, resp.Headers[SLOStatusHeader])
	assert.InDelta(t, before+1, promtestutil.ToFloat64(sloAdmissions.WithLabelValues(team.label, "critical", sloStatusViolated)), 0)

	// A downgraded request is outside the contract, so no promise applies to it.
	over := newRequest("team-a", 5000, 10)
	require.NoError(t, p.RequestHeader(context.Background(), over))
	resp = &fwkrh.Response{Headers: map[string]string{}}
	p.ResponseHeader(context.Background(), over, resp, nil)
	assert.Equal(t, "standard", resp.Headers[ClassHeader])
	assert.NotContains(t, resp.Headers, SLOStatusHeader)

	// A team without a promise gets no status.
	other := newRequest("team-b", 100, 10)
	require.NoError(t, p.RequestHeader(context.Background(), other))
	resp = &fwkrh.Response{Headers: map[string]string{}}
	p.ResponseHeader(context.Background(), other, resp, nil)
	assert.NotContains(t, resp.Headers, SLOStatusHeader)
}
