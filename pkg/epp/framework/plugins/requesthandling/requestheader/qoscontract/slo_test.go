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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// fakeQuerier answers each query from a table; an absent query returns no sample.
type fakeQuerier struct {
	answers map[string]fakeAnswer
	queries []string
}

type fakeAnswer struct {
	value float64
	err   error
}

func (f *fakeQuerier) Query(_ context.Context, query string, _ time.Time) (float64, bool, error) {
	f.queries = append(f.queries, query)
	a, ok := f.answers[query]
	if !ok {
		return 0, false, nil
	}
	if a.err != nil {
		return 0, false, a.err
	}
	return a.value, true, nil
}

func sloParams(p *Parameters) {
	p.SLOValidation = &SLOValidationConfig{PrometheusURL: "http://prometheus:9090", MinRequests: 10, LabelSelector: `namespace="llm-d"`}
	p.Teams[0].Promise = &PromiseConfig{
		TTFT: &LatencyObjective{Percentile: "p95", ThresholdMs: 2000},
		TPOT: &LatencyObjective{Percentile: "p95", ThresholdMs: 33},
	}
}

func newSLOTestPlugin(t *testing.T, mutate func(*Parameters)) (*Plugin, *fakeClock, *fakeQuerier) {
	t.Helper()
	p, clock := newTestPlugin(t, func(params *Parameters) {
		sloParams(params)
		if mutate != nil {
			mutate(params)
		}
	})
	require.NotNil(t, p.validator)
	q := &fakeQuerier{answers: map[string]fakeAnswer{}}
	p.validator.querier = q
	return p, clock, q
}

func checkFor(t *testing.T, st *teamState, slo string) sloCheck {
	t.Helper()
	for _, c := range st.slo.checks {
		if c.slo == slo {
			return c
		}
	}
	t.Fatalf("no %s check", slo)
	return sloCheck{}
}

// answer sets the results of a check's queries.
func (f *fakeQuerier) answer(c sloCheck, count, observed, observedAll float64) {
	f.answers[c.countQuery] = fakeAnswer{value: count}
	f.answers[c.observedQuery] = fakeAnswer{value: observed}
	f.answers[c.observedAllQuery] = fakeAnswer{value: observedAll}
}

func TestSLOConfig_Invalid(t *testing.T) {
	cases := map[string]func(*Parameters){
		"promise without sloValidation": func(p *Parameters) { p.SLOValidation = nil },
		"empty promise":                 func(p *Parameters) { p.Teams[0].Promise = &PromiseConfig{} },
		"bad percentile":                func(p *Parameters) { p.Teams[0].Promise.TTFT.Percentile = "95" },
		"p100":                          func(p *Parameters) { p.Teams[0].Promise.TTFT.Percentile = "p100" },
		"zero threshold":                func(p *Parameters) { p.Teams[0].Promise.TPOT.ThresholdMs = 0 },
		"custom team header":            func(p *Parameters) { p.TeamHeader = "x-team" },
		"bad url":                       func(p *Parameters) { p.SLOValidation.PrometheusURL = "prometheus:9090" },
		"bad window":                    func(p *Parameters) { p.SLOValidation.Window = "5 minutes" },
		"zero interval":                 func(p *Parameters) { p.SLOValidation.Interval = "0s" },
		"negative min requests":         func(p *Parameters) { p.SLOValidation.MinRequests = -1 },
		"selector with braces":          func(p *Parameters) { p.SLOValidation.LabelSelector = `{job="epp"}` },
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

func TestSLOConfig_NoPromiseNoValidator(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) {
		params.SLOValidation = &SLOValidationConfig{PrometheusURL: "http://prometheus:9090"}
	})
	assert.Nil(t, p.validator, "validation runs only for teams with a promise")
}

func TestSLOConfig_FactoryParsesPromise(t *testing.T) {
	raw := `{"classes":[{"name":"critical","priority":100},{"name":"batch","priority":-1}],
		"defaultTeam":{"class":"batch"},
		"teams":[{"id":"a","class":"critical","maxConcurrency":3,
			"promise":{"ttft":{"percentile":"p95","thresholdMs":2000},"itl":{"percentile":"p99","thresholdMs":50}}}],
		"sloValidation":{"prometheusURL":"https://thanos:9091","window":"10m","interval":"30s"}}`
	got, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	p := got.(*Plugin)
	require.NotNil(t, p.validator)
	assert.Equal(t, 30*time.Second, p.validator.cfg.interval)
	assert.Equal(t, "10m", p.validator.cfg.window)
	checks := p.teams["a"].slo.checks
	require.Len(t, checks, 2)
	assert.Equal(t, sloTTFT, checks[0].slo)
	assert.Equal(t, sloITL, checks[1].slo)
	assert.InDelta(t, 0.99, checks[1].quantile, 1e-12)
	assert.InDelta(t, 0.05, checks[1].threshold, 1e-12)
}

func TestSLOQueries(t *testing.T) {
	p, _, _ := newSLOTestPlugin(t, func(params *Parameters) {
		params.DefaultTeam.Promise = &PromiseConfig{ITL: &LatencyObjective{Percentile: "p99.9", ThresholdMs: 100}}
	})

	ttft := checkFor(t, p.teams["team-a"], sloTTFT)
	assert.Equal(t,
		`histogram_quantile(0.95, sum by (le) (rate(llm_d_epp_request_ttft_seconds_bucket{fairness_id=~"(team-a)(/.*)?",priority="100",streaming="true",namespace="llm-d"}[5m])))`,
		ttft.observedQuery)
	assert.Equal(t,
		`histogram_quantile(0.95, sum by (le) (rate(llm_d_epp_request_ttft_seconds_bucket{fairness_id=~"(team-a)(/.*)?",streaming="true",namespace="llm-d"}[5m])))`,
		ttft.observedAllQuery, "the all scope covers the team's traffic at any class")
	assert.Equal(t,
		`sum(increase(llm_d_epp_request_ttft_seconds_count{fairness_id=~"(team-a)(/.*)?",priority="100",streaming="true",namespace="llm-d"}[5m]))`,
		ttft.countQuery)

	tpot := checkFor(t, p.teams["team-a"], sloTPOT)
	assert.Contains(t, tpot.observedQuery, `llm_d_epp_request_streaming_tpot_seconds_bucket{fairness_id=~"(team-a)(/.*)?",priority="100",namespace="llm-d"}`)

	// The default team covers every fairness ID the contract does not list.
	itl := checkFor(t, p.defaultTeam, sloITL)
	assert.Equal(t,
		`histogram_quantile(0.999, sum by (le) (rate(llm_d_epp_request_streaming_itl_seconds_bucket{fairness_id!~"(team-a|team-b|team-d)(/.*)?",priority="10",namespace="llm-d"}[5m])))`,
		itl.observedQuery)
}

func TestValidate_Verdicts(t *testing.T) {
	cases := []struct {
		name     string
		count    float64
		observed float64
		want     string
	}{
		{"within threshold", 100, 1.5, sloStatusMet},
		{"at threshold", 100, 2, sloStatusMet},
		{"above threshold", 100, 2.5, sloStatusViolated},
		{"too few requests", 9, 2.5, sloStatusInsufficientData},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _, q := newSLOTestPlugin(t, func(params *Parameters) { params.Teams[0].Promise.TPOT = nil })
			team := p.teams["team-a"]
			c := checkFor(t, team, sloTTFT)
			q.answer(c, tc.count, tc.observed, 3)

			p.validator.validate(context.Background())

			snap := team.slo.snapshot.Load()
			require.NotNil(t, snap)
			assert.Equal(t, tc.want, snap.status)
			r := snap.results[sloTTFT]
			assert.Equal(t, tc.want, r.status)
			assert.InDelta(t, tc.observed, r.observed, 1e-12)
			assert.InDelta(t, 3, r.observedAll, 1e-12)
			assert.InDelta(t, 1, promtestutil.ToFloat64(sloPromiseStatus.WithLabelValues(team.label, "critical", sloTTFT, tc.want)), 0)
			assert.InDelta(t, tc.observed, promtestutil.ToFloat64(sloObservedSeconds.WithLabelValues(team.label, "critical", sloTTFT, scopeGuaranteed)), 1e-12)
		})
	}
}

func TestValidate_NoDataIsInsufficient(t *testing.T) {
	p, _, _ := newSLOTestPlugin(t, nil)
	p.validator.validate(context.Background())
	snap := p.teams["team-a"].slo.snapshot.Load()
	require.NotNil(t, snap)
	assert.Equal(t, sloStatusInsufficientData, snap.status)
	assert.True(t, math.IsNaN(snap.results[sloTTFT].observed))
}

func TestValidate_QueryErrorIsUnknown(t *testing.T) {
	p, _, q := newSLOTestPlugin(t, nil)
	team := p.teams["team-a"]
	ttft, tpot := checkFor(t, team, sloTTFT), checkFor(t, team, sloTPOT)
	q.answer(ttft, 100, 1, 1)
	q.answer(tpot, 100, 0.02, 0.02)
	q.answers[tpot.countQuery] = fakeAnswer{err: errors.New("prometheus down")}
	before := promtestutil.ToFloat64(sloValidationErrors.WithLabelValues(team.label, sloTPOT))

	p.validator.validate(context.Background())

	snap := team.slo.snapshot.Load()
	assert.Equal(t, sloStatusMet, snap.results[sloTTFT].status)
	assert.Equal(t, sloStatusUnknown, snap.results[sloTPOT].status)
	assert.Equal(t, sloStatusUnknown, snap.status, "an unknown objective outranks a met one")
	assert.InDelta(t, before+1, promtestutil.ToFloat64(sloValidationErrors.WithLabelValues(team.label, sloTPOT)), 0)
}

func TestValidate_ViolationOutranksUnknown(t *testing.T) {
	p, _, q := newSLOTestPlugin(t, nil)
	team := p.teams["team-a"]
	ttft, tpot := checkFor(t, team, sloTTFT), checkFor(t, team, sloTPOT)
	q.answer(ttft, 100, 5, 5)
	q.answers[tpot.countQuery] = fakeAnswer{err: errors.New("prometheus down")}

	p.validator.validate(context.Background())
	assert.Equal(t, sloStatusViolated, team.slo.snapshot.Load().status)
}

func TestValidateSLO_StatusIsStaleAfterThreeIntervals(t *testing.T) {
	p, clock, q := newSLOTestPlugin(t, func(params *Parameters) { params.Teams[0].Promise.TPOT = nil })
	team := p.teams["team-a"]
	assert.Equal(t, sloStatusUnknown, team.slo.current(clock.now()), "unknown before the first validation")

	q.answer(checkFor(t, team, sloTTFT), 100, 1, 1)
	p.validator.validate(context.Background())
	assert.Equal(t, sloStatusMet, team.slo.current(clock.now()))

	clock.t = clock.t.Add(3 * time.Minute)
	assert.Equal(t, sloStatusMet, team.slo.current(clock.now()))
	clock.t = clock.t.Add(time.Second)
	assert.Equal(t, sloStatusUnknown, team.slo.current(clock.now()))
}

func TestRequestHeader_ReportsSLOStatusWithinContract(t *testing.T) {
	p, _, q := newSLOTestPlugin(t, func(params *Parameters) { params.Teams[0].Promise.TPOT = nil })
	team := p.teams["team-a"]
	q.answer(checkFor(t, team, sloTTFT), 100, 5, 5)
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

func TestPrometheusQuerier(t *testing.T) {
	tokenFile := filepath.Join(t.TempDir(), "token")
	require.NoError(t, os.WriteFile(tokenFile, []byte("secret\n"), 0o600))

	responses := map[string]string{
		"one":   `{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1700000000,"0.25"]}]}}`,
		"none":  `{"status":"success","data":{"resultType":"vector","result":[]}}`,
		"nan":   `{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1700000000,"NaN"]}]}}`,
		"many":  `{"status":"success","data":{"resultType":"vector","result":[{"metric":{"a":"1"},"value":[1700000000,"1"]},{"metric":{"a":"2"},"value":[1700000000,"2"]}]}}`,
		"error": `{"status":"error","errorType":"bad_data","error":"parse error"}`,
	}
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		require.NoError(t, r.ParseForm())
		body := responses[r.Form.Get("query")]
		w.Header().Set("Content-Type", "application/json")
		if r.Form.Get("query") == "error" {
			w.WriteHeader(http.StatusBadRequest)
		}
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	q, err := newPrometheusQuerier(SLOValidationConfig{PrometheusURL: srv.URL, BearerTokenFile: tokenFile})
	require.NoError(t, err)
	ctx := context.Background()

	v, ok, err := q.Query(ctx, "one", time.Now())
	require.NoError(t, err)
	assert.True(t, ok)
	assert.InDelta(t, 0.25, v, 1e-12)
	assert.Equal(t, "Bearer secret", auth)

	_, ok, err = q.Query(ctx, "none", time.Now())
	require.NoError(t, err)
	assert.False(t, ok)

	_, ok, err = q.Query(ctx, "nan", time.Now())
	require.NoError(t, err)
	assert.False(t, ok, "histogram_quantile over no data is NaN")

	_, _, err = q.Query(ctx, "many", time.Now())
	assert.Error(t, err)

	_, _, err = q.Query(ctx, "error", time.Now())
	assert.Error(t, err)
}
