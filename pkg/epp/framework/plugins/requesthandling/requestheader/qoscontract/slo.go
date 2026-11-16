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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"math"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	promapi "github.com/prometheus/client_golang/api"
	promv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	"sigs.k8s.io/controller-runtime/pkg/log"

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

// sloCheck is one objective of one team's promise, with the queries that validate it.
type sloCheck struct {
	slo       string
	quantile  float64
	threshold float64 // seconds
	// observedQuery and observedAllQuery return the latency at the promised percentile for the
	// guaranteed and the all scope. countQuery returns the guaranteed requests in the window.
	observedQuery    string
	observedAllQuery string
	countQuery       string
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
	checks     []sloCheck
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
	var fairness string
	switch {
	case !isDefault:
		fairness = "fairness_id=~" + strconv.Quote(fairnessPattern([]string{team.cfg.ID}))
	case len(listedLabels) > 0:
		fairness = "fairness_id!~" + strconv.Quote(fairnessPattern(listedLabels))
	}
	s := &teamSLO{class: class.Name, staleAfter: staleIntervals * v.interval}
	for _, o := range team.cfg.Promise.objectives() {
		q, err := parsePercentile(o.Percentile)
		if err != nil {
			return nil, err
		}
		series := sloSeries[o.slo]
		all := matchers(fairness, series.streamingOnly, v.cfg.LabelSelector, "")
		guaranteed := matchers(fairness, series.streamingOnly, v.cfg.LabelSelector, strconv.Itoa(class.Priority))
		s.checks = append(s.checks, sloCheck{
			slo:              o.slo,
			quantile:         q,
			threshold:        o.ThresholdMs / 1000,
			observedQuery:    quantileQuery(q, series.metric, guaranteed, v.window),
			observedAllQuery: quantileQuery(q, series.metric, all, v.window),
			countQuery:       fmt.Sprintf("sum(increase(%s_count{%s}[%s]))", series.metric, guaranteed, v.window),
		})
	}
	return s, nil
}

// fairnessPattern matches the given fairness IDs, alone or as the tenant of a composed
// "<tenant>/<program>" fairness ID. PromQL anchors the regex at both ends.
func fairnessPattern(labels []string) string {
	quoted := make([]string, len(labels))
	for i, l := range labels {
		quoted[i] = regexp.QuoteMeta(l)
	}
	return "(" + strings.Join(quoted, "|") + ")(/.*)?"
}

func matchers(fairness string, streamingOnly bool, extra, priority string) string {
	var m []string
	if fairness != "" {
		m = append(m, fairness)
	}
	if priority != "" {
		m = append(m, "priority="+strconv.Quote(priority))
	}
	if streamingOnly {
		m = append(m, `streaming="true"`)
	}
	if extra = strings.TrimSpace(extra); extra != "" {
		m = append(m, extra)
	}
	return strings.Join(m, ",")
}

func quantileQuery(q float64, metric, sel, window string) string {
	return fmt.Sprintf("histogram_quantile(%s, sum by (le) (rate(%s_bucket{%s}[%s])))",
		strconv.FormatFloat(q, 'g', -1, 64), metric, sel, window)
}

// promQuerier evaluates instant queries that return at most one sample.
type promQuerier interface {
	// Query returns the query's single sample. ok is false when the query returns no sample, or
	// a NaN one.
	Query(ctx context.Context, query string, ts time.Time) (value float64, ok bool, err error)
}

type prometheusQuerier struct{ api promv1.API }

func newPrometheusQuerier(cfg SLOValidationConfig) (promQuerier, error) {
	tlsConfig := &tls.Config{InsecureSkipVerify: cfg.InsecureSkipVerify} //nolint:gosec // opt-in, for self-signed in-cluster Prometheus.
	if cfg.CAFile != "" {
		pem, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("read caFile: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("caFile %q holds no PEM certificate", cfg.CAFile)
		}
		tlsConfig.RootCAs = pool
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.TLSClientConfig = tlsConfig
	var rt http.RoundTripper = transport
	if cfg.BearerTokenFile != "" {
		rt = &bearerTokenFileRoundTripper{file: cfg.BearerTokenFile, next: rt}
	}
	client, err := promapi.NewClient(promapi.Config{Address: cfg.PrometheusURL, RoundTripper: rt})
	if err != nil {
		return nil, fmt.Errorf("prometheus client: %w", err)
	}
	return &prometheusQuerier{api: promv1.NewAPI(client)}, nil
}

func (q *prometheusQuerier) Query(ctx context.Context, query string, ts time.Time) (float64, bool, error) {
	v, _, err := q.api.Query(ctx, query, ts)
	if err != nil {
		return 0, false, err
	}
	vec, ok := v.(model.Vector)
	if !ok {
		return 0, false, fmt.Errorf("query returned %s, want a vector", v.Type())
	}
	switch len(vec) {
	case 0:
		return 0, false, nil
	case 1:
		f := float64(vec[0].Value)
		if math.IsNaN(f) {
			return 0, false, nil
		}
		return f, true, nil
	default:
		return 0, false, fmt.Errorf("query returned %d series, want at most 1", len(vec))
	}
}

// bearerTokenFileRoundTripper reads the token for every request, so a rotated token is used.
type bearerTokenFileRoundTripper struct {
	file string
	next http.RoundTripper
}

func (rt *bearerTokenFileRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := os.ReadFile(rt.file)
	if err != nil {
		return nil, fmt.Errorf("read bearerTokenFile: %w", err)
	}
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(string(token)))
	return rt.next.RoundTrip(req)
}

// sloValidator periodically validates team promises against Prometheus.
type sloValidator struct {
	cfg     sloValidation
	querier promQuerier
	now     func() time.Time
	teams   []*teamState // teams with a promise
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

// validate runs one validation cycle, bounded by the query timeout.
func (v *sloValidator) validate(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, v.cfg.queryTimeout)
	defer cancel()
	ts := v.now()
	for _, team := range v.teams {
		s := team.slo
		snap := &sloSnapshot{status: sloStatusMet, results: make(map[string]sloResult, len(s.checks)), at: ts}
		for _, c := range s.checks {
			r := v.evaluate(ctx, team, c, ts)
			snap.results[c.slo] = r
			if sloStatusRank[r.status] > sloStatusRank[snap.status] {
				snap.status = r.status
			}
			recordSLOResult(team.cfg.ID, s.class, c.slo, r)
		}
		s.snapshot.Store(snap)
	}
}

// evaluate validates one check. The promise is met when the observed latency at the promised
// percentile, over the guaranteed traffic in the window, is within the threshold.
func (v *sloValidator) evaluate(ctx context.Context, team *teamState, c sloCheck, ts time.Time) sloResult {
	r := sloResult{status: sloStatusUnknown, observed: math.NaN(), observedAll: math.NaN()}
	query := func(q, what string) (float64, bool, bool) {
		val, ok, err := v.querier.Query(ctx, q, ts)
		if err != nil {
			sloValidationErrors.WithLabelValues(team.cfg.ID, c.slo).Inc()
			log.FromContext(ctx).Error(err, "QoS promise validation query failed", "team", team.cfg.ID, "slo", c.slo, "query", what)
			return 0, false, false
		}
		return val, ok, true
	}
	count, _, okCount := query(c.countQuery, "count")
	observed, hasObserved, okObserved := query(c.observedQuery, scopeGuaranteed)
	if all, hasAll, _ := query(c.observedAllQuery, scopeAll); hasAll {
		r.observedAll = all
	}
	if !okCount || !okObserved {
		return r
	}
	r.requests = count
	if hasObserved {
		r.observed = observed
	}
	switch {
	case !hasObserved || count < v.cfg.minRequests:
		r.status = sloStatusInsufficientData
	case observed <= c.threshold:
		r.status = sloStatusMet
	default:
		r.status = sloStatusViolated
	}
	return r
}

// newSLOValidator builds the validator for the teams with a promise, and sets their checks. It
// returns nil when no team has a promise.
func newSLOValidator(params Parameters, listed []*teamState, defaultTeam *teamState, now func() time.Time) (*sloValidator, error) {
	if params.SLOValidation == nil {
		return nil, nil
	}
	v, err := params.SLOValidation.resolve()
	if err != nil {
		return nil, fmt.Errorf("sloValidation: %w", err)
	}
	listedLabels := make([]string, len(listed))
	for i, t := range listed {
		listedLabels[i] = t.cfg.ID
	}
	validator := &sloValidator{cfg: v, now: now}
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
	if validator.querier, err = newPrometheusQuerier(v.cfg); err != nil {
		return nil, fmt.Errorf("sloValidation: %w", err)
	}
	return validator, nil
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
