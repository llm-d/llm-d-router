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
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time { return c.t }

func testParams() Parameters {
	p := defaultParameters()
	p.Mode = ModeEnforceLimits
	p.Classes = []ClassConfig{
		{Name: "critical", Priority: 100},
		{Name: "standard", Priority: 10},
		{Name: "batch", Priority: -10},
	}
	p.DefaultTeam = TeamConfig{Class: "standard"}
	p.Teams = []TeamConfig{
		{ID: "team-a", Class: "critical", MaxConcurrency: 2, MaxPromptTokens: 1000, MaxOutputTokens: 100},
		{ID: "team-b", Class: "batch", MaxConcurrency: 1, MaxPromptTokens: 1000, MaxOutputTokens: 100},
		{ID: "team-d", Class: "batch", DailyTokens: 500},
	}
	return p
}

func newTestPlugin(t *testing.T, mutate func(*Parameters)) (*Plugin, *fakeClock) {
	t.Helper()
	params := testParams()
	if mutate != nil {
		mutate(&params)
	}
	clock := &fakeClock{t: time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)}
	p, err := newPlugin("qos", params, clock.now)
	require.NoError(t, err)
	return p, clock
}

// usage returns a team's committed in-flight requests and daily weighted tokens.
func usage(p *Plugin, team string) (inflight int, daily float64) {
	st := p.teams[team]
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.inflight, st.dailyUsed
}

var nextRequestID atomic.Int64

// newRequest builds a request of roughly promptTokens prompt tokens and an output cap of maxOut.
func newRequest(team string, promptTokens int, maxOut int64) *fwksched.InferenceRequest {
	headers := map[string]string{}
	if team != "" {
		headers[metadata.FlowFairnessIDKey] = team
	}
	return &fwksched.InferenceRequest{
		RequestID:        "req-" + strconv.FormatInt(nextRequestID.Add(1), 10),
		Headers:          headers,
		RequestSizeBytes: promptTokens * fwkrequest.BytesPerToken,
		Body:             &fwkrequest.InferenceRequestBody{MaxOutputTokens: &maxOut},
	}
}

func complete(p *Plugin, req *fwksched.InferenceRequest, prompt, completion int) {
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = prompt
	resp.Usage.CompletionTokens = completion
	p.ResponseBody(context.Background(), req, resp, nil)
}

func requireCode(t *testing.T, err error, code string) errcommon.Error {
	t.Helper()
	var e errcommon.Error
	require.True(t, errors.As(err, &e), "want errcommon.Error, got %v", err)
	require.Equal(t, code, e.Code)
	return e
}

func TestFactory(t *testing.T) {
	raw := `{"classes":[{"name":"critical","priority":100},{"name":"batch","priority":-1}],
		"defaultTeam":{"class":"batch"},
		"teams":[{"id":"a","class":"critical","maxConcurrency":3,"dailyTokens":1000}]}`
	got, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	p := got.(*Plugin)
	assert.Equal(t, "qos", p.TypedName().Name)
	assert.Equal(t, 3, p.teams["a"].cfg.MaxConcurrency)
	assert.InDelta(t, 1000, p.teams["a"].cfg.DailyTokens, 1e-9)
	assert.True(t, p.reportOnly, "decisions are recorded, not applied, unless mode is enforce-limits")
}

func TestFactory_InvalidConfig(t *testing.T) {
	cases := map[string]string{
		"no classes":           `{"defaultTeam":{"class":"x"}}`,
		"unknown default":      `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"b"}}`,
		"non-decreasing":       `{"classes":[{"name":"a","priority":1},{"name":"b","priority":1}],"defaultTeam":{"class":"a"}}`,
		"duplicate class":      `{"classes":[{"name":"a","priority":2},{"name":"a","priority":1}],"defaultTeam":{"class":"a"}}`,
		"bad mode":             `{"mode":"dry","classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"}}`,
		"zero output weight":   `{"outputTokenWeight":0,"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"}}`,
		"negative limit":       `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},"teams":[{"id":"t","class":"a","maxConcurrency":-1}]}`,
		"duplicate team":       `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},"teams":[{"id":"t","class":"a"},{"id":"t","class":"a"}]}`,
		"team without id":      `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},"teams":[{"class":"a"}]}`,
		"unknown field strict": `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},"bogus":1}`,
		"removed rate commit":  `{"classes":[{"name":"a","priority":1}],"defaultTeam":{"class":"a"},"teams":[{"id":"t","class":"a","commitTokensPerMinute":1000}]}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			require.Error(t, err)
		})
	}
}

func TestRequestHeader_ServesAtCommittedClass(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 50)

	require.NoError(t, p.RequestHeader(t.Context(), req))

	assert.Equal(t, 100, req.Objectives.Priority)
	assert.Equal(t, "team-a", req.FairnessID)
	inflight, daily := usage(p, "team-a")
	assert.Equal(t, 1, inflight)
	assert.InDelta(t, 200, daily, 1e-9, "reserve = 100 + 2*50")

	resp := &fwkrh.Response{Headers: map[string]string{}}
	p.ResponseHeader(t.Context(), req, resp, nil)
	assert.Equal(t, "critical", resp.Headers[ClassHeader])
	assert.Equal(t, "1", resp.Headers[ConcurrencyRemainingHeader])
}

func TestRequestHeader_UnknownTeamUsesDefault(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("stranger", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	assert.Equal(t, 10, req.Objectives.Priority)
	assert.Equal(t, "stranger", req.FairnessID, "the team header still identifies the flow")

	anon := newRequest("", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), anon))
	assert.Equal(t, 10, anon.Objectives.Priority)
	assert.Empty(t, anon.FairnessID, "the director assigns the default flow")
}

func TestRequestHeader_ExplicitFairnessIDIsKept(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 10, 10)
	req.FairnessID = "session-42"
	require.NoError(t, p.RequestHeader(t.Context(), req))
	assert.Equal(t, "session-42", req.FairnessID)
}

func TestRequestHeader_OverSizeLimitDowngradesOneClass(t *testing.T) {
	p, _ := newTestPlugin(t, nil)

	for name, req := range map[string]*fwksched.InferenceRequest{
		"prompt": newRequest("team-a", 2000, 10),
		"output": newRequest("team-a", 10, 500),
	} {
		require.NoError(t, p.RequestHeader(t.Context(), req), name)
		assert.Equal(t, 10, req.Objectives.Priority, "%s above the size limit moves to the next class", name)

		resp := &fwkrh.Response{Headers: map[string]string{}}
		p.ResponseHeader(t.Context(), req, resp, nil)
		assert.Equal(t, "standard", resp.Headers[ClassHeader], name)
		complete(p, req, 10, 10)
	}

	inflight, daily := usage(p, "team-a")
	assert.Equal(t, 0, inflight, "an oversized request holds no slot")
	assert.InDelta(t, 0, daily, 1e-9, "an oversized request is not charged to the budget")
}

func TestRequestHeader_LowestClassOverSizeLimitRejects(t *testing.T) {
	p, _ := newTestPlugin(t, nil)

	e := requireCode(t, p.RequestHeader(t.Context(), newRequest("team-b", 2000, 10)), errcommon.BadRequest)
	assert.Equal(t, string(errcommon.RequestDroppedReasonSizeLimit), e.Headers[errcommon.RequestDroppedReasonHeaderKey])
	assert.Equal(t, "batch", e.Headers[ClassHeader])

	_ = requireCode(t, p.RequestHeader(t.Context(), newRequest("team-b", 10, 500)), errcommon.BadRequest)

	inflight, _ := usage(p, "team-b")
	assert.Equal(t, 0, inflight, "a rejected request holds no slot")
}

func TestRequestHeader_MissingOutputCapReservesTeamMax(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 0)
	req.Body.MaxOutputTokens = nil
	require.NoError(t, p.RequestHeader(t.Context(), req))
	_, daily := usage(p, "team-a")
	assert.InDelta(t, 300, daily, 1e-9, "reserve = 100 + 2*100 (team maxOutputTokens)")
}

func TestRequestHeader_OverConcurrencyCommitDowngradesOneClass(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	r1, r2 := newRequest("team-a", 10, 10), newRequest("team-a", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), r1))
	require.NoError(t, p.RequestHeader(t.Context(), r2))

	over := newRequest("team-a", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), over))
	assert.Equal(t, 10, over.Objectives.Priority, "over commit moves to the next class")

	resp := &fwkrh.Response{Headers: map[string]string{}}
	p.ResponseHeader(t.Context(), over, resp, nil)
	assert.Equal(t, "standard", resp.Headers[ClassHeader])
	assert.Equal(t, "0", resp.Headers[ConcurrencyRemainingHeader])

	// The downgraded request holds no slot and no charge.
	complete(p, over, 10, 10)
	inflight, daily := usage(p, "team-a")
	assert.Equal(t, 2, inflight)
	assert.InDelta(t, 60, daily, 1e-9)

	complete(p, r1, 10, 10)
	next := newRequest("team-a", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), next))
	assert.Equal(t, 100, next.Objectives.Priority, "a completion frees a committed slot")
}

func TestRequestHeader_OverDailyCommitDowngradesOneClass(t *testing.T) {
	p, clock := newTestPlugin(t, func(params *Parameters) {
		params.Teams[0].MaxConcurrency = 0
		params.Teams[0].DailyTokens = 500
	})
	first := newRequest("team-a", 100, 100) // 300
	require.NoError(t, p.RequestHeader(t.Context(), first))
	require.Equal(t, 100, first.Objectives.Priority)

	over := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), over))
	assert.Equal(t, 10, over.Objectives.Priority, "300 + 300 exceeds the 500 daily commit")
	complete(p, over, 100, 100)
	_, daily := usage(p, "team-a")
	assert.InDelta(t, 300, daily, 1e-9, "the downgraded request is not charged to the budget")

	clock.t = clock.t.Add(12 * time.Hour)
	next := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), next))
	assert.Equal(t, 100, next.Objectives.Priority, "the budget resets at the UTC day boundary")
}

func TestRequestHeader_LowestClassOverConcurrencyRejects(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	first := newRequest("team-b", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), first))

	e := requireCode(t, p.RequestHeader(t.Context(), newRequest("team-b", 10, 10)), errcommon.ResourceExhausted)
	assert.Equal(t, string(errcommon.RequestDroppedReasonOverQuota), e.Headers[errcommon.RequestDroppedReasonHeaderKey])
	assert.Equal(t, "batch", e.Headers[ClassHeader])
	assert.Equal(t, "1", e.Headers[RetryAfterHeader])
	assert.Equal(t, "0", e.Headers[ConcurrencyRemainingHeader])

	complete(p, first, 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-b", 10, 10)), "a completion frees a slot")
}

func TestRequestHeader_LowestClassOverDailyRejects(t *testing.T) {
	p, clock := newTestPlugin(t, nil)
	// default reservation is 1000 output tokens; use explicit caps to stay small.
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100))) // 300
	e := requireCode(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100)), errcommon.ResourceExhausted)
	assert.Equal(t, string(errcommon.RequestDroppedReasonOverQuota), e.Headers[errcommon.RequestDroppedReasonHeaderKey])
	assert.Equal(t, "43200", e.Headers[RetryAfterHeader], "noon UTC waits 12h for the next day")

	clock.t = clock.t.Add(12 * time.Hour)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100)), "the budget resets at the UTC day boundary")
}

func TestResponseBody_RefundsUnusedReservation(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	_, daily := usage(p, "team-d")
	require.InDelta(t, 300, daily, 1e-9)

	complete(p, req, 100, 20) // actual = 100 + 2*20 = 140; refund 160
	inflight, daily := usage(p, "team-d")
	assert.InDelta(t, 140, daily, 1e-9)
	assert.Equal(t, 0, inflight)

	complete(p, req, 100, 20)
	_, daily = usage(p, "team-d")
	assert.InDelta(t, 140, daily, 1e-9, "settlement happens once")
}

func TestResponseBody_MissingUsageKeepsReservation(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	complete(p, req, 0, 0)
	_, daily := usage(p, "team-d")
	assert.InDelta(t, 300, daily, 1e-9)
}

func TestResponseBody_UsageAboveReservationIsCharged(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	complete(p, req, 900, 400) // actual 1700 against a reserve of 300
	_, daily := usage(p, "team-d")
	assert.InDelta(t, 1700, daily, 1e-9)
	_ = requireCode(t, p.RequestHeader(t.Context(), newRequest("team-d", 1, 1)), errcommon.ResourceExhausted)
}

func TestStreamEndReleasesUncompletedRequest(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(ctx, req))
	inflight, _ := usage(p, "team-d")
	require.Equal(t, 1, inflight)

	cancel()
	require.Eventually(t, func() bool {
		inflight, _ := usage(p, "team-d")
		return inflight == 0
	}, time.Second, 5*time.Millisecond)
	_, daily := usage(p, "team-d")
	assert.InDelta(t, 0, daily, 1e-9, "an unserved request is refunded in full")
}

func TestCompletedRequestIsNotReleasedAgainOnStreamEnd(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(ctx, req))
	complete(p, req, 100, 100)
	cancel()
	time.Sleep(20 * time.Millisecond)
	inflight, daily := usage(p, "team-d")
	assert.Equal(t, 0, inflight)
	assert.InDelta(t, 300, daily, 1e-9)
}

// TestReportOnlyServesCommittedClass verifies that report-only mode serves every request at its team's
// committed class: downgrades and rejections are recorded, not applied.
func TestReportOnlyServesCommittedClass(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = ModeReportOnly })

	oversized := newRequest("team-a", 5000, 10)
	require.NoError(t, p.RequestHeader(t.Context(), oversized), "a would-be downgrade is recorded, not applied")
	assert.Equal(t, 100, oversized.Objectives.Priority, "team-a stays at its committed critical class")
	assert.Equal(t, "team-a", oversized.FairnessID)

	reqs := make([]*fwksched.InferenceRequest, 0, 5)
	for range 5 {
		req := newRequest("team-b", 10, 10)
		require.NoError(t, p.RequestHeader(t.Context(), req), "a would-be rejection is recorded, not applied")
		assert.Equal(t, -10, req.Objectives.Priority, "team-b stays at its committed batch class")
		resp := &fwkrh.Response{Headers: map[string]string{}}
		p.ResponseHeader(t.Context(), req, resp, nil)
		assert.Empty(t, resp.Headers, "report-only mode adds no response headers")
		reqs = append(reqs, req)
	}
	inflight, _ := usage(p, "team-b")
	assert.Equal(t, 1, inflight, "a would-be rejection holds no slot, as in enforce-limits mode")
	for _, req := range reqs {
		complete(p, req, 10, 10)
	}
	inflight, _ = usage(p, "team-b")
	assert.Equal(t, 0, inflight)
}

// testutilDecisions returns the decision counter values for mode, keyed by
// team|class|served_class|decision|reason.
func testutilDecisions(mode string) map[string]float64 {
	ch := make(chan prometheus.Metric, 1024)
	go func() { contractDecisions.Collect(ch); close(ch) }()
	out := map[string]float64{}
	for m := range ch {
		var pb dto.Metric
		if err := m.Write(&pb); err != nil {
			continue
		}
		lb := map[string]string{}
		for _, l := range pb.GetLabel() {
			lb[l.GetName()] = l.GetValue()
		}
		if lb["mode"] != mode {
			continue
		}
		out[strings.Join([]string{lb["team"], lb["class"], lb["served_class"], lb["decision"], lb["reason"]}, "|")] = pb.GetCounter().GetValue()
	}
	return out
}

// diffDecisions returns the one label set whose count grew between before and after.
func diffDecisions(before, after map[string]float64) string {
	for k, v := range after {
		if v > before[k] {
			return k
		}
	}
	return ""
}

// TestReportOnlyAndEnforceLimitsSetTheSamePriority verifies that both modes serve a request within its
// contract at the team's class, and that the only difference is that report-only mode does not move
// a request outside its contract to a lower class.
func TestReportOnlyAndEnforceLimitsSetTheSamePriority(t *testing.T) {
	cases := []struct {
		name                string
		req                 func() *fwksched.InferenceRequest
		setup               int // team-a requests served within the contract first, to fill its concurrency commit of 2
		enforceLimitsPrio   int
		reportOnlyPrio      int
		enforceLimitsErrors bool
	}{
		{name: "critical within contract", req: func() *fwksched.InferenceRequest { return newRequest("team-a", 10, 10) }, enforceLimitsPrio: 100, reportOnlyPrio: 100},
		{name: "standard within contract", req: func() *fwksched.InferenceRequest { return newRequest("stranger", 10, 10) }, enforceLimitsPrio: 10, reportOnlyPrio: 10},
		{name: "batch within contract", req: func() *fwksched.InferenceRequest { return newRequest("team-b", 10, 10) }, enforceLimitsPrio: -10, reportOnlyPrio: -10},
		{name: "critical over size", req: func() *fwksched.InferenceRequest { return newRequest("team-a", 5000, 10) }, enforceLimitsPrio: 10, reportOnlyPrio: 100},
		{name: "critical over concurrency", req: func() *fwksched.InferenceRequest { return newRequest("team-a", 10, 10) }, setup: 2, enforceLimitsPrio: 10, reportOnlyPrio: 100},
		{name: "batch over size", req: func() *fwksched.InferenceRequest { return newRequest("team-b", 5000, 10) }, enforceLimitsErrors: true, reportOnlyPrio: -10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []string{ModeEnforceLimits, ModeReportOnly} {
				p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = mode })
				for range tc.setup {
					require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-a", 10, 10)))
				}
				req := tc.req()
				err := p.RequestHeader(t.Context(), req)
				if mode == ModeEnforceLimits && tc.enforceLimitsErrors {
					require.Error(t, err)
					continue
				}
				require.NoError(t, err)
				want := tc.enforceLimitsPrio
				if mode == ModeReportOnly {
					want = tc.reportOnlyPrio
				}
				assert.Equal(t, want, req.Objectives.Priority, mode)
			}
		})
	}
}

// TestReportOnlyDecisionsMatchEnforceLimits verifies that report-only mode records the same
// decisions that enforce-limits mode applies, for the same requests.
func TestReportOnlyDecisionsMatchEnforceLimits(t *testing.T) {
	run := func(mode string) []string {
		p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = mode })
		var got []string
		decide := func(req *fwksched.InferenceRequest) {
			before := testutilDecisions(mode)
			_ = p.RequestHeader(t.Context(), req)
			got = append(got, diffDecisions(before, testutilDecisions(mode)))
		}
		for range 3 {
			decide(newRequest("team-a", 10, 10)) // the third is over concurrency 2
		}
		decide(newRequest("team-a", 5000, 10)) // over the size limit

		// team-b is in the lowest class with a concurrency commit of 1. A would-be rejection must
		// hold no slot, so once the in-contract request completes the next one is within the contract again.
		first := newRequest("team-b", 10, 10)
		decide(first)
		decide(newRequest("team-b", 10, 10)) // over concurrency
		complete(p, first, 10, 10)
		decide(newRequest("team-b", 10, 10))

		decide(newRequest("team-d", 100, 100))
		decide(newRequest("team-d", 100, 100)) // over the daily budget, lowest class
		return got
	}
	assert.Equal(t, run(ModeEnforceLimits), run(ModeReportOnly))
}

// decisions returns the decision counter for one label combination in enforce-limits mode.
func decisions(team, class, servedClass, decision, reason string) float64 {
	return promtestutil.ToFloat64(contractDecisions.WithLabelValues(
		team, class, servedClass, decision, reason, ModeEnforceLimits))
}

// TestDecisionReason verifies that a downgraded or rejected request records the limit it
// exceeded, and that a request within the contract records none.
func TestDecisionReason(t *testing.T) {
	cases := []struct {
		name        string
		mutate      func(*Parameters)
		team        string
		setup       []*fwksched.InferenceRequest // served within the contract first, to fill a commit
		req         *fwksched.InferenceRequest
		servedClass string
		decision    string
		reason      string
	}{
		{
			name: "within contract", team: "team-a", req: newRequest("team-a", 10, 10),
			servedClass: "critical", decision: decisionWithinContract, reason: "",
		},
		{
			name: "downgraded over prompt size", team: "team-a", req: newRequest("team-a", 2000, 10),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonSize,
		},
		{
			name: "downgraded over output size", team: "team-a", req: newRequest("team-a", 10, 500),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonSize,
		},
		{
			name: "downgraded over concurrency", team: "team-a",
			setup:       []*fwksched.InferenceRequest{newRequest("team-a", 10, 10), newRequest("team-a", 10, 10)},
			req:         newRequest("team-a", 10, 10),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonConcurrency,
		},
		{
			name:   "downgraded over daily budget",
			mutate: func(p *Parameters) { p.Teams[0].MaxConcurrency = 0; p.Teams[0].DailyTokens = 500 },
			team:   "team-a", setup: []*fwksched.InferenceRequest{newRequest("team-a", 100, 100)},
			req:         newRequest("team-a", 100, 100),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonDaily,
		},
		{
			name:   "size is reported before daily budget and concurrency",
			mutate: func(p *Parameters) { p.Teams[0].MaxConcurrency = 1; p.Teams[0].DailyTokens = 500 },
			team:   "team-a", setup: []*fwksched.InferenceRequest{newRequest("team-a", 100, 100)},
			req:         newRequest("team-a", 2000, 10),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonSize,
		},
		{
			name:   "daily budget is reported before concurrency",
			mutate: func(p *Parameters) { p.Teams[0].MaxConcurrency = 1; p.Teams[0].DailyTokens = 500 },
			team:   "team-a", setup: []*fwksched.InferenceRequest{newRequest("team-a", 100, 100)},
			req:         newRequest("team-a", 100, 100),
			servedClass: "standard", decision: decisionDowngraded, reason: reasonDaily,
		},
		{
			name: "rejected over size", team: "team-b", req: newRequest("team-b", 2000, 10),
			servedClass: "", decision: decisionRejectedSize, reason: reasonSize,
		},
		{
			name: "rejected over concurrency", team: "team-b",
			setup:       []*fwksched.InferenceRequest{newRequest("team-b", 10, 10)},
			req:         newRequest("team-b", 10, 10),
			servedClass: "", decision: decisionRejectedConcurrency, reason: reasonConcurrency,
		},
		{
			name: "rejected over daily budget", team: "team-d",
			setup:       []*fwksched.InferenceRequest{newRequest("team-d", 100, 100)},
			req:         newRequest("team-d", 100, 100),
			servedClass: "", decision: decisionRejectedDaily, reason: reasonDaily,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := newTestPlugin(t, tc.mutate)
			for _, req := range tc.setup {
				require.NoError(t, p.RequestHeader(t.Context(), req))
			}
			class := p.classes[p.teams[tc.team].classIndex].Name
			before := decisions(tc.team, class, tc.servedClass, tc.decision, tc.reason)
			_ = p.RequestHeader(t.Context(), tc.req)
			assert.InDelta(t, 1, decisions(tc.team, class, tc.servedClass, tc.decision, tc.reason)-before, 1e-9)
		})
	}
}
