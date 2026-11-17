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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
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

// servedPriority returns the priority the director applies after the plugin runs: the produced
// PriorityKey, or the request's own priority when the plugin produced none.
func servedPriority(req *fwksched.InferenceRequest) int {
	if p, ok := fwksched.ReadRequestAttribute[int](req, fwkrh.PriorityKey); ok {
		return p
	}
	return req.Objectives.Priority
}

// servedFairnessID returns the fairness ID the director applies after the plugin runs: the
// request's own fairness ID, or the produced TenantKey when the request has none.
func servedFairnessID(req *fwksched.InferenceRequest) string {
	if req.FairnessID != "" {
		return req.FairnessID
	}
	tenant, _ := fwksched.ReadRequestAttribute[string](req, fwkrh.TenantKey)
	return tenant
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

	assert.Equal(t, 100, servedPriority(req))
	assert.Equal(t, "team-a", servedFairnessID(req))
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
	assert.Equal(t, 10, servedPriority(req))
	assert.Equal(t, "stranger", servedFairnessID(req), "the team header still identifies the flow")

	anon := newRequest("", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), anon))
	assert.Equal(t, 10, servedPriority(anon))
	assert.Empty(t, servedFairnessID(anon), "the director assigns the default flow")
}

func TestRequestHeader_ExplicitFairnessIDIsKept(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 10, 10)
	req.FairnessID = "session-42"
	require.NoError(t, p.RequestHeader(t.Context(), req))
	assert.Equal(t, "session-42", servedFairnessID(req))
	_, produced := req.GetAttribute(fwkrh.TenantKey)
	assert.False(t, produced, "a request with a fairness ID gets no tenant")
}

func TestProducesPriorityAndTenant(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	assert.Equal(t, map[fwkplugin.DataKey]any{fwkrh.PriorityKey: 0, fwkrh.TenantKey: ""}, p.Produces())
}

func TestRequestHeader_OverSizeLimitDowngradesOneClass(t *testing.T) {
	p, _ := newTestPlugin(t, nil)

	for name, req := range map[string]*fwksched.InferenceRequest{
		"prompt": newRequest("team-a", 2000, 10),
		"output": newRequest("team-a", 10, 500),
	} {
		require.NoError(t, p.RequestHeader(t.Context(), req), name)
		assert.Equal(t, 10, servedPriority(req), "%s above the size limit moves to the next class", name)

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
	assert.Equal(t, 10, servedPriority(over), "over commit moves to the next class")

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
	assert.Equal(t, 100, servedPriority(next), "a completion frees a committed slot")
}

func TestRequestHeader_OverDailyCommitDowngradesOneClass(t *testing.T) {
	p, clock := newTestPlugin(t, func(params *Parameters) {
		params.Teams[0].MaxConcurrency = 0
		params.Teams[0].DailyTokens = 500
	})
	first := newRequest("team-a", 100, 100) // 300
	require.NoError(t, p.RequestHeader(t.Context(), first))
	require.Equal(t, 100, servedPriority(first))

	over := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), over))
	assert.Equal(t, 10, servedPriority(over), "300 + 300 exceeds the 500 daily commit")
	complete(p, over, 100, 100)
	_, daily := usage(p, "team-a")
	assert.InDelta(t, 300, daily, 1e-9, "the downgraded request is not charged to the budget")

	clock.t = clock.t.Add(12 * time.Hour)
	next := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), next))
	assert.Equal(t, 100, servedPriority(next), "the budget resets at the UTC day boundary")
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

// TestSettlementByTerminationCause verifies the charge for each way a dispatched request can
// end. Each request reserves 100 + 2*100 = 300 weighted tokens.
func TestSettlementByTerminationCause(t *testing.T) {
	cases := []struct {
		name     string
		status   string // response status; empty sends no response headers
		cause    fwkrh.TerminationCause
		prompt   int
		output   int
		events   int
		want     float64
		wantNote string
	}{
		{name: "natural with usage", status: "200", cause: fwkrh.TerminationCauseNatural, prompt: 100, output: 20, want: 140, wantNote: "the reported usage"},
		{name: "natural without usage", status: "200", cause: fwkrh.TerminationCauseNatural, events: 50, want: 300, wantNote: "the reservation"},
		{name: "upstream error status", status: "503", cause: fwkrh.TerminationCauseNatural, want: 0, wantNote: "nothing for an error that reports no usage"},
		{name: "upstream error status with usage", status: "400", cause: fwkrh.TerminationCauseNatural, prompt: 100, want: 100, wantNote: "the reported usage"},
		{name: "client disconnect mid-stream", status: "200", cause: fwkrh.TerminationCauseClientDisconnect, events: 10, want: 120, wantNote: "the prompt plus the streamed events"},
		{name: "client disconnect before output", status: "200", cause: fwkrh.TerminationCauseClientDisconnect, want: 0, wantNote: "nothing when nothing was streamed"},
		{name: "client disconnect with usage", status: "200", cause: fwkrh.TerminationCauseClientDisconnect, prompt: 100, output: 90, events: 10, want: 280, wantNote: "the reported usage"},
		{name: "evicted before output", cause: fwkrh.TerminationCauseEvicted, want: 0, wantNote: "nothing when nothing was streamed"},
		{name: "evicted mid-stream", status: "200", cause: fwkrh.TerminationCauseEvicted, events: 5, want: 110, wantNote: "the prompt plus the streamed events"},
		{name: "error before output", cause: fwkrh.TerminationCauseError, want: 0, wantNote: "nothing when nothing was streamed"},
		{name: "error mid-stream", status: "200", cause: fwkrh.TerminationCauseError, events: 5, want: 110, wantNote: "the prompt plus the streamed events"},
		{name: "response not observed", cause: fwkrh.TerminationCauseNotObserved, want: 300, wantNote: "the reservation"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := newTestPlugin(t, nil)
			req := newRequest("team-d", 100, 100)
			require.NoError(t, p.RequestHeader(t.Context(), req))
			if tc.status != "" {
				p.ResponseHeader(t.Context(), req, &fwkrh.Response{Headers: map[string]string{":status": tc.status}}, nil)
			}
			resp := &fwkrh.Response{EndOfStream: true, TerminationCause: tc.cause, StreamedEvents: tc.events}
			resp.Usage.PromptTokens = tc.prompt
			resp.Usage.CompletionTokens = tc.output
			p.ResponseBody(t.Context(), req, resp, nil)

			inflight, daily := usage(p, "team-d")
			assert.Equal(t, 0, inflight, "the slot is released")
			assert.InDelta(t, tc.want, daily, 1e-9, "charged %s", tc.wantNote)
		})
	}
}

func TestIsErrorStatus(t *testing.T) {
	for status, want := range map[string]bool{"200": false, "204": false, "400": true, "429": true, "503": true, "": false, "abc": false} {
		assert.Equal(t, want, isErrorStatus(map[string]string{":status": status}), status)
	}
	assert.True(t, isErrorStatus(map[string]string{"status": "500"}), "the status header is read when :status is absent")
}

func TestRequestAbandonedRefundsInFull(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	inflight, _ := usage(p, "team-d")
	require.Equal(t, 1, inflight)

	p.RequestAbandoned(t.Context(), req)
	inflight, daily := usage(p, "team-d")
	assert.Equal(t, 0, inflight)
	assert.InDelta(t, 0, daily, 1e-9, "a request that was never dispatched is refunded in full")

	complete(p, req, 100, 90)
	_, daily = usage(p, "team-d")
	assert.InDelta(t, 0, daily, 1e-9, "a later end of stream does not settle again")
}

func TestRequestAbandonedAfterCompletionIsNoOp(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	complete(p, req, 100, 100)
	p.RequestAbandoned(t.Context(), req)
	inflight, daily := usage(p, "team-d")
	assert.Equal(t, 0, inflight)
	assert.InDelta(t, 300, daily, 1e-9)
}

// TestDisconnectIsSettledAtEndOfStream verifies that the end of the request's stream context does
// not settle the request: a client that reads a generation and disconnects before the usage
// chunk is still charged when the forced end of stream arrives.
func TestDisconnectIsSettledAtEndOfStream(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	req := newRequest("team-d", 100, 100)
	require.NoError(t, p.RequestHeader(ctx, req))
	cancel()
	inflight, _ := usage(p, "team-d")
	assert.Equal(t, 1, inflight, "the slot is held until end of stream")

	complete(p, req, 100, 90)
	inflight, daily := usage(p, "team-d")
	assert.Equal(t, 0, inflight)
	assert.InDelta(t, 280, daily, 1e-9, "100 + 2*90 weighted tokens served")
}

// TestConcurrentSettlement runs requests that settle by end of stream and by abandonment in
// parallel. Run under -race.
func TestConcurrentSettlement(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) {
		params.Teams = append(params.Teams, TeamConfig{ID: "team-x", Class: "batch", MaxConcurrency: 1000})
	})
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				req := newRequest("team-x", 1, 1)
				if err := p.RequestHeader(t.Context(), req); err != nil {
					continue
				}
				if i%2 == 0 {
					p.RequestAbandoned(t.Context(), req)
				} else {
					complete(p, req, 1, 1)
				}
			}
		}()
	}
	wg.Wait()
	inflight, _ := usage(p, "team-x")
	assert.Equal(t, 0, inflight)
}

// objectivePriority is the priority the InferenceObjective sets in the report-only tests. It is
// sheddable (negative), so a test catches report-only mode making it non-sheddable.
const objectivePriority = -5

// newObjectiveRequest is newRequest with an output cap of 10 and the InferenceObjective priority set.
func newObjectiveRequest(team string, promptTokens int) *fwksched.InferenceRequest {
	req := newRequest(team, promptTokens, 10)
	req.Objectives.Priority = objectivePriority
	return req
}

// requireUnchanged asserts that the plugin produced no priority and no tenant for req.
func requireUnchanged(t *testing.T, req *fwksched.InferenceRequest, msgAndArgs ...any) {
	t.Helper()
	_, priority := req.GetAttribute(fwkrh.PriorityKey)
	_, tenant := req.GetAttribute(fwkrh.TenantKey)
	assert.False(t, priority, msgAndArgs...)
	assert.False(t, tenant, msgAndArgs...)
	assert.Equal(t, objectivePriority, servedPriority(req), msgAndArgs...)
}

// TestReportOnlyLeavesRequestUnchanged verifies that report-only mode records downgrades and
// rejections without applying them, and leaves the request's priority and fairness ID alone.
func TestReportOnlyLeavesRequestUnchanged(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = ModeReportOnly })

	oversized := newObjectiveRequest("team-a", 5000)
	require.NoError(t, p.RequestHeader(t.Context(), oversized), "a would-be downgrade is recorded, not applied")
	requireUnchanged(t, oversized, "team-a keeps its objective priority")
	assert.Empty(t, servedFairnessID(oversized), "the team is not made the fairness ID")

	reqs := make([]*fwksched.InferenceRequest, 0, 5)
	for range 5 {
		req := newObjectiveRequest("team-b", 10)
		require.NoError(t, p.RequestHeader(t.Context(), req), "a would-be rejection is recorded, not applied")
		requireUnchanged(t, req, "team-b keeps its objective priority")
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

// TestOnlyEnforceLimitsSetsTheClassPriority verifies that enforce-limits mode serves each request
// at its team's class or the class below, and that report-only mode keeps the objective
// priority in every case, so a sheddable objective stays sheddable.
func TestOnlyEnforceLimitsSetsTheClassPriority(t *testing.T) {
	cases := []struct {
		name                string
		req                 func() *fwksched.InferenceRequest
		setup               int // team-a requests served within the contract first, to fill its concurrency commit of 2
		enforceLimitsPrio   int
		enforceLimitsErrors bool
	}{
		{name: "critical within contract", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("team-a", 10) }, enforceLimitsPrio: 100},
		{name: "unlisted team within contract", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("stranger", 10) }, enforceLimitsPrio: 10},
		{name: "batch within contract", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("team-b", 10) }, enforceLimitsPrio: -10},
		{name: "critical over size", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("team-a", 5000) }, enforceLimitsPrio: 10},
		{name: "critical over concurrency", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("team-a", 10) }, setup: 2, enforceLimitsPrio: 10},
		{name: "batch over size", req: func() *fwksched.InferenceRequest { return newObjectiveRequest("team-b", 5000) }, enforceLimitsErrors: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []string{ModeEnforceLimits, ModeReportOnly} {
				p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = mode })
				for range tc.setup {
					require.NoError(t, p.RequestHeader(t.Context(), newObjectiveRequest("team-a", 10)))
				}
				req := tc.req()
				err := p.RequestHeader(t.Context(), req)
				if mode == ModeEnforceLimits && tc.enforceLimitsErrors {
					require.Error(t, err)
					continue
				}
				require.NoError(t, err)
				if mode == ModeReportOnly {
					requireUnchanged(t, req, mode)
					continue
				}
				assert.Equal(t, tc.enforceLimitsPrio, servedPriority(req), mode)
			}
		})
	}
}

// TestReportOnlyDecisionsMatchEnforceLimits verifies that report-only mode records the same
// decisions that enforce-limits mode applies, for the same requests.
func TestReportOnlyDecisionsMatchEnforceLimits(t *testing.T) {
	for _, mode := range []string{ModeEnforceLimits, ModeReportOnly} {
		t.Run(mode, func(t *testing.T) {
			p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = mode })
			decide := func(req *fwksched.InferenceRequest, class, servedClass, decision, reason string) {
				t.Helper()
				team := req.Headers[metadata.FlowFairnessIDKey]
				before := decisions(mode, team, class, servedClass, decision, reason)
				_ = p.RequestHeader(t.Context(), req)
				assert.InDelta(t, 1, decisions(mode, team, class, servedClass, decision, reason)-before, 1e-9)
			}
			for range 2 {
				decide(newRequest("team-a", 10, 10), "critical", "critical", decisionWithinContract, "")
			}
			decide(newRequest("team-a", 10, 10), "critical", "standard", decisionDowngraded, reasonConcurrency)
			decide(newRequest("team-a", 5000, 10), "critical", "standard", decisionDowngraded, reasonSize)

			// team-b is in the lowest class with a concurrency commit of 1. A would-be rejection must
			// hold no slot, so once the in-contract request completes the next one is within the contract again.
			first := newRequest("team-b", 10, 10)
			decide(first, "batch", "batch", decisionWithinContract, "")
			decide(newRequest("team-b", 10, 10), "batch", "", decisionRejectedConcurrency, reasonConcurrency)
			complete(p, first, 10, 10)
			decide(newRequest("team-b", 10, 10), "batch", "batch", decisionWithinContract, "")

			decide(newRequest("team-d", 100, 100), "batch", "batch", decisionWithinContract, "")
			decide(newRequest("team-d", 100, 100), "batch", "", decisionRejectedDaily, reasonDaily)
		})
	}
}

// decisions returns the decision counter for one label combination.
func decisions(mode, team, class, servedClass, decision, reason string) float64 {
	return promtestutil.ToFloat64(contractDecisions.WithLabelValues(
		team, class, servedClass, decision, reason, mode))
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
			before := decisions(ModeEnforceLimits, tc.team, class, tc.servedClass, tc.decision, tc.reason)
			_ = p.RequestHeader(t.Context(), tc.req)
			assert.InDelta(t, 1, decisions(ModeEnforceLimits, tc.team, class, tc.servedClass, tc.decision, tc.reason)-before, 1e-9)
		})
	}
}

func TestUntilNextDay(t *testing.T) {
	ist := time.FixedZone("IST", 2*60*60)
	cases := []struct {
		name string
		now  time.Time
		want time.Duration
	}{
		{name: "midday", now: time.Date(2026, 11, 12, 12, 0, 0, 0, time.UTC), want: 12 * time.Hour},
		{name: "at midnight", now: time.Date(2026, 11, 12, 0, 0, 0, 0, time.UTC), want: 24 * time.Hour},
		{name: "end of month", now: time.Date(2026, 11, 30, 23, 59, 30, 0, time.UTC), want: 30 * time.Second},
		{name: "non-UTC location", now: time.Date(2026, 11, 13, 1, 0, 0, 0, ist), want: time.Hour},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, untilNextDay(tc.now))
		})
	}
}
