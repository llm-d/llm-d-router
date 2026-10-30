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

package qosadmission

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time { return c.t }

func testParams() Parameters {
	p := defaultParameters()
	p.Classes = []ClassConfig{
		{Name: "critical", Priority: 100},
		{Name: "standard", Priority: 10},
		{Name: "batch", Priority: -10},
	}
	p.DefaultTeam = TeamConfig{Class: "standard"}
	p.Teams = []TeamConfig{
		// 6000 tokens/min = 100/s, 10s burst = 1000 tokens.
		{ID: "team-a", Class: "critical", CommitTokensPerMinute: 6000, BurstSeconds: 10, MaxConcurrency: 2, MaxPromptTokens: 1000, MaxOutputTokens: 100},
		{ID: "team-b", Class: "batch", CommitTokensPerMinute: 6000, BurstSeconds: 10},
		{ID: "team-d", Class: "standard", DailyTokens: 500},
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
		RequestSizeBytes: promptTokens * fwkrh.BytesPerToken,
		Body:             &fwkrh.InferenceRequestBody{MaxOutputTokens: &maxOut},
	}
}

func complete(p *Plugin, req *fwksched.InferenceRequest, prompt, completion int) {
	resp := &fwkrc.Response{EndOfStream: true}
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
		"teams":[{"id":"a","class":"critical","commitTokensPerMinute":1000}]}`
	got, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.NoError(t, err)
	p := got.(*Plugin)
	assert.Equal(t, "qos", p.TypedName().Name)
	assert.Equal(t, float64(defaultBurstSeconds), p.teams["a"].cfg.BurstSeconds)
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
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("qos", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			require.Error(t, err)
		})
	}
}

func TestRequestHeader_AdmitsAtCommittedClass(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 50)

	require.NoError(t, p.RequestHeader(t.Context(), req))

	assert.Equal(t, 100, req.Objectives.Priority)
	assert.Equal(t, "team-a", req.FairnessID)
	// reserve = 100 + 2*50 = 200 of 1000
	assert.InDelta(t, 800, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
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

func TestRequestHeader_SizeCaps(t *testing.T) {
	p, _ := newTestPlugin(t, nil)

	err := p.RequestHeader(t.Context(), newRequest("team-a", 2000, 10))
	e := requireCode(t, err, errcommon.BadRequest)
	assert.Equal(t, string(errcommon.RequestDroppedReasonSizeLimit), e.Headers[errcommon.RequestDroppedReasonHeaderKey])

	err = p.RequestHeader(t.Context(), newRequest("team-a", 10, 500))
	_ = requireCode(t, err, errcommon.BadRequest)

	assert.Equal(t, 0, p.teams["team-a"].inflight, "a rejected request holds no slot")
}

func TestRequestHeader_MissingOutputCapReservesTeamMax(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 0)
	req.Body.MaxOutputTokens = nil
	require.NoError(t, p.RequestHeader(t.Context(), req))
	// reserve = 100 + 2*100 (team maxOutputTokens)
	assert.InDelta(t, 700, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
}

func TestRequestHeader_ConcurrencyCapRejects(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	r1, r2, r3 := newRequest("team-a", 10, 10), newRequest("team-a", 10, 10), newRequest("team-a", 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), r1))
	require.NoError(t, p.RequestHeader(t.Context(), r2))

	e := requireCode(t, p.RequestHeader(t.Context(), r3), errcommon.ResourceExhausted)
	assert.Equal(t, "1", e.Headers[RetryAfterHeader])
	assert.Equal(t, "0", e.Headers[ConcurrencyRemainingHeader])
	assert.Equal(t, "critical", e.Headers[ClassHeader])

	complete(p, r1, 10, 10)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-a", 10, 10)), "a completion frees a slot")
}

func TestRequestHeader_OverCommitDowngradesOneClass(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) { params.Teams[0].MaxConcurrency = 0 })
	// Each reserves 100 + 2*100 = 300; the 1000-token bucket admits three.
	for range 3 {
		req := newRequest("team-a", 100, 100)
		require.NoError(t, p.RequestHeader(t.Context(), req))
		require.Equal(t, 100, req.Objectives.Priority)
	}
	over := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), over))
	assert.Equal(t, 10, over.Objectives.Priority, "over commit moves to the next class")

	resp := &fwkrc.Response{Headers: map[string]string{}}
	p.ResponseHeader(t.Context(), over, resp, nil)
	assert.Equal(t, "standard", resp.Headers[ClassHeader])
	assert.Equal(t, "100", resp.Headers[RemainingTokensHeader])
	assert.Equal(t, "9", resp.Headers[ResetTokensHeader])

	// The downgraded request was not charged, so its refund must not credit the bucket.
	complete(p, over, 0, 0)
	assert.InDelta(t, 100, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
}

func TestRequestHeader_LowestClassOverCommitRejects(t *testing.T) {
	p, clock := newTestPlugin(t, nil)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-b", 500, 250)))

	e := requireCode(t, p.RequestHeader(t.Context(), newRequest("team-b", 100, 50)), errcommon.ResourceExhausted)
	assert.Equal(t, string(errcommon.RequestDroppedReasonOverQuota), e.Headers[errcommon.RequestDroppedReasonHeaderKey])
	assert.Equal(t, "batch", e.Headers[ClassHeader])
	assert.Equal(t, "2", e.Headers[RetryAfterHeader], "200 tokens at 100/s")
	assert.Equal(t, "0", e.Headers[RemainingTokensHeader])

	clock.t = clock.t.Add(2 * time.Second)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-b", 100, 50)))
}

func TestResponseBody_RefundsUnusedReservation(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	require.InDelta(t, 700, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)

	complete(p, req, 100, 20) // actual = 100 + 2*20 = 140; refund 160
	assert.InDelta(t, 860, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
	assert.Equal(t, 0, p.teams["team-a"].inflight)

	complete(p, req, 100, 20)
	assert.InDelta(t, 860, p.teams["team-a"].bucket.remaining(p.now()), 1e-9, "settlement happens once")
}

func TestResponseBody_MissingUsageKeepsReservation(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	complete(p, req, 0, 0)
	assert.InDelta(t, 700, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
}

func TestResponseBody_UsageAboveReservationIsDebt(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	req := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(t.Context(), req))
	complete(p, req, 900, 400) // actual 1700 against a reserve of 300: 700 - 1400 = -700
	assert.InDelta(t, 0, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
	assert.Equal(t, 7*time.Second, p.teams["team-a"].bucket.waitFor(0, p.now()))
}

func TestStreamEndReleasesUncompletedRequest(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	req := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(ctx, req))
	require.Equal(t, 1, p.teams["team-a"].inflight)

	cancel()
	require.Eventually(t, func() bool {
		st := p.teams["team-a"]
		st.mu.Lock()
		defer st.mu.Unlock()
		return st.inflight == 0
	}, time.Second, 5*time.Millisecond)
	assert.InDelta(t, 1000, p.teams["team-a"].bucket.remaining(p.now()), 1e-9, "an unserved request is refunded in full")
}

func TestCompletedRequestIsNotReleasedAgainOnStreamEnd(t *testing.T) {
	p, _ := newTestPlugin(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	req := newRequest("team-a", 100, 100)
	require.NoError(t, p.RequestHeader(ctx, req))
	complete(p, req, 100, 100)
	cancel()
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, 0, p.teams["team-a"].inflight)
	assert.InDelta(t, 700, p.teams["team-a"].bucket.remaining(p.now()), 1e-9)
}

func TestDailyBudget(t *testing.T) {
	p, clock := newTestPlugin(t, nil)
	// default reservation is 1000 output tokens; use explicit caps to stay small.
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100))) // 300
	e := requireCode(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100)), errcommon.ResourceExhausted)
	assert.Equal(t, "43200", e.Headers[RetryAfterHeader], "noon UTC waits 12h for the next day")

	clock.t = clock.t.Add(12 * time.Hour)
	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-d", 100, 100)), "the budget resets at the UTC day boundary")
}

func TestShadowModeNeverActs(t *testing.T) {
	p, _ := newTestPlugin(t, func(params *Parameters) { params.Mode = ModeShadow })

	require.NoError(t, p.RequestHeader(t.Context(), newRequest("team-a", 5000, 10)), "size caps are recorded, not enforced")
	for range 5 {
		req := newRequest("team-b", 500, 250)
		require.NoError(t, p.RequestHeader(t.Context(), req))
		assert.Equal(t, 0, req.Objectives.Priority, "shadow mode leaves the priority alone")
		resp := &fwkrc.Response{Headers: map[string]string{}}
		p.ResponseHeader(t.Context(), req, resp, nil)
		assert.Empty(t, resp.Headers)
		complete(p, req, 500, 250)
	}
	assert.Equal(t, 0, p.teams["team-b"].inflight)
}
