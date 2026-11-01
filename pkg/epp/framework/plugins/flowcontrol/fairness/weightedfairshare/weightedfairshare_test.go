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

package weightedfairshare

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var _ fwkplugin.Plugin = &weightedFairShare{}

// newPolicy builds a policy through the factory from a JSON parameters blob ("" for defaults).
func newPolicy(t *testing.T, params string) *weightedFairShare {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(bytes.NewReader([]byte(params)))
	}
	p, err := WeightedFairShareFairnessPolicyFactory("test", dec, nil)
	require.NoError(t, err)
	return p.(*weightedFairShare)
}

// queue is a one-request queue for flow id at the given priority.
func queue(id string, priority int, enqueued time.Time, headers map[string]string) flowcontrol.FlowQueueAccessor {
	key := flowcontrol.FlowKey{ID: id, Priority: priority}
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: key,
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV: enqueued,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{
				FlowKeyV:          key,
				InferenceRequestV: &fwksched.InferenceRequest{FairnessID: id, Headers: headers},
			},
		},
	}
}

// band is a priority band holding queues. Pass the same state to keep it across picks.
func band(state *minVirtualService, queues ...flowcontrol.FlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		PolicyStateV: state,
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !cb(q) {
					return
				}
			}
		},
	}
}

// complete reports a finished request for flow id, as the model server's usage block would.
func complete(p *weightedFairShare, id string, priority, promptTokens, completionTokens int) {
	req := &fwksched.InferenceRequest{FairnessID: id, Objectives: fwksched.RequestObjectives{Priority: priority}}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = promptTokens
	resp.Usage.CompletionTokens = completionTokens
	p.ResponseBody(context.Background(), req, resp, nil)
}

// pickAndComplete runs n dispatches, completing each picked request with 100 prompt tokens,
// and returns how many times each flow was picked.
func pickAndComplete(t *testing.T, p *weightedFairShare, b flowcontrol.PriorityBandAccessor, n int) map[string]int {
	t.Helper()
	served := map[string]int{}
	for range n {
		got, err := p.Pick(context.Background(), b)
		require.NoError(t, err)
		require.NotNil(t, got)
		served[got.FlowKey().ID]++
		complete(p, got.FlowKey().ID, 0, 100, 0)
	}
	return served
}

func virtualService(p *weightedFairShare, id string) float64 {
	return p.stateFor(flowcontrol.FlowKey{ID: id}).virtualService
}

func TestConfig_Defaults(t *testing.T) {
	cfg := newPolicy(t, "").cfg

	assert.Equal(t, 1.0, cfg.DefaultFlowWeight)
	assert.Equal(t, []float64{0, 1, 2}, []float64{cfg.CostPerRequest, cfg.CostPerInputToken, cfg.CostPerOutputToken})
}

func TestConfig_RejectsInvalid(t *testing.T) {
	for name, params := range map[string]string{
		"negative idle TTL":    `{"idleTtlSeconds":-1}`,
		"zero sweep interval":  `{"sweepSeconds":0}`,
		"negative cost":        `{"costPerOutputToken":-1}`,
		"all-zero cost":        `{"costPerRequest":0,"costPerInputToken":0,"costPerOutputToken":0}`,
		"zero flow weight":     `{"flowWeights":{"team-a":0}}`,
		"negative flow weight": `{"flowWeights":{"team-a":-3}}`,
		"zero default weight":  `{"defaultFlowWeight":0}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := WeightedFairShareFairnessPolicyFactory("x", json.NewDecoder(bytes.NewReader([]byte(params))), nil)
			require.Error(t, err)
		})
	}
}

// The cost of a completed request is costPerRequest + costPerInputToken*prompt +
// costPerOutputToken*completion. At weight 1 the flow's virtual service equals that cost.
func TestCost(t *testing.T) {
	for name, tc := range map[string]struct {
		params             string
		prompt, completion int
		want               float64
	}{
		"default 0/1/2":      {"", 100, 10, 120},
		"output tokens only": {`{"costPerInputToken":0,"costPerOutputToken":1}`, 100000, 250, 250},
		"flat per request":   {`{"costPerRequest":1,"costPerInputToken":0,"costPerOutputToken":0}`, 100000, 5000, 1},
		"blend of all three": {`{"costPerRequest":50,"costPerInputToken":0.5,"costPerOutputToken":3}`, 200, 10, 180},
	} {
		t.Run(name, func(t *testing.T) {
			p := newPolicy(t, tc.params)
			complete(p, "alpha", 0, tc.prompt, tc.completion)
			assert.InDelta(t, tc.want, virtualService(p, "alpha"), 1e-9)
		})
	}
}

// Weights come from flowWeights by fairness ID, unlisted IDs take defaultFlowWeight, and
// nothing on the request changes them.
func TestWeight_FromConfigOnly(t *testing.T) {
	p := newPolicy(t, `{"flowWeights":{"team-a":7,"team-b":3},"defaultFlowWeight":2}`)

	assert.Equal(t, 7.0, p.weightFor(flowcontrol.FlowKey{ID: "team-a"}))
	assert.Equal(t, 3.0, p.weightFor(flowcontrol.FlowKey{ID: "team-b"}))
	assert.Equal(t, 2.0, p.weightFor(flowcontrol.FlowKey{ID: "unlisted"}))
	assert.Equal(t, 7.0, p.weightFor(flowcontrol.FlowKey{ID: "team-a", Priority: -10}), "weight does not depend on priority")

	forged := map[string]string{"x-llm-d-inference-fairness-weight": "999"}
	_, err := p.Pick(context.Background(), band(&minVirtualService{}, queue("team-a", 0, time.Now(), forged)))
	require.NoError(t, err)
	assert.Equal(t, 7.0, testutil.ToFloat64(p.weightGauge.WithLabelValues("team-a", "0")))
}

// Completed cost is divided by weight: 400 tokens at weight 4 is virtual service 100, and the
// gauge publishes it.
func TestResponseBody_PublishesVirtualService(t *testing.T) {
	p := newPolicy(t, `{"flowWeights":{"alpha":4},"costPerOutputToken":0}`)

	complete(p, "alpha", 3, 400, 0)

	assert.InDelta(t, 100.0, testutil.ToFloat64(p.vserviceGauge.WithLabelValues("alpha", "3")), 1e-9)
}

// alpha (weight 3, cost 300) is at 100 and beta (weight 1, cost 150) at 150, so alpha is picked
// even though its raw cost is higher.
func TestPick_SmallestVirtualServiceWins(t *testing.T) {
	p := newPolicy(t, `{"flowWeights":{"alpha":3,"beta":1}}`)
	complete(p, "alpha", 0, 300, 0)
	complete(p, "beta", 0, 150, 0)
	now := time.Now()

	got, err := p.Pick(context.Background(), band(&minVirtualService{}, queue("alpha", 0, now, nil), queue("beta", 0, now, nil)))

	require.NoError(t, err)
	assert.Equal(t, "alpha", got.FlowKey().ID)
}

// Among flows with the same virtual service, the one whose head request waited longest wins.
// A flow further ahead loses even if it waited longer still.
func TestPick_HeadWaitOnlyBreaksTies(t *testing.T) {
	p := newPolicy(t, "")
	complete(p, "ahead", 0, 200, 0)
	complete(p, "tied-new", 0, 100, 0)
	complete(p, "tied-old", 0, 100, 0)
	now := time.Now()

	got, err := p.Pick(context.Background(), band(&minVirtualService{},
		queue("ahead", 0, now.Add(-time.Hour), nil),
		queue("tied-new", 0, now, nil),
		queue("tied-old", 0, now.Add(-time.Minute), nil)))

	require.NoError(t, err)
	assert.Equal(t, "tied-old", got.FlowKey().ID)
}

// With equal-cost requests, picks split 4:2:1 for weights 4:2:1.
func TestPick_ConvergesToWeightRatio(t *testing.T) {
	p := newPolicy(t, `{"flowWeights":{"big":4,"mid":2,"small":1}}`)
	now := time.Now()
	b := band(&minVirtualService{}, queue("big", 0, now, nil), queue("mid", 0, now, nil), queue("small", 0, now, nil))

	served := pickAndComplete(t, p, b, 700)

	assert.InDelta(t, 400, served["big"], 2)
	assert.InDelta(t, 200, served["mid"], 2)
	assert.InDelta(t, 100, served["small"], 2)
}

// "busy" is served alone for 100 requests. A flow that then joins (new, or idle since early on)
// starts at the band's minimum virtual service instead of far behind, so it shares the next 20
// picks evenly rather than taking all of them.
func TestPick_JoiningFlowDoesNotBurst(t *testing.T) {
	for name, idleEarlier := range map[string]bool{"new flow": false, "flow returning from idle": true} {
		t.Run(name, func(t *testing.T) {
			p := newPolicy(t, "")
			now := time.Now()
			state := &minVirtualService{}
			if idleEarlier {
				complete(p, "joiner", 0, 100, 0)
			}
			pickAndComplete(t, p, band(state, queue("busy", 0, now, nil)), 100)

			served := pickAndComplete(t, p, band(state, queue("busy", 0, now, nil), queue("joiner", 0, now, nil)), 20)

			assert.InDelta(t, 10, served["joiner"], 1, "served: %v", served)
		})
	}
}

// When Pick raises a joining flow, the gauge shows the raised value before any of the
// flow's requests complete.
func TestPick_PublishesRaise(t *testing.T) {
	p := newPolicy(t, "")
	now := time.Now()
	state := &minVirtualService{}
	pickAndComplete(t, p, band(state, queue("busy", 0, now, nil)), 10)

	_, err := p.Pick(context.Background(), band(state, queue("busy", 0, now, nil), queue("joiner", 0, now, nil)))

	require.NoError(t, err)
	assert.InDelta(t, state.virtualService, testutil.ToFloat64(p.vserviceGauge.WithLabelValues("joiner", "0")), 1e-9)
	assert.Positive(t, state.virtualService)
}

// Raising a flow to the minimum never lowers one that is ahead of it.
func TestCatchUp_NeverLowers(t *testing.T) {
	s := &flowState{}
	s.charge(900, time.Now())

	vs, raised := s.catchUp(100, time.Now())
	assert.InDelta(t, 900.0, vs, 1e-9)
	assert.False(t, raised)
}

// A tenant's flows at different priorities have separate virtual service, so heavy use in band 0
// does not count against it in band 10.
func TestPick_BandsAreSeparate(t *testing.T) {
	p := newPolicy(t, "")
	complete(p, "tenant", 0, 500, 0)
	complete(p, "other", 10, 100, 0)
	now := time.Now()

	got, err := p.Pick(context.Background(), band(&minVirtualService{}, queue("tenant", 10, now, nil), queue("other", 10, now, nil)))

	require.NoError(t, err)
	assert.Equal(t, "tenant", got.FlowKey().ID)
}

func TestPick_NoCandidates(t *testing.T) {
	p := newPolicy(t, "")

	got, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got, "nil band")

	empty := &fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "drained"}}
	got, err = p.Pick(context.Background(), band(&minVirtualService{}, empty))
	require.NoError(t, err)
	assert.Nil(t, got, "empty queue")

	_, err = p.Pick(context.Background(), band(nil, queue("alpha", 0, time.Now(), nil)))
	require.Error(t, err, "band without policy state")
}

// Flows untouched for longer than idleTtlSeconds are dropped.
func TestPrune_DropsIdleFlows(t *testing.T) {
	p := newPolicy(t, `{"idleTtlSeconds":100}`)
	later := time.Now().Add(150 * time.Second)
	complete(p, "stale", 0, 10, 0)
	complete(p, "fresh", 0, 10, 0)
	_, _ = p.stateFor(flowcontrol.FlowKey{ID: "fresh"}).catchUp(0, later)

	p.prune(later)

	_, staleOK := p.state.Load(flowcontrol.FlowKey{ID: "stale"})
	_, freshOK := p.state.Load(flowcontrol.FlowKey{ID: "fresh"})
	assert.False(t, staleOK)
	assert.True(t, freshOK)
}

// One instance per priority band is normal, so two instances must register their collectors in
// one registry without colliding.
func TestTwoInstances_ShareARegistry(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(),
		func() []types.NamespacedName { return nil }, fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))

	_, err := WeightedFairShareFairnessPolicyFactory("band-a", nil, handle)
	require.NoError(t, err)
	_, err = WeightedFairShareFairnessPolicyFactory("band-b", nil, handle)
	require.NoError(t, err)
}
