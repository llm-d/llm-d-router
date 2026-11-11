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

package thunderagent

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	clocktesting "k8s.io/utils/clock/testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

// testConfig uses round numbers so tests can assert exact values: 1000-token
// fallback capacity, byte estimates at 4 bytes per token, and the default 30 s
// idleGracePeriod.
func testConfig() Config {
	cfg := defaultConfig()
	cfg.FallbackCapacityTokens = 1000
	return cfg
}

// testLedger simulates session-state-producer's published attributes against a
// controllable clock so unit tests can exercise ThunderAgent deterministically.
type testLedger struct {
	clock    *clocktesting.FakeClock
	agent    *ThunderAgent
	sessions map[string]*sessionstate.SessionState
	pods     map[string]fwkdl.Endpoint
}

func newTestLedger(cfg Config) *testLedger {
	clk := clocktesting.NewFakeClock(time.Date(2026, 11, 11, 12, 0, 0, 0, time.UTC))
	return &testLedger{
		clock:    clk,
		agent:    newThunderAgentWithClock("test", cfg, clk),
		sessions: make(map[string]*sessionstate.SessionState),
		pods:     make(map[string]fwkdl.Endpoint),
	}
}

func (l *testLedger) now() time.Time {
	return l.clock.Now()
}

func (l *testLedger) step(d time.Duration) {
	l.clock.Step(d)
}

func (l *testLedger) pod(name string, blockSize, numBlocks int) fwkdl.Endpoint {
	nn := types.NamespacedName{Namespace: "default", Name: name}
	podID := nn.String()
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: nn}, &fwkdl.Metrics{
		CacheBlockSize: blockSize,
		CacheNumBlocks: numBlocks,
	})
	ep.GetAttributes().Put(
		sessionstate.EndpointSessionsDataKey.WithNonEmptyProducerName(sessionstate.SessionStateProducerType),
		&fwkdl.DynamicAttribute{
			Get: func() fwkdl.Cloneable {
				var out sessionstate.EndpointSessions
				for _, s := range l.sessions {
					if s.Endpoint == podID {
						out = append(out, *s)
					}
				}
				slices.SortFunc(out, func(a, b sessionstate.SessionState) int {
					return strings.Compare(a.SessionID, b.SessionID)
				})
				return out
			},
		},
	)
	l.pods[podID] = ep
	return ep
}

func (l *testLedger) ensurePod(name string) {
	if _, ok := l.pods["default/"+name]; !ok {
		l.pod(name, 0, 0)
	}
}

func (l *testLedger) removePod(name string) {
	podID := "default/" + name
	delete(l.pods, podID)
	for _, s := range l.sessions {
		if s.Endpoint == podID {
			s.Endpoint = ""
		}
	}
}

func (l *testLedger) prime(eps ...fwkdl.Endpoint) {
	l.agent.Saturation(context.Background(), eps)
}

func (l *testLedger) ensureSession(id string) *sessionstate.SessionState {
	s, ok := l.sessions[id]
	if !ok {
		s = &sessionstate.SessionState{SessionID: id}
		l.sessions[id] = s
	}
	return s
}

func (l *testLedger) startTurn(id, podName string, sizeBytes int) {
	l.ensurePod(podName)
	s := l.ensureSession(id)
	s.TurnsTaken++
	s.InFlightRequests++
	s.InFlightTokens += sessionstate.EstimatePromptTokens(sizeBytes)
	s.Endpoint = "default/" + podName
	s.LastSeenAt = l.now()
}

func (l *testLedger) runTurn(id, podName string, sizeBytes, totalTokens int) {
	l.ensurePod(podName)
	s := l.ensureSession(id)
	s.TurnsTaken++
	s.CompletedRequests++
	s.InFlightRequests = 0
	s.InFlightTokens = 0
	s.Endpoint = "default/" + podName
	now := l.now()
	s.LastSeenAt = now
	s.LastResponseAt = now
	if totalTokens > 0 {
		s.ContextTokens = int64(totalTokens)
	} else {
		est := sessionstate.EstimatePromptTokens(sizeBytes)
		if est > s.ContextTokens {
			s.ContextTokens = est
		}
	}
}

func (l *testLedger) seed(id, podName string, committed int) {
	l.runTurn(id, podName, 400, committed)
}

func (l *testLedger) idleFor(id string, d time.Duration) {
	s := l.sessions[id]
	t := l.now().Add(-d)
	s.LastSeenAt = t
	s.LastResponseAt = t
}

func (l *testLedger) evict(id string) {
	delete(l.sessions, id)
}

func (l *testLedger) pause(id string) {
	s := l.sessions[id]
	l.agent.mu.Lock()
	local := l.agent.ensureLocalLocked(id)
	local.paused = true
	local.pausedAtTurns = s.TurnsTaken
	l.agent.mu.Unlock()
}

func (l *testLedger) isPaused(id string) bool {
	s, ok := l.sessions[id]
	if !ok {
		return false
	}
	l.agent.mu.Lock()
	defer l.agent.mu.Unlock()
	return l.agent.local[id].pausedLive(*s)
}

func (l *testLedger) endpointTokens(podName string) float64 {
	snap := l.agent.snapshot()
	ep, ok := snap.endpoints["default/"+podName]
	if !ok {
		return -1
	}
	return ep.workingSet
}

func (l *testLedger) request(id string, sizeBytes int) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		RequestID:        "req-" + id,
		RequestSizeBytes: sizeBytes,
	}
	if id != "" {
		req.PutAttribute(agentidentity.AgentIdentityKey, id)
		if s, ok := l.sessions[id]; ok {
			req.PutAttribute(
				sessionstate.SessionStateDataKey.WithNonEmptyProducerName(sessionstate.SessionStateProducerType),
				*s,
			)
		}
	}
	return req
}

func (l *testLedger) makeQueue(id string, headEnqueue time.Time, headBytes uint64) *fwkfcmocks.MockFlowQueueAccessor {
	return l.makeFlowQueue(id, id, headEnqueue, headBytes)
}

// makeFlowQueue builds a one-item queue named flowID whose head request
// carries the given agent-identity session; an empty session leaves it
// anonymous.
func (l *testLedger) makeFlowQueue(flowID, session string, headEnqueue time.Time, headBytes uint64) *fwkfcmocks.MockFlowQueueAccessor {
	req := fwkfcmocks.NewMockFlowControlRequest(headBytes, "req-"+flowID, fwkfc.FlowKey{ID: flowID})
	req.InferenceRequestV = l.request(session, int(headBytes))
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: fwkfc.FlowKey{ID: flowID},
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     headEnqueue,
			OriginalRequestV: req,
		},
	}
}

func schedEndpoint(name string) fwksched.Endpoint {
	nn := types.NamespacedName{Namespace: "default", Name: name}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: nn}, &fwkdl.Metrics{}, nil)
}

func bandOf(queues ...fwkfc.FlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(callback func(flow fwkfc.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !callback(q) {
					return
				}
			}
		},
	}
}

func pick(t *testing.T, a *ThunderAgent, queues ...fwkfc.FlowQueueAccessor) fwkfc.FlowQueueAccessor {
	t.Helper()
	got, err := a.Pick(context.Background(), bandOf(queues...))
	require.NoError(t, err)
	return got
}
