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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// testConfig uses round numbers so tests can assert exact values: 1000-token
// fallback capacity, byte estimates at 4 bytes per token, the default 30 s
// idle lease.
func testConfig() Config {
	cfg := defaultConfig()
	cfg.CapacityTokens = 1000
	return cfg
}

func newTestAgent(cfg Config) *ThunderAgent {
	return newThunderAgent("test", cfg)
}

// schedEndpoint builds a scheduling endpoint; blockSize and numBlocks of 0
// leave the endpoint without scraped capacity, selecting the fallback.
func schedEndpoint(name string, blockSize, numBlocks int) fwksched.Endpoint {
	nn := types.NamespacedName{Namespace: "default", Name: name}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: nn},
		&fwkdl.Metrics{CacheBlockSize: blockSize, CacheNumBlocks: numBlocks}, nil)
}

func schedulingResultFor(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func newRequest(sessionID string, sizeBytes int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:        "req-" + sessionID,
		FairnessID:       sessionID,
		RequestSizeBytes: sizeBytes,
	}
}

func endOfStream(totalTokens, promptTokens int) *fwkrc.Response {
	return &fwkrc.Response{
		EndOfStream: true,
		Usage:       requesthandling.Usage{TotalTokens: totalTokens, PromptTokens: promptTokens},
	}
}

// startTurn dispatches one turn without completing it.
func startTurn(t *testing.T, a *ThunderAgent, id string, endpoint fwksched.Endpoint, sizeBytes int) *fwksched.InferenceRequest {
	t.Helper()
	req := newRequest(id, sizeBytes)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoint)))
	return req
}

// runTurn dispatches and completes one turn reporting totalTokens.
func runTurn(t *testing.T, a *ThunderAgent, id string, endpoint fwksched.Endpoint, sizeBytes, totalTokens int) {
	t.Helper()
	req := startTurn(t, a, id, endpoint, sizeBytes)
	a.ResponseBody(context.Background(), req, endOfStream(totalTokens, totalTokens), nil)
}

func sessionOf(a *ThunderAgent, id string) (session, bool) {
	a.mgr.mu.Lock()
	defer a.mgr.mu.Unlock()
	s, ok := a.mgr.sessions[id]
	if !ok {
		return session{}, false
	}
	return *s, true
}

func endpointTokens(a *ThunderAgent, id string) float64 {
	a.mgr.mu.Lock()
	defer a.mgr.mu.Unlock()
	p, ok := a.mgr.endpoints[id]
	if !ok {
		return -1
	}
	return p.occupancy(time.Now())
}

// idleFor backdates a session's last response, so it looks idle for d.
func idleFor(a *ThunderAgent, id string, d time.Duration) {
	a.mgr.mu.Lock()
	a.mgr.sessions[id].lastResponseAt = time.Now().Add(-d)
	a.mgr.mu.Unlock()
}

// pause marks a session paused, as a reclaim would.
func pause(a *ThunderAgent, id string) {
	a.mgr.mu.Lock()
	a.mgr.sessions[id].paused = true
	a.mgr.mu.Unlock()
}

// forceMaintenance backdates the rate limiter so the next hook call runs the
// full-table maintenance.
func forceMaintenance(a *ThunderAgent) {
	a.mgr.mu.Lock()
	a.mgr.lastMaintenance = time.Time{}
	a.mgr.mu.Unlock()
}

// dlEndpoint builds a datalayer endpoint without scraped capacity, selecting
// the fallback.
func dlEndpoint(name string) fwkdl.Endpoint {
	nn := types.NamespacedName{Namespace: "default", Name: name}
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: nn}, &fwkdl.Metrics{})
}

// primeFitView refreshes the pod ledger the way the flow controller does
// each dispatch cycle.
func primeFitView(a *ThunderAgent, endpoints ...fwkdl.Endpoint) {
	a.Saturation(context.Background(), endpoints)
}

func isPaused(a *ThunderAgent, id string) bool {
	a.mgr.mu.Lock()
	defer a.mgr.mu.Unlock()
	s, ok := a.mgr.sessions[id]
	return ok && s.paused
}

func makeQueue(id string, headEnqueue time.Time, headBytes uint64) *fwkfcmocks.MockFlowQueueAccessor {
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: fwkfc.FlowKey{ID: id},
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     headEnqueue,
			OriginalRequestV: fwkfcmocks.NewMockFlowControlRequest(headBytes, "req-"+id, fwkfc.FlowKey{ID: id}),
		},
	}
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

// seed runs one 400-byte turn for the session on pod and completes it with
// the given committed total.
func seed(t *testing.T, a *ThunderAgent, id string, pod string, committed int) {
	t.Helper()
	ep := schedEndpoint(pod, 0, 0)
	req := startTurn(t, a, id, ep, 400)
	a.ResponseBody(context.Background(), req, endOfStream(committed, 100), nil)
}

func pick(t *testing.T, a *ThunderAgent, queues ...fwkfc.FlowQueueAccessor) fwkfc.FlowQueueAccessor {
	t.Helper()
	got, err := a.Pick(context.Background(), bandOf(queues...))
	require.NoError(t, err)
	return got
}
