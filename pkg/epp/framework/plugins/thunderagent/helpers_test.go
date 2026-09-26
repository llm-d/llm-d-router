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

// testConfig returns a config with round numbers and no growth buffer so
// tests can assert exact utilization values.
func testConfig() Config {
	cfg := defaultConfig()
	cfg.CapacityTokens = 1000
	cfg.BufferTokensPerProgram = 0
	return cfg
}

func newTestAgent(cfg Config) *ThunderAgent {
	return newThunderAgent("test", cfg)
}

func newTestEndpoints(names ...string) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, 0, len(names))
	for _, name := range names {
		nn := types.NamespacedName{Namespace: "default", Name: name}
		endpoints = append(endpoints, fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: nn}, &fwkdl.Metrics{}, nil))
	}
	return endpoints
}

func dlEndpoints(names ...string) []fwkdl.Endpoint {
	endpoints := make([]fwkdl.Endpoint, 0, len(names))
	for _, name := range names {
		endpoints = append(endpoints, fwkdl.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: name}}, &fwkdl.Metrics{}))
	}
	return endpoints
}

func schedulingResultFor(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func newRequest(programID string, sizeBytes int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:        "req-" + programID,
		FairnessID:       programID,
		RequestSizeBytes: sizeBytes,
	}
}

func endOfStream(totalTokens, promptTokens int) *fwkrc.Response {
	return &fwkrc.Response{
		EndOfStream: true,
		Usage:       requesthandling.Usage{TotalTokens: totalTokens, PromptTokens: promptTokens},
	}
}

// seedProgram runs one full turn for a program on the given endpoint, leaving
// it dispatched (REASONING class) with totalTokens committed.
func seedProgram(t *testing.T, a *ThunderAgent, id string, endpoint fwksched.Endpoint, totalTokens int) {
	t.Helper()
	req := newRequest(id, 0)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoint)))
	a.ResponseBody(context.Background(), req, endOfStream(totalTokens, totalTokens), nil)
}

func makeQueue(id string, length int, headEnqueue time.Time) *fwkfcmocks.MockFlowQueueAccessor {
	return makeQueueWithBytes(id, length, headEnqueue, 0)
}

func makeQueueWithBytes(id string, length int, headEnqueue time.Time, headBytes uint64) *fwkfcmocks.MockFlowQueueAccessor {
	q := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     length,
		FlowKeyV: fwkfc.FlowKey{ID: id},
	}
	if length > 0 {
		q.PeekV = &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV:     headEnqueue,
			OriginalRequestV: fwkfcmocks.NewMockFlowControlRequest(headBytes, "req-"+id, fwkfc.FlowKey{ID: id}),
		}
	}
	return q
}

// primeFitView refreshes the plugin's per-pod fit view the way the flow
// controller does each dispatch cycle.
func primeFitView(a *ThunderAgent, pods ...string) {
	a.Saturation(context.Background(), dlEndpoints(pods...))
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
