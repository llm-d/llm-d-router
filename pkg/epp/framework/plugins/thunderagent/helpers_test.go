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

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// testConfig uses round numbers so tests can assert exact values: 1000-token
// fallback capacity, byte estimates at 4 bytes per token.
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

// newRequest builds a request whose session id is published the way the
// agent-identity plugin does; an empty sessionID leaves the request
// anonymous.
func newRequest(sessionID string, sizeBytes int) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		RequestID:        "req-" + sessionID,
		RequestSizeBytes: sizeBytes,
	}
	if sessionID != "" {
		req.PutAttribute(agentidentity.AgentIdentityKey, sessionID)
	}
	return req
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
	return p.workingSetTokens()
}
