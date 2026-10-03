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
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreRequest sets the request's estimated prompt tokens as its session's
// in-flight tokens, binds the session to the picked endpoint.
func (a *ThunderAgent) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	id := sessionID(request)
	if id == "" {
		return nil
	}
	endpoint := pickedEndpoint(schedulingResult)
	if endpoint == nil {
		return nil
	}
	md := endpoint.GetMetadata()
	if md == nil {
		return nil
	}
	capacity := a.endpointCapacity(endpoint.GetMetrics())

	now := time.Now()
	m := a.mgr
	m.mu.Lock()
	m.maintainLocked(now)
	s := m.bindLocked(id, m.ensureEndpointLocked(md.ID.String(), capacity, now))
	s.inflightTokens = estimateTokens(request.RequestSizeBytes)
	s.turnCount++
	s.lastActivity = now
	m.mu.Unlock()
	return nil
}

// ResponseBody updates the session in thunder agent session manager.
func (a *ThunderAgent) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := sessionID(request)
	if id == "" {
		return
	}

	now := time.Now()
	m := a.mgr
	m.mu.Lock()
	m.maintainLocked(now)
	s, ok := m.sessions[id]
	if !ok {
		m.mu.Unlock()
		return
	}
	switch {
	case response.Usage.TotalTokens > 0:
		s.committedTokens = int64(response.Usage.TotalTokens)
	case s.inflightTokens > s.committedTokens:
		s.committedTokens = s.inflightTokens
	}
	// Clear the session inflight token since it's completed.
	s.inflightTokens = 0
	s.lastResponseAt = now
	s.lastActivity = now
	m.mu.Unlock()
}

func pickedEndpoint(schedulingResult *fwksched.SchedulingResult) fwksched.Endpoint {
	if schedulingResult == nil {
		return nil
	}
	result, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 {
		return nil
	}
	return result.TargetEndpoints[0]
}

// endpointCapacity returns a pod's KV cache capacity in tokens.
func (a *ThunderAgent) endpointCapacity(m *datalayer.Metrics) float64 {
	if m != nil && m.CacheBlockSize > 0 && m.CacheNumBlocks > 0 {
		return float64(m.CacheBlockSize) * float64(m.CacheNumBlocks)
	}
	return a.capacityTokens
}
