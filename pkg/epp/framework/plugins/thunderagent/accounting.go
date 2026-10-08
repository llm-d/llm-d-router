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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreRequest adds the request's estimated prompt tokens to its session's
// in-flight tokens, binds the session to the picked endpoint, and stashes the
// estimate on the request for ResponseBody to remove. A session can have
// several turns in flight at once (parallel sub-agents), so each turn charges
// and later removes only its own estimate.
func (a *ThunderAgent) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
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
	movedFrom := ""
	if prev, ok := m.sessions[id]; ok && prev.class() == classAdmitted && prev.endpoint.id != md.ID.String() {
		movedFrom = prev.endpoint.id
	}
	s := m.bindLocked(id, m.ensureEndpointLocked(md.ID.String(), capacity, now))
	resumed := s.paused
	s.paused = false
	s.reservedUntil = time.Time{}
	// At least one token, so inflightTokens is nonzero exactly while a turn
	// is in flight.
	estimate := max(estimateTokens(request.RequestSizeBytes), 1)
	s.inflightTokens += estimate
	s.turnCount++
	s.lastActivity = now
	m.mu.Unlock()

	request.PutAttribute(inflightEstimateKey, estimate)

	if resumed {
		a.metrics.resumes.Inc()
	}
	if movedFrom != "" {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Session moved off its pod: the pod was filtered out, or other scorers outweigh thunder-agent",
			"fromPod", movedFrom, "toPod", md.ID.String())
	}
	return nil
}

// ResponseBody settles a completed turn: it removes the turn's own in-flight
// estimate and replaces the session's committed tokens with the usage total.
// Requests PreRequest did not charge are ignored.
func (a *ThunderAgent) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := sessionID(request)
	if id == "" {
		return
	}
	estimate, ok := fwksched.ReadRequestAttribute[int64](request, inflightEstimateKey)
	if !ok {
		return
	}

	now := time.Now()
	m := a.mgr
	m.mu.Lock()
	s, ok := m.sessions[id]
	if !ok {
		m.mu.Unlock()
		return
	}
	s.inflightTokens = max(s.inflightTokens-estimate, 0)
	switch {
	case response.Usage.TotalTokens > 0:
		s.committedTokens = int64(response.Usage.TotalTokens)
	case estimate > s.committedTokens:
		s.committedTokens = estimate
	}
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
