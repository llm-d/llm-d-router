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

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

func (a *ThunderAgent) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Score pins a session to its reserved or bound pod: that pod scores 1.0 and
// every other candidate 0.0, so a session's turns keep hitting the KV blocks
// its history already occupies.
func (a *ThunderAgent) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	id := sessionID(request)
	if id == "" {
		return nil
	}

	now := a.clock.Now()
	state, hasState := sessionstate.ReadSessionState(request)

	a.mu.Lock()
	l := a.local[id]
	res := resolvedSession{state: state, hasState: hasState}
	if !hasState {
		hint := ""
		if l != nil {
			hint = l.endpointHint
		}
		if s, ok := a.boundSessionLocked(id, hint); ok {
			res = resolvedSession{state: s, hasState: true, isBound: true}
		}
	}
	resumed := false
	if res.hasState && l != nil {
		resumed = a.refreshOneLocked(id, l, res, false, now)
		l = a.local[id]
	}
	target := ""
	switch {
	case res.hasState && l.reservationLive(res.state, now):
		target = l.reservedEndpoint
	case res.hasState:
		target = res.state.Endpoint
	case l != nil && l.reservedUnbound && !l.reservedUntil.IsZero() && now.Before(l.reservedUntil):
		target = l.reservedEndpoint
	}
	a.mu.Unlock()
	if resumed {
		a.metrics.resumes.Inc()
	}
	if target == "" {
		return nil
	}

	var match fwksched.Endpoint
	for _, endpoint := range endpoints {
		if md := endpoint.GetMetadata(); md != nil && md.ID.String() == target {
			match = endpoint
			break
		}
	}
	if match == nil {
		return nil
	}
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0.0
	}
	scores[match] = 1.0
	return scores
}
