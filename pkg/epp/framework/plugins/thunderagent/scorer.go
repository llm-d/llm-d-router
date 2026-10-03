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
)

func (a *ThunderAgent) Category() fwksched.ScorerCategory {
	return fwksched.Balance
}

// Score pins a session to the pod recorded in the ledger: that pod scores
// 1.0 and every other candidate 0.0, so a session's turns keep hitting the
// KV blocks its history already occupies.
func (a *ThunderAgent) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	id := sessionID(request)
	if id == "" {
		return nil
	}

	m := a.mgr
	m.mu.Lock()
	target := ""
	if s, ok := m.sessions[id]; ok && s.endpoint != nil {
		target = s.endpoint.id
	}
	m.mu.Unlock()
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
