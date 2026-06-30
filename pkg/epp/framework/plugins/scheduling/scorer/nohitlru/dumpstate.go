/*
Copyright 2025 The Kubernetes Authors.

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

package nohitlru

import (
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// maxDebugDumpEndpoints bounds the endpoint sample emitted by DumpState so the
// debug payload stays small when many endpoints are tracked.
const maxDebugDumpEndpoints = 100

var _ plugin.StateDumper = &NoHitLRU{}

// noHitLRUState is the sanitized, bounded snapshot returned by DumpState.
type noHitLRUState struct {
	TotalEndpoints int  `json:"totalEndpoints"`
	MaxEndpoints   int  `json:"maxEndpoints"`
	Truncated      bool `json:"truncated"`
	// Endpoints in LRU order, least-recently-used first -- the order the scorer
	// favors for cold requests. Endpoint names only; no request data.
	Endpoints []string `json:"endpoints"`
}

// DumpState implements [plugin.StateDumper] for the /debug/plugins/state
// endpoint, exposing the scorer's LRU of endpoint names. The list is capped to
// keep the payload bounded; TotalEndpoints reports the true count so operators
// can tell when the dump is partial.
func (s *NoHitLRU) DumpState() (json.RawMessage, error) {
	state := noHitLRUState{MaxEndpoints: maxDebugDumpEndpoints}
	if s.lruCache != nil {
		keys := s.lruCache.Keys()
		state.TotalEndpoints = len(keys)
		state.Endpoints = keys
		if len(state.Endpoints) > maxDebugDumpEndpoints {
			state.Endpoints = state.Endpoints[:maxDebugDumpEndpoints]
			state.Truncated = true
		}
	}
	return json.Marshal(state)
}
