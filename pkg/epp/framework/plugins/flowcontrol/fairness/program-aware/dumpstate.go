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

package programaware

import (
	"encoding/json"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// maxDebugDumpPrograms bounds the per-program sample emitted by DumpState so the
// debug payload stays small when many programs are tracked.
const maxDebugDumpPrograms = 100

var _ plugin.StateDumper = &ProgramAwarePlugin{}

// fairnessState is the sanitized, bounded snapshot returned by DumpState.
type fairnessState struct {
	TotalPrograms int            `json:"totalPrograms"`
	MaxPrograms   int            `json:"maxPrograms"`
	Truncated     bool           `json:"truncated"`
	Programs      []programState `json:"programs"`
}

type programState struct {
	ProgramID     string  `json:"programID"`
	Dispatched    int64   `json:"dispatched"`
	InFlight      int64   `json:"inFlight"`
	AverageWaitMs float64 `json:"averageWaitMs"`
}

// DumpState implements [plugin.StateDumper] for the /debug/plugins/state
// endpoint, exposing per-program fairness counters. The program list is sorted
// by dispatched count and capped; TotalPrograms reports the true count so
// operators can tell when the dump is partial.
func (p *ProgramAwarePlugin) DumpState() (json.RawMessage, error) {
	state := fairnessState{MaxPrograms: maxDebugDumpPrograms}
	p.programMetrics.Range(func(key, value any) bool {
		id, _ := key.(string)
		m, ok := value.(*ProgramMetrics)
		if !ok {
			return true
		}
		state.Programs = append(state.Programs, programState{
			ProgramID:     id,
			Dispatched:    m.DispatchedCount(),
			InFlight:      m.InFlight(),
			AverageWaitMs: m.AverageWaitTime(),
		})
		return true
	})
	state.TotalPrograms = len(state.Programs)

	sort.Slice(state.Programs, func(a, b int) bool {
		if state.Programs[a].Dispatched != state.Programs[b].Dispatched {
			return state.Programs[a].Dispatched > state.Programs[b].Dispatched
		}
		return state.Programs[a].ProgramID < state.Programs[b].ProgramID
	})
	if len(state.Programs) > maxDebugDumpPrograms {
		state.Programs = state.Programs[:maxDebugDumpPrograms]
		state.Truncated = true
	}
	return json.Marshal(state)
}
