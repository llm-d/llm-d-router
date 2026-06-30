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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func decodeFairnessDump(t *testing.T, p *ProgramAwarePlugin) fairnessState {
	t.Helper()
	raw, err := p.DumpState()
	require.NoError(t, err)
	var state fairnessState
	require.NoError(t, json.Unmarshal(raw, &state))
	return state
}

func TestDumpState_EmptyHasNoPrograms(t *testing.T) {
	state := decodeFairnessDump(t, &ProgramAwarePlugin{})

	assert.Equal(t, 0, state.TotalPrograms)
	assert.Empty(t, state.Programs)
	assert.False(t, state.Truncated)
	assert.Equal(t, maxDebugDumpPrograms, state.MaxPrograms)
}

func TestDumpState_SummarizesPerProgramCounters(t *testing.T) {
	p := &ProgramAwarePlugin{}
	a := p.getOrCreateMetrics("prog-a")
	a.RecordDispatched(time.Time{})
	a.RecordDispatched(time.Time{})
	b := p.getOrCreateMetrics("prog-b")
	b.RecordDispatched(time.Time{})

	state := decodeFairnessDump(t, p)

	assert.Equal(t, 2, state.TotalPrograms)
	assert.False(t, state.Truncated)

	// Sorted by dispatched count, busiest first.
	require.Len(t, state.Programs, 2)
	assert.Equal(t, programState{ProgramID: "prog-a", Dispatched: 2, InFlight: 2, AverageWaitMs: 0}, state.Programs[0])
	assert.Equal(t, programState{ProgramID: "prog-b", Dispatched: 1, InFlight: 1, AverageWaitMs: 0}, state.Programs[1])
}

func TestDumpState_CapsAndFlagsTruncation(t *testing.T) {
	p := &ProgramAwarePlugin{}
	const total = maxDebugDumpPrograms + 50
	for i := 0; i < total; i++ {
		p.getOrCreateMetrics(fmt.Sprintf("prog-%04d", i)).RecordDispatched(time.Time{})
	}

	state := decodeFairnessDump(t, p)

	assert.Equal(t, total, state.TotalPrograms, "total reflects the true count, not the capped sample")
	assert.Len(t, state.Programs, maxDebugDumpPrograms)
	assert.True(t, state.Truncated)
	// Equal dispatched counts, so the sort tiebreaks by program ID.
	assert.Equal(t, "prog-0000", state.Programs[0].ProgramID)
}

func TestProgramAwarePlugin_ImplementsStateDumper(t *testing.T) {
	require.Implements(t, (*interface {
		DumpState() (json.RawMessage, error)
	})(nil), &ProgramAwarePlugin{})
}
