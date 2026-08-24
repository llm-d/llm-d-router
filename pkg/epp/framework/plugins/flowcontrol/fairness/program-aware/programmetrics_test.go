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

package programaware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestProgramMetrics_LastCompletionTime_ZeroBeforeAnyCompletion(t *testing.T) {
	m := &ProgramMetrics{}
	assert.True(t, m.LastCompletionTime().IsZero())
}

func TestProgramMetrics_RecordCompletion_StampsTimeAndDecrementsInFlight(t *testing.T) {
	m := &ProgramMetrics{}
	m.inFlight.Store(1)
	when := time.Date(2026, 6, 4, 12, 0, 0, 0, time.UTC)
	m.RecordCompletion(when)
	assert.Equal(t, when, m.LastCompletionTime())
	assert.Equal(t, int64(0), m.InFlight())
}

func TestProgramMetrics_RecordDispatched_WithEnqueueTime_UpdatesWaitMean(t *testing.T) {
	m := &ProgramMetrics{}
	m.RecordDispatched(time.Now().Add(-50 * time.Millisecond))
	m.RecordDispatched(time.Now().Add(-150 * time.Millisecond))
	assert.Equal(t, int64(2), m.WaitCount())
	assert.InDelta(t, 100.0, m.AverageWaitTime(), 20.0)
	assert.Equal(t, int64(2), m.DispatchedCount())
	assert.Equal(t, int64(2), m.InFlight())
}

func TestProgramMetrics_RecordDispatched_ZeroEnqueueTime_SkipsWaitUpdate(t *testing.T) {
	m := &ProgramMetrics{}
	m.RecordDispatched(time.Time{})
	assert.Equal(t, int64(0), m.WaitCount())
	assert.Equal(t, float64(0), m.AverageWaitTime())
	assert.Equal(t, int64(1), m.DispatchedCount())
	assert.Equal(t, int64(1), m.InFlight())
}
