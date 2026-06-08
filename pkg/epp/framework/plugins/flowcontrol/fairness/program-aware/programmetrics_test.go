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
