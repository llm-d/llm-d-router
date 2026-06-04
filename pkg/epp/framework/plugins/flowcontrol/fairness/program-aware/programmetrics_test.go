package programaware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestProgramMetrics_LastCompletionTime_ZeroBeforeAnyCompletion(t *testing.T) {
	m := &ProgramMetrics{}
	assert.True(t, m.LastCompletionTime().IsZero(),
		"a fresh ProgramMetrics has no completion time")
}

func TestProgramMetrics_LastCompletionTime_RecordedOnFirstCompletion(t *testing.T) {
	m := &ProgramMetrics{}
	when := time.Date(2026, 6, 4, 12, 0, 0, 0, time.UTC)
	m.RecordServiceRate(100.0, when)
	assert.Equal(t, when, m.LastCompletionTime())
}

func TestProgramMetrics_LastCompletionTime_AdvancesOnSubsequentCompletion(t *testing.T) {
	m := &ProgramMetrics{}
	first := time.Date(2026, 6, 4, 12, 0, 0, 0, time.UTC)
	second := first.Add(5 * time.Second)
	m.RecordServiceRate(100.0, first)
	m.RecordServiceRate(200.0, second)
	assert.Equal(t, second, m.LastCompletionTime())
}
