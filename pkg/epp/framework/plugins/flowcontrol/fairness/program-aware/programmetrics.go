package programaware

import (
	"sync"
	"sync/atomic"
	"time"
)

type ProgramMetrics struct {
	// mu guards the fields below.
	mu                 sync.Mutex
	averageWaitTime    float64
	waitCount          int64
	lastCompletionTime time.Time

	dispatchedCount atomic.Int64
	inFlight        atomic.Int64
}

// RecordDispatched accepts a zero enqueueTime when no queue wait was observed.
func (m *ProgramMetrics) RecordDispatched(enqueueTime time.Time) {
	m.inFlight.Add(1)
	m.dispatchedCount.Add(1)
	if enqueueTime.IsZero() {
		return
	}
	waitMs := float64(time.Since(enqueueTime).Milliseconds())
	m.mu.Lock()
	defer m.mu.Unlock()
	m.waitCount++
	m.averageWaitTime += (waitMs - m.averageWaitTime) / float64(m.waitCount)
}

func (m *ProgramMetrics) RecordCompletion(now time.Time) {
	m.inFlight.Add(-1)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastCompletionTime = now
}

func (m *ProgramMetrics) DispatchedCount() int64 { return m.dispatchedCount.Load() }
func (m *ProgramMetrics) InFlight() int64        { return m.inFlight.Load() }

func (m *ProgramMetrics) AverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageWaitTime
}

func (m *ProgramMetrics) WaitCount() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.waitCount
}

func (m *ProgramMetrics) LastCompletionTime() time.Time {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastCompletionTime
}
