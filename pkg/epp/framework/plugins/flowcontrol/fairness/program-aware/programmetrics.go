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
