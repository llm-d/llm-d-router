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
	evicted            bool

	// aggregates is shared with every ProgramMetrics of a plugin and is set
	// before the value is published to the program map.
	aggregates *fairnessAggregates

	dispatchedCount atomic.Int64
	inFlight        atomic.Int64
}

// fairnessAggregates maintains the sums behind Jain's fairness index over the
// average wait time of programs with wait observations. RecordDispatched and
// eviction update the sums, so Pick reads the index without visiting every
// tracked program.
type fairnessAggregates struct {
	mu    sync.Mutex
	sum   float64
	sumSq float64
	n     int64
}

// observe applies a change to one program's average wait time. firstWait
// reports whether the program has just started contributing to the index.
func (a *fairnessAggregates) observe(previous, current float64, firstWait bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sum += current - previous
	a.sumSq += current*current - previous*previous
	if firstWait {
		a.n++
	}
}

// forget removes a program's contribution, if any.
func (a *fairnessAggregates) forget(average float64, contributed bool) {
	if !contributed {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sum -= average
	a.sumSq -= average * average
	a.n--
}

func (a *fairnessAggregates) index() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	// Accumulated rounding error can drive sumSq below zero even when every
	// tracked program is equally served.
	if a.sumSq <= 0 {
		return 1.0
	}
	return min(1.0, max(0.0, jainFairnessIndex(a.sum, a.sumSq, float64(a.n))))
}

// reset replaces the accumulated sums. The eviction sweep uses it to discard
// the floating-point error left behind by incremental updates.
func (a *fairnessAggregates) reset(sum, sumSq float64, n int64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sum = sum
	a.sumSq = sumSq
	a.n = n
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
	previous := m.averageWaitTime
	firstWait := m.waitCount == 0
	m.waitCount++
	m.averageWaitTime += (waitMs - m.averageWaitTime) / float64(m.waitCount)
	current := m.averageWaitTime
	track := m.aggregates != nil && !m.evicted
	m.mu.Unlock()
	if track {
		m.aggregates.observe(previous, current, firstWait)
	}
}

// forgetAggregates removes this program's wait contribution from the shared
// fairness aggregates. A later RecordDispatched on the same value does not
// re-add it.
func (m *ProgramMetrics) forgetAggregates() {
	if m.aggregates == nil {
		return
	}
	m.mu.Lock()
	m.evicted = true
	average := m.averageWaitTime
	contributed := m.waitCount > 0
	m.mu.Unlock()
	m.aggregates.forget(average, contributed)
}

func (m *ProgramMetrics) RecordCompletion(now time.Time) {
	m.inFlight.Add(-1)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastCompletionTime = now
}

func (m *ProgramMetrics) DispatchedCount() int64 { return m.dispatchedCount.Load() }
func (m *ProgramMetrics) InFlight() int64        { return m.inFlight.Load() }

// waitStats returns the average wait time and observation count together.
func (m *ProgramMetrics) waitStats() (average float64, count int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageWaitTime, m.waitCount
}

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
