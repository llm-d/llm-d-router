package programaware

import (
	"sync"
	"sync/atomic"
	"time"
)

const ewmaAlpha = 0.5

// Token cost weights: output tokens are ~2× more expensive than input tokens
const (
	weightInputToken  = 1
	weightOutputToken = 2
)

// ProgramMetrics holds aggregated metrics for a single program (identified by its fairness ID).
// All methods are goroutine-safe.
type ProgramMetrics struct {
	mu              sync.Mutex
	averageWaitTime float64 // cumulative mean of wait time in milliseconds
	waitCount       int64   // number of wait time observations

	averageTokens float64 // EWMA of per-request token usage (input+output)

	// Service rate: EWMA of weighted tokens per second, updated on each completion.
	serviceRate        float64
	lastCompletionTime time.Time

	totalRequests   atomic.Int64
	dispatchedCount atomic.Int64

	// inFlight counts requests that have been dispatched but whose response
	// has not yet completed. Used to gate deficit decay so that a queue with
	// Len==0 but an outstanding dispatch is not treated as inactive.
	inFlight atomic.Int64
}

// IncrementRequests atomically increments the total request counter.
func (m *ProgramMetrics) IncrementRequests() {
	m.totalRequests.Add(1)
}

// IncrementDispatched atomically increments the dispatched counter.
func (m *ProgramMetrics) IncrementDispatched() {
	m.dispatchedCount.Add(1)
}

// IncrementInFlight atomically increments the in-flight request counter.
// Called when a request is dispatched (PreRequest hook).
func (m *ProgramMetrics) IncrementInFlight() {
	m.inFlight.Add(1)
}

// DecrementInFlight atomically decrements the in-flight request counter.
// Called when a request completes (ResponseBody hook).
func (m *ProgramMetrics) DecrementInFlight() {
	m.inFlight.Add(-1)
}

// InFlight returns the current count of dispatched-but-not-completed requests.
func (m *ProgramMetrics) InFlight() int64 {
	return m.inFlight.Load()
}

// RecordWaitTime adds a new wait-time observation and updates the cumulative
// mean using Welford's incremental form: avg += (x - avg) / n.
func (m *ProgramMetrics) RecordWaitTime(waitMs float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.waitCount++
	m.averageWaitTime += (waitMs - m.averageWaitTime) / float64(m.waitCount)
}

// AverageWaitTime returns the cumulative mean of wait time in milliseconds.
// Returns 0 if no observations have been recorded.
func (m *ProgramMetrics) AverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageWaitTime
}

// WaitCount returns the number of wait-time observations recorded.
func (m *ProgramMetrics) WaitCount() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.waitCount
}

// RecordTokens updates the EWMA of per-request weighted token cost so recent
// heavy/light requests carry more weight.
func (m *ProgramMetrics) RecordTokens(input, output int64) {
	cost := weightInputToken*float64(input) + weightOutputToken*float64(output)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.averageTokens == 0 {
		m.averageTokens = cost
	} else {
		m.averageTokens = ewmaAlpha*cost + (1-ewmaAlpha)*m.averageTokens
	}
}

// AverageTokens returns the EWMA of per-request token usage.
func (m *ProgramMetrics) AverageTokens() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageTokens
}

// RecordServiceRate updates the EWMA of service rate (weighted tokens/sec)
// using the elapsed time since the last completion.
func (m *ProgramMetrics) RecordServiceRate(weightedTokens float64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastCompletionTime.IsZero() {
		m.lastCompletionTime = now
		return // no rate from a single point
	}
	elapsed := now.Sub(m.lastCompletionTime).Seconds()
	if elapsed <= 0 {
		return
	}
	instantRate := weightedTokens / elapsed
	if m.serviceRate == 0 {
		m.serviceRate = instantRate
	} else {
		m.serviceRate = ewmaAlpha*instantRate + (1-ewmaAlpha)*m.serviceRate
	}
	m.lastCompletionTime = now
}

// ServiceRate returns the EWMA of weighted tokens per second.
func (m *ProgramMetrics) ServiceRate() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.serviceRate
}

// LastCompletionTime returns the wall-clock time of the most recent response
// completion, or the zero time if the program has never completed a request.
// Used by the eviction sweeper to identify idle programs.
func (m *ProgramMetrics) LastCompletionTime() time.Time {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastCompletionTime
}

// TotalRequests returns the total number of requests seen for this program.
func (m *ProgramMetrics) TotalRequests() int64 {
	return m.totalRequests.Load()
}

// DispatchedCount returns the total number of dispatched requests for this program.
func (m *ProgramMetrics) DispatchedCount() int64 {
	return m.dispatchedCount.Load()
}


