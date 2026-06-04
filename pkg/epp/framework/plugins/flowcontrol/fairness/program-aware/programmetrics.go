package programaware

import (
	"math"
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
	mu                sync.Mutex
	averageWaitTime   float64 // EWMA in milliseconds
	accumulatedWaitMs float64 // total accumulated wait time in milliseconds
	waitCount         int64   // number of wait time observations
	hasWaitData       bool

	averageTokens float64 // EWMA of per-request token usage (input+output)

	// Attained service: time-decayed accumulator of weighted tokens consumed.
	// Increased on each completion, decayed on each Pick() cycle.
	attainedService float64
	lastDecayTime   time.Time // last wall-clock time DecayServiceTimed was called

	// Service rate: EWMA of weighted tokens per second, updated on each completion.
	// Used for Jain's fairness index (equalizing rates across programs = fair).
	serviceRate        float64
	lastCompletionTime time.Time

	totalRequests     atomic.Int64
	dispatchedCount   atomic.Int64
	totalInputTokens  atomic.Int64
	totalOutputTokens atomic.Int64

	// inFlight counts requests that have been dispatched but whose response
	// has not yet completed. Used to gate deficit decay so that a queue with
	// Len==0 but an outstanding dispatch is not treated as inactive.
	inFlight atomic.Int64

	// deficitTokens is the DRR deficit counter: positive means the program is owed
	// service; negative means it has been overserved relative to its quantum.
	// Only used by DRRStrategy.
	deficitTokens    atomic.Int64
	lastDeficitDecay time.Time // last wall-clock time DecayDeficitTimed was called
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

// RecordWaitTime updates the EWMA of wait time with a new observation
// and accumulates the total wait time for lifetime average computation.
func (m *ProgramMetrics) RecordWaitTime(waitMs float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasWaitData {
		m.averageWaitTime = waitMs
		m.hasWaitData = true
	} else {
		m.averageWaitTime = ewmaAlpha*waitMs + (1-ewmaAlpha)*m.averageWaitTime
	}
	m.accumulatedWaitMs += waitMs
	m.waitCount++
}

// HasWaitData returns true if at least one wait time observation has been recorded.
func (m *ProgramMetrics) HasWaitData() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.hasWaitData
}

// AverageWaitTime returns the current EWMA of wait time in milliseconds.
func (m *ProgramMetrics) AverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.averageWaitTime
}

// TotalAverageWaitTime returns the lifetime average wait time in milliseconds
// (accumulated wait time / total observations). Returns 0 if no data.
func (m *ProgramMetrics) TotalAverageWaitTime() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.waitCount == 0 {
		return 0
	}
	return m.accumulatedWaitMs / float64(m.waitCount)
}

// RecordTokens adds token counts from a completed request and updates the
// EWMA of per-request token usage so recent heavy/light requests carry more weight.
func (m *ProgramMetrics) RecordTokens(input, output int64) {
	m.totalInputTokens.Add(input)
	m.totalOutputTokens.Add(output)

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

// AddService accumulates weighted token cost into the attained service counter.
func (m *ProgramMetrics) AddService(weightedTokens float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.attainedService += weightedTokens
}

// DecayService multiplies the attained service counter by the given factor,
// causing old service to be gradually forgotten.
func (m *ProgramMetrics) DecayService(factor float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.attainedService *= factor
}

// DecayServiceTimed applies time-based exponential decay using a half-life.
// The decay factor is computed as 0.5^(elapsed/halfLifeSeconds), so service
// halves every halfLifeSeconds regardless of how frequently this is called.
func (m *ProgramMetrics) DecayServiceTimed(halfLifeSeconds float64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastDecayTime.IsZero() {
		m.lastDecayTime = now
		return
	}
	elapsed := now.Sub(m.lastDecayTime).Seconds()
	if elapsed <= 0 {
		return
	}
	factor := math.Pow(0.5, elapsed/halfLifeSeconds)
	m.attainedService *= factor
	m.lastDecayTime = now
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

// AttainedService returns the current time-decayed attained service value.
func (m *ProgramMetrics) AttainedService() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.attainedService
}

// TotalRequests returns the total number of requests seen for this program.
func (m *ProgramMetrics) TotalRequests() int64 {
	return m.totalRequests.Load()
}

// DispatchedCount returns the total number of dispatched requests for this program.
func (m *ProgramMetrics) DispatchedCount() int64 {
	return m.dispatchedCount.Load()
}

// TotalInputTokens returns the total number of input tokens across all requests.
func (m *ProgramMetrics) TotalInputTokens() int64 {
	return m.totalInputTokens.Load()
}

// TotalOutputTokens returns the total number of output tokens across all requests.
func (m *ProgramMetrics) TotalOutputTokens() int64 {
	return m.totalOutputTokens.Load()
}

// --- DRR deficit counter ---

// AddDeficit increases the deficit counter by n tokens (quantum allocation).
func (m *ProgramMetrics) AddDeficit(n int64) {
	m.deficitTokens.Add(n)
}

// DeductTokens decreases the deficit counter by n tokens (actual cost deduction).
func (m *ProgramMetrics) DeductTokens(n int64) {
	m.deficitTokens.Add(-n)
}

// ResetDeficit sets the deficit counter to zero (called when the queue drains).
func (m *ProgramMetrics) ResetDeficit() {
	m.deficitTokens.Store(0)
}

// Deficit returns the current deficit counter value in tokens.
func (m *ProgramMetrics) Deficit() int64 {
	return m.deficitTokens.Load()
}

// DecayDeficit multiplies the deficit counter by the given factor.
//
// Uses a CompareAndSwap retry loop so concurrent AddDeficit/DeductTokens
// updates between the read and the write are not lost.
func (m *ProgramMetrics) DecayDeficit(factor float64) {
	for {
		current := m.deficitTokens.Load()
		decayed := int64(float64(current) * factor)
		if m.deficitTokens.CompareAndSwap(current, decayed) {
			return
		}
	}
}

// DecayDeficitTimed applies time-based exponential decay to the deficit counter
// using a half-life. The decay factor is 0.5^(elapsed/halfLifeSeconds), so the
// deficit halves every halfLifeSeconds regardless of call frequency.
//
// The mu lock guards lastDeficitDecay; the deficit value itself is updated via
// a CompareAndSwap retry loop so concurrent AddDeficit/DeductTokens are not lost.
func (m *ProgramMetrics) DecayDeficitTimed(halfLifeSeconds float64, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastDeficitDecay.IsZero() {
		m.lastDeficitDecay = now
		return
	}
	elapsed := now.Sub(m.lastDeficitDecay).Seconds()
	if elapsed <= 0 {
		return
	}
	factor := math.Pow(0.5, elapsed/halfLifeSeconds)
	for {
		current := m.deficitTokens.Load()
		decayed := int64(float64(current) * factor)
		if m.deficitTokens.CompareAndSwap(current, decayed) {
			break
		}
	}
	m.lastDeficitDecay = now
}
