package programaware

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestDecayDeficit_NoLostUpdates verifies that concurrent AddDeficit calls
// during a DecayDeficit pass are not silently overwritten by the decay's
// read-modify-write. With factor=1.0 decay is a no-op, so any lost AddDeficit
// would show up as a final value below the expected sum.
func TestDecayDeficit_NoLostUpdates(t *testing.T) {
	m := &ProgramMetrics{}

	const adders = 32
	const addsPerGoroutine = 1000
	const decayCalls = 1000
	const addAmount int64 = 1

	var wg sync.WaitGroup
	wg.Add(adders + 1)

	// Adders: each performs addsPerGoroutine atomic AddDeficit(1) calls.
	for range adders {
		go func() {
			defer wg.Done()
			for range addsPerGoroutine {
				m.AddDeficit(addAmount)
			}
		}()
	}

	// Decayer: hammers DecayDeficit with factor=1.0 (mathematical no-op)
	// concurrently with the adders. With the pre-CAS Load/Store impl this
	// would race and drop adds; with CAS it must preserve every add.
	go func() {
		defer wg.Done()
		for range decayCalls {
			m.DecayDeficit(1.0)
		}
	}()

	wg.Wait()

	expected := int64(adders) * addsPerGoroutine * addAmount
	assert.Equal(t, expected, m.Deficit(), "DecayDeficit must not lose concurrent AddDeficit updates")
}

// TestDecayDeficitTimed_NoLostUpdates exercises the time-based path under
// concurrency. lastDeficitDecay is guarded by mu; the deficit value itself is
// updated via CompareAndSwap, so concurrent AddDeficit calls must not be lost.
//
// The decayer fires ONE DecayDeficitTimed call concurrently with the adders.
// We use a single call (not a tight loop) because each decay multiplies by a
// factor < 1.0, and int64 truncation accumulates across many calls — that is
// expected math, not a race. A single decay call cannot truncate more than the
// deficit's current value's worth of tokens, so the assertion can be tight.
func TestDecayDeficitTimed_NoLostUpdates(t *testing.T) {
	m := &ProgramMetrics{}
	// Prime lastDeficitDecay so the goroutine's call performs the decay branch
	// (otherwise the first call just records lastDeficitDecay and returns).
	m.DecayDeficitTimed(1, time.Now().Add(-time.Hour))

	const adders = 32
	const addsPerGoroutine = 1000
	const addAmount int64 = 1
	// Half-life of 1e9 seconds → factor over a few ms is float-indistinguishable from 1.0.
	const longHalfLife = 1e9

	var wg sync.WaitGroup
	wg.Add(adders + 1)

	for range adders {
		go func() {
			defer wg.Done()
			for range addsPerGoroutine {
				m.AddDeficit(addAmount)
			}
		}()
	}

	go func() {
		defer wg.Done()
		// Single decay call concurrent with the adders.
		m.DecayDeficitTimed(longHalfLife, time.Now())
	}()

	wg.Wait()

	expected := int64(adders) * addsPerGoroutine * addAmount
	got := m.Deficit()
	// Tight bound: with a long half-life, a single decay call's truncation is
	// at most 1 token. Anything beyond that signals a lost AddDeficit.
	assert.GreaterOrEqual(t, got, expected-1,
		"DecayDeficitTimed lost concurrent AddDeficit updates (expected ~%d, got %d)", expected, got)
	assert.LessOrEqual(t, got, expected,
		"deficit overshot expected sum (expected %d, got %d)", expected, got)
}

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
