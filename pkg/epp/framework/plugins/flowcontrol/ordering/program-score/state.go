package programscore

import (
	"math"
	"sync"
	"time"
)

// programState tracks one program's decayed turn count and decayed token cost. The two accumulate
// and decay independently so Less can weight them independently, and so a future third signal
// (e.g. prefix-match) is one more accumulator rather than a redesign.
//
// Decay mirrors programaware's lasState: folded lazily on read/write so an idle program's cost
// ages out in wall-clock time without the program ever being visited.
type programState struct {
	mu          sync.Mutex
	turns       float64
	tokens      float64
	decayAnchor time.Time
	lastActive  time.Time
}

// decayLocked folds in the decay accrued since decayAnchor and advances the anchor to now.
// halfLifeSeconds <= 0 disables decay. The caller must hold mu.
func (s *programState) decayLocked(now time.Time, halfLifeSeconds float64) {
	if s.decayAnchor.IsZero() {
		s.decayAnchor = now
		return
	}
	elapsed := now.Sub(s.decayAnchor).Seconds()
	if elapsed <= 0 {
		return
	}
	if halfLifeSeconds > 0 {
		factor := math.Exp2(-elapsed / halfLifeSeconds)
		s.turns *= factor
		s.tokens *= factor
	}
	s.decayAnchor = now
}

// Cost returns the decayed turns and tokens as of now, with decay folded in.
func (s *programState) Cost(now time.Time, halfLifeSeconds float64) (turns, tokens float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	return s.turns, s.tokens
}

// AddTurn folds in pending decay, increments the turn count by one, and returns the new total.
func (s *programState) AddTurn(now time.Time, halfLifeSeconds float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	s.turns++
	s.lastActive = now
	return s.turns
}

// AddTokens folds in pending decay, accumulates the token cost, and returns the new total.
func (s *programState) AddTokens(cost float64, now time.Time, halfLifeSeconds float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked(now, halfLifeSeconds)
	s.tokens += cost
	s.lastActive = now
	return s.tokens
}

// LastActive reports the last time this program's state was touched, used by idle eviction.
func (s *programState) LastActive() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastActive
}
