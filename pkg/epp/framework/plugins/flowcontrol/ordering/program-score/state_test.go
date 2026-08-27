package programscore

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestProgramState_AddAndDecay(t *testing.T) {
	t.Parallel()
	now := time.Now()

	s := &programState{}
	turns := s.AddTurn(now, 60)
	assert.Equal(t, 1.0, turns)
	tokens := s.AddTokens(100, now, 60)
	assert.Equal(t, 100.0, tokens)

	// One half-life later, both accumulators should have halved.
	later := now.Add(60 * time.Second)
	gotTurns, gotTokens := s.Cost(later, 60)
	assert.InDelta(t, 0.5, gotTurns, 1e-9)
	assert.InDelta(t, 50.0, gotTokens, 1e-9)
}

func TestProgramState_ZeroHalfLifeDisablesDecay(t *testing.T) {
	t.Parallel()
	now := time.Now()

	s := &programState{}
	s.AddTurn(now, 0)
	gotTurns, _ := s.Cost(now.Add(time.Hour), 0)
	assert.Equal(t, 1.0, gotTurns, "halfLifeSeconds <= 0 must disable decay")
}

func TestProgramState_LastActive(t *testing.T) {
	t.Parallel()
	now := time.Now()

	s := &programState{}
	assert.Zero(t, s.LastActive())
	s.AddTurn(now, 60)
	assert.Equal(t, now, s.LastActive())
}
