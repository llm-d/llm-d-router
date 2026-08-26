package programscore

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var testFlowKey = flowcontrol.FlowKey{ID: "test-flow", Priority: 0}

// itemFor builds a mock queue item whose program is identified by fairnessID.
func itemFor(id, fairnessID string, enqueue time.Time) *mocks.MockQueueItemAccessor {
	req := mocks.NewMockFlowControlRequest(10, id, testFlowKey)
	req.InferenceRequestV = &fwksched.InferenceRequest{FairnessID: fairnessID}
	return &mocks.MockQueueItemAccessor{OriginalRequestV: req, EnqueueTimeV: enqueue}
}

func TestConfig_Validate(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{"defaults are valid", func(*Config) {}, false},
		{"negative WeightTurns", func(c *Config) { c.WeightTurns = -1 }, true},
		{"negative WeightTokens", func(c *Config) { c.WeightTokens = -1 }, true},
		{"negative HalfLifeSeconds", func(c *Config) { c.HalfLifeSeconds = -1 }, true},
		{"negative RescoreIntervalSeconds", func(c *Config) { c.RescoreIntervalSeconds = -1 }, true},
		{"negative EvictionTTLSeconds", func(c *Config) { c.EvictionTTLSeconds = -1 }, true},
		{"zero EvictionSweepSeconds", func(c *Config) { c.EvictionSweepSeconds = 0 }, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := DefaultConfig()
			tc.mutate(&cfg)
			err := cfg.validate()
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestFactory_NilHandle(t *testing.T) {
	t.Parallel()
	p, err := ProgramScoreOrderingPolicyFactory(ProgramScoreOrderingPolicyType, nil, nil)
	require.NoError(t, err)
	require.NotNil(t, p)
	assert.Equal(t, ProgramScoreOrderingPolicyType, p.TypedName().Name)
}

func TestProgramScorePolicy_Less(t *testing.T) {
	t.Parallel()
	now := time.Now()

	t.Run("fewer turns dispatches first", func(t *testing.T) {
		t.Parallel()
		p := newProgramScorePolicy(DefaultConfig())
		busy := itemFor("busy-req", "busy-program", now)
		idle := itemFor("idle-req", "idle-program", now)

		// Give "busy-program" three turns; "idle-program" stays at zero.
		for range 3 {
			require.NoError(t, p.PreRequest(context.Background(), busy.OriginalRequestV.InferenceRequest(), nil))
		}

		assert.True(t, p.Less(idle, busy), "the program with fewer turns should dispatch first")
		assert.False(t, p.Less(busy, idle))
	})

	t.Run("tokens consumed also weigh in", func(t *testing.T) {
		t.Parallel()
		p := newProgramScorePolicy(DefaultConfig())
		heavy := itemFor("heavy-req", "heavy-program", now)
		light := itemFor("light-req", "light-program", now)

		p.ResponseBody(context.Background(), heavy.OriginalRequestV.InferenceRequest(),
			&fwkrc.Response{EndOfStream: true, Usage: fwkrh.Usage{PromptTokens: 10000, CompletionTokens: 10000}}, nil)

		assert.True(t, p.Less(light, heavy), "the program that consumed fewer tokens should dispatch first")
	})

	t.Run("equal cost breaks tie by enqueue time", func(t *testing.T) {
		t.Parallel()
		p := newProgramScorePolicy(DefaultConfig())
		earlier := itemFor("earlier", "same-program-a", now.Add(-time.Second))
		later := itemFor("later", "same-program-b", now)

		assert.True(t, p.Less(earlier, later))
		assert.False(t, p.Less(later, earlier))
	})

	t.Run("intermediate stream chunks do not add token cost", func(t *testing.T) {
		t.Parallel()
		p := newProgramScorePolicy(DefaultConfig())
		req := &fwksched.InferenceRequest{FairnessID: "streaming-program"}
		p.ResponseBody(context.Background(), req,
			&fwkrc.Response{EndOfStream: false, Usage: fwkrh.Usage{PromptTokens: 10000}}, nil)

		turns, tokens := p.getOrCreateState("streaming-program").Cost(now, p.cfg.HalfLifeSeconds)
		assert.Zero(t, turns)
		assert.Zero(t, tokens, "a non-final chunk must not be charged")
	})
}

func TestProgramScorePolicy_RescoreInterval(t *testing.T) {
	t.Parallel()
	p := newProgramScorePolicy(Config{RescoreIntervalSeconds: 2.5})
	assert.Equal(t, 2500*time.Millisecond, p.RescoreInterval())
}

// TestProgramScorePolicy_EqualizesShareAcrossManyAgentsInOneQueue is the regression test for the
// issue's motivating claim: a tenant running many agents under one shared queue must not out-
// dispatch a tenant running few, because dispatch order tracks per-program turns/tokens rather
// than how many of a tenant's agents occupy the queue.
func TestProgramScorePolicy_EqualizesShareAcrossManyAgentsInOneQueue(t *testing.T) {
	t.Parallel()
	p := newProgramScorePolicy(DefaultConfig())
	now := time.Now()

	// tenant-many has 5 agents, each already dispatched once (5 turns total). tenant-few has 1
	// agent, never dispatched (0 turns). All 6 requests share one queue/FlowKey.
	for i := range 5 {
		agentID := "tenant-many-agent-" + string(rune('a'+i))
		require.NoError(t, p.PreRequest(context.Background(), &fwksched.InferenceRequest{FairnessID: agentID}, nil))
	}
	newTurn := itemFor("tenant-many-req", "tenant-many-agent-a", now)
	fewTurn := itemFor("tenant-few-req", "tenant-few-agent", now)

	assert.True(t, p.Less(fewTurn, newTurn),
		"tenant-few's single, never-dispatched agent must go before tenant-many's already-served agent")
}

func TestProgramScorePolicy_EvictIdle(t *testing.T) {
	t.Parallel()
	p := newProgramScorePolicy(DefaultConfig())
	p.getOrCreateState("idle-program").AddTurn(time.Now().Add(-time.Hour), 0)

	p.evictIdle(time.Minute)

	_, loaded := p.states.Load("idle-program")
	assert.False(t, loaded, "an idle-past-TTL program must be evicted")
}
