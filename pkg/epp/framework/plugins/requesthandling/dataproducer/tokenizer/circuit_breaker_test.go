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

package tokenizer

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeClock is a manually advanced clock for the circuit breaker.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newTestBreaker(t *testing.T, cfg circuitBreakerConfig) (*renderCircuitBreaker, *fakeClock) {
	t.Helper()
	b, err := newRenderCircuitBreaker(&cfg)
	require.NoError(t, err)
	clock := &fakeClock{t: time.Unix(1_800_000_000, 0)}
	b.now = clock.now
	return b, clock
}

// call admits one render call and records result for it.
func call(t *testing.T, b *renderCircuitBreaker, result string) circuitTransition {
	t.Helper()
	allowed, probe, _ := b.admit()
	require.True(t, allowed)
	return b.record(probe, result)
}

// tripBreaker opens b with consecutive timeouts.
func tripBreaker(t *testing.T, b *renderCircuitBreaker) {
	t.Helper()
	for i := 0; i < b.consecutiveTimeouts; i++ {
		call(t, b, renderResultTimeout)
	}
	require.Equal(t, circuitOpen, b.state)
}

func TestRenderCircuitBreaker_Config(t *testing.T) {
	b, err := newRenderCircuitBreaker(&circuitBreakerConfig{})
	require.NoError(t, err)
	assert.Equal(t, defaultCircuitConsecutiveTimeouts, b.consecutiveTimeouts)
	assert.Equal(t, defaultCircuitOpenDuration, b.openDuration)
	assert.Equal(t, defaultCircuitMaxOpenDuration, b.maxOpenDuration)

	// An open duration above the default cap raises the default cap with it.
	b, err = newRenderCircuitBreaker(&circuitBreakerConfig{OpenDuration: "5m"})
	require.NoError(t, err)
	assert.Equal(t, 5*time.Minute, b.maxOpenDuration)

	for _, tc := range []struct {
		name string
		cfg  circuitBreakerConfig
	}{
		{"negative consecutive timeouts", circuitBreakerConfig{ConsecutiveTimeouts: -1}},
		{"unparsable open duration", circuitBreakerConfig{OpenDuration: "soon"}},
		{"zero open duration", circuitBreakerConfig{OpenDuration: "0s"}},
		{"negative max open duration", circuitBreakerConfig{MaxOpenDuration: "-1s"}},
		{"max below open duration", circuitBreakerConfig{OpenDuration: "30s", MaxOpenDuration: "10s"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newRenderCircuitBreaker(&tc.cfg)
			require.Error(t, err)
			_, err = newVLLMHTTPRenderer(&vllmConfig{CircuitBreaker: &tc.cfg})
			require.Error(t, err)
		})
	}
}

func TestRenderCircuitBreaker_OpensOnlyOnConsecutiveTimeouts(t *testing.T) {
	b, _ := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 3})

	assert.Equal(t, circuitNoTransition, call(t, b, renderResultTimeout))
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultTimeout))
	// A success resets the count.
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultSuccess))
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultTimeout))
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultTimeout))
	// Fast failures and cancellations neither count nor reset.
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultError))
	assert.Equal(t, circuitNoTransition, call(t, b, renderResultCanceled))
	assert.Equal(t, circuitOpened, call(t, b, renderResultTimeout))

	allowed, _, _ := b.admit()
	assert.False(t, allowed)
	assert.Equal(t, defaultCircuitOpenDuration, b.openFor())
}

func TestRenderCircuitBreaker_HalfOpenAdmitsOneProbe(t *testing.T) {
	b, clock := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 1, OpenDuration: "10s"})
	tripBreaker(t, b)

	clock.advance(10*time.Second - time.Nanosecond)
	allowed, _, _ := b.admit()
	require.False(t, allowed)

	clock.advance(time.Nanosecond)
	allowed, probe, transition := b.admit()
	require.True(t, allowed)
	require.True(t, probe)
	assert.Equal(t, circuitHalfOpened, transition)

	allowed, _, transition = b.admit()
	assert.False(t, allowed, "a second call is skipped while the probe is in flight")
	assert.Equal(t, circuitNoTransition, transition)

	assert.Equal(t, circuitClosedAgain, b.record(true, renderResultSuccess))
	allowed, probe, _ = b.admit()
	assert.True(t, allowed)
	assert.False(t, probe)
}

func TestRenderCircuitBreaker_ConcurrentHalfOpenAdmitsOneProbe(t *testing.T) {
	b, clock := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 1})
	tripBreaker(t, b)
	clock.advance(defaultCircuitOpenDuration)

	var admitted atomic.Int32
	var wg sync.WaitGroup
	for range 50 {
		wg.Go(func() {
			if allowed, _, _ := b.admit(); allowed {
				admitted.Add(1)
			}
		})
	}
	wg.Wait()
	assert.Equal(t, int32(1), admitted.Load())
}

func TestRenderCircuitBreaker_ProbeOutcome(t *testing.T) {
	for _, tc := range []struct {
		result     string
		transition circuitTransition
		state      circuitState
	}{
		{renderResultSuccess, circuitClosedAgain, circuitClosed},
		// A fast failure shows the endpoint answers within the budget.
		{renderResultError, circuitClosedAgain, circuitClosed},
		{renderResultTimeout, circuitOpened, circuitOpen},
		{renderResultCanceled, circuitNoTransition, circuitHalfOpen},
	} {
		t.Run(tc.result, func(t *testing.T) {
			b, clock := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 1})
			tripBreaker(t, b)
			clock.advance(defaultCircuitOpenDuration)
			allowed, probe, _ := b.admit()
			require.True(t, allowed && probe)

			assert.Equal(t, tc.transition, b.record(true, tc.result))
			assert.Equal(t, tc.state, b.state)
			if tc.state == circuitHalfOpen {
				allowed, probe, _ := b.admit()
				assert.True(t, allowed && probe, "the next call probes after a canceled probe")
			}
		})
	}
}

func TestRenderCircuitBreaker_OpenDurationBackoff(t *testing.T) {
	b, clock := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 1, OpenDuration: "10s", MaxOpenDuration: "30s"})
	tripBreaker(t, b)
	require.Equal(t, 10*time.Second, b.openFor())

	// Each failed probe doubles the open duration up to the cap.
	for _, want := range []time.Duration{20 * time.Second, 30 * time.Second, 30 * time.Second} {
		clock.advance(b.openFor())
		allowed, probe, _ := b.admit()
		require.True(t, allowed && probe)
		require.Equal(t, circuitOpened, b.record(true, renderResultTimeout))
		assert.Equal(t, want, b.openFor())
	}

	// Opening again soon after closing keeps the grown duration.
	clock.advance(b.openFor())
	call(t, b, renderResultSuccess)
	require.Equal(t, circuitClosed, b.state)
	clock.advance(time.Second)
	tripBreaker(t, b)
	assert.Equal(t, 30*time.Second, b.openFor())

	// Opening after maxOpenDuration of closed time starts from openDuration.
	clock.advance(b.openFor())
	call(t, b, renderResultSuccess)
	clock.advance(30 * time.Second)
	tripBreaker(t, b)
	assert.Equal(t, 10*time.Second, b.openFor())
}

func TestRenderCircuitBreaker_IgnoresCallsAdmittedBeforeOpening(t *testing.T) {
	b, _ := newTestBreaker(t, circuitBreakerConfig{ConsecutiveTimeouts: 1})
	allowed, probe, _ := b.admit()
	require.True(t, allowed)
	require.False(t, probe)
	tripBreaker(t, b)

	assert.Equal(t, circuitNoTransition, b.record(false, renderResultSuccess))
	assert.Equal(t, circuitOpen, b.state)
}

// TestProduce_RenderCircuitBreaker drives the circuit through the HTTP render
// path: a saturated endpoint opens it, requests then skip rendering without an
// error, and a probe against the recovered endpoint closes it.
func TestProduce_RenderCircuitBreaker(t *testing.T) {
	var saturated atomic.Bool
	saturated.Store(true)
	var requests atomic.Int32
	release := make(chan struct{})
	r := newRenderServer(t, vllmConfig{
		Timeout:        "20ms",
		MMTimeout:      "20ms",
		CircuitBreaker: &circuitBreakerConfig{ConsecutiveTimeouts: 2, OpenDuration: "10s"},
	}, func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		if saturated.Load() {
			<-release
			return
		}
		_, _ = io.WriteString(w, `{"token_ids":[1,2]}`)
	})
	// Registered after the server so held requests return before it closes.
	t.Cleanup(func() { close(release) })
	clock := &fakeClock{t: time.Unix(1_800_000_000, 0)}
	r.breaker.now = clock.now
	p := newTestPlugin(r)
	logs := &logCapture{}
	ctx := logs.context(context.Background())
	gauge := func() float64 { return testutil.ToFloat64(renderCircuitOpen.WithLabelValues(PluginType, r.pluginName)) }

	for range 2 {
		require.ErrorIs(t, p.Produce(ctx, chatRequest(), nil), context.DeadlineExceeded)
	}
	assert.Equal(t, 1.0, gauge())
	assert.Equal(t, 1.0, testutil.ToFloat64(renderCircuitOpensTotal.WithLabelValues(PluginType, r.pluginName)))
	assert.True(t, logged(logs, "token-producer render circuit opened"))

	request := chatRequest()
	require.NoError(t, p.Produce(ctx, request, nil))
	assert.Nil(t, request.Body.TokenizedRequest)
	assert.Equal(t, int32(2), requests.Load(), "the skipped call does not reach the render endpoint")
	assert.Equal(t, 1.0, testutil.ToFloat64(renderSkippedTotal.WithLabelValues(PluginType, r.pluginName)))

	saturated.Store(false)
	clock.advance(10 * time.Second)
	request = chatRequest()
	require.NoError(t, p.Produce(ctx, request, nil))
	require.NotNil(t, request.Body.TokenizedRequest)
	assert.Equal(t, []uint32{1, 2}, request.Body.TokenizedRequest.Prompts[0].TokenIDs)
	assert.Equal(t, 0.0, gauge())
	assert.True(t, logged(logs, "token-producer render circuit closed"))
}

func TestRenderCircuitBreaker_UnobservedCallsBypass(t *testing.T) {
	var requests atomic.Int32
	r := newRenderServer(t, vllmConfig{CircuitBreaker: &circuitBreakerConfig{ConsecutiveTimeouts: 1}}, func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = io.WriteString(w, `{"token_ids":[1]}`)
	})
	r.breaker.record(false, renderResultTimeout)
	require.Equal(t, circuitOpen, r.breaker.state)

	require.ErrorIs(t, renderPath(context.Background(), r, chatRenderPath), errRenderCircuitOpen)
	require.NoError(t, renderPath(withUnobservedRender(context.Background()), r, chatRenderPath))
	assert.Equal(t, int32(1), requests.Load())
	assert.Equal(t, circuitOpen, r.breaker.state, "an unobserved call does not change the circuit")
}

func logged(logs *logCapture, msg string) bool {
	for _, line := range logs.snapshot() {
		if strings.Contains(line, `"msg"="`+msg+`"`) {
			return true
		}
	}
	return false
}
