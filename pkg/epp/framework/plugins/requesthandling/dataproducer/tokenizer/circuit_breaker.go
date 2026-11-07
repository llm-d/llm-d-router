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
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	defaultCircuitConsecutiveTimeouts = 5
	defaultCircuitOpenDuration        = 10 * time.Second
	defaultCircuitMaxOpenDuration     = 2 * time.Minute
)

// renderPaths are the render paths that each get a circuit breaker. The paths
// have different render budgets, so a fast success on one says nothing about
// timeouts on another.
var renderPaths = []string{completionsRenderPath, chatRenderPath, messagesRenderPath, responsesRenderPath}

// errRenderCircuitOpen is returned for a render call that the circuit breaker skipped.
var errRenderCircuitOpen = errors.New("render circuit breaker is open")

// circuitBreakerConfig enables the render circuit breaker. Empty fields use the defaults.
type circuitBreakerConfig struct {
	// ConsecutiveTimeouts is the number of consecutive timed-out render calls
	// that opens the circuit. Defaults to 5.
	ConsecutiveTimeouts int `json:"consecutiveTimeouts,omitempty"`
	// OpenDuration is how long the circuit stays open each time it opens from
	// closed (Go duration string). Defaults to 10s.
	OpenDuration string `json:"openDuration,omitempty"`
	// MaxOpenDuration caps the open duration, which doubles on each failed
	// probe. Defaults to 2m.
	MaxOpenDuration string `json:"maxOpenDuration,omitempty"`
}

type circuitState int

const (
	circuitClosed circuitState = iota
	circuitOpen
	circuitHalfOpen
)

// circuitTransition is a state change reported to the caller, which owns the
// logs and metrics for it.
type circuitTransition int

const (
	circuitNoTransition circuitTransition = iota
	circuitOpened
	circuitHalfOpened
	circuitClosedAgain
)

// renderCircuitBreaker tracks render timeouts for one render path. While
// closed, consecutive timeouts open it. While open, render calls are skipped.
// When the open duration elapses it is half-open: one render call probes the
// endpoint and the others are skipped until the probe completes.
type renderCircuitBreaker struct {
	consecutiveTimeouts int
	openDuration        time.Duration
	maxOpenDuration     time.Duration
	now                 func() time.Time

	mu    sync.Mutex
	state circuitState
	// generation changes on every state change. A call is recorded only if
	// it was admitted in the current state.
	generation uint64
	timeouts   int
	current    time.Duration
	openUntil  time.Time
	probing    bool
}

func newRenderCircuitBreaker(cfg *circuitBreakerConfig) (*renderCircuitBreaker, error) {
	if cfg.ConsecutiveTimeouts < 0 {
		return nil, fmt.Errorf("invalid 'circuitBreaker.consecutiveTimeouts' %d: must not be negative", cfg.ConsecutiveTimeouts)
	}
	consecutiveTimeouts := cfg.ConsecutiveTimeouts
	if consecutiveTimeouts == 0 {
		consecutiveTimeouts = defaultCircuitConsecutiveTimeouts
	}
	openDuration, err := parsePositiveDuration(cfg.OpenDuration, defaultCircuitOpenDuration)
	if err != nil {
		return nil, fmt.Errorf("invalid 'circuitBreaker.openDuration': %w", err)
	}
	maxOpenDuration, err := parsePositiveDuration(cfg.MaxOpenDuration, max(defaultCircuitMaxOpenDuration, openDuration))
	if err != nil {
		return nil, fmt.Errorf("invalid 'circuitBreaker.maxOpenDuration': %w", err)
	}
	if maxOpenDuration < openDuration {
		return nil, fmt.Errorf("invalid 'circuitBreaker.maxOpenDuration' %s: must not be less than openDuration %s", maxOpenDuration, openDuration)
	}
	return &renderCircuitBreaker{
		consecutiveTimeouts: consecutiveTimeouts,
		openDuration:        openDuration,
		maxOpenDuration:     maxOpenDuration,
		now:                 time.Now,
	}, nil
}

// newRenderCircuitBreakers returns one breaker per render path.
func newRenderCircuitBreakers(cfg *circuitBreakerConfig) (map[string]*renderCircuitBreaker, error) {
	breakers := make(map[string]*renderCircuitBreaker, len(renderPaths))
	for _, path := range renderPaths {
		b, err := newRenderCircuitBreaker(cfg)
		if err != nil {
			return nil, err
		}
		breakers[path] = b
	}
	return breakers, nil
}

func parsePositiveDuration(s string, def time.Duration) (time.Duration, error) {
	d, err := parseHTTPDuration(s, def)
	if err != nil {
		return 0, err
	}
	if d <= 0 {
		return 0, fmt.Errorf("%q must be a positive duration", s)
	}
	return d, nil
}

// admit reports whether a render call may proceed. An admitted call passes
// the returned generation to record.
func (b *renderCircuitBreaker) admit() (allowed bool, generation uint64, transition circuitTransition) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case circuitClosed:
		return true, b.generation, circuitNoTransition
	case circuitOpen:
		if b.now().Before(b.openUntil) {
			return false, 0, circuitNoTransition
		}
		b.setState(circuitHalfOpen)
		transition = circuitHalfOpened
	}
	if b.probing {
		return false, 0, transition
	}
	b.probing = true
	return true, b.generation, transition
}

// record applies the result label of an admitted render call.
func (b *renderCircuitBreaker) record(generation uint64, result string) circuitTransition {
	b.mu.Lock()
	defer b.mu.Unlock()
	// A call admitted before the last state change, such as one that was in
	// flight when the circuit opened, says nothing about the current state.
	if generation != b.generation {
		return circuitNoTransition
	}
	switch b.state {
	case circuitHalfOpen:
		// The only call admitted while half-open is the probe.
		b.probing = false
		switch result {
		case renderResultTimeout:
			b.open(min(2*b.current, b.maxOpenDuration))
			return circuitOpened
		case renderResultCanceled:
			// The caller went away; the next admitted call probes instead.
			return circuitNoTransition
		default:
			// Any other completion shows the endpoint answers within the budget.
			b.setState(circuitClosed)
			b.timeouts = 0
			return circuitClosedAgain
		}
	case circuitClosed:
		switch result {
		case renderResultTimeout:
			b.timeouts++
			if b.timeouts < b.consecutiveTimeouts {
				return circuitNoTransition
			}
			b.open(b.openDuration)
			return circuitOpened
		case renderResultSuccess:
			b.timeouts = 0
		}
	}
	return circuitNoTransition
}

func (b *renderCircuitBreaker) setState(s circuitState) {
	b.state = s
	b.generation++
}

func (b *renderCircuitBreaker) open(d time.Duration) {
	b.setState(circuitOpen)
	b.timeouts = 0
	b.current = d
	b.openUntil = b.now().Add(d)
}

// openFor returns the current open duration, for logging.
func (b *renderCircuitBreaker) openFor() time.Duration {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.current
}
