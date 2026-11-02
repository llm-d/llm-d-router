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

// errRenderCircuitOpen is returned for a render call that the circuit breaker skipped.
var errRenderCircuitOpen = errors.New("render circuit breaker is open")

// circuitBreakerConfig enables the render circuit breaker. Empty fields use the defaults.
type circuitBreakerConfig struct {
	// ConsecutiveTimeouts is the number of consecutive timed-out render calls
	// that opens the circuit. Defaults to 5.
	ConsecutiveTimeouts int `json:"consecutiveTimeouts,omitempty"`
	// OpenDuration is how long the circuit stays open after it first opens
	// (Go duration string). Defaults to 10s.
	OpenDuration string `json:"openDuration,omitempty"`
	// MaxOpenDuration caps the open duration, which doubles each time the
	// circuit opens again soon after closing. Defaults to 2m.
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

// renderCircuitBreaker tracks render timeouts for one renderer. While closed,
// consecutive timeouts open it. While open, render calls are skipped. When the
// open duration elapses it is half-open: one render call probes the endpoint
// and the others are skipped until the probe completes.
type renderCircuitBreaker struct {
	consecutiveTimeouts int
	openDuration        time.Duration
	maxOpenDuration     time.Duration
	now                 func() time.Time

	mu        sync.Mutex
	state     circuitState
	timeouts  int
	current   time.Duration
	openUntil time.Time
	closedAt  time.Time
	probing   bool
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

// admit reports whether a render call may proceed and whether it is the
// half-open probe, whose outcome must be passed to record.
func (b *renderCircuitBreaker) admit() (allowed, probe bool, transition circuitTransition) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case circuitClosed:
		return true, false, circuitNoTransition
	case circuitOpen:
		if b.now().Before(b.openUntil) {
			return false, false, circuitNoTransition
		}
		b.state = circuitHalfOpen
		transition = circuitHalfOpened
	}
	if b.probing {
		return false, false, transition
	}
	b.probing = true
	return true, true, transition
}

// record applies the result label of an admitted render call.
func (b *renderCircuitBreaker) record(probe bool, result string) circuitTransition {
	b.mu.Lock()
	defer b.mu.Unlock()
	if probe {
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
			b.state = circuitClosed
			b.timeouts = 0
			b.closedAt = b.now()
			return circuitClosedAgain
		}
	}
	// Calls admitted before the circuit opened can complete after it.
	if b.state != circuitClosed {
		return circuitNoTransition
	}
	switch result {
	case renderResultTimeout:
		b.timeouts++
		if b.timeouts < b.consecutiveTimeouts {
			return circuitNoTransition
		}
		next := b.openDuration
		// Opening again soon after closing means the endpoint is still
		// overloaded, so the open duration keeps growing.
		if !b.closedAt.IsZero() && b.now().Sub(b.closedAt) < b.maxOpenDuration {
			next = min(2*b.current, b.maxOpenDuration)
		}
		b.open(next)
		return circuitOpened
	case renderResultSuccess:
		b.timeouts = 0
	}
	return circuitNoTransition
}

func (b *renderCircuitBreaker) open(d time.Duration) {
	b.state = circuitOpen
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
