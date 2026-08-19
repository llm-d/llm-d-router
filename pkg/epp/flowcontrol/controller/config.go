/*
Copyright 2025 The Kubernetes Authors.

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

package controller

import (
	"fmt"
	"time"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
)

const (
	// defaultRequestTTL is the default Time-To-Live applied to queued requests when the configuration
	// does not specify one. It is a queue-wait budget: a request still undispatched after this long is
	// shed with a retryable backpressure signal instead of being served with severely degraded
	// time-to-first-token. With no client or gateway deadline (the well-lit guides configure no gateway
	// request timeout) it is the only bound on waiting against a pool that has endpoints; where such
	// deadlines exist and fire sooner, context cancellation evicts the request first.
	defaultRequestTTL = 60 * time.Second
	// defaultNoEndpointRequestTTL is the queue-wait budget applied while the candidate pool has no endpoints when
	// neither budget is configured. It is deliberately equal to `defaultRequestTTL`, so splitting the budget is opt-in:
	// sizing the two regimes apart (a cold start is minutes, a time-to-first-token SLO is seconds) is a
	// deployment-specific decision, not one a default can make.
	defaultNoEndpointRequestTTL = 60 * time.Second
	// defaultExpiryCleanupInterval is the default frequency for scanning for expired items.
	defaultExpiryCleanupInterval = 1 * time.Second
	// defaultEnqueueChannelBufferSize is the default size of a worker's incoming request buffer.
	defaultEnqueueChannelBufferSize = 100
	// defaultMaxRevocationsPerDecision caps revocations per reclamation decision, bounding the
	// damage from mean-footprint misestimation.
	defaultMaxRevocationsPerDecision = 2
	// defaultEvictionConfirmationGrace covers the reclaiming stage (engine abort, KV GC, scrape,
	// staleness window) for the default utilization detector. Deployments pairing eviction with the
	// concurrency detector can lower it substantially.
	defaultEvictionConfirmationGrace = 500 * time.Millisecond
	// defaultEvictionConfirmationTimeout bounds how long an unconfirmed revocation can hold the
	// reclamation pacing gate closed.
	defaultEvictionConfirmationTimeout = 10 * time.Second
)

// Config holds the configuration for the `FlowController`.
type Config struct {
	// DefaultRequestTTL is the default Time-To-Live applied to requests that do not specify their own
	// TTL hint. Because the admission adapter does not currently plumb a per-request hint, this value
	// governs every request entering flow control while the candidate pool has endpoints.
	// Optional: Defaults to `defaultRequestTTL` (60s). An explicit zero disables eviction in that
	// regime, and, unless `NoEndpointRequestTTL` overrides it, in the empty-pool regime as well; such
	// requests are then bounded only by request context cancellation (client disconnect or gateway
	// timeout).
	DefaultRequestTTL time.Duration

	// NoEndpointRequestTTL is the queue-wait budget that replaces `DefaultRequestTTL` while the candidate
	// pool has no endpoints. Which budget is in force is re-evaluated as the request waits, and each
	// change of regime starts a fresh budget, so a request queued against an empty pool is not shed the
	// instant an endpoint appears and makes it dispatchable.
	// Optional: Follows `DefaultRequestTTL` when the configuration sets that alone, and defaults to
	// `defaultNoEndpointRequestTTL` (60s) when it sets neither. An explicit zero disables eviction while the pool is
	// empty, in which case such requests are bounded only by request context cancellation.
	NoEndpointRequestTTL time.Duration

	// ExpiryCleanupInterval is the interval at which each processor scans its queues for expired items.
	// Optional: Defaults to `defaultExpiryCleanupInterval` (1 second).
	ExpiryCleanupInterval time.Duration

	// EnqueueChannelBufferSize is the size of the buffered channel that accepts incoming requests for each
	// processor. This buffer acts as a shock absorber, decoupling the high-frequency distributor from the processor's
	// serial execution loop and allowing the system to handle short bursts of traffic without blocking.
	// Optional: Defaults to `defaultEnqueueChannelBufferSize` (100).
	EnqueueChannelBufferSize int

	// EnableEviction enables demand-driven in-flight eviction: when higher-priority requests are
	// blocked by pool saturation, lower-priority in-flight requests may be terminated to reclaim
	// capacity. Requires the eviction plumbing to be wired (see Deps.InFlightEvictor).
	// See docs/flow-control-eviction.md.
	// Optional: Defaults to false.
	EnableEviction bool

	// MaxRevocationsPerDecision caps how many revocations a single reclamation decision may issue.
	// Not exposed through the API configuration: benchmark data shows sizing is deficit-bound, so
	// this cap rarely binds and is not worth a user-facing knob.
	// Optional: Defaults to `defaultMaxRevocationsPerDecision` (2).
	MaxRevocationsPerDecision int

	// EvictionConfirmationGrace is how long a confirmed revocation's pending-reclaim debit keeps
	// suppressing further reclamation, covering the saturation signal's confirmation-to-visibility
	// lag. Not exposed through the API configuration: the EPP wiring derives it from the selected
	// saturation detector, since the correct value is a property of the sensor, not a preference.
	// Optional: Defaults to `defaultEvictionConfirmationGrace` (500ms), the conservative value for
	// scraped sensors.
	EvictionConfirmationGrace time.Duration

	// EvictionConfirmationTimeout bounds how long an unconfirmed revocation can hold the
	// reclamation pacing gate closed before being treated as confirmed. Not exposed through the
	// API configuration.
	// Optional: Defaults to `defaultEvictionConfirmationTimeout` (10s).
	EvictionConfirmationTimeout time.Duration
}

func (c *Config) String() string {
	if c == nil {
		return "<nil>"
	}
	// Define a local type definition to prevent infinite recursion when calling Sprintf("%+v").
	// A new type definition inherits the struct fields but does not copy its methods,
	// bypassing the Stringer check and allowing a safe reflection-based field dump.
	type temp Config
	return fmt.Sprintf("%+v", temp(*c))
}

// ConfigOption is a functional option for configuring the FlowController.
type ConfigOption func(*Config)

// NewConfigFromAPI creates a new Config from the API configuration.
func NewConfigFromAPI(apiConfig *configapi.FlowControlConfig) (*Config, error) {
	opts := make([]ConfigOption, 0, 4)
	if apiConfig != nil {
		if apiConfig.DefaultRequestTTL != nil {
			opts = append(opts, WithDefaultRequestTTL(apiConfig.DefaultRequestTTL.Duration))
		}
		switch {
		case apiConfig.NoEndpointRequestTTL != nil:
			opts = append(opts, WithNoEndpointRequestTTL(apiConfig.NoEndpointRequestTTL.Duration))
		case apiConfig.DefaultRequestTTL != nil:
			// Left unset, the no-endpoint budget follows `DefaultRequestTTL`, so splitting the regimes is opt-in: a
			// deployment that bounded queue wait with the single field it had keeps that bound in both regimes, and an
			// explicit "0s" still disables eviction outright.
			opts = append(opts, WithNoEndpointRequestTTL(apiConfig.DefaultRequestTTL.Duration))
		}
		if apiConfig.EnableEviction {
			opts = append(opts, WithEnableEviction(true))
		}
	}
	return NewConfig(opts...)
}

// NewConfig creates a new Config with the given options, applying defaults and validation.
func NewConfig(opts ...ConfigOption) (*Config, error) {
	c := &Config{
		DefaultRequestTTL:           defaultRequestTTL,
		NoEndpointRequestTTL:        defaultNoEndpointRequestTTL,
		ExpiryCleanupInterval:       defaultExpiryCleanupInterval,
		EnqueueChannelBufferSize:    defaultEnqueueChannelBufferSize,
		MaxRevocationsPerDecision:   defaultMaxRevocationsPerDecision,
		EvictionConfirmationGrace:   defaultEvictionConfirmationGrace,
		EvictionConfirmationTimeout: defaultEvictionConfirmationTimeout,
	}

	for _, opt := range opts {
		opt(c)
	}

	if err := c.validate(); err != nil {
		return nil, err
	}
	return c, nil
}

// WithDefaultRequestTTL sets the default request TTL.
func WithDefaultRequestTTL(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.DefaultRequestTTL = d
	}
}

// WithNoEndpointRequestTTL sets the queue-wait budget applied while the pool has no endpoints.
func WithNoEndpointRequestTTL(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.NoEndpointRequestTTL = d
	}
}

// WithExpiryCleanupInterval sets the expiry cleanup interval.
func WithExpiryCleanupInterval(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.ExpiryCleanupInterval = d
	}
}

// WithEnqueueChannelBufferSize sets the size of the enqueue channel buffer.
func WithEnqueueChannelBufferSize(size int) ConfigOption {
	return func(c *Config) {
		c.EnqueueChannelBufferSize = size
	}
}

// WithEnableEviction enables demand-driven in-flight eviction.
func WithEnableEviction(enabled bool) ConfigOption {
	return func(c *Config) {
		c.EnableEviction = enabled
	}
}

// WithMaxRevocationsPerDecision sets the per-decision revocation cap.
func WithMaxRevocationsPerDecision(n int) ConfigOption {
	return func(c *Config) {
		c.MaxRevocationsPerDecision = n
	}
}

// WithEvictionConfirmationGrace sets the post-confirmation grace period.
func WithEvictionConfirmationGrace(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.EvictionConfirmationGrace = d
	}
}

// WithEvictionConfirmationTimeout sets the confirmation timeout.
func WithEvictionConfirmationTimeout(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.EvictionConfirmationTimeout = d
	}
}

// validate checks the configuration for validity.
func (c *Config) validate() error {
	if c.DefaultRequestTTL < 0 {
		return fmt.Errorf("DefaultRequestTTL cannot be negative, but got %v", c.DefaultRequestTTL)
	}
	if c.NoEndpointRequestTTL < 0 {
		return fmt.Errorf("NoEndpointRequestTTL cannot be negative, but got %v", c.NoEndpointRequestTTL)
	}
	if c.ExpiryCleanupInterval <= 0 {
		return fmt.Errorf("ExpiryCleanupInterval must be positive, but got %v", c.ExpiryCleanupInterval)
	}
	if c.EnqueueChannelBufferSize < 0 {
		return fmt.Errorf("EnqueueChannelBufferSize cannot be negative, but got %d", c.EnqueueChannelBufferSize)
	}
	if c.MaxRevocationsPerDecision < 1 {
		return fmt.Errorf("MaxRevocationsPerDecision must be at least 1, but got %d", c.MaxRevocationsPerDecision)
	}
	if c.EvictionConfirmationGrace < 0 {
		return fmt.Errorf("EvictionConfirmationGrace cannot be negative, but got %v", c.EvictionConfirmationGrace)
	}
	if c.EvictionConfirmationTimeout <= 0 {
		return fmt.Errorf("EvictionConfirmationTimeout must be positive, but got %v", c.EvictionConfirmationTimeout)
	}
	return nil
}
