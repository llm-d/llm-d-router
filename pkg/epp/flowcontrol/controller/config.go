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
	// time-to-first-token. It is also the only bound on queue wait when neither the client nor the
	// gateway enforces a request deadline (the well-lit guides configure no gateway request timeout);
	// where such deadlines exist and fire sooner, context cancellation evicts the request first.
	defaultRequestTTL = 60 * time.Second
	// defaultNoEndpointRequestTTL is the default queue-wait budget applied while the candidate pool has
	// no endpoints. It is deliberately equal to `defaultRequestTTL`, so splitting the budget is opt-in:
	// sizing the two regimes apart (a cold start is minutes, a time-to-first-token SLO is seconds) is a
	// deployment-specific decision, not one a default can make.
	defaultNoEndpointRequestTTL = 60 * time.Second
	// defaultExpiryCleanupInterval is the default frequency for scanning for expired items.
	defaultExpiryCleanupInterval = 1 * time.Second
	// defaultEnqueueChannelBufferSize is the default size of a worker's incoming request buffer.
	defaultEnqueueChannelBufferSize = 100
)

// Config holds the configuration for the `FlowController`.
type Config struct {
	// DefaultRequestTTL is the default Time-To-Live applied to requests that do not specify their own
	// TTL hint. Because the admission adapter does not currently plumb a per-request hint, this value
	// governs every request entering flow control.
	// Optional: Defaults to `defaultRequestTTL` (60s). An explicit zero disables the TTL entirely, in
	// which case queued requests are bounded only by request context cancellation (client disconnect
	// or gateway timeout).
	DefaultRequestTTL time.Duration

	// NoEndpointRequestTTL is the queue-wait budget that replaces `DefaultRequestTTL` while the candidate
	// pool has no endpoints. Which budget is in force is re-evaluated as the request waits, and each
	// change of regime starts a fresh budget, so a request queued against an empty pool is not shed the
	// instant an endpoint appears and makes it dispatchable.
	// Optional: Defaults to `defaultNoEndpointRequestTTL` (60s). An explicit zero disables eviction while
	// the pool is empty, in which case such requests are bounded only by request context cancellation.
	NoEndpointRequestTTL time.Duration

	// ExpiryCleanupInterval is the interval at which each processor scans its queues for expired items.
	// Optional: Defaults to `defaultExpiryCleanupInterval` (1 second).
	ExpiryCleanupInterval time.Duration

	// EnqueueChannelBufferSize is the size of the buffered channel that accepts incoming requests for each
	// processor. This buffer acts as a shock absorber, decoupling the high-frequency distributor from the processor's
	// serial execution loop and allowing the system to handle short bursts of traffic without blocking.
	// Optional: Defaults to `defaultEnqueueChannelBufferSize` (100).
	EnqueueChannelBufferSize int
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
	opts := make([]ConfigOption, 0, 2)
	if apiConfig != nil {
		if apiConfig.DefaultRequestTTL != nil {
			opts = append(opts, WithDefaultRequestTTL(apiConfig.DefaultRequestTTL.Duration))
		}
		if apiConfig.NoEndpointRequestTTL != nil {
			opts = append(opts, WithNoEndpointRequestTTL(apiConfig.NoEndpointRequestTTL.Duration))
		}
	}
	return NewConfig(opts...)
}

// NewConfig creates a new Config with the given options, applying defaults and validation.
func NewConfig(opts ...ConfigOption) (*Config, error) {
	c := &Config{
		DefaultRequestTTL:        defaultRequestTTL,
		NoEndpointRequestTTL:     defaultNoEndpointRequestTTL,
		ExpiryCleanupInterval:    defaultExpiryCleanupInterval,
		EnqueueChannelBufferSize: defaultEnqueueChannelBufferSize,
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
	return nil
}
