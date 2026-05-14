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

// Note on Time-Based Lifecycle Tests:
// Tests validating the controller's handling of request TTLs (e.g., OnReqCtxTimeout*) rely on real-time timers
// (context.WithDeadline). The injected testclock.FakeClock is used to control the timing of internal loops (like
// reconciliation), but it cannot manipulate the timers used by the standard context package. Therefore, these specific
// tests use time.Sleep or assertions on real-time durations.

package controller

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"
	testclock "k8s.io/utils/clock/testing"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/controller/internal"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

// --- Test Harness & Fixtures ---

type mockSaturationDetector struct {
	flowcontrol.SaturationDetector
}

func (m *mockSaturationDetector) Saturation(_ context.Context, _ []datalayer.Endpoint) float64 {
	return 0.0
}

// testHarness holds the `FlowController` and its dependencies under test.
type testHarness struct {
	fc  *FlowController
	cfg *Config
	// clock is the clock interface used by the controller.
	clock        clock.WithTicker
	mockRegistry *mockRegistryClient
	// mockClock provides access to FakeClock methods (Step, HasWaiters) if and only if the underlying clock is a
	// FakeClock.
	mockClock            *testclock.FakeClock
	mockProcessorFactory *mockShardProcessorFactory
}

// unitHarnessOption allows configuring the test harness.
type unitHarnessOption func(*testHarnessOpts)

type testHarnessOpts struct {
	clock clock.WithTicker
}

func withHarnessClock(c clock.WithTicker) unitHarnessOption {
	return func(o *testHarnessOpts) {
		o.clock = c
	}
}

// newUnitHarness creates a test environment with a mock processor factory, suitable for focused unit tests of the
// controller's logic. It starts the controller's run loop using the provided context for lifecycle management.
func newUnitHarness(
	ctx context.Context,
	t *testing.T,
	cfg *Config,
	registry *mockRegistryClient,
	opts ...unitHarnessOption,
) *testHarness {
	t.Helper()

	harnessOpts := &testHarnessOpts{
		clock: testclock.NewFakeClock(time.Now()),
	}
	for _, opt := range opts {
		opt(harnessOpts)
	}

	mockDetector := &mockSaturationDetector{}
	mockEndpointCandidates := &mocks.MockEndpointCandidates{}

	mockProcessorFactory := &mockShardProcessorFactory{}

	usageLimitPolicy := usagelimits.DefaultPolicy()

	// Default the registry if nil, simplifying tests that don't focus on registry interaction.
	if registry == nil {
		registry = &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}
	}

	fc, err := NewFlowController(ctx, "test-pool", cfg, Deps{
		Registry:           registry,
		SaturationDetector: mockDetector,
		EndpointCandidates: mockEndpointCandidates,
		UsageLimitPolicy:   usageLimitPolicy,
		Clock:              harnessOpts.clock,
	})
	require.NoError(t, err, "failed to create FlowController for unit test harness")
	fc.shardProcessorFactory = mockProcessorFactory.new

	h := &testHarness{
		fc:                   fc,
		cfg:                  cfg,
		clock:                harnessOpts.clock,
		mockRegistry:         registry,
		mockProcessorFactory: mockProcessorFactory,
	}

	if fc, ok := harnessOpts.clock.(*testclock.FakeClock); ok {
		h.mockClock = fc
	}

	return h
}

// newIntegrationHarness creates a test environment that uses real `ShardProcessor`s, suitable for integration tests
// validating the controller-processor interaction.
func newIntegrationHarness(ctx context.Context, t *testing.T, cfg *Config, registry *mockRegistryClient) *testHarness {
	t.Helper()
	mockDetector := &mockSaturationDetector{}
	mockEndpointCandidates := &mocks.MockEndpointCandidates{}
	usageLimitPolicy := usagelimits.DefaultPolicy()

	// Align FakeClock with system time. See explanation in newUnitHarness.
	mockClock := testclock.NewFakeClock(time.Now())
	if registry == nil {
		registry = &mockRegistryClient{
			FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{},
		}
	}

	fc, err := NewFlowController(ctx, "test-pool", cfg, Deps{
		Registry:           registry,
		SaturationDetector: mockDetector,
		EndpointCandidates: mockEndpointCandidates,
		UsageLimitPolicy:   usageLimitPolicy,
		Clock:              mockClock,
	})
	require.NoError(t, err, "failed to create FlowController for integration test harness")

	h := &testHarness{
		fc:           fc,
		cfg:          cfg,
		clock:        mockClock,
		mockRegistry: registry,
		mockClock:    mockClock,
	}
	return h
}

// mockActiveFlowConnection is a local mock for the `contracts.ActiveFlowConnection` interface.
type mockActiveFlowConnection struct {
	RegistryV    contracts.FlowRegistry
	RegistryFunc func() contracts.FlowRegistry
	FlowKeyV     flowcontrol.FlowKey
}

func (m *mockActiveFlowConnection) GetDataPlane() contracts.FlowRegistryDataPlane {
	if m.RegistryFunc != nil {
		return m.RegistryFunc()
	}
	return m.RegistryV
}

func (m *mockActiveFlowConnection) FlowKey() flowcontrol.FlowKey {
	return m.FlowKeyV
}

// mockRegistryClient is a mock for the private `registryClient` interface.
type mockRegistryClient struct {
	contracts.FlowRegistryObserver
	contracts.FlowRegistryDataPlane
	WithConnectionFunc func(key flowcontrol.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error
	StatsFunc          func() contracts.AggregateStats
}

func (m *mockRegistryClient) WithConnection(
	key flowcontrol.FlowKey,
	fn func(conn contracts.ActiveFlowConnection) error,
) error {
	if m.WithConnectionFunc != nil {
		return m.WithConnectionFunc(key, fn)
	}
	return fn(&mockActiveFlowConnection{RegistryV: m})
}

func (m *mockRegistryClient) Stats() contracts.AggregateStats {
	if m.StatsFunc != nil {
		return m.StatsFunc()
	}
	return contracts.AggregateStats{}
}

// mockShardProcessor is a mock for the internal `shardProcessor` interface.
type mockShardProcessor struct {
	SubmitFunc        func(item *internal.FlowItem) error
	SubmitOrBlockFunc func(ctx context.Context, item *internal.FlowItem) error
	// runCtx captures the context provided to the Run method for lifecycle assertions.
	runCtx   context.Context
	runCtxMu sync.RWMutex
	// runStarted is closed when the Run method is called, allowing tests to synchronize with worker startup.
	runStarted chan struct{}
}

func (m *mockShardProcessor) Submit(item *internal.FlowItem) error {
	if m.SubmitFunc != nil {
		return m.SubmitFunc(item)
	}
	return nil
}

func (m *mockShardProcessor) SubmitOrBlock(ctx context.Context, item *internal.FlowItem) error {
	if m.SubmitOrBlockFunc != nil {
		return m.SubmitOrBlockFunc(ctx, item)
	}
	return nil
}

func (m *mockShardProcessor) Run(ctx context.Context) {
	m.runCtxMu.Lock()
	m.runCtx = ctx
	m.runCtxMu.Unlock()
	if m.runStarted != nil {
		close(m.runStarted)
	}
	// Block until the context is cancelled, simulating a running worker.
	<-ctx.Done()
}

// Context returns the context captured during the Run method call.
func (m *mockShardProcessor) Context() context.Context {
	m.runCtxMu.RLock()
	defer m.runCtxMu.RUnlock()
	return m.runCtx
}

// mockShardProcessorFactory allows tests to inject specific `mockShardProcessor` instances.
type mockShardProcessorFactory struct {
	processor atomic.Pointer[mockShardProcessor]
}

// new is the factory function conforming to the `shardProcessorFactory` signature.
func (f *mockShardProcessorFactory) new(
	_ context.Context, // The factory does not use the lifecycle context; it's passed to the processor's Run method later.
	_ contracts.FlowRegistry,
	_ flowcontrol.SaturationDetector,
	_ contracts.EndpointCandidates,
	_ flowcontrol.UsageLimitPolicy,
	_ clock.WithTicker,
	_ time.Duration,
	_ int,
	_ logr.Logger,
) processor {
	if proc := f.processor.Load(); proc != nil {
		return proc
	}
	// Return a default mock processor if one is not explicitly registered by the test.
	return &mockShardProcessor{}
}

var defaultFlowKey = flowcontrol.FlowKey{ID: "test-flow", Priority: 100}

func newTestRequest(key flowcontrol.FlowKey) *fwkfcmocks.MockFlowControlRequest {
	return &fwkfcmocks.MockFlowControlRequest{
		FlowKeyV:  key,
		ByteSizeV: 100,
		IDV:       "req-" + key.ID,
	}
}

// --- Test Cases ---

// TestFlowController_EnqueueAndWait covers the primary API entry point, focusing on validation, distribution logic,
// retries, and the request lifecycle (including post-distribution cancellation/timeout).
func TestFlowController_EnqueueAndWait(t *testing.T) {
	t.Parallel()

	t.Run("Rejections", func(t *testing.T) {
		t.Parallel()

		t.Run("OnReqCtxExpiredBeforeDistribution", func(t *testing.T) {
			t.Parallel()
			// Test that if the request context provided to EnqueueAndWait is already expired, it returns immediately.
			h := newUnitHarness(t.Context(), t, &Config{DefaultRequestTTL: 1 * time.Minute}, nil)

			h.mockRegistry.WithConnectionFunc = func(key flowcontrol.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{
					RegistryV: h.mockRegistry,
					FlowKeyV:  key,
				})
			}
			h.mockRegistry.FlowRegistryDataPlane = &mocks.MockRegistryDataPlane{}
			// Configure processor to block until context expiry.
			h.mockProcessorFactory.processor.Store(&mockShardProcessor{
				SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
					<-ctx.Done()              // Wait for the context to be done.
					return context.Cause(ctx) // Return the cause.
				},
			})

			req := newTestRequest(defaultFlowKey)
			// Use a context with a deadline in the past.
			reqCtx, cancel := context.WithDeadlineCause(
				context.Background(),
				h.clock.Now().Add(-1*time.Second),
				types.ErrTTLExpired)
			defer cancel()

			outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
			require.Error(t, err, "EnqueueAndWait must fail if request context deadline is exceeded")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "error should wrap types.ErrTTLExpired from the context cause")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "outcome should be QueueOutcomeRejectedOther")
		})
		t.Run("OnControllerShutdown", func(t *testing.T) {
			t.Parallel()
			// Create a context specifically for the controller's lifecycle.
			ctx, cancel := context.WithCancel(t.Context())
			h := newUnitHarness(ctx, t, &Config{}, nil)
			cancel() // Immediately stop the controller.

			// Wait for the controller's run loop and all workers (none in this case) to exit.
			// We need to wait because the shutdown process is asynchronous.
			h.fc.wg.Wait()

			req := newTestRequest(defaultFlowKey)
			// The request context is valid, but the controller itself is stopped.
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if controller is not running")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "error should wrap ErrFlowControllerNotRunning")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther on shutdown")
		})

		t.Run("OnRegistryConnectionError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}
			h := newUnitHarness(t.Context(), t, &Config{}, mockRegistry)

			expectedErr := errors.New("simulated connection failure")
			// Configure the registry to fail when attempting to retrieve ActiveFlowConnection.
			mockRegistry.WithConnectionFunc = func(
				_ flowcontrol.FlowKey,
				_ func(conn contracts.ActiveFlowConnection) error,
			) error {
				return expectedErr
			}

			req := newTestRequest(defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if registry connection fails")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, expectedErr, "error should wrap the underlying connection error")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther for transient registry errors")
		})

		t.Run("OnManagedQueueError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}
			h := newUnitHarness(t.Context(), t, &Config{}, mockRegistry)

			// Create a faulty shard that successfully leases the flow but fails to return the
			// ManagedQueue. This shard should be considered as unavailable.
			faultyRegistry := &mocks.MockRegistryDataPlane{
				ManagedQueueFunc: func(_ flowcontrol.FlowKey) (contracts.ManagedQueue, error) {
					return nil, errors.New("invariant violation: queue retrieval failed")
				},
			}
			mockRegistry.WithConnectionFunc = func(
				key flowcontrol.FlowKey,
				fn func(conn contracts.ActiveFlowConnection) error,
			) error {
				return fn(&mockActiveFlowConnection{
					RegistryV: faultyRegistry,
					FlowKeyV:  key,
				})
			}

			req := newTestRequest(defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if no shards are available")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome,
				"outcome should be QueueOutcomeRejectedCapacity when no shards exist for the flow")
		})
	})

	// Distribution tests validate the JSQ-Bytes algorithm, the two-phase submission strategy, and error handling during
	// the handoff, including time-based failures during blocking fallback.
	t.Run("Distribution", func(t *testing.T) {
		t.Parallel()

		// Define a long default TTL to prevent unexpected timeouts unless a test case explicitly sets a shorter one.
		const defaultTestTTL = 5 * time.Second

		testCases := []struct {
			name           string
			setupProcessor func(t *testing.T, h *testHarness)
			// requestTTL overrides the default TTL for time-sensitive tests.
			requestTTL      time.Duration
			expectedOutcome types.QueueOutcome
			expectErr       bool
			expectErrIs     error
		}{
			{
				name: "SubmitSucceeds_NonBlocking",
				setupProcessor: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processor.Store(&mockShardProcessor{
						SubmitFunc: func(item *internal.FlowItem) error {
							// Simulate asynchronous processing and successful dispatch.
							go item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
							return nil
						},
					})
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				// Validates the scenario where the request's TTL expires while the controller is blocked waiting for capacity.
				// NOTE: This relies on real time passing, as context.WithDeadline timers cannot be controlled by FakeClock.
				name:       "Rejects_AfterBlocking_WhenTTL_Expires",
				requestTTL: 50 * time.Millisecond, // Short TTL to keep the test fast.
				setupProcessor: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processor.Store(&mockShardProcessor{
						// Reject the non-blocking attempt.
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						// Block the fallback attempt until the context (carrying the TTL deadline) expires.
						SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
							<-ctx.Done()
							return ctx.Err()
						},
					})
				},
				// No runActions needed; we rely on the real-time timer to expire.
				// When the blocking call fails due to context expiry, the outcome is RejectedOther.
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				// The error must reflect the specific cause of the context cancellation (ErrTTLExpired).
				expectErrIs: types.ErrTTLExpired,
			},
			{
				name: "Rejects_OnProcessorShutdownDuringSubmit",
				setupProcessor: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processor.Store(&mockShardProcessor{
						// Simulate the processor shutting down during the non-blocking handoff.
						SubmitFunc: func(_ *internal.FlowItem) error { return types.ErrFlowControllerNotRunning },
						SubmitOrBlockFunc: func(_ context.Context, _ *internal.FlowItem) error {
							return types.ErrFlowControllerNotRunning
						},
					})
				},
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				expectErrIs:     types.ErrFlowControllerNotRunning,
			},
			{
				name: "Rejects_OnProcessorShutdownDuringSubmitOrBlock",
				setupProcessor: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processor.Store(&mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						// Simulate the processor shutting down during the blocking handoff.
						SubmitOrBlockFunc: func(_ context.Context, _ *internal.FlowItem) error {
							return types.ErrFlowControllerNotRunning
						},
					})
				},
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				expectErrIs:     types.ErrFlowControllerNotRunning,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				// Arrange
				mockRegistry := &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}

				// Configure the harness with the appropriate TTL.
				harnessConfig := &Config{DefaultRequestTTL: defaultTestTTL}
				if tc.requestTTL > 0 {
					harnessConfig.DefaultRequestTTL = tc.requestTTL
				}
				h := newUnitHarness(t.Context(), t, harnessConfig, mockRegistry)

				// Configure the registry to return the specified shards.
				mockRegistry.WithConnectionFunc = func(
					key flowcontrol.FlowKey,
					fn func(conn contracts.ActiveFlowConnection) error,
				) error {
					return fn(&mockActiveFlowConnection{
						RegistryV: h.mockRegistry,
						FlowKeyV:  key,
					})
				}
				tc.setupProcessor(t, h)

				// Act
				var outcome types.QueueOutcome
				var err error

				startTime := time.Now() // Capture real start time for duration checks.
				// Use a background context for the parent; the request lifecycle is governed by the config/derived context.
				outcome, err = h.fc.EnqueueAndWait(context.Background(), newTestRequest(defaultFlowKey))

				// Assert
				if tc.expectErr {
					require.Error(t, err, "expected an error during EnqueueAndWait but got nil")
					assert.ErrorIs(t, err, tc.expectErrIs, "error should wrap the expected underlying cause")
					// All failures during the distribution phase (capacity, timeout, shutdown) should result in a rejection.
					assert.ErrorIs(t, err, types.ErrRejected, "rejection errors must wrap types.ErrRejected")

					// Specific assertion for real-time TTL tests.
					if errors.Is(tc.expectErrIs, types.ErrTTLExpired) {
						duration := time.Since(startTime)
						// Ensure the test didn't return instantly. Use a tolerance for CI environments.
						// This validates that the real-time wait actually occurred.
						assert.GreaterOrEqual(t, duration, tc.requestTTL-30*time.Millisecond,
							"EnqueueAndWait returned faster than the TTL allows, indicating the timer did not function correctly")
					}

				} else {
					require.NoError(t, err, "expected no error during EnqueueAndWait but got: %v", err)
				}
				assert.Equal(t, tc.expectedOutcome, outcome, "outcome did not match expected value")
			})
		}
	})

	t.Run("Retry", func(t *testing.T) {
		t.Parallel()

		// This test specifically validates the behavior when the request context is cancelled externally while the
		// controller is blocked in the SubmitOrBlock phase.
		t.Run("Rejects_OnRequestContextCancelledWhileBlocking", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}
			mockRegistry.WithConnectionFunc = func(
				key flowcontrol.FlowKey,
				fn func(conn contracts.ActiveFlowConnection,
				) error) error {
				return fn(&mockActiveFlowConnection{
					RegistryV: mockRegistry,
					FlowKeyV:  key,
				})
			}
			// Use a long TTL to ensure the failure is due to cancellation, not timeout.
			h := newUnitHarness(t.Context(), t, &Config{DefaultRequestTTL: 10 * time.Second}, mockRegistry)
			h.mockProcessorFactory.processor.Store(&mockShardProcessor{
				// Reject non-blocking attempt.
				SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				// Block the fallback attempt until the context is cancelled.
				SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
					<-ctx.Done()
					return ctx.Err()
				},
			})

			// Create a cancellable context for the request.
			reqCtx, cancelReq := context.WithCancel(context.Background())
			// Cancel the request shortly after starting the operation.
			// We use real time sleep here as we are testing external cancellation signals interacting with the context.
			go func() { time.Sleep(10 * time.Millisecond); cancelReq() }()

			outcome, err := h.fc.EnqueueAndWait(reqCtx, newTestRequest(defaultFlowKey))

			require.Error(t, err, "EnqueueAndWait must fail when context is cancelled during a blocking submit")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, context.Canceled, "error should wrap the underlying ctx.Err() (context.Canceled)")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther when cancelled during distribution")
		})
	})

	// Lifecycle covers the post-distribution phase, focusing on how the controller handles context cancellation and TTL
	// expiry while the request is buffered or queued by the processor (Asynchronous Finalization).
	t.Run("Lifecycle", func(t *testing.T) {
		t.Parallel()

		// Validates that the controller correctly initiates asynchronous finalization when the request context is cancelled
		// after ownership has been transferred to the processor.
		t.Run("OnReqCtxCancelledAfterDistribution", func(t *testing.T) {
			t.Parallel()
			// Use a long TTL to ensure the failure is due to cancellation.
			h := newUnitHarness(t.Context(), t, &Config{DefaultRequestTTL: 10 * time.Second}, nil)

			h.mockRegistry.WithConnectionFunc = func(key flowcontrol.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{
					RegistryV: h.mockRegistry,
					FlowKeyV:  key,
				})
			}
			h.mockRegistry.FlowRegistryDataPlane = &mocks.MockRegistryDataPlane{}

			// Channel for synchronization.
			itemSubmitted := make(chan *internal.FlowItem, 1)

			// Configure the processor to accept the item but never finalize it, simulating a queued request.
			h.mockProcessorFactory.processor.Store(&mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					item.SetHandle(&fwkfcmocks.MockQueueItemHandle{})
					itemSubmitted <- item
					return nil
				},
			})

			reqCtx, cancelReq := context.WithCancel(context.Background())
			req := newTestRequest(defaultFlowKey)

			var outcome types.QueueOutcome
			var err error
			done := make(chan struct{})
			go func() {
				outcome, err = h.fc.EnqueueAndWait(reqCtx, req)
				close(done)
			}()

			// 1. Wait for the item to be successfully distributed.
			var item *internal.FlowItem
			select {
			case item = <-itemSubmitted:
				// Success. Ownership has transferred. EnqueueAndWait is now in the select loop.
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for item to be submitted to the processor")
			}

			// 2. Cancel the request context.
			cancelReq()

			// 3. Wait for EnqueueAndWait to return.
			select {
			case <-done:
				// Success. The controller detected the cancellation and unblocked the caller.
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for EnqueueAndWait to return after cancellation")
			}

			// 4. Assertions for EnqueueAndWait's return values.
			require.Error(t, err, "EnqueueAndWait should return an error when the request is cancelled post-distribution")
			// The outcome should be Evicted (as the handle was set).
			assert.ErrorIs(t, err, types.ErrEvicted, "error should wrap ErrEvicted")
			// The underlying cause must be propagated.
			assert.ErrorIs(t, err, types.ErrContextCancelled, "error should wrap ErrContextCancelled")
			assert.Equal(t, types.QueueOutcomeEvictedContextCancelled, outcome, "outcome should be EvictedContextCancelled")

			// 5. Assert that the FlowItem itself was indeed finalized by the controller.
			finalState := item.FinalState()
			require.NotNil(t, finalState, "Item should have been finalized asynchronously by the controller")
			assert.Equal(t, types.QueueOutcomeEvictedContextCancelled, finalState.Outcome,
				"Item's internal outcome must match the returned outcome")
		})

		// Validates the asynchronous finalization path due to TTL expiry.
		// Note: This relies on real time passing, as context.WithDeadline timers cannot be controlled by FakeClock.
		t.Run("OnReqCtxTimeoutAfterDistribution", func(t *testing.T) {
			t.Parallel()
			// Configure a short TTL to keep the test reasonably fast.
			const requestTTL = 50 * time.Millisecond
			h := newUnitHarness(t.Context(), t, &Config{
				DefaultRequestTTL:               requestTTL,
				ProcessorReconciliationInterval: time.Minute,
				ExpiryCleanupInterval:           time.Minute,
			}, nil, withHarnessClock(clock.RealClock{}))

			h.mockRegistry.WithConnectionFunc = func(key flowcontrol.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{
					RegistryV: h.mockRegistry,
					FlowKeyV:  key,
				})
			}

			itemSubmitted := make(chan *internal.FlowItem, 1)

			// Configure the processor to accept the item but never finalize it.
			h.mockProcessorFactory.processor.Store(&mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					item.SetHandle(&fwkfcmocks.MockQueueItemHandle{})
					itemSubmitted <- item
					return nil
				},
			})

			req := newTestRequest(defaultFlowKey)
			// Use a context for the call itself that won't time out independently.
			enqueueCtx, enqueueCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer enqueueCancel()

			var outcome types.QueueOutcome
			var err error
			done := make(chan struct{})

			startTime := time.Now() // Capture start time to validate duration.
			go func() {
				outcome, err = h.fc.EnqueueAndWait(enqueueCtx, req)
				close(done)
			}()

			// 1. Wait for the item to be submitted.
			var item *internal.FlowItem
			select {
			case item = <-itemSubmitted:
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for item to be submitted to the processor")
			}

			// 2.Wait for the TTL to expire (Real time). We do NOT call Step().
			// Wait for EnqueueAndWait to return due to the TTL expiry.
			select {
			case <-done:
				// Success. Now validate that enough time actually passed.
				duration := time.Since(startTime)
				assert.GreaterOrEqual(t, duration, requestTTL-30*time.Millisecond, // tolerance for CI environments
					"EnqueueAndWait returned faster than the TTL allows, indicating the timer did not function correctly")
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for EnqueueAndWait to return after TTL expiry")
			}

			// 4. Assertions for EnqueueAndWait's return values.
			require.Error(t, err, "EnqueueAndWait should return an error when TTL expires post-distribution")
			assert.ErrorIs(t, err, types.ErrEvicted, "error should wrap ErrEvicted")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "error should wrap the underlying cause (types.ErrTTLExpired)")
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "outcome should be EvictedTTL")

			// 5. Assert FlowItem final state.
			finalState := item.FinalState()
			require.NotNil(t, finalState, "Item should have been finalized asynchronously by the controller")
			assert.Equal(t, types.QueueOutcomeEvictedTTL, finalState.Outcome,
				"Item's internal outcome must match the returned outcome")
		})

		// Validates that the Flow Registry lease is held (pinned) for the entire duration of the request, including the
		// time spent blocking in the processor's queue. If the lease is released early, the Garbage Collector could delete
		// the flow while requests are queued.
		t.Run("LeaseHeldDuringQueueing", func(t *testing.T) {
			t.Parallel()

			// Synchronization channels
			leaseReleased := make(chan struct{})
			processorEntered := make(chan struct{})
			unblockProcessor := make(chan struct{})

			// 1. Setup Registry: Trace when the lease is released.
			mockRegistry := &mockRegistryClient{FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{}}
			mockRegistry.WithConnectionFunc = func(
				key flowcontrol.FlowKey,
				fn func(conn contracts.ActiveFlowConnection) error,
			) error {
				// Execute the controller's logic.
				err := fn(&mockActiveFlowConnection{
					RegistryV: mockRegistry,
					FlowKeyV:  key,
				})
				// Signal that the closure has finished and the lease is about to be released.
				close(leaseReleased)
				return err
			}

			h := newUnitHarness(t.Context(), t, &Config{}, mockRegistry)

			// 2. Setup Processor: Simulate a long wait in the queue.
			h.mockProcessorFactory.processor.Store(&mockShardProcessor{
				SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				SubmitOrBlockFunc: func(ctx context.Context, item *internal.FlowItem) error {
					close(processorEntered) // Signal that we are now "queued"

					// Block until the test allows us to proceed.
					select {
					case <-unblockProcessor:
						item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
						return nil
					case <-ctx.Done():
						return ctx.Err()
					}
				},
			})

			// 3. Run EnqueueAndWait in the background.
			go func() {
				_, _ = h.fc.EnqueueAndWait(context.Background(), newTestRequest(defaultFlowKey))
			}()

			// 4. Wait for the request to enter the queue (Blocking phase).
			select {
			case <-processorEntered:
				// Success: The request is now blocked inside the processor.
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for request to enter processor")
			}

			// 5. Verify the lease is still held.
			// If leaseReleased is closed, it means the controller returned from WithConnection while the request was still
			// inside SubmitOrBlock.
			select {
			case <-leaseReleased:
				t.Fatal("registry lease was released while the request was still queued.")
			default:
				// Success: The lease is still held.
			}

			// 6. Cleanup: Unblock the processor and allow the lease to release.
			close(unblockProcessor)

			// Verify that the lease is eventually released after processing finishes.
			select {
			case <-leaseReleased:
				// Success
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for lease to release after processing finished")
			}
		})
	})
}

// TestFlowController_WorkerManagement covers the lifecycle of the processor (worker), including startup
// and shutdown.
func TestFlowController_WorkerManagement(t *testing.T) {
	t.Parallel()

	// Reconciliation validates that the controller correctly identifies and shuts down workers whose shards no longer
	// exist in the registry.
	t.Run("Reconciliation", func(t *testing.T) {
		t.Parallel()

		mockRegistry := &mockRegistryClient{
			FlowRegistryDataPlane: &mocks.MockRegistryDataPlane{},
			StatsFunc: func() contracts.AggregateStats {
				// The current state of the world according to the registry.
				return contracts.AggregateStats{}
			}}
		h := newUnitHarness(t.Context(), t, &Config{}, mockRegistry)

		// Pre-populate the controller with initial worker, simulating a previous state.

		// Initialize the processor mock with the channel needed to synchronize startup.
		h.mockProcessorFactory.processor.Store(&mockShardProcessor{runStarted: make(chan struct{})})
		// Start the worker using the internal mechanism.
		h.fc.getOrStartWorker()

		require.NotNil(t, h.mockProcessorFactory.processor.Load(), "pre-condition: initial worker not set up correctly")

		// Wait for the worker goroutine to have started and captured its context.
		proc := h.mockProcessorFactory.processor.Load()
		select {
		case <-proc.runStarted:
			// Worker is running.
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for worker to start")
		}
	})

	// Validates the atomicity of worker creation and ensures resource cleanup for the loser of the race.
	t.Run("WorkerCreationRace", func(t *testing.T) {
		t.Parallel()

		// This test orchestrates a deterministic race condition.
		factoryEntered := make(chan *mockShardProcessor, 2)
		continueFactory := make(chan struct{})
		// Map to store the construction context for each processor instance, allowing us to verify cleanup.
		constructionContexts := sync.Map{}

		h := newUnitHarness(t.Context(), t, &Config{}, nil)

		// Inject a custom factory to control the timing of worker creation.
		h.fc.shardProcessorFactory = func(
			ctx context.Context, // The context created by getOrStartWorker for the potential new processor.
			registry contracts.FlowRegistry,
			_ flowcontrol.SaturationDetector,
			_ contracts.EndpointCandidates,
			_ flowcontrol.UsageLimitPolicy,
			_ clock.WithTicker,
			_ time.Duration,
			_ int,
			_ logr.Logger,
		) processor {
			// This function is called by getOrStartWorker before the LoadOrStore check.
			proc := &mockShardProcessor{runStarted: make(chan struct{})}
			constructionContexts.Store(proc, ctx) // Capture the construction context.

			// Signal entry and then block, allowing another goroutine to enter.
			factoryEntered <- proc
			<-continueFactory
			return proc
		}

		var wg sync.WaitGroup
		wg.Add(2)

		// Start two goroutines that will race to create the same worker.
		go func() {
			defer wg.Done()
			h.fc.getOrStartWorker()
		}()
		go func() {
			defer wg.Done()
			h.fc.getOrStartWorker()
		}()

		// 1. Wait for both goroutines to enter the factory and create their respective processor instances.
		proc1 := <-factoryEntered
		proc2 := <-factoryEntered

		// 2. Unblock both goroutines, allowing them to race to workers.LoadOrStore.
		close(continueFactory)
		wg.Wait()

		// 3. Identify the winner and the loser.
		storedWorker := h.fc.worker.Load()
		require.NotNil(t, storedWorker, "a worker must have been successfully stored in the map")

		winnerProc := storedWorker.processor.(*mockShardProcessor)

		var loserProc *mockShardProcessor
		if winnerProc == proc1 {
			loserProc = proc2
		} else {
			loserProc = proc1
		}

		// 4. Validate the state of the winning processor.
		// Wait for the Run method to be called on the winner (only the winner should start).
		select {
		case <-winnerProc.runStarted:
			// Success.
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for the winning worker's Run method to be called")
		}

		// The winning processor's context must remain active.
		require.NotNil(t, winnerProc.Context(), "winner's context should not be nil (Run was called)")
		select {
		case <-winnerProc.Context().Done():
			t.Error("context of the winning worker should not be cancelled")
		default:
			// Success
		}

		// 5. Validate the state of the losing processor and resource cleanup.
		// The losing processor's Run method must NOT be called.
		select {
		case <-loserProc.runStarted:
			t.Error("Run was incorrectly called on the losing worker")
		default:
			// Success
		}
	})
}

// Helper function to create a realistic mock registry environment for integration/concurrency tests.
func setupRegistryForConcurrency(t *testing.T, flowKey flowcontrol.FlowKey) *mockRegistryClient {
	t.Helper()
	mockRegistry := &mockRegistryClient{}

	// Configure the shard and its dependencies required by the real ShardProcessor implementation.

	// Use high-fidelity mock queues (MockManagedQueue) that implement the necessary interfaces and synchronization.
	currentQueue := &mocks.MockManagedQueue{FlowKeyV: flowKey}

	dataplane := &mocks.MockRegistryDataPlane{
		ManagedQueueFunc: func(_ flowcontrol.FlowKey) (contracts.ManagedQueue, error) {
			return currentQueue, nil
		},
		// Configuration required for ShardProcessor initialization and dispatch logic.
		AllOrderedPriorityLevelsFunc: func() []int { return []int{flowKey.Priority} },
		PriorityBandAccessorFunc: func(priority int) (flowcontrol.PriorityBandAccessor, error) {
			if priority == flowKey.Priority {
				return &fwkfcmocks.MockPriorityBandAccessor{
					PriorityV: priority,
					IterateQueuesFunc: func(f func(flowcontrol.FlowQueueAccessor) bool) {
						f(currentQueue.FlowQueueAccessor())
					},
				}, nil
			}
			return nil, fmt.Errorf("unexpected priority %d", priority)
		},
		FairnessPolicyFunc: func(_ int) (flowcontrol.FairnessPolicy, error) {
			return &fwkfcmocks.MockFairnessPolicy{
				PickFunc: func(_ context.Context, _ flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
					return currentQueue.FlowQueueAccessor(), nil
				},
			}, nil
		},
		// Configure stats reporting based on the live state of the mock queues.
		StatsFunc: func() contracts.AggregateStats {
			return contracts.AggregateStats{
				TotalLen:      uint64(currentQueue.Len()),
				TotalByteSize: currentQueue.ByteSize(),
				PerPriorityBandStats: map[int]contracts.PriorityBandStats{
					flowKey.Priority: {
						Len:           uint64(currentQueue.Len()),
						ByteSize:      currentQueue.ByteSize(),
						CapacityBytes: 1e9, // Effectively unlimited capacity to ensure dispatch success.
					},
				},
			}
		},
	}

	// Configure the registry connection.
	mockRegistry.WithConnectionFunc = func(key flowcontrol.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error {
		return fn(&mockActiveFlowConnection{
			RegistryV: dataplane,
			FlowKeyV:  key,
		})
	}
	mockRegistry.StatsFunc = func() contracts.AggregateStats {
		return dataplane.Stats()
	}
	mockRegistry.FlowRegistryDataPlane = dataplane
	return mockRegistry
}

// TestFlowController_Concurrency_Distribution performs an integration test under high contention, using real
// ShardProcessors.
// It validates the thread-safety of the distribution logic and the overall system throughput.
func TestFlowController_Concurrency_Distribution(t *testing.T) {
	const (
		numGoroutines = 50
		numRequests   = 200
	)

	// Arrange
	mockRegistry := setupRegistryForConcurrency(t, defaultFlowKey)

	// Initialize the integration harness with real ShardProcessors.
	h := newIntegrationHarness(t.Context(), t, &Config{
		// Use a generous buffer to focus the test on distribution logic rather than backpressure.
		EnqueueChannelBufferSize: numRequests,
		DefaultRequestTTL:        5 * time.Second,
		ExpiryCleanupInterval:    100 * time.Millisecond,
	}, mockRegistry)

	// Act: Hammer the controller concurrently.
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	outcomes := make(chan types.QueueOutcome, numRequests)

	for i := range numGoroutines {
		goroutineID := i
		go func() {
			defer wg.Done()
			for j := range numRequests / numGoroutines {
				req := newTestRequest(defaultFlowKey)
				req.IDV = fmt.Sprintf("req-distrib-%d-%d", goroutineID, j)

				// Use a reasonable timeout for the individual request context.
				reqCtx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()

				ctx := logr.NewContext(reqCtx, logr.Discard())
				outcome, err := h.fc.EnqueueAndWait(ctx, req)
				if err != nil {
					// Use t.Errorf for concurrent tests to report failures without halting execution.
					t.Errorf("EnqueueAndWait failed unexpectedly under load: %v", err)
				}
				outcomes <- outcome
			}
		}()
	}

	// Wait for all requests to complete.
	wg.Wait()
	close(outcomes)

	// Assert: All requests should be successfully dispatched.
	successCount := 0
	for outcome := range outcomes {
		if outcome == types.QueueOutcomeDispatched {
			successCount++
		}
	}
	require.Equal(t, numRequests, successCount,
		"all concurrent requests must be dispatched successfully without errors or data races")
}

// TestFlowController_Concurrency_Backpressure specifically targets the blocking submission path (SubmitOrBlock) by
// configuring the processors with zero buffer capacity.
func TestFlowController_Concurrency_Backpressure(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrency integration test in short mode.")
	}
	t.Parallel()

	const (
		numGoroutines = 20
		// Fewer requests than the distribution test, as the blocking path is inherently slower.
		numRequests = 40
	)

	// Arrange: Set up the registry environment.
	mockRegistry := setupRegistryForConcurrency(t, defaultFlowKey)

	// Use the integration harness with a configuration designed to induce backpressure.
	h := newIntegrationHarness(t.Context(), t, &Config{
		// Zero buffer forces immediate use of SubmitOrBlock if the processor loop is busy.
		EnqueueChannelBufferSize: 0,
		// Generous TTL to ensure timeouts are not the cause of failure.
		DefaultRequestTTL:     10 * time.Second,
		ExpiryCleanupInterval: 100 * time.Millisecond,
	}, mockRegistry)

	// Act: Concurrently submit requests.
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	outcomes := make(chan types.QueueOutcome, numRequests)

	for i := range numGoroutines {
		goroutineID := i
		go func() {
			defer wg.Done()
			for j := range numRequests / numGoroutines {
				req := newTestRequest(defaultFlowKey)
				req.IDV = fmt.Sprintf("req-backpressure-%d-%d", goroutineID, j)

				// Use a reasonable timeout for the individual request context to ensure the test finishes promptly if a
				// deadlock occurs.
				reqCtx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()

				outcome, err := h.fc.EnqueueAndWait(logr.NewContext(reqCtx, logr.Discard()), req)
				if err != nil {
					t.Errorf("EnqueueAndWait failed unexpectedly under backpressure for request %s: %v", req.ID(), err)
				}
				outcomes <- outcome
			}
		}()
	}
	wg.Wait()
	close(outcomes)

	// Assert: Verify successful dispatch despite high contention and zero buffer.
	successCount := 0
	for outcome := range outcomes {
		if outcome == types.QueueOutcomeDispatched {
			successCount++
		}
	}
	require.Equal(t, numRequests, successCount,
		"all concurrent requests should be dispatched successfully even under high contention and zero buffer capacity")
}
