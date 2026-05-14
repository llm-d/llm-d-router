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

package registry

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/framework/plugins/queue"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol/mocks"
)

const (
	// highPriority is the priority level for the "High" priority band in the test harness config.
	highPriority int = 20
	// lowPriority is the priority level for the "Low" priority band in the test harness config.
	lowPriority int = 10
	// nonExistentPriority is a priority that is known not to exist in the test harness config.
	nonExistentPriority int = 99
)

// --- Test Harness and Mocks ---

// testHarness holds all components for a `registry` test.
type testHarness struct {
	t                *testing.T
	registry         *FlowRegistry
	statsPropagator  *mockStatsPropagator
	highPriorityKey1 flowcontrol.FlowKey
	highPriorityKey2 flowcontrol.FlowKey
	lowPriorityKey   flowcontrol.FlowKey
}

// newTestHarness initializes a `testHarness` with a default configuration.
func newTestHarness(t *testing.T) *testHarness {
	t.Helper()

	globalConfig, err := NewConfig(
		newTestPluginsHandle(t),
		WithPriorityBand(&PriorityBandConfig{Priority: highPriority}),
		WithPriorityBand(&PriorityBandConfig{Priority: lowPriority}),
	)
	require.NoError(t, err, "Test setup: validating and defaulting config should not fail")

	statsPropagator := &mockStatsPropagator{}
	fakeClock := testclock.NewFakeClock(time.Now())
	registryOpts := []RegistryOption{withClock(fakeClock)}
	registry, _ := NewFlowRegistry(globalConfig, logr.Discard(), registryOpts...)

	h := &testHarness{
		t:                t,
		registry:         registry,
		statsPropagator:  statsPropagator,
		highPriorityKey1: flowcontrol.FlowKey{ID: "hp-flow-1", Priority: highPriority},
		highPriorityKey2: flowcontrol.FlowKey{ID: "hp-flow-2", Priority: highPriority},
		lowPriorityKey:   flowcontrol.FlowKey{ID: "lp-flow-1", Priority: lowPriority},
	}
	// Automatically sync some default flows for convenience.
	h.synchronizeFlow(h.highPriorityKey1)
	h.synchronizeFlow(h.highPriorityKey2)
	h.synchronizeFlow(h.lowPriorityKey)
	return h
}

// synchronizeFlow simulates the registry synchronizing a flow with a real queue.
func (h *testHarness) synchronizeFlow(key flowcontrol.FlowKey) {
	h.t.Helper()
	policy := h.registry.config.PriorityBands[key.Priority].OrderingPolicy
	q, err := queue.NewQueueFromName(defaultQueue, policy)
	assert.NoError(h.t, err, "Helper synchronizeFlow: failed to create real queue for synchronization")
	h.registry.synchronizeFlow(key, policy, q)
}

// addItem adds an item to a specific flow's queue on the shard.
func (h *testHarness) addItem(key flowcontrol.FlowKey, size uint64) flowcontrol.QueueItemAccessor {
	h.t.Helper()
	mq, err := h.registry.ManagedQueue(key)
	require.NoError(h.t, err, "Helper addItem: failed to get queue for flow %s; ensure flow is synchronized", key)
	item := mocks.NewMockQueueItemAccessor(size, "req", key)
	require.NoError(h.t, mq.Add(item), "Helper addItem: failed to add item to queue for flow %s", key)
	return item
}

// removeItem removes an item from a specific flow's queue.
func (h *testHarness) removeItem(key flowcontrol.FlowKey, item flowcontrol.QueueItemAccessor) {
	h.t.Helper()
	mq, err := h.registry.ManagedQueue(key)
	require.NoError(h.t, err, "Helper removeItem: failed to get queue for flow %s; ensure flow is synchronized", key)
	_, err = mq.Remove(item.Handle())
	require.NoError(h.t, err, "Helper removeItem: failed to remove item from queue for flow %s", key)
}

// --- Basic Tests ---

func TestShard_New(t *testing.T) {
	t.Parallel()

	t.Run("ShouldInitializeCorrectly_WithDefaultConfig", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)

		assert.Equal(t, []int{highPriority, lowPriority}, h.registry.AllOrderedPriorityLevels(),
			"Registry must report configured priority levels sorted numerically (highest priority first)")

		val, ok := h.registry.priorityBands.Load(highPriority)
		bandHigh := val.(*priorityBand)
		require.True(t, ok, "Priority band %d (High) must be initialized", highPriority)
		require.NotNil(t, bandHigh.fairnessPolicy, "Fairness policy must be instantiated during construction")
		assert.Equal(t, DefaultFairnessPolicyRef, bandHigh.fairnessPolicy.TypedName().Name,
			"Must match the configured fairness policy implementation")
	})
}

func TestShard_Stats(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t)
	h.addItem(h.highPriorityKey1, 100)
	h.addItem(h.highPriorityKey1, 50)

	stats := h.registry.Stats()

	assert.Equal(t, uint64(2), stats.TotalLen, "Total length must aggregate counts from all bands")
	assert.Equal(t, uint64(150), stats.TotalByteSize, "Total byte size must aggregate sizes from all bands")

	bandHighStats, ok := stats.PerPriorityBandStats[highPriority]
	require.True(t, ok, "Stats snapshot must include entries for all configured priority bands (e.g., %d)", highPriority)
	assert.Equal(t, uint64(2), bandHighStats.Len, "Priority band length must reflect the items queued at that level")
	assert.Equal(t, uint64(150), bandHighStats.ByteSize,
		"Priority band byte size must reflect the items queued at that level")
}

func TestShard_Accessors(t *testing.T) {
	t.Parallel()

	t.Run("SuccessPaths", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)

		t.Run("ManagedQueue", func(t *testing.T) {
			t.Parallel()
			mq, err := h.registry.ManagedQueue(h.highPriorityKey1)
			require.NoError(t, err, "ManagedQueue accessor must succeed for a synchronized flow")
			require.NotNil(t, mq, "Returned ManagedQueue must not be nil")
			assert.Equal(t, h.highPriorityKey1, mq.FlowQueueAccessor().FlowKey(),
				"The returned queue instance must correspond to the requested FlowKey")
		})

		t.Run("FairnessPolicy", func(t *testing.T) {
			t.Parallel()
			policy, err := h.registry.FairnessPolicy(highPriority)
			require.NoError(t, err, "InterFlowDispatchPolicy accessor must succeed for a configured priority band")
			require.NotNil(t, policy, "Returned policy must not be nil (guaranteed by contract)")
			assert.Equal(t, DefaultFairnessPolicyRef, policy.TypedName().Name,
				"Must return the configured fairness policy implementation")
		})
	})

	t.Run("ErrorPaths", func(t *testing.T) {
		t.Parallel()
		testCases := []struct {
			name      string
			action    func(fr contracts.FlowRegistry) error
			expectErr error
		}{
			{
				name: "ManagedQueue_PriorityNotFound",
				action: func(fr contracts.FlowRegistry) error {
					_, err := fr.ManagedQueue(flowcontrol.FlowKey{Priority: nonExistentPriority})
					return err
				},
				expectErr: contracts.ErrPriorityBandNotFound,
			},
			{
				name: "ManagedQueue_FlowNotFound",
				action: func(fr contracts.FlowRegistry) error {
					_, err := fr.ManagedQueue(flowcontrol.FlowKey{ID: "missing", Priority: highPriority})
					return err
				},
				expectErr: contracts.ErrFlowInstanceNotFound,
			},
			{
				name: "FairnessPolicy_PriorityNotFound",
				action: func(fr contracts.FlowRegistry) error {
					_, err := fr.FairnessPolicy(nonExistentPriority)
					return err
				},
				expectErr: contracts.ErrPriorityBandNotFound,
			},
		}
		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t)
				err := tc.action(h.registry)
				require.Error(t, err, "The accessor method must return an error for this scenario")
				assert.ErrorIs(t, err, tc.expectErr,
					"The error must wrap the specific sentinel error defined in the contracts package")
			})
		}
	})
}

func TestShard_PriorityBandAccessor(t *testing.T) {
	t.Parallel()

	t.Run("ShouldFail_WhenPriorityDoesNotExist", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)
		_, err := h.registry.PriorityBandAccessor(nonExistentPriority)
		assert.ErrorIs(t, err, contracts.ErrPriorityBandNotFound,
			"Requesting an accessor for an unconfigured priority must fail with ErrPriorityBandNotFound")
	})

	t.Run("ShouldSucceed_WhenPriorityExists", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)
		accessor, err := h.registry.PriorityBandAccessor(h.highPriorityKey1.Priority)
		require.NoError(t, err, "Requesting an accessor for a configured priority must succeed")
		require.NotNil(t, accessor, "The returned accessor instance must not be nil")

		t.Run("Properties_ShouldReturnCorrectValues", func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, h.highPriorityKey1.Priority, accessor.Priority(),
				"Accessor Priority() must match the configured numerical priority")
		})

		t.Run("FlowKeys_ShouldReturnAllKeysInBand", func(t *testing.T) {
			t.Parallel()
			keys := accessor.FlowKeys()
			expectedKeys := []flowcontrol.FlowKey{h.highPriorityKey1, h.highPriorityKey2}
			assert.ElementsMatch(t, expectedKeys, keys,
				"FlowKeys() must return a complete snapshot of all flows registered in this band")
		})

		t.Run("Queue_ShouldReturnCorrectAccessor", func(t *testing.T) {
			t.Parallel()
			q := accessor.Queue(h.highPriorityKey1.ID)
			require.NotNil(t, q, "Queue() must return a non-nil accessor for a registered flow ID")
			assert.Equal(t, h.highPriorityKey1, q.FlowKey(), "The returned queue accessor must have the correct FlowKey")
			assert.Nil(t, accessor.Queue("non-existent"), "Queue() must return nil if the flow ID is not found in this band")
		})

		t.Run("IterateQueues", func(t *testing.T) {
			t.Parallel()

			t.Run("ShouldVisitAllQueuesInBand", func(t *testing.T) {
				t.Parallel()
				var iteratedKeys []flowcontrol.FlowKey
				accessor.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
					iteratedKeys = append(iteratedKeys, queue.FlowKey())
					return true
				})
				expectedKeys := []flowcontrol.FlowKey{h.highPriorityKey1, h.highPriorityKey2}
				assert.ElementsMatch(t, expectedKeys, iteratedKeys,
					"IterateQueues must visit every registered flow in the band exactly once")
			})

			t.Run("ShouldExitEarly_WhenCallbackReturnsFalse", func(t *testing.T) {
				t.Parallel()
				var iterationCount int
				accessor.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
					iterationCount++
					return false
				})
				assert.Equal(t, 1, iterationCount, "IterateQueues must terminate immediately when the callback returns false")
			})

			t.Run("ShouldBeSafe_DuringConcurrentMapModification", func(t *testing.T) {
				t.Parallel()
				h := newTestHarness(t) // Isolated harness to avoid corrupting the state for other parallel tests
				accessor, err := h.registry.PriorityBandAccessor(highPriority)
				require.NoError(t, err)

				var wg sync.WaitGroup
				wg.Add(2)

				// Goroutine A: The Iterator (constantly reading)
				go func() {
					defer wg.Done()
					for range 100 {
						accessor.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
							// Accessing data should not panic or race.
							_ = queue.FlowKey()
							return true
						})
					}
				}()

				// Goroutine B: The Modifier (constantly writing)
				go func() {
					defer wg.Done()
					for i := range 100 {
						key := flowcontrol.FlowKey{ID: fmt.Sprintf("new-flow-%d", i), Priority: highPriority}
						h.synchronizeFlow(key)
						h.registry.mu.Lock()
						h.registry.deleteFlow(key)
						h.registry.mu.Unlock()
					}
				}()

				// The primary assertion is that this test completes without the race detector firing, which proves the
				// `RLock/WLock` separation is correct.
				wg.Wait()
			})
		})

		t.Run("OnEmptyBand", func(t *testing.T) {
			t.Parallel()
			h := newTestHarness(t)
			h.registry.mu.Lock()
			h.registry.deleteFlow(h.lowPriorityKey)
			h.registry.mu.Unlock()
			accessor, err := h.registry.PriorityBandAccessor(lowPriority)
			require.NoError(t, err, "Setup: getting an accessor for an empty band must succeed")

			keys := accessor.FlowKeys()
			assert.NotNil(t, keys, "FlowKeys() on an empty band must return a non-nil slice")
			assert.Empty(t, keys, "FlowKeys() on an empty band must return an empty slice")

			var callbackExecuted bool
			accessor.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
				callbackExecuted = true
				return true
			})
			assert.False(t, callbackExecuted, "IterateQueues must not execute the callback for an empty band")
		})
	})
}

// --- Lifecycle and State Management Tests ---

func TestShard_SynchronizeFlow(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t)
	flowKey := flowcontrol.FlowKey{ID: "flow1", Priority: highPriority}

	h.synchronizeFlow(flowKey)
	mq1, err := h.registry.ManagedQueue(flowKey)
	require.NoError(t, err, "Flow instance should be accessible after synchronization")

	h.synchronizeFlow(flowKey)
	mq2, err := h.registry.ManagedQueue(flowKey)
	require.NoError(t, err, "Flow instance should remain accessible after idempotent re-synchronization")
	assert.Same(t, mq1, mq2, "Idempotent synchronization must not replace the existing queue instance")
}

func TestShard_DeleteFlow(t *testing.T) {
	t.Parallel()
	h := newTestHarness(t)
	_, err := h.registry.ManagedQueue(h.highPriorityKey1)
	require.NoError(t, err, "Test setup: flow instance must exist before deletion")

	h.registry.mu.Lock()
	h.registry.deleteFlow(h.highPriorityKey1)
	h.registry.mu.Unlock()

	_, err = h.registry.ManagedQueue(h.highPriorityKey1)
	require.Error(t, err, "Flow instance should not be accessible after deletion")
	assert.ErrorIs(t, err, contracts.ErrFlowInstanceNotFound,
		"Accessing a deleted flow must return ErrFlowInstanceNotFound")
}

func TestShard_DynamicProvisioning(t *testing.T) {
	t.Parallel()

	t.Run("ShouldAddBandDynamically", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)

		// Update the config definition first (simulating the Registry's job).
		dynamicPrio := 15
		newBandCfg, err := NewPriorityBandConfig(newTestPluginsHandle(t), dynamicPrio)
		require.NoError(t, err)
		h.registry.config.PriorityBands[dynamicPrio] = newBandCfg

		h.registry.mu.Lock()
		h.registry.addPriorityBand(dynamicPrio)
		h.registry.mu.Unlock()

		expectedLevels := []int{highPriority, dynamicPrio, lowPriority} // 20, 15, 10
		assert.Equal(t, expectedLevels, h.registry.AllOrderedPriorityLevels(),
			"New priority must be inserted into the sorted order correctly")

		_, err = h.registry.PriorityBandAccessor(dynamicPrio)
		require.NoError(t, err, "Accessor should be available for the new band")
	})

	t.Run("ShouldBeIdempotent", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)

		// Prepare config.
		dynamicPrio := 15
		newBandCfg, err := NewPriorityBandConfig(newTestPluginsHandle(t), dynamicPrio)
		require.NoError(t, err)
		h.registry.config.PriorityBands[dynamicPrio] = newBandCfg

		// Call twice.
		h.registry.mu.Lock()
		h.registry.addPriorityBand(dynamicPrio)
		h.registry.addPriorityBand(dynamicPrio)
		h.registry.mu.Unlock()

		levelCount := 0
		for _, p := range h.registry.AllOrderedPriorityLevels() {
			if p == dynamicPrio {
				levelCount++
			}
		}
		assert.Equal(t, 1, levelCount, "Priority level should appear exactly once in ordered list")
	})

	t.Run("ShouldPanic_WhenConfigMissing", func(t *testing.T) {
		t.Parallel()
		h := newTestHarness(t)

		// Try to add a band that is not in h.shard.config.
		assert.Panics(t, func() { h.registry.addPriorityBand(nonExistentPriority) },
			"Should fail if the definition layer hasn't been updated first")
	})
}

// --- Concurrency Test ---

// TestShard_Concurrency_MixedWorkload is a general stability test that simulates a realistic workload by having
// concurrent readers (e.g., dispatchers) and writers operating on the same shard.
// It provides high confidence that the fine-grained locking strategy is free of deadlocks and data races under
// sustained, mixed contention.
func TestShard_Concurrency_MixedWorkload(t *testing.T) {
	t.Parallel()
	const (
		numReaders   = 5
		numWriters   = 2
		opsPerWriter = 100
	)

	h := newTestHarness(t)
	stopCh := make(chan struct{})
	var readersWg, writersWg sync.WaitGroup

	readersWg.Add(numReaders)
	for range numReaders {
		go func() {
			defer readersWg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					for _, priority := range h.registry.AllOrderedPriorityLevels() {
						accessor, err := h.registry.PriorityBandAccessor(priority)
						if err == nil {
							accessor.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool { return true })
						}
					}
				}
			}
		}()
	}

	writersWg.Add(numWriters)
	for range numWriters {
		go func() {
			defer writersWg.Done()
			for j := range opsPerWriter {
				// Alternate writing to different flows and priorities to increase contention.
				if j%2 == 0 {
					item := h.addItem(h.highPriorityKey1, 10)
					h.removeItem(h.highPriorityKey1, item)
				} else {
					item := h.addItem(h.lowPriorityKey, 5)
					h.removeItem(h.lowPriorityKey, item)
				}
			}
		}()
	}

	// Wait for all writers to complete first.
	writersWg.Wait()

	// Now stop the readers and wait for them to exit.
	close(stopCh)
	readersWg.Wait()

	// The primary assertion is that this test completes without the race detector firing; however, we can make some final
	// assertions on state consistency.
	finalStats := h.registry.Stats()
	assert.Zero(t, finalStats.TotalLen, "After all paired add/remove operations, the total length should be zero")
	assert.Zero(t, finalStats.TotalByteSize, "After all paired add/remove operations, the total byte size should be zero")
}

// TestShard_Concurrency_AllOrderedPriorityLevels_RaceSafety verifies that AllOrderedPriorityLevels() is safe to call
// concurrently with addPriorityBand() and deletePriorityBand(). This test is designed to trigger the Go race detector
// if the implementation returns the internal slice without proper synchronization.
func TestShard_Concurrency_AllOrderedPriorityLevels_RaceSafety(t *testing.T) {
	t.Parallel()
	const (
		numReaders = 5
		iterations = 200
	)

	h := newTestHarness(t)

	dynamicPrio := 15
	newBandCfg, err := NewPriorityBandConfig(newTestPluginsHandle(t), dynamicPrio)
	require.NoError(t, err)
	h.registry.config.PriorityBands[dynamicPrio] = newBandCfg

	stopCh := make(chan struct{})
	var readersWg, writerWg sync.WaitGroup

	// Readers: continuously call AllOrderedPriorityLevels() and iterate
	readersWg.Add(numReaders)
	for range numReaders {
		go func() {
			defer readersWg.Done()
			for {
				select {
				case <-stopCh:
					return
				default:
					levels := h.registry.AllOrderedPriorityLevels()
					// Force iteration over the returned slice to surface races on the backing array.
					sum := 0
					for _, p := range levels {
						sum += p
					}
				}
			}
		}()
	}

	// Writer: repeatedly add and delete a priority band, modifying orderedPriorityLevels.
	// deletePriorityBand also removes the config entry, so we must restore it before each add.
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		for range iterations {
			h.registry.mu.Lock()
			h.registry.addPriorityBand(dynamicPrio)
			h.registry.mu.Unlock()

			h.registry.deletePriorityBand(dynamicPrio)

			h.registry.mu.Lock()
			h.registry.config.PriorityBands[dynamicPrio] = newBandCfg
			h.registry.mu.Unlock()
		}
	}()

	writerWg.Wait()
	close(stopCh)
	readersWg.Wait()

	// If we reach here without the race detector firing, the implementation is safe.
	// Final sanity: dynamic band should be removed, only the original two remain.
	assert.Equal(t, []int{highPriority, lowPriority}, h.registry.AllOrderedPriorityLevels(),
		"After all add/delete cycles, only original priority levels should remain")
}
