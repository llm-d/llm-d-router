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

package internal

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// maxCleanupWorkers caps the number of concurrent workers for background cleanup tasks. This prevents the processor
// from overwhelming the Go scheduler with too many goroutines.
const maxCleanupWorkers = 4

// ErrProcessorBusy is a sentinel error returned by the processor's Submit method indicating that the processor's
// internal buffer is momentarily full and cannot accept new work.
var ErrProcessorBusy = errors.New("processor is busy")

// Processor is the core worker of the FlowController.
//
// A single Processor owns the entire request data plane and is responsible for all request lifecycle operations from
// the point an item is successfully submitted to it.
//
// # Request Lifecycle Management & Ownership
//
// The Processor takes ownership of a FlowItem only after it has been successfully sent to its internal enqueueChan
// via Submit or SubmitOrBlock (i.e., when these methods return nil).
// Once the Processor takes ownership, it is solely responsible for ensuring that item.Finalize() or
// item.FinalizeWithOutcome() is called exactly once for that item, under all circumstances (dispatch, rejection, sweep,
// or shutdown).
//
// If Submit or SubmitOrBlock return an error, ownership remains with the caller (the Controller), which must then
// handle the finalization.
//
// # Concurrency Model
//
// To ensure correctness and high performance, the processor uses a single-goroutine, actor-based model. The main run
// loop is the sole writer for all state-mutating operations. This makes complex transactions (like capacity checks)
// inherently atomic without coarse-grained locks.
type Processor struct {
	poolName             string
	registry             contracts.FlowRegistry
	registryBackground   contracts.FlowRegistryBackground
	saturationDetector   flowcontrol.SaturationDetector
	endpointCandidates   contracts.EndpointCandidates
	usageLimitPolicy     flowcontrol.UsageLimitPolicy
	clock                clock.WithTicker
	cleanupSweepInterval time.Duration
	logger               logr.Logger

	// lifecycleCtx controls the processor's lifetime. Monitored by Submit* methods for safe shutdown.
	lifecycleCtx context.Context

	// enqueueChan is the entry point for new requests.
	enqueueChan chan *FlowItem

	// poolEmpty caches whether the candidate pool had zero endpoints as of the most recent dispatchCycle. enqueue reads
	// it to distinguish a queue-capacity rejection caused by genuine unavailability (no backends, e.g. scale-from-zero)
	// from one caused by backpressure against a contended but non-empty pool. Only accessed from the Run goroutine, so
	// it needs no synchronization.
	poolEmpty bool

	// wg is used to wait for background tasks (cleanup sweep) to complete on shutdown.
	wg             sync.WaitGroup
	isShuttingDown atomic.Bool
	shutdownOnce   sync.Once
}

// NewProcessor creates a new Processor instance.
func NewProcessor(
	ctx context.Context,
	poolName string,
	registry contracts.FlowRegistry,
	registryBackground contracts.FlowRegistryBackground,
	saturationDetector flowcontrol.SaturationDetector,
	endpointCandidates contracts.EndpointCandidates,
	usageLimitPolicy flowcontrol.UsageLimitPolicy,
	clock clock.WithTicker,
	cleanupSweepInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) *Processor {
	return &Processor{
		registry:             registry,
		registryBackground:   registryBackground,
		poolName:             poolName,
		saturationDetector:   saturationDetector,
		endpointCandidates:   endpointCandidates,
		usageLimitPolicy:     usageLimitPolicy,
		clock:                clock,
		cleanupSweepInterval: cleanupSweepInterval,
		logger:               logger,
		lifecycleCtx:         ctx,
		enqueueChan:          make(chan *FlowItem, enqueueChannelBufferSize),
	}
}

// Submit attempts a non-blocking handoff of an item to the processor's internal enqueue channel.
//
// Ownership Contract:
//   - Returns nil: The item was successfully handed off.
//     The Processor takes responsibility for calling Finalize on the item.
//   - Returns error: The item was not handed off.
//     Ownership of the FlowItem remains with the caller, who is responsible for calling Finalize.
//
// Possible errors:
//   - ErrProcessorBusy: The processor's input channel is full.
//   - types.ErrFlowControllerNotRunning: The processor is shutting down.
func (p *Processor) Submit(item *FlowItem) error {
	if p.isShuttingDown.Load() {
		return types.ErrFlowControllerNotRunning
	}
	select { // The default case makes this select non-blocking.
	case p.enqueueChan <- item:
		return nil // Ownership transferred.
	case <-p.lifecycleCtx.Done():
		return types.ErrFlowControllerNotRunning
	default:
		return ErrProcessorBusy
	}
}

// SubmitOrBlock performs a blocking handoff of an item to the processor's internal enqueue channel.
// It waits until the item is handed off, the caller's context is cancelled, or the processor shuts down.
//
// Ownership Contract:
//   - Returns nil: The item was successfully handed off.
//     The Processor takes responsibility for calling Finalize on the item.
//   - Returns error: The item was not handed off.
//     Ownership of the FlowItem remains with the caller, who is responsible for calling Finalize.
//
// Possible errors:
//   - ctx.Err(): The provided context was cancelled or its deadline exceeded.
//   - types.ErrFlowControllerNotRunning: The processor is shutting down.
func (p *Processor) SubmitOrBlock(ctx context.Context, item *FlowItem) error {
	if p.isShuttingDown.Load() {
		return types.ErrFlowControllerNotRunning
	}

	select { // The absence of a default case makes this call blocking.
	case p.enqueueChan <- item:
		return nil // Ownership transferred.
	case <-ctx.Done():
		return ctx.Err()
	case <-p.lifecycleCtx.Done():
		return types.ErrFlowControllerNotRunning
	}
}

// Run is the main operational loop for the processor. It must be run as a goroutine.
// It uses a `select` statement to interleave accepting new requests with dispatching existing ones, balancing
// responsiveness with throughput.
func (p *Processor) Run(ctx context.Context) {
	p.logger.V(logutil.DEFAULT).Info("Processor run loop starting.")
	defer p.logger.V(logutil.DEFAULT).Info("Processor run loop stopped.")

	p.wg.Add(1)
	go p.runCleanupSweep(ctx)

	// Create a ticker for periodic dispatch attempts to avoid tight loops
	dispatchTicker := p.clock.NewTicker(time.Millisecond)
	defer dispatchTicker.Stop()

	var gcCh <-chan time.Time
	var priorityBandUpdateCh <-chan map[int]struct{}
	if p.registryBackground != nil {
		gcTicker := p.clock.NewTicker(p.registryBackground.FlowGCTimeout())
		defer gcTicker.Stop()
		gcCh = gcTicker.C()
		priorityBandUpdateCh = p.registryBackground.PriorityBandUpdateChannel()
	}

	// This is the main worker loop. It continuously processes incoming requests and dispatches queued requests until the
	// context is cancelled. The `select` statement has these cases:
	//
	//  1. Context Cancellation: The highest priority is shutting down. If the context's `Done` channel is closed, the
	//     loop will drain all queues and exit. This is the primary exit condition.
	//  2. New Item Arrival: If an item is available on `enqueueChan`, it will be processed. This ensures that the
	//     processor is responsive to new work.
	//  3. Dispatch Ticker: Periodically triggers a dispatch cycle to attempt to dispatch items from existing queues,
	//     ensuring that queued work is processed even when no new items arrive.
	//  4. Priority Band Updates: Applies control-plane priority band topology changes.
	//  5. Registry GC: Periodically garbage-collects idle flows and priority bands.
	for {
		select {
		case <-ctx.Done():
			p.shutdown()
			p.wg.Wait()
			return
		case item, ok := <-p.enqueueChan:
			if !ok { // Should not happen in practice, but is a clean shutdown signal.
				p.shutdown()
				p.wg.Wait()
				return
			}
			// This is a safeguard against logic errors in the distributor.
			if item == nil {
				p.logger.Error(nil, "Logic error: nil item received on processor enqueue channel, ignoring.")
				continue
			}
			p.enqueue(item)
			p.dispatchCycle(ctx) // Process immediately when an item arrives
		case <-dispatchTicker.C():
			p.dispatchCycle(ctx) // Periodically attempt to dispatch from queues
		case desired := <-priorityBandUpdateCh:
			p.registryBackground.ApplyDesiredPriorities(desired)
		case <-gcCh:
			p.registryBackground.ExecuteGCCycle()
		}
	}
}

// enqueue processes an item received from the enqueueChan.
// It handles capacity checks, checks for external finalization, and either admits the item to a queue or rejects it.
func (p *Processor) enqueue(item *FlowItem) {

	req := item.OriginalRequest()
	key := req.FlowKey()
	priorityStr := strconv.Itoa(key.Priority)
	outcome := item.FinalState()

	startTime := time.Now()

	defer func() {
		outcomeStr := "NotYetFinalized"
		if fs := item.FinalState(); fs != nil {
			outcomeStr = fs.Outcome.String()
		}
		metrics.RecordFlowControlRequestEnqueueDuration(key.ID, priorityStr, outcomeStr, time.Since(startTime))
	}()

	// --- Optimistic External Finalization Check ---
	// Check if the item was finalized by the Controller (due to TTL/cancellation) while it was buffered in enqueueChan.
	// This is an optimistic check to avoid unnecessary processing on items already considered dead.
	// The ultimate guarantee of cleanup for any races is the runCleanupSweep mechanism.
	if finalState := outcome; finalState != nil {
		p.logger.V(logutil.TRACE).Info("Item finalized externally before processing, discarding.",
			"outcome", finalState.Outcome, "err", finalState.Err, "flowKey", key, "requestID", req.ID())
		return
	}

	// --- Configuration Validation ---
	managedQ, err := p.registry.ManagedQueue(key)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get queue for flow key %s: %w", key, err)
		p.logger.Error(finalErr, "Rejecting request, queue lookup failed", "flowKey", key, "requestID", req.ID())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}

	_, err = p.registry.PriorityBandAccessor(key.Priority)
	if err != nil {
		finalErr := fmt.Errorf("configuration error: failed to get priority band for priority %d: %w", key.Priority, err)
		p.logger.Error(finalErr, "Rejecting request, priority band lookup failed", "flowKey", key, "requestID", req.ID())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}

	// --- Capacity Check ---
	// This check is safe because it is performed by the single-writer Run goroutine.
	if ok, stats := p.hasCapacity(key.Priority, req.ByteSize()); !ok {
		// When the pool has no endpoints, the queue is acting as a scale-from-zero waiting room. A capacity rejection in
		// that state reflects genuine unavailability (surfaced as 503), not backpressure against a contended pool (429).
		if p.poolEmpty {
			p.logger.V(logutil.DEBUG).Info("Rejecting request, queue at capacity with no endpoints",
				"flowKey", key, "requestID", req.ID(), "reqByteSize", req.ByteSize())
			item.FinalizeWithOutcome(types.QueueOutcomeRejectedNoEndpoints, fmt.Errorf("%w: %w",
				types.ErrRejected, types.ErrNoEndpoints))
			return
		}
		p.logger.V(logutil.DEBUG).Info("Rejecting request, queue at capacity",
			"flowKey", key, "requestID", req.ID(), "reqByteSize", req.ByteSize(),
			"totalLen", stats.TotalLen, "totalCapacityRequests", stats.TotalCapacityRequests,
			"totalByteSize", stats.TotalByteSize, "totalCapacityBytes", stats.TotalCapacityBytes)
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedCapacity, fmt.Errorf("%w: %w",
			types.ErrRejected, types.ErrQueueAtCapacity))
		return
	}

	// --- Commitment Point ---
	// The item is admitted. The ManagedQueue.Add implementation is responsible for calling item.SetHandle() atomically.
	if err := managedQ.Add(item); err != nil {
		finalErr := fmt.Errorf("failed to add item to queue for flow key %s: %w", key, err)
		p.logger.Error(finalErr, "Rejecting request, queue add failed",
			"flowKey", key, "requestID", req.ID())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, finalErr))
		return
	}
	p.logger.V(logutil.TRACE).Info("Item enqueued.",
		"flowKey", key, "requestID", req.ID())
}

// hasCapacity checks if the global limits and the specific priority band have enough capacity.
// This check reflects actual resource utilization, including "zombie" items (finalized but unswept), to prevent
// physical resource overcommitment.
func (p *Processor) hasCapacity(priority int, itemByteSize uint64) (bool, contracts.AggregateStats) {
	stats := p.registry.Stats()
	if stats.TotalCapacityBytes > 0 && stats.TotalByteSize+itemByteSize > stats.TotalCapacityBytes {
		return false, stats
	}
	if stats.TotalCapacityRequests > 0 && stats.TotalLen+1 > stats.TotalCapacityRequests {
		return false, stats
	}

	bandStats, ok := stats.PerPriorityBandStats[priority]
	if !ok {
		return false, stats
	}
	if bandStats.CapacityBytes > 0 && bandStats.ByteSize+itemByteSize > bandStats.CapacityBytes {
		return false, stats
	}
	if bandStats.CapacityRequests > 0 && bandStats.Len+1 > bandStats.CapacityRequests {
		return false, stats
	}

	return true, stats
}

// dispatchCycle attempts to dispatch a single item by iterating through priority bands from highest to lowest.
// It applies the configured policies for each band to select an item and then attempts to dispatch it.
// It returns true if an item was successfully dispatched, and false otherwise.
// It enforces Head-of-Line (HoL) blocking if the selected item is saturated.
//
// # Work Conservation and Head-of-Line (HoL) Blocking
//
// The cycle attempts to be work-conserving by skipping bands where selection fails.
// However, if a selected item is saturated (cannot be scheduled), the cycle stops immediately. This enforces HoL
// blocking to respect the policy's decision and prevent priority inversion, where dispatching lower-priority work might
// exacerbate the saturation affecting the high-priority item.
func (p *Processor) dispatchCycle(ctx context.Context) bool {
	dispatchCycleStart := time.Now()
	defer func() {
		metrics.RecordFlowControlDispatchCycleDuration(time.Since(dispatchCycleStart))
	}()

	pool := p.endpointCandidates.Locate(ctx, nil)
	p.poolEmpty = len(pool) == 0

	saturation := p.saturationDetector.Saturation(ctx, pool)

	// Record pool saturation metric
	metrics.RecordFlowControlPoolSaturation(p.poolName, saturation)

	priorities := p.registry.AllOrderedPriorityLevels()
	ceilings := p.usageLimitPolicy.ComputeLimit(ctx, saturation, priorities)

	for i, priority := range priorities {
		// --- Viability Check (Saturation/HoL Blocking) ---
		// Check before selecting an item: if we are already saturated for this priority, stop immediately.
		usageLimit := ceilings[i]
		if saturation >= usageLimit {
			p.logger.V(logutil.DEBUG).Info("Priority band is saturated; enforcing HoL blocking.",
				"priority", priority, "saturation", saturation, "usageLimit", usageLimit)
			// Stop the dispatch cycle entirely to respect strict policy decision and prevent priority inversion where
			// lower-priority work might exacerbate the saturation affecting high-priority work.
			return false
		}

		originalBand, err := p.registry.PriorityBandAccessor(priority)
		if err != nil {
			p.logger.Error(err, "Failed to get PriorityBandAccessor, skipping band", "priority", priority)
			continue
		}

		item, err := p.selectItem(ctx, originalBand)
		if err != nil {
			p.logger.Error(err, "Failed to select item, skipping priority band for this cycle",
				"priority", priority)
			continue // Continue to the next band to maximize work conservation.
		}
		if item == nil {
			continue
		}

		// --- Dispatch ---
		req := item.OriginalRequest()
		if err := p.dispatchItem(item); err != nil {
			p.logger.Error(err, "Failed to dispatch item, skipping priority band for this cycle",
				"flowKey", req.FlowKey(), "requestID", req.ID())
			continue // Continue to the next band to maximize work conservation.
		}
		return true
	}
	return false
}

// selectItem applies the configured fairness and ordering policies to select a single item.
func (p *Processor) selectItem(
	ctx context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.QueueItemAccessor, error) {
	fairnessP, err := p.registry.FairnessPolicy(flowGroup.Priority())
	if err != nil {
		return nil, fmt.Errorf("could not get FairnessPolicy: %w", err)
	}
	queue, err := fairnessP.Pick(ctx, flowGroup)
	if err != nil {
		return nil, fmt.Errorf("FairnessPolicy %q failed to select queue: %w", fairnessP.TypedName(), err)
	}
	if queue == nil {
		// nothing to select
		return nil, nil //nolint:nilnil
	}
	// The queue itself is responsible for explicit ordering via its configured OrderingPolicy.
	// We simply peek at the head.
	return queue.Peek(), nil
}

// dispatchItem handles the final steps of dispatching an item: removing it from the queue and finalizing its outcome.
func (p *Processor) dispatchItem(itemAcc flowcontrol.QueueItemAccessor) error {
	req := itemAcc.OriginalRequest()
	key := req.FlowKey()
	managedQ, err := p.registry.ManagedQueue(key)
	if err != nil {
		return fmt.Errorf("failed to get ManagedQueue for flow %s: %w", key, err)
	}

	removedItemAcc, err := managedQ.Remove(itemAcc.Handle())
	if err != nil {
		// This happens benignly if the item was already removed by the cleanup sweep loop.
		// We log it at a low level for visibility but return nil so the dispatch cycle proceeds.
		p.logger.V(logutil.DEBUG).Info("Failed to remove item during dispatch (likely already finalized and swept).",
			"flowKey", key, "requestID", req.ID(), "error", err)
		return nil
	}

	removedItem := removedItemAcc.(*FlowItem)
	p.logger.V(logutil.TRACE).Info("Item dispatched.", "flowKey", req.FlowKey(), "requestID", req.ID())
	removedItem.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
	return nil
}

// runCleanupSweep starts a background goroutine that periodically scans all queues for externally finalized items
// ("zombie" items) and removes them in batches.
func (p *Processor) runCleanupSweep(ctx context.Context) {
	defer p.wg.Done()
	logger := p.logger.WithName("runCleanupSweep")
	logger.V(logutil.DEFAULT).Info("Cleanup sweep goroutine starting.")
	defer logger.V(logutil.DEFAULT).Info("Cleanup sweep goroutine stopped.")

	ticker := p.clock.NewTicker(p.cleanupSweepInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C():
			p.sweepFinalizedItems()
		}
	}
}

// sweepFinalizedItems performs a single scan of all queues, removing finalized items in batch and releasing their
// memory.
func (p *Processor) sweepFinalizedItems() {
	processFn := func(managedQ contracts.ManagedQueue, logger logr.Logger) {
		predicate := func(itemAcc flowcontrol.QueueItemAccessor) bool {
			return itemAcc.(*FlowItem).FinalState() != nil
		}
		removedItems := managedQ.Cleanup(predicate)
		if len(removedItems) > 0 {
			logger.V(logutil.TRACE).Info("Swept finalized items and released capacity.",
				"count", len(removedItems))
		}
	}
	p.processAllQueuesConcurrently("sweepFinalizedItems", processFn)
}

// shutdown handles the graceful termination of the processor, ensuring all pending items (in channel and queues) are
// Finalized.
func (p *Processor) shutdown() {
	p.shutdownOnce.Do(func() {
		p.isShuttingDown.Store(true)
		p.logger.V(logutil.DEFAULT).Info("Processor shutting down.")

	DrainLoop: // Drain the enqueueChan to finalize buffered items.
		for {
			select {
			case item := <-p.enqueueChan:
				if item == nil {
					continue
				}
				// Finalize buffered items.
				item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther,
					fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning))
			default:
				break DrainLoop
			}
		}
		// We do not close enqueueChan because external goroutines (Controller) send on it.
		// The channel will be garbage collected when the processor terminates.
		p.evictAll()
	})
}

// evictAll drains all queues, finalizes every item, and releases their memory.
func (p *Processor) evictAll() {
	processFn := func(managedQ contracts.ManagedQueue, logger logr.Logger) {
		key := managedQ.FlowQueueAccessor().FlowKey()
		removedItems := managedQ.Drain()

		outcome := types.QueueOutcomeEvictedOther
		errShutdown := fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrFlowControllerNotRunning)
		for _, i := range removedItems {
			item, ok := i.(*FlowItem)
			if !ok {
				logger.Error(fmt.Errorf("internal error: unexpected type %T", i),
					"Panic condition detected during shutdown", "flowKey", key)
				continue
			}

			// Finalization is idempotent; safe to call even if already finalized externally.
			// The per-request log is emitted by EnqueueAndWait when it unblocks.
			item.FinalizeWithOutcome(outcome, errShutdown)
		}
	}
	p.processAllQueuesConcurrently("evictAll", processFn)
}

// processAllQueuesConcurrently iterates over all queues in all priority bands and executes the given
// `processFn` for each queue using a dynamically sized worker pool.
func (p *Processor) processAllQueuesConcurrently(
	ctxName string,
	processFn func(mq contracts.ManagedQueue, logger logr.Logger),
) {
	logger := p.logger.WithName(ctxName)

	type resolvedQueue struct {
		mq     contracts.ManagedQueue
		logger logr.Logger
	}

	// Phase 1: Collect all queues and resolve ManagedQueue handles in one pass.
	// This avoids holding registry locks while processing, and allows us to determine the optimal number of workers.
	var resolvedQueues []resolvedQueue
	for _, priority := range p.registry.AllOrderedPriorityLevels() {
		band, err := p.registry.PriorityBandAccessor(priority)
		if err != nil {
			logger.Error(err, "Failed to get PriorityBandAccessor", "priority", priority)
			continue
		}
		band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
			key := queue.FlowKey()
			mq, err := p.registry.ManagedQueue(key)
			if err != nil {
				logger.V(logutil.DEBUG).Info("Skipping queue; ManagedQueue no longer resolvable",
					"flowKey", key, "err", err)
				return true
			}
			resolvedQueues = append(resolvedQueues, resolvedQueue{
				mq: mq,
				logger: logger.WithValues(
					"flowKey", key,
					"flowID", key.ID,
					"flowPriority", key.Priority),
			})
			return true
		})
	}

	if len(resolvedQueues) == 0 {
		return
	}

	// Phase 2: Determine the optimal number of workers.
	numWorkers := min(maxCleanupWorkers, len(resolvedQueues))

	// Phase 3: Create a worker pool to process the resolved queues.
	tasks := make(chan resolvedQueue)

	var wg sync.WaitGroup
	for range numWorkers {
		wg.Go(func() {
			for task := range tasks {
				processFn(task.mq, task.logger)
			}
		})
	}

	// Feed the channel with all the queues to be processed.
	for _, task := range resolvedQueues {
		tasks <- task
	}
	close(tasks) // Close the channel to signal workers to exit.
	wg.Wait()    // Wait for all workers to finish.
}
