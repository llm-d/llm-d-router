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

// Package controller contains the implementation of the FlowController engine.
//
// The FlowController is the central processing engine of the Flow Control layer. It is a high-throughput
// component responsible for managing the lifecycle of all incoming requests. It achieves this by acting as a stateless
// supervisor that orchestrates a  stateful worker (SProcessor).
package controller

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/utils/clock"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller/internal"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// registryClient defines the minimal interface that the FlowController needs to interact with the FlowRegistry.
type registryClient interface {
	contracts.FlowRegistryObserver
	contracts.FlowRegistryDataPlane
}

// processor is the minimal internal interface that the FlowController requires from its workers.
type processor interface {
	Run(ctx context.Context)
	Submit(item *internal.FlowItem) error
	SubmitOrBlock(ctx context.Context, item *internal.FlowItem) error
}

// processorFactory defines the signature for creating a Processor.
type processorFactory func(
	ctx context.Context,
	registry contracts.FlowRegistry,
	saturationDetector flowcontrol.SaturationDetector,
	endpointCandidates contracts.EndpointCandidates,
	usageLimitPolicy flowcontrol.UsageLimitPolicy,
	clock clock.WithTicker,
	cleanupSweepInterval time.Duration,
	enqueueChannelBufferSize int,
	logger logr.Logger,
) processor

var _ processor = &internal.Processor{}

// FlowController is the central, high-throughput engine of the Flow Control layer.
// It is designed as a stateless distributor that orchestrates a stateful worker (Processor), following a
// supervisor-worker pattern.
//
// Request Lifecycle Management:
//
//  1. Asynchronous Finalization (Controller-Owned): The Controller actively monitors the request Context
//     (TTL/Cancellation) in EnqueueAndWait. If the Context expires, the Controller immediately Finalizes the item and
//     unblocks the caller.
//  2. Synchronous Finalization (Processor-Owned): The Processor handles Dispatch, Capacity Rejection, and Shutdown.
//  3. Cleanup (Processor-Owned): The Processor periodically sweeps externally finalized items to reclaim capacity.
type FlowController struct {
	// --- Immutable dependencies (set at construction) ---

	config             *Config
	registry           registryClient
	saturationDetector flowcontrol.SaturationDetector
	endpointCandidates contracts.EndpointCandidates
	usageLimitPolicy   flowcontrol.UsageLimitPolicy
	clock              clock.WithTicker
	logger             logr.Logger
	processorFactory   processorFactory
	processor          processor

	// --- Lifecycle state ---

	// parentCtx is the root context for the controller's lifecycle, established when NewFlowController is called.
	// It is the parent for all long-lived worker goroutines.
	parentCtx context.Context
}

// Deps groups the external FlowController build dependencies to construct a FlowController.
type Deps struct {
	Registry           contracts.FlowRegistry
	SaturationDetector flowcontrol.SaturationDetector
	EndpointCandidates contracts.EndpointCandidates
	UsageLimitPolicy   flowcontrol.UsageLimitPolicy
	Clock              clock.WithTicker
	ProcessorFactory   processorFactory
}

// NewFlowController creates and starts a new FlowController instance.
// The provided context governs the lifecycle of the controller and all its workers.
func NewFlowController(
	ctx context.Context,
	poolName string,
	config *Config,
	deps Deps,
) *FlowController {
	if deps.Clock == nil {
		deps.Clock = clock.RealClock{}
	}
	fc := &FlowController{
		config:             config,
		registry:           deps.Registry,
		saturationDetector: deps.SaturationDetector,
		endpointCandidates: deps.EndpointCandidates,
		usageLimitPolicy:   deps.UsageLimitPolicy,
		clock:              deps.Clock,
		logger:             log.FromContext(ctx).WithName("flow-controller"),
		parentCtx:          ctx,
	}

	if deps.ProcessorFactory == nil {
		fc.processorFactory = func(
			ctx context.Context,
			registry contracts.FlowRegistry,
			saturationDetector flowcontrol.SaturationDetector,
			endpointCandidates contracts.EndpointCandidates,
			usageLimitPolicy flowcontrol.UsageLimitPolicy,
			clock clock.WithTicker,
			cleanupSweepInterval time.Duration,
			enqueueChannelBufferSize int,
			logger logr.Logger,
		) processor {
			return internal.NewProcessor(
				ctx,
				poolName,
				registry,
				saturationDetector,
				endpointCandidates,
				usageLimitPolicy,
				clock,
				cleanupSweepInterval,
				enqueueChannelBufferSize,
				logger,
			)
		}
	} else {
		fc.processorFactory = deps.ProcessorFactory
	}

	// Construct a new worker, but do not start its goroutine yet.
	fc.processor = fc.processorFactory(
		fc.parentCtx,
		fc.registry,
		fc.saturationDetector,
		fc.endpointCandidates,
		fc.usageLimitPolicy,
		fc.clock,
		fc.config.ExpiryCleanupInterval,
		fc.config.EnqueueChannelBufferSize,
		fc.logger,
	)

	return fc
}

func (fc *FlowController) Start() {
	fc.logger.V(logutil.DEFAULT).Info("Starting the Processor.")

	go fc.processor.Run(fc.parentCtx)

}

// EnqueueAndWait is the primary, synchronous entry point to the Flow Control system. It submits a request and blocks
// until the request reaches a terminal outcome (dispatched, rejected, or evicted).
//
// # Design Rationale: The Synchronous Model
//
// This blocking model is deliberately chosen for its simplicity and robustness, especially in the context of Envoy
// External Processing (ext_proc), which operates on a stream-based protocol.
//
//   - ext_proc Alignment: A single goroutine typically manages the stream for a given HTTP request.
//     EnqueueAndWait fits this perfectly: the request-handling goroutine calls it, blocks, and upon return, has a
//     definitive outcome to act upon.
//   - Simplified State Management: The state of a "waiting" request is implicitly managed by the blocked goroutine's
//     stack and its Context. The system only needs to signal this specific goroutine to unblock it.
//   - Direct Backpressure: If queues are full, EnqueueAndWait returns an error immediately, providing direct
//     backpressure to the caller.
func (fc *FlowController) EnqueueAndWait(
	ctx context.Context,
	req flowcontrol.FlowControlRequest,
) (types.QueueOutcome, error) {
	flowKey := req.FlowKey()
	priority := strconv.Itoa(flowKey.Priority)
	reqBytes := req.ByteSize()
	metrics.IncFlowControlQueueSize(
		flowKey.ID, priority,
		req.InferencePoolName(),
		req.ModelName(), req.TargetModelName())
	defer metrics.DecFlowControlQueueSize(
		flowKey.ID, priority,
		req.InferencePoolName(),
		req.ModelName(), req.TargetModelName())
	metrics.AddFlowControlQueueBytes(
		flowKey.ID, priority,
		req.InferencePoolName(),
		req.ModelName(), req.TargetModelName(), reqBytes)
	defer metrics.SubFlowControlQueueBytes(
		flowKey.ID, priority,
		req.InferencePoolName(),
		req.ModelName(), req.TargetModelName(), reqBytes)

	// 1. Create the derived context that governs this request's lifecycle (Parent Cancellation + TTL).
	reqCtx, cancel, enqueueTime := fc.createRequestContext(ctx, req)
	defer cancel()

	var finalOutcome types.QueueOutcome

	// 2. Acquire a lease for the Flow.
	// We hold this lease for the entire duration of the request (Distribution + Queueing).
	err := fc.registry.WithConnection(flowKey, func(conn contracts.ActiveFlowConnection) error {

		select { // Non-blocking check on controller lifecycle.
		case <-fc.parentCtx.Done():
			finalOutcome = types.QueueOutcomeRejectedOther
			return fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning)
		default:
		}

		// Attempt to distribute the request once, passing the active connection.
		item, err := fc.tryDistribution(reqCtx, req, enqueueTime, conn)
		if err != nil {
			// Distribution failed terminally (e.g., context cancelled during blocking submit).
			// The item has already been finalized by tryDistribution.
			finalState := item.FinalState()
			finalOutcome = finalState.Outcome
			return finalState.Err
		}

		// Distribution was successful; ownership of the item has been transferred to a processor.
		// Now, we block here in awaitFinalization until the request is finalized by either the processor (e.g., dispatched,
		// rejected) or the controller itself (e.g., caller's context cancelled/TTL expired).
		outcome, err := fc.awaitFinalization(reqCtx, item)

		// The outcome is terminal (Dispatched, Evicted, or another rejection).
		finalOutcome = outcome
		return err
	})

	// If WithConnection returned an error (e.g. connection failure, context cancelled before lease), we must ensure we
	// return a valid rejection outcome.
	// In the success case (where the closure ran), finalOutcome is set inside the closure.
	if err != nil && finalOutcome == types.QueueOutcomeNotYetFinalized {
		return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: %w", types.ErrRejected, err)
	}

	return finalOutcome, err
}

// tryDistribution handles a single attempt to select a shard and submit a request.
// It uses the provided `conn` to access the registry data plane.
// If this function returns an error, it guarantees that the provided `item` has been finalized.
func (fc *FlowController) tryDistribution(
	reqCtx context.Context,
	req flowcontrol.FlowControlRequest,
	enqueueTime time.Time,
	conn contracts.ActiveFlowConnection,
) (*internal.FlowItem, error) {
	// Calculate effective TTL for item initialization (reqCtx is the enforcement mechanism).
	effectiveTTL := fc.config.DefaultRequestTTL
	if deadline, ok := reqCtx.Deadline(); ok {
		if ttl := deadline.Sub(enqueueTime); ttl > 0 {
			effectiveTTL = ttl
		}
	}

	// We must create a fresh FlowItem on each attempt as finalization is per-lifecycle.
	item := internal.NewItem(req, effectiveTTL, enqueueTime)

	dp := conn.GetDataPlane()
	_, err := dp.ManagedQueue(conn.FlowKey())
	if err != nil {
		fc.logger.Error(err,
			"Invariant violation. Failed to get ManagedQueue for a leased flow.",
			"flowKey", conn.FlowKey())
		item.FinalizeWithOutcome(types.QueueOutcomeRejectedCapacity, types.ErrRejected)
		return item, err
	}

	outcome, err := fc.distributeRequest(reqCtx, item)
	if err == nil {
		// Success: Ownership of the item has been transferred to the processor.
		return item, nil
	}

	// For any distribution error, the controller retains ownership and must finalize the item.
	var finalErr error
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		// We propagate the original context error here, EnqueueAndWait will rely on item.FinalState().Err.
		finalErr = err
		item.Finalize(context.Cause(reqCtx))
	} else { // e.g.,
		finalErr = fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
		item.FinalizeWithOutcome(outcome, finalErr)
	}
	return item, finalErr
}

// awaitFinalization blocks until an item is finalized, either by the processor (synchronously) or by the controller
// itself due to context expiry (asynchronously).
func (fc *FlowController) awaitFinalization(
	reqCtx context.Context,
	item *internal.FlowItem,
) (types.QueueOutcome, error) {
	select {
	case <-reqCtx.Done():
		// Asynchronous Finalization (Controller-initiated):
		// The request Context expired (Cancellation/TTL) while the item was being processed.
		cause := context.Cause(reqCtx)
		item.Finalize(cause)

		// The processor will eventually discard this "zombie" item during its cleanup sweep.
		finalState := item.FinalState()
		return finalState.Outcome, finalState.Err

	case finalState := <-item.Done():
		// Synchronous Finalization (Processor-initiated):
		// The processor finalized the item (Dispatch, Reject, Shutdown).
		return finalState.Outcome, finalState.Err
	}
}

// createRequestContext derives the context that governs a request's lifecycle, enforcing the TTL deadline.
func (fc *FlowController) createRequestContext(
	ctx context.Context,
	req flowcontrol.FlowControlRequest,
) (context.Context, context.CancelFunc, time.Time) {
	enqueueTime := fc.clock.Now()
	effectiveTTL := req.InitialEffectiveTTL()
	if effectiveTTL <= 0 {
		effectiveTTL = fc.config.DefaultRequestTTL
	}

	if effectiveTTL > 0 {
		reqCtx, cancel := context.WithDeadlineCause(ctx, enqueueTime.Add(effectiveTTL), types.ErrTTLExpired)
		return reqCtx, cancel, enqueueTime
	}
	reqCtx, cancel := context.WithCancel(ctx)
	return reqCtx, cancel, enqueueTime
}

// distributeRequest implements a flow-aware, two-phase "Join-Shortest-Queue-by-Bytes" (JSQ-Bytes) distribution strategy
// with graceful backpressure. It attempts to submit an item to the best-ranked candidate from the provided list.
//
// The algorithm operates as follows:
//  1. Phase 1 (Non-blocking Fast Failover): It iterates through the ranked candidates and attempts a non-blocking
//     submission. The first successful submission wins.
//  2. Phase 2 (Blocking Fallback): If all non-blocking attempts fail, it performs a single blocking submission to the
//     least-loaded candidate, providing backpressure.
//
// The provided context (ctx) is used for the blocking submission phase (SubmitOrBlock).
//
// Ownership Contract:
//   - Returns nil: Success. Ownership transferred to Processor.
//   - Returns error: Failure (Context expiry, shutdown,, etc.).
//     Ownership retained by Controller. The Controller MUST finalize the item.
func (fc *FlowController) distributeRequest(
	ctx context.Context,
	item *internal.FlowItem,
) (types.QueueOutcome, error) {
	reqID := item.OriginalRequest().ID()
	if err := fc.processor.Submit(item); err == nil {
		return types.QueueOutcomeNotYetFinalized, nil
	}

	// processor is busy. Attempt a single blocking submission to the candidate.
	fc.logger.V(logutil.TRACE).Info("Processor is busy, attempting blocking submit", "requestID", reqID)
	err := fc.processor.SubmitOrBlock(ctx, item)
	if err != nil {
		return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
	}
	return types.QueueOutcomeNotYetFinalized, nil // Success, ownership transferred.
}
