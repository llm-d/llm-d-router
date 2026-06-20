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
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// registryClient defines the minimal interface that the FlowController needs to interact with the FlowRegistry.
type registryClient interface {
	contracts.FlowRegistryObserver
	contracts.FlowRegistryDataPlane
}

// processor is the minimal internal interface that the FlowController requires from its worker.
type processor interface {
	Run(ctx context.Context)
	Submit(item *internal.FlowItem) error
	SubmitOrBlock(ctx context.Context, item *internal.FlowItem) error
}

// processorFactory defines the signature for creating a Processor.
type processorFactory func(
	ctx context.Context,
	registry contracts.FlowRegistry,
	registryBackground contracts.FlowRegistryBackground,
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
// It is the request-facing front end that hands each request to a single stateful worker (the Processor) and blocks
// until the request reaches a terminal outcome.
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
	flowRegistry       contracts.FlowRegistry
	registryBackground contracts.FlowRegistryBackground
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
	var registryBackground contracts.FlowRegistryBackground
	if bg, ok := deps.Registry.(contracts.FlowRegistryBackground); ok {
		registryBackground = bg
	}
	fc := &FlowController{
		config:             config,
		registry:           deps.Registry,
		flowRegistry:       deps.Registry,
		registryBackground: registryBackground,
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
			registryBackground contracts.FlowRegistryBackground,
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
				registryBackground,
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
		fc.flowRegistry,
		fc.registryBackground,
		fc.saturationDetector,
		fc.endpointCandidates,
		fc.usageLimitPolicy,
		fc.clock,
		fc.config.ExpiryCleanupInterval,
		fc.config.EnqueueChannelBufferSize,
		fc.logger,
	)

	fc.logger.V(logutil.DEFAULT).Info("Starting the Processor.")

	go fc.processor.Run(fc.parentCtx)

	return fc
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
	sloClass := metrics.SLOClassNone
	if r := req.InferenceRequest(); r != nil {
		sloClass = fwkrequest.GetHeader(r.Headers, metadata.ObjectiveKey)
		if sloClass == "" {
			sloClass = metrics.SLOClassNone
		}
	}
	if sloClass == "" {
		sloClass = metrics.SLOClassNone
	}
	metrics.RecordFlowControlSLOIncomingRequest(sloClass, req.InferencePoolName())
	metrics.IncFlowControlQueueSize(
		flowKey.ID, priority,
		req.InferencePoolName(),
		sloClass,
		req.ModelName(), req.TargetModelName())
	defer metrics.DecFlowControlQueueSize(
		flowKey.ID, priority,
		req.InferencePoolName(),
		sloClass,
		req.ModelName(), req.TargetModelName())
	metrics.AddFlowControlQueueBytes(
		flowKey.ID, priority,
		req.InferencePoolName(),
		sloClass,
		req.ModelName(), req.TargetModelName(), reqBytes)
	defer metrics.SubFlowControlQueueBytes(
		flowKey.ID, priority,
		req.InferencePoolName(),
		sloClass,
		req.ModelName(), req.TargetModelName(), reqBytes)

	// 1. Create the derived context that governs this request's lifecycle (Parent Cancellation + TTL).
	reqCtx, cancel, enqueueTime := fc.createRequestContext(ctx, req)
	defer cancel()

	var finalOutcome types.QueueOutcome

	// 2. Acquire a lease for the Flow.
	// We hold this lease for the entire duration of the request (Distribution + Queueing).
	err := fc.withConnectionWithFallback(req, func(conn contracts.ActiveFlowConnection, effectiveReq flowcontrol.FlowControlRequest) error {

		select { // Non-blocking check on controller lifecycle.
		case <-fc.parentCtx.Done():
			finalOutcome = types.QueueOutcomeRejectedOther
			return fmt.Errorf("%w: %w", types.ErrRejected, types.ErrFlowControllerNotRunning)
		default:
		}

		// Attempt to distribute the request once, passing the active connection.
		// effectiveReq carries the fallback flow key when the requested band was not provisioned, so the
		// item is enqueued under the band that was actually leased.
		item, err := fc.tryDistribution(reqCtx, effectiveReq, enqueueTime, conn)
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
		finalOutcome = types.QueueOutcomeRejectedOther
		err = fmt.Errorf("%w: %w", types.ErrRejected, err)
	}

	if finalOutcome != types.QueueOutcomeDispatched {
		fc.logger.V(logutil.VERBOSE).Info("Request dropped",
			"requestID", req.ID(), "flowKey", flowKey, "outcome", finalOutcome, "err", err)
	}

	metrics.IncFlowControlRequestsTotal(finalOutcome.String(), priority, req.InferencePoolName())

	return finalOutcome, err
}

// fallbackRequest wraps a FlowControlRequest to override its flow key, so a request that falls back to a different
// priority is enqueued under the band that was actually leased rather than its original (unprovisioned) band.
//
// Trade-off: downstream consumers see this wrapper, so item.OriginalRequest().FlowKey() reports the fallback
// priority rather than the requested one — despite the method name. This is intentional, since the item must be
// leased, distributed, and enqueued consistently at the fallback priority. The originally requested priority
// therefore survives only in the withConnectionWithFallback log; surfacing it in metrics is left as a follow-up
// (a dedicated fallback counter labeled with the original priority).
type fallbackRequest struct {
	flowcontrol.FlowControlRequest
	key flowcontrol.FlowKey
}

func (r fallbackRequest) FlowKey() flowcontrol.FlowKey { return r.key }

// withConnectionWithFallback acquires a flow connection, falling back to priority 0 when the requested band is not yet
// provisioned. On fallback, the callback receives a request whose FlowKey reports priority 0, ensuring the item is
// leased, distributed, and enqueued consistently under priority 0; otherwise it receives the original request.
//
// Note: relative to the requested priority this is a demotion for positive priorities but a promotion for negative
// ones. It is an availability-first fallback for the brief window before the control plane provisions the band.
func (fc *FlowController) withConnectionWithFallback(
	req flowcontrol.FlowControlRequest,
	fn func(conn contracts.ActiveFlowConnection, effectiveReq flowcontrol.FlowControlRequest) error,
) error {
	key := req.FlowKey()
	err := fc.registry.WithConnection(key, func(conn contracts.ActiveFlowConnection) error {
		return fn(conn, req)
	})
	if err == nil || !errors.Is(err, contracts.ErrPriorityBandNotFound) || key.Priority == 0 {
		return err
	}

	fc.logger.V(logutil.DEFAULT).Info(
		"Priority band not provisioned, falling back to priority 0",
		"originalPriority", key.Priority,
		"flowID", key.ID,
	)
	fallbackKey := key
	fallbackKey.Priority = 0
	fallback := fallbackRequest{FlowControlRequest: req, key: fallbackKey}
	return fc.registry.WithConnection(fallbackKey, func(conn contracts.ActiveFlowConnection) error {
		return fn(conn, fallback)
	})
}

// tryDistribution handles a single attempt to submit a request to the processor.
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

// distributeRequest submits an item to the processor with graceful backpressure.
//
// It operates in two phases:
//  1. Non-blocking submit: a fast Submit that succeeds immediately if the processor's enqueue channel has capacity.
//  2. Blocking fallback: if the processor is busy, a single blocking SubmitOrBlock that applies backpressure until the
//     item is accepted, the context expires, or the processor shuts down.
//
// The provided context (ctx) is used for the blocking submission phase (SubmitOrBlock).
//
// Ownership Contract:
//   - Returns nil: Success. Ownership transferred to the Processor.
//   - Returns error: Failure (context expiry, shutdown, etc.).
//     Ownership retained by the Controller, which MUST finalize the item.
func (fc *FlowController) distributeRequest(
	ctx context.Context,
	item *internal.FlowItem,
) (types.QueueOutcome, error) {
	reqID := item.OriginalRequest().ID()
	if err := fc.processor.Submit(item); err == nil {
		return types.QueueOutcomeNotYetFinalized, nil
	}

	// processor is busy. Attempt a single blocking submission to the candidate.
	fc.logger.V(logutil.DEBUG).Info("Processor is busy, attempting blocking submit", "requestID", reqID)
	err := fc.processor.SubmitOrBlock(ctx, item)
	if err != nil {
		return types.QueueOutcomeRejectedOther, fmt.Errorf("%w: request not accepted: %w", types.ErrRejected, err)
	}
	return types.QueueOutcomeNotYetFinalized, nil // Success, ownership transferred.
}
