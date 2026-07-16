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

package requestcontrol

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/statestore"
	requtil "github.com/llm-d/llm-d-router/pkg/epp/util/request"
)

// AdmissionController defines the interface for making admission control decisions.
// Implementations of this interface determine whether an incoming inference request should be accepted or rejected
// based on various criteria such as system load, fairness, priority, and available capacity.
type AdmissionController interface {
	// Admit determines if a request should be admitted.
	// It is called by the Director for each incoming request.
	//
	// Args:
	//   ctx: The request context, carrying deadlines, cancellation signals, and logger.
	//   reqCtx: The handlers.RequestContext containing details about the incoming request.
	//   priority: The priority level of the request, as determined by the InferenceObjective.
	//
	// Returns:
	//   - nil: If the request is admitted and should proceed to scheduling.
	//   - errcommon.Error: If the request is rejected.
	Admit(
		ctx context.Context,
		reqCtx *handlers.RequestContext,
		priority int,
	) error
}

// flowController defines the minimal interface required by FlowControlAdmissionController for enqueuing requests and
// waiting for an admission outcome.
type flowController interface {
	EnqueueAndWait(ctx context.Context, req flowcontrol.FlowControlRequest) (types.QueueOutcome, error)
}

// rejectIfSheddableAndSaturated checks if a request should be immediately rejected.
func rejectIfSheddableAndSaturated(
	ctx context.Context,
	sd flowcontrol.SaturationDetector,
	endpointCandidates contracts.EndpointCandidates,
	reqCtx *handlers.RequestContext,
	priority int,
	logger logr.Logger,
) error {
	if requtil.IsSheddable(priority) {
		if sd.Saturation(ctx, endpointCandidates.Locate(ctx, reqCtx.Request.Metadata)) >= 1.0 {
			logger.V(logutil.TRACE).Info("Request rejected: system saturated and request is sheddable",
				"requestID", reqCtx.SchedulingRequest.RequestID)
			return errcommon.Error{
				Code: errcommon.ResourceExhausted,
				Msg:  "system saturated, sheddable request dropped",
			}
		}
	}
	return nil
}

// --- LegacyAdmissionController ---

// LegacyAdmissionController implements saturation-based admission control.
// It rejects sheddable requests (priority < 0) if the saturationDetector indicates that the system is currently
// saturated. Non-sheddable requests always bypass the saturation check.
type LegacyAdmissionController struct {
	saturationDetector flowcontrol.SaturationDetector
	endpointCandidates contracts.EndpointCandidates
}

// NewLegacyAdmissionController creates a new LegacyAdmissionController.
func NewLegacyAdmissionController(
	sd flowcontrol.SaturationDetector,
	endpointCandidates contracts.EndpointCandidates,
) *LegacyAdmissionController {
	return &LegacyAdmissionController{
		saturationDetector: sd,
		endpointCandidates: endpointCandidates,
	}
}

// Admit implements the AdmissionController interface for the legacy strategy.
// It checks for saturation only for requests with priority < 0.
func (lac *LegacyAdmissionController) Admit(
	ctx context.Context,
	reqCtx *handlers.RequestContext,
	priority int,
) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Executing LegacyAdmissionController",
		"priority", priority, "fairnessID", reqCtx.SchedulingRequest.FairnessID)
	if err := rejectIfSheddableAndSaturated(
		ctx,
		lac.saturationDetector,
		lac.endpointCandidates,
		reqCtx, priority,
		logger,
	); err != nil {
		return err
	}
	logger.V(logutil.TRACE).Info("Request admitted", "requestID", reqCtx.SchedulingRequest.RequestID)
	return nil
}

// --- FlowControlAdmissionController ---

// FlowControlAdmissionController delegates admission decisions to the Flow Control layer.
// It uses the provided Flow Controller to enqueue the request and await an outcome.
type FlowControlAdmissionController struct {
	flowController flowController
	poolName       string

	// state, when non-nil, additionally gates admission on a fleet-wide
	// concurrency lease after the local FlowController admits a request
	// (RFC #1593 feasibility spike, stateless mode only). nil in classic and
	// stateful mode: those keep today's exact admission semantics and error
	// granularity (translateFlowControlOutcome), never going through this path.
	state statestore.FlowControlState
}

// NewFlowControlAdmissionController creates a new FlowControlAdmissionController.
func NewFlowControlAdmissionController(fc flowController, poolName string, opts ...FlowControlAdmissionControllerOption) *FlowControlAdmissionController {
	fcac := &FlowControlAdmissionController{
		flowController: fc,
		poolName:       poolName,
	}
	for _, opt := range opts {
		opt(fcac)
	}
	return fcac
}

// FlowControlAdmissionControllerOption configures optional behavior of a
// FlowControlAdmissionController constructed via NewFlowControlAdmissionController.
type FlowControlAdmissionControllerOption func(*FlowControlAdmissionController)

// WithConcurrencyLease additionally gates admission on a fleet-wide
// concurrency lease (pkg/epp/statestore.ConcurrencyState) after the local
// FlowController admits a request, per the RFC's LocalFallback degradation
// strategy: local queue admission always runs unchanged; the remote check
// runs only after, and on remote failure a local concurrency cap
// (localMaxConcurrency) is used instead. localMaxConcurrency <= 0 means no
// local fallback cap.
//
// fc (the flowController already held by the controller) is reused directly
// as the local pkg/epp/statestore.FlowControlBackend: its EnqueueAndWait
// signature already matches, so it is passed in here rather than
// re-resolved from the controller to keep construction order explicit.
func WithConcurrencyLease(fc flowController, remote statestore.ConcurrencyState, timeout time.Duration, localMaxConcurrency int64) FlowControlAdmissionControllerOption {
	return func(fcac *FlowControlAdmissionController) {
		local := statestore.NewLocalFlowControlState(fc)
		fcac.state = statestore.NewLocalFallbackFlowControlState(local, remote, timeout, localMaxConcurrency)
	}
}

// ConcurrencyReleaser is implemented by AdmissionController implementations
// that hold a fleet-wide concurrency lease requiring release at response
// completion (RFC #1593 feasibility spike, stateless mode only). Director
// checks for this via a type assertion, since AdmissionController itself has
// no Release method — classic/stateful admission never needs one.
type ConcurrencyReleaser interface {
	ReleaseConcurrency(ctx context.Context, requestID, fairnessID string, priority int)
}

var _ ConcurrencyReleaser = (*FlowControlAdmissionController)(nil)

// Admit implements the AdmissionController interface by checking for saturation on sheddable requests first, then
// deferring to the Flow Control system.
func (fcac *FlowControlAdmissionController) Admit(
	ctx context.Context,
	reqCtx *handlers.RequestContext,
	priority int,
) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Executing FlowControlAdmissionController",
		"requestID", reqCtx.SchedulingRequest.RequestID, "priority", priority, "fairnessID", reqCtx.SchedulingRequest.FairnessID)

	fcReq := &flowControlRequest{
		fairnessID:        reqCtx.SchedulingRequest.FairnessID,
		priority:          priority,
		requestByteSize:   uint64(reqCtx.RequestSize),
		inferenceRequest:  reqCtx.SchedulingRequest,
		receivedTimestamp: reqCtx.RequestReceivedTimestamp,
		reqMetadata:       reqCtx.Request.Metadata,
		inferencePoolName: fcac.poolName,
		modelName:         reqCtx.IncomingModelName,
	}

	if fcac.state != nil {
		outcome, err := fcac.state.Admit(ctx, fcReq)
		logger.V(logutil.DEBUG).Info("Flow control + concurrency lease outcome",
			"requestID", reqCtx.SchedulingRequest.RequestID, "outcome", outcome, "error", err)
		return translateConcurrencyOutcome(outcome, err)
	}

	outcome, err := fcac.flowController.EnqueueAndWait(ctx, fcReq)
	logger.V(logutil.DEBUG).Info("Flow control outcome",
		"requestID", reqCtx.SchedulingRequest.RequestID, "outcome", outcome, "error", err)
	return translateFlowControlOutcome(outcome, err)
}

// ReleaseConcurrency releases a previously admitted concurrency lease. A
// no-op when no concurrency lease is configured (classic/stateful mode, or
// this request was rejected before a lease was ever held).
func (fcac *FlowControlAdmissionController) ReleaseConcurrency(ctx context.Context, requestID, fairnessID string, priority int) {
	if fcac.state == nil {
		return
	}
	_ = fcac.state.Release(ctx, requestID, statestore.FlowControlKey{ID: fairnessID, Priority: priority})
}

// translateConcurrencyOutcome maps the coarse outcome of the concurrency-lease
// composition (pkg/epp/statestore.FlowControlOutcome) to the public
// errcommon.Error contract used by the Director. Unlike
// translateFlowControlOutcome, this collapses the local FlowController's
// richer rejection/eviction reasons (capacity, no-endpoints, TTL, context
// cancellation) into a single "rejected" signal, since
// statestore.localFallbackConcurrencyState's local step reports only
// Admitted/Rejected — a documented simplification for this feasibility spike.
func translateConcurrencyOutcome(outcome statestore.FlowControlOutcome, err error) error {
	switch outcome {
	case statestore.FlowControlOutcomeAdmitted, statestore.FlowControlOutcomeDegraded:
		return nil
	case statestore.FlowControlOutcomeRejected:
		msg := "request rejected by flow control"
		if err != nil {
			msg = err.Error()
		}
		return errcommon.Error{Code: errcommon.ResourceExhausted, Msg: msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSaturated)}}
	default:
		return errcommon.Error{Code: errcommon.Internal, Msg: fmt.Sprintf("unhandled concurrency outcome: %v", outcome)}
	}
}

// flowControlRequest is an adapter that implements the FlowControlRequest interface.
type flowControlRequest struct {
	fairnessID        string
	priority          int
	requestByteSize   uint64
	inferenceRequest  *scheduling.InferenceRequest
	receivedTimestamp time.Time
	reqMetadata       map[string]any
	inferencePoolName string
	modelName         string
}

var _ flowcontrol.FlowControlRequest = &flowControlRequest{}

func (r *flowControlRequest) ID() string {
	if r.inferenceRequest == nil {
		return ""
	}
	return r.inferenceRequest.RequestID
}
func (r *flowControlRequest) InitialEffectiveTTL() time.Duration { return 0 } // Use controller default.
func (r *flowControlRequest) ByteSize() uint64                   { return r.requestByteSize }

func (r *flowControlRequest) InferenceRequest() *scheduling.InferenceRequest {
	return r.inferenceRequest
}
func (r *flowControlRequest) ReceivedTimestamp() time.Time { return r.receivedTimestamp }
func (r *flowControlRequest) GetMetadata() map[string]any  { return r.reqMetadata }
func (r *flowControlRequest) InferencePoolName() string    { return r.inferencePoolName }
func (r *flowControlRequest) ModelName() string            { return r.modelName }
func (r *flowControlRequest) TargetModelName() string {
	if r.inferenceRequest == nil {
		return ""
	}
	return r.inferenceRequest.TargetModel
}

func (r *flowControlRequest) FlowKey() flowcontrol.FlowKey {
	return flowcontrol.FlowKey{ID: r.fairnessID, Priority: r.priority}
}

// translateFlowControlOutcome maps the context-rich outcome of the Flow Control layer to the public errcommon.Error
// contract used by the Director.
func translateFlowControlOutcome(outcome types.QueueOutcome, err error) error {
	msg := "request rejected by flow control"
	if err != nil {
		msg = err.Error()
	}

	switch outcome {
	case types.QueueOutcomeDispatched:
		return nil
	case types.QueueOutcomeRejectedCapacity:
		return errcommon.Error{Code: errcommon.ResourceExhausted, Msg: msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSaturated)}}
	case types.QueueOutcomeRejectedNoEndpoints:
		// No serving capacity exists (e.g. pool scaled to zero): signal genuine unavailability rather than backpressure.
		return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no endpoints available: " + msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonNoEndpoints)}}
	case types.QueueOutcomeEvictedTTL:
		return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "request timed out in queue: " + msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonTTLExpired)}}
	case types.QueueOutcomeEvictedContextCancelled:
		return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "client disconnected: " + msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonContextCancelled)}}
	case types.QueueOutcomeRejectedOther, types.QueueOutcomeEvictedOther:
		if errors.Is(err, types.ErrFlowControllerNotRunning) {
			return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "flow controller shutting down: " + msg, Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonShuttingDown)}}
		}
		return errcommon.Error{Code: errcommon.Internal, Msg: "internal flow control error: " + msg}
	default:
		return errcommon.Error{Code: errcommon.Internal, Msg: "unhandled flow control outcome: " + msg}
	}
}
