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
}

// NewFlowControlAdmissionController creates a new FlowControlAdmissionController.
func NewFlowControlAdmissionController(fc flowController, poolName string) *FlowControlAdmissionController {
	return &FlowControlAdmissionController{
		flowController: fc,
		poolName:       poolName,
	}
}

// Admit implements the AdmissionController interface by deferring the admission decision to the Flow Control system
// via EnqueueAndWait. Saturation is enforced downstream by the dispatch cycle, which gates lower-priority bands as
// pool saturation approaches their usage limits; queued requests may be rejected on capacity or evicted on TTL expiry.
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

	outcome, err := fcac.flowController.EnqueueAndWait(ctx, fcReq)
	logger.V(logutil.DEBUG).Info("Flow control outcome",
		"requestID", reqCtx.SchedulingRequest.RequestID, "outcome", outcome, "error", err)
	return translateFlowControlOutcome(outcome, err)
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

// InitialEffectiveTTL returns 0 to defer to the controller-level default TTL, which is therefore the only TTL
// source for every request today.
// TODO(https://github.com/llm-d/llm-d-router/issues/1090): plumb more specific TTL scopes and resolve
// most-specific-wins: per-request (clamped), then per-band (effectively priority band config, eventually
// codifiable in the InferenceObjective CRD), then the controller default.
func (r *flowControlRequest) InitialEffectiveTTL() time.Duration { return 0 }
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
