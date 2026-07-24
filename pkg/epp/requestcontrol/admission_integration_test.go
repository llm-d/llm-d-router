/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	fccontroller "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	fcregistry "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// This file covers the admission-to-controller seam with a REAL FlowController and FlowRegistry
// behind FlowControlAdmissionController, complementing admission_test.go (which mocks the flow
// controller) and the flowcontrol integration suite (which enters at EnqueueAndWait). It asserts
// the externally visible contract of Admit for every terminal flow control outcome: the
// errcommon.Error code and the x-llm-d-request-dropped-reason header.

// realFlowControlHarness wires a FlowControlAdmissionController to a real FlowController and
// FlowRegistry, following the construction pattern of the flowcontrol integration suite
// (pkg/epp/flowcontrol/integration_helpers_test.go).
type realFlowControlHarness struct {
	cancel context.CancelFunc
	ac     *FlowControlAdmissionController
	reg    *fcregistry.FlowRegistry
}

// realFlowControlOpts selects the knobs each outcome needs: a fixed saturation level (1.0 keeps
// requests queued), the candidate pool (empty vs. non-empty steers capacity-rejection
// classification), the controller-default request TTL, and an optional per-band request cap.
type realFlowControlOpts struct {
	saturation      float64
	candidates      []fwkdl.Endpoint
	requestTTL      time.Duration // Defaults to one minute (effectively "never" for these tests).
	bandMaxRequests uint64        // Zero means unbounded.
}

func newRealFlowControlHarness(t *testing.T, opts realFlowControlOpts) *realFlowControlHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	handle := testutils.NewTestHandle(ctx)
	orderingPlugin, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	require.NoError(t, err)
	fairnessPlugin, err := globalstrict.GlobalStrictFairnessPolicyFactory("global-strict", nil, handle)
	require.NoError(t, err)
	defaults := fcregistry.PriorityBandPolicyDefaults{
		OrderingPolicy: orderingPlugin.(flowcontrol.OrderingPolicy),
		FairnessPolicy: fairnessPlugin.(flowcontrol.FairnessPolicy),
	}

	bandOpts := []fcregistry.PriorityBandConfigOption{fcregistry.WithBandMaxBytes(10_000_000)}
	if opts.bandMaxRequests > 0 {
		bandOpts = append(bandOpts, fcregistry.WithBandMaxRequests(opts.bandMaxRequests))
	}
	band, err := fcregistry.NewPriorityBandConfig(0, defaults, bandOpts...)
	require.NoError(t, err)
	regCfg, err := fcregistry.NewConfig(defaults, fcregistry.WithPriorityBand(band))
	require.NoError(t, err)
	reg := fcregistry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.RunMaintenanceLoop(ctx)

	requestTTL := opts.requestTTL
	if requestTTL == 0 {
		requestTTL = time.Minute
	}
	detector := &mockSaturationDetector{
		SaturationFunc: func(context.Context, []fwkdl.Endpoint) float64 { return opts.saturation },
	}
	fc := fccontroller.NewFlowController(ctx, "test-pool", &fccontroller.Config{
		DefaultRequestTTL:        requestTTL,
		ExpiryCleanupInterval:    10 * time.Millisecond,
		EnqueueChannelBufferSize: 100,
	}, fccontroller.Deps{
		Registry:           reg,
		SaturationDetector: detector,
		EndpointCandidates: &mocks.MockEndpointCandidates{Candidates: opts.candidates},
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})

	return &realFlowControlHarness{
		cancel: cancel,
		ac:     NewFlowControlAdmissionController(fc, "test-pool"),
		reg:    reg,
	}
}

// newSeamRequestContext builds the minimal RequestContext the admission controller adapts into a
// FlowControlRequest. FairnessID must be non-empty: the registry rejects empty flow IDs.
func newSeamRequestContext(id string) *handlers.RequestContext {
	return &handlers.RequestContext{
		SchedulingRequest:        &fwksched.InferenceRequest{RequestID: id, FairnessID: "seam-flow"},
		Request:                  &handlers.Request{Metadata: map[string]any{}},
		RequestSize:              100,
		RequestReceivedTimestamp: time.Now(),
		IncomingModelName:        "test-model",
	}
}

// admitAsync runs Admit in a goroutine so the test can drive queue state (e.g. cancel a context
// or submit an overflow request) while the call blocks inside EnqueueAndWait.
func admitAsync(ctx context.Context, ac *FlowControlAdmissionController, id string) <-chan error {
	result := make(chan error, 1)
	go func() { result <- ac.Admit(ctx, newSeamRequestContext(id), 0) }()
	return result
}

// waitAdmit unblocks a pending admitAsync result. The generous timeout mirrors the flowcontrol
// integration suite; every path under test finalizes deterministically well before it.
func waitAdmit(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("Admit did not return a terminal outcome within 10s")
		return nil
	}
}

// requireDropped asserts the errcommon.Error contract for a dropped request: the canonical error
// code (which drives the HTTP status) and the dropped-reason response header.
func requireDropped(t *testing.T, err error, wantCode string, wantReason errcommon.RequestDroppedReason) {
	t.Helper()
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e, "error should be of type errcommon.Error")
	assert.Equal(t, wantCode, e.Code, "incorrect error code")
	assert.Equal(t, string(wantReason), e.Headers[errcommon.RequestDroppedReasonHeaderKey],
		"incorrect dropped-reason header")
}

func nonEmptyEndpoints() []fwkdl.Endpoint {
	return []fwkdl.Endpoint{fwkdl.NewEndpoint(nil, nil)}
}

// TestFlowControlAdmissionController_RealControllerSeam drives every terminal flow control
// outcome through the real controller stack and asserts the status code + header contract that
// admission_test.go only covers against a mock.
func TestFlowControlAdmissionController_RealControllerSeam(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	t.Run("dispatched_returns_nil", func(t *testing.T) {
		t.Parallel()
		// Unsaturated detector + non-empty pool: the dispatch cycle admits immediately.
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation: 0.0,
			candidates: nonEmptyEndpoints(),
		})

		err := waitAdmit(t, admitAsync(ctx, h.ac, "dispatch-req"))
		require.NoError(t, err, "an unsaturated system should admit the request")
	})

	t.Run("rejected_capacity_returns_429_saturated", func(t *testing.T) {
		t.Parallel()
		// A saturated detector parks a filler request in the only queue slot (bandMaxRequests=1);
		// the next request overflows band capacity against a non-empty pool -> RejectedCapacity.
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation:      1.0,
			candidates:      nonEmptyEndpoints(),
			bandMaxRequests: 1,
		})

		filler := admitAsync(ctx, h.ac, "filler-req")
		require.Eventually(t, func() bool { return h.reg.Stats().TotalLen == 1 },
			time.Second, time.Millisecond, "filler request should be queued before the overflow request")

		err := waitAdmit(t, admitAsync(ctx, h.ac, "overflow-req"))
		requireDropped(t, err, errcommon.ResourceExhausted, errcommon.RequestDroppedReasonSaturated)

		// Shut the controller down and drain the filler so the goroutine ends inside the test.
		h.cancel()
		require.Error(t, waitAdmit(t, filler), "queued filler should be evicted on shutdown")
	})

	t.Run("rejected_no_endpoints_returns_503_no_endpoints", func(t *testing.T) {
		t.Parallel()
		// Same capacity overflow, but against an EMPTY candidate pool the processor classifies
		// the rejection as RejectedNoEndpoints (genuine unavailability, e.g. scale-from-zero).
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation:      1.0,
			candidates:      nil,
			bandMaxRequests: 1,
		})

		filler := admitAsync(ctx, h.ac, "filler-req")
		require.Eventually(t, func() bool { return h.reg.Stats().TotalLen == 1 },
			time.Second, time.Millisecond, "filler request should be queued before the overflow request")

		err := waitAdmit(t, admitAsync(ctx, h.ac, "overflow-req"))
		requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonNoEndpoints)

		h.cancel()
		require.Error(t, waitAdmit(t, filler), "queued filler should be evicted on shutdown")
	})

	t.Run("evicted_ttl_returns_503_ttl_expired", func(t *testing.T) {
		t.Parallel()
		// The admission adapter always defers to the controller-default TTL
		// (flowControlRequest.InitialEffectiveTTL() == 0), so a short DefaultRequestTTL plus a
		// saturated detector forces a TTL eviction of the queued request.
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation: 1.0,
			candidates: nonEmptyEndpoints(),
			requestTTL: 100 * time.Millisecond,
		})

		err := waitAdmit(t, admitAsync(ctx, h.ac, "ttl-req"))
		requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonTTLExpired)
	})

	t.Run("evicted_context_cancelled_returns_503_context_cancelled", func(t *testing.T) {
		t.Parallel()
		// Cancelling the caller's context while the request sits in the queue (saturated
		// detector, long TTL) simulates a client disconnect.
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation: 1.0,
			candidates: nonEmptyEndpoints(),
		})

		admitCtx, admitCancel := context.WithCancel(ctx)
		defer admitCancel()
		result := admitAsync(admitCtx, h.ac, "cancel-req")
		require.Eventually(t, func() bool { return h.reg.Stats().TotalLen == 1 },
			time.Second, time.Millisecond, "request should be queued before cancelling")
		admitCancel()

		err := waitAdmit(t, result)
		requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonContextCancelled)
	})

	t.Run("shutdown_returns_503_shutting_down", func(t *testing.T) {
		t.Parallel()
		// A request admitted after the controller's context is cancelled is rejected with
		// ErrFlowControllerNotRunning, which maps to 503 + the shutting-down reason.
		h := newRealFlowControlHarness(t, realFlowControlOpts{
			saturation: 0.0,
			candidates: nonEmptyEndpoints(),
		})
		h.cancel()

		err := waitAdmit(t, admitAsync(ctx, h.ac, "shutdown-req"))
		requireDropped(t, err, errcommon.ServiceUnavailable, errcommon.RequestDroppedReasonShuttingDown)
	})
}
