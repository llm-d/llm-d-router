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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	fctypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

// --- Mocks ---

type mockSaturationDetector struct {
	flowcontrol.SaturationDetector
	SaturationFunc func(ctx context.Context, candidatePods []fwkdl.Endpoint) float64
}

func (m *mockSaturationDetector) Saturation(ctx context.Context, candidatePods []fwkdl.Endpoint) float64 {
	if m.SaturationFunc != nil {
		return m.SaturationFunc(ctx, candidatePods)
	}
	return 0.0
}

type mockFlowController struct {
	outcome fctypes.QueueOutcome
	err     error
	called  bool
}

func (m *mockFlowController) EnqueueAndWait(
	_ context.Context,
	_ flowcontrol.FlowControlRequest,
) (fctypes.QueueOutcome, error) {
	m.called = true
	return m.outcome, m.err
}

// --- Legacy Controller Tests ---

func TestLegacyAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &fwksched.InferenceRequest{RequestID: "test-req"},
		Request: &handlers.Request{
			Metadata: map[string]any{},
		},
	}

	mockPods := []fwkdl.Endpoint{fwkdl.NewEndpoint(nil, nil)}

	testCases := []struct {
		name            string
		priority        int
		isSaturated     bool
		locatorPods     []fwkdl.Endpoint
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
	}{
		{
			name:        "non_sheddable_saturated_admit",
			priority:    0,
			isSaturated: true,
			locatorPods: mockPods,
			expectErr:   false,
		},
		{
			name:        "sheddable_not_saturated_admit",
			priority:    -1,
			isSaturated: false,
			locatorPods: mockPods,
			expectErr:   false,
		},
		{
			name:            "sheddable_saturated_reject",
			priority:        -1,
			isSaturated:     true,
			locatorPods:     mockPods,
			expectErr:       true,
			expectErrCode:   errcommon.ResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
		},
		{
			name:            "sheddable_no_pods_reject",
			priority:        -1,
			isSaturated:     true,
			locatorPods:     []fwkdl.Endpoint{},
			expectErr:       true,
			expectErrCode:   errcommon.ResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			mockDetector := &mockSaturationDetector{
				SaturationFunc: func(_ context.Context, _ []fwkdl.Endpoint) float64 {
					if tc.isSaturated {
						return 1.0
					}
					return 0.0
				},
			}
			endpointCandidates := &mocks.MockEndpointCandidates{Candidates: tc.locatorPods}
			ac := NewLegacyAdmissionController(mockDetector, endpointCandidates)

			err := ac.Admit(ctx, reqCtx, tc.priority)

			if !tc.expectErr {
				assert.NoError(t, err, "Admit() should not have returned an error for scenario: %s", tc.name)
			} else {
				require.Error(t, err, "Admit() should have returned an error for scenario: %s", tc.name)
				var e errcommon.Error
				if assert.ErrorAs(t, err, &e, "error should be of type errcommon.Error") {
					assert.Equal(t, tc.expectErrCode, e.Code, "incorrect error code for scenario: %s", tc.name)
					assert.Contains(t, e.Msg, tc.expectErrSubstr, "incorrect error message substring for scenario: %s", tc.name)
				}
			}
		})
	}
}

// --- Flow Control Controller Tests ---

func TestFlowControlRequestAdapter(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name            string
		requestID       string
		fairnessID      string
		priority        int
		requestByteSize uint64
		expectFlowKey   flowcontrol.FlowKey
	}{
		{
			name:            "simple",
			requestID:       "req-1",
			fairnessID:      "flow-1",
			priority:        10,
			requestByteSize: 1024,
			expectFlowKey:   flowcontrol.FlowKey{ID: "flow-1", Priority: 10},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fcReq := &flowControlRequest{
				fairnessID:       tc.fairnessID,
				priority:         tc.priority,
				requestByteSize:  tc.requestByteSize,
				inferenceRequest: &fwksched.InferenceRequest{RequestID: tc.requestID},
			}

			assert.Equal(t, tc.requestID, fcReq.ID(), "ID() mismatch")
			assert.Equal(t, tc.requestByteSize, fcReq.ByteSize(), "ByteSize() mismatch")
			assert.Equal(t, tc.expectFlowKey, fcReq.FlowKey(), "FlowKey() mismatch")
			assert.Zero(t, fcReq.InitialEffectiveTTL(), "InitialEffectiveTTL() should be zero")
		})
	}
}

func TestFlowControlAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &fwksched.InferenceRequest{RequestID: "test-req"},
		Request: &handlers.Request{
			Metadata: map[string]any{},
		},
	}

	testCases := []struct {
		name            string
		priority        int
		fcOutcome       fctypes.QueueOutcome
		fcErr           error
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
		expectHeaders   map[string]string
	}{
		{
			name:      "sheddable_dispatched",
			priority:  -1,
			fcOutcome: fctypes.QueueOutcomeDispatched,
			expectErr: false,
		},
		{
			name:      "non_sheddable_dispatched",
			priority:  0,
			fcOutcome: fctypes.QueueOutcomeDispatched,
			expectErr: false,
		},
		{
			name:            "fc_reject_capacity",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeRejectedCapacity,
			expectErr:       true,
			expectErrCode:   errcommon.ResourceExhausted,
			expectErrSubstr: "request rejected by flow control",
			expectHeaders:   map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSaturated)},
		},
		{
			name:            "fc_reject_no_endpoints",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeRejectedNoEndpoints,
			fcErr:           fctypes.ErrNoEndpoints,
			expectErr:       true,
			expectErrCode:   errcommon.ServiceUnavailable,
			expectErrSubstr: "no endpoints available",
			expectHeaders:   map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonNoEndpoints)},
		},
		{
			name:            "fc_evict_ttl",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedTTL,
			fcErr:           errors.New("timeout"),
			expectErr:       true,
			expectErrCode:   errcommon.ServiceUnavailable,
			expectErrSubstr: "request timed out in queue: timeout",
			expectHeaders:   map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonTTLExpired)},
		},
		{
			name:            "fc_evict_context_cancelled",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedContextCancelled,
			expectErr:       true,
			expectErrCode:   errcommon.ServiceUnavailable,
			expectErrSubstr: "client disconnected",
			expectHeaders:   map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonContextCancelled)},
		},
		{
			name:            "fc_reject_other",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeRejectedOther,
			expectErr:       true,
			expectErrCode:   errcommon.Internal,
			expectErrSubstr: "internal flow control error",
		},
		{
			name:            "fc_evict_other",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedOther,
			fcErr:           errors.New("internal error"),
			expectErr:       true,
			expectErrCode:   errcommon.Internal,
			expectErrSubstr: "internal flow control error: internal error",
		},
		{
			name:            "fc_unhandled_outcome",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeNotYetFinalized,
			expectErr:       true,
			expectErrCode:   errcommon.Internal,
			expectErrSubstr: "unhandled flow control outcome",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fc := &mockFlowController{outcome: tc.fcOutcome, err: tc.fcErr}
			ac := NewFlowControlAdmissionController(fc, "pool")

			err := ac.Admit(ctx, reqCtx, tc.priority)

			assert.True(t, fc.called, "FlowController should have been called for scenario: %s", tc.name)

			if !tc.expectErr {
				assert.NoError(t, err, "Admit() returned an unexpected error for scenario: %s", tc.name)
			} else {
				require.Error(t, err, "Admit() should have returned an error for scenario: %s", tc.name)
				var e errcommon.Error
				if assert.ErrorAs(t, err, &e, "error should be of type errcommon.Error") {
					assert.Equal(t, tc.expectErrCode, e.Code, "incorrect error code for scenario: %s", tc.name)
					assert.Contains(t, e.Msg, tc.expectErrSubstr, "incorrect error message substring for scenario: %s", tc.name)
					assert.Equal(t, tc.expectHeaders, e.Headers, "incorrect headers for scenario: %s", tc.name)
				}
			}
		})
	}
}

func TestTranslateFlowControlOutcome(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		outcome    fctypes.QueueOutcome
		err        error
		wantCode   string
		wantReason string
		wantNil    bool
	}{
		{
			name:    "dispatched returns nil",
			outcome: fctypes.QueueOutcomeDispatched,
			err:     nil,
			wantNil: true,
		},
		{
			name:       "capacity rejection returns 429",
			outcome:    fctypes.QueueOutcomeRejectedCapacity,
			err:        fctypes.ErrQueueAtCapacity,
			wantCode:   errcommon.ResourceExhausted,
			wantReason: string(errcommon.RequestDroppedReasonSaturated),
		},
		{
			name:       "TTL expiry returns 503",
			outcome:    fctypes.QueueOutcomeEvictedTTL,
			err:        fctypes.ErrTTLExpired,
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonTTLExpired),
		},
		{
			name:       "context cancellation returns 503",
			outcome:    fctypes.QueueOutcomeEvictedContextCancelled,
			err:        fctypes.ErrContextCancelled,
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonContextCancelled),
		},
		{
			name:       "shutdown eviction returns 503",
			outcome:    fctypes.QueueOutcomeEvictedOther,
			err:        fctypes.ErrFlowControllerNotRunning,
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonShuttingDown),
		},
		{
			name:       "shutdown rejection returns 503",
			outcome:    fctypes.QueueOutcomeRejectedOther,
			err:        fctypes.ErrFlowControllerNotRunning,
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonShuttingDown),
		},
		{
			name:       "pre-admission TTL rejection maps like TTL eviction",
			outcome:    fctypes.QueueOutcomeRejectedOther,
			err:        fmt.Errorf("%w: %w", fctypes.ErrRejected, fctypes.ErrTTLExpired),
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonTTLExpired),
		},
		{
			name:       "pre-admission cancellation rejection maps like cancellation eviction",
			outcome:    fctypes.QueueOutcomeRejectedOther,
			err:        fmt.Errorf("%w: %w", fctypes.ErrRejected, fctypes.ErrContextCancelled),
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonContextCancelled),
		},
		{
			name:       "other TTL eviction maps like TTL eviction",
			outcome:    fctypes.QueueOutcomeEvictedOther,
			err:        fmt.Errorf("%w: %w", fctypes.ErrEvicted, fctypes.ErrTTLExpired),
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonTTLExpired),
		},
		{
			name:       "other cancellation eviction maps like cancellation eviction",
			outcome:    fctypes.QueueOutcomeEvictedOther,
			err:        fmt.Errorf("%w: %w", fctypes.ErrEvicted, fctypes.ErrContextCancelled),
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonContextCancelled),
		},
		{
			name:       "shutdown takes precedence over TTL",
			outcome:    fctypes.QueueOutcomeRejectedOther,
			err:        fmt.Errorf("%w: %w", fctypes.ErrFlowControllerNotRunning, fctypes.ErrTTLExpired),
			wantCode:   errcommon.ServiceUnavailable,
			wantReason: string(errcommon.RequestDroppedReasonShuttingDown),
		},
		{
			name:     "internal error returns 500",
			outcome:  fctypes.QueueOutcomeRejectedOther,
			err:      errors.New("unexpected failure"),
			wantCode: errcommon.Internal,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := translateFlowControlOutcome(tt.outcome, tt.err)
			if tt.wantNil {
				require.NoError(t, result)
				return
			}
			require.Error(t, result)
			var e errcommon.Error
			require.ErrorAs(t, result, &e)
			assert.Equal(t, tt.wantCode, e.Code)
			if tt.wantReason != "" {
				assert.Equal(t, tt.wantReason, e.Headers[errcommon.RequestDroppedReasonHeaderKey],
					"drop reason header should match")
			}
		})
	}
}
