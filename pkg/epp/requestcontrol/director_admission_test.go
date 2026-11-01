/*
Copyright 2026 The llm-d Authors.

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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	fctypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

func TestHandleRequestRecordsAdmissionDecision(t *testing.T) {
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		Address: "192.168.1.100",
		Port:    "8000",
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
	}, nil, fwkdl.NewAttributes())
	scheduled := &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
	poolWithEndpoint := &mocks.MockEndpointCandidates{Candidates: []fwkdl.Endpoint{makeMockEndpoint("pod1", "192.168.1.100")}}

	tests := []struct {
		name         string
		setup        func(*Director)
		wantDecision string // empty: the attribute must be absent
		wantReason   string // empty: the attribute must be absent
		wantQueued   bool
		wantErr      bool
	}{
		{
			name:         "admitted without flow control",
			setup:        func(*Director) {},
			wantDecision: "admitted",
		},
		{
			name: "admitted by flow control after queueing",
			setup: func(d *Director) {
				d.admissionController = NewFlowControlAdmissionController(
					&mockFlowController{outcome: fctypes.QueueOutcomeDispatched, delay: 5 * time.Millisecond}, "pool", poolWithEndpoint)
			},
			wantDecision: "admitted",
			wantQueued:   true,
		},
		{
			name: "flow control rejects when the queue wait expires",
			setup: func(d *Director) {
				d.admissionController = NewFlowControlAdmissionController(
					&mockFlowController{outcome: fctypes.QueueOutcomeEvictedTTL, err: fmt.Errorf("%w: %w", fctypes.ErrEvicted, fctypes.ErrTTLExpired)},
					"pool", poolWithEndpoint)
			},
			wantDecision: "rejected",
			wantReason:   string(errcommon.RequestDroppedReasonTTLExpired),
			wantErr:      true,
		},
		{
			name: "rejection without a dropped reason falls back to the error code",
			setup: func(d *Director) {
				d.admissionController = &mockAdmissionController{admitErr: errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "system saturated, sheddable request dropped"}}
			},
			wantDecision: "rejected",
			wantReason:   errcommon.ResourceExhausted,
			wantErr:      true,
		},
		{
			name: "no endpoints after the admission controller admits",
			setup: func(d *Director) {
				d.endpointCandidates = &mocks.MockEndpointCandidates{}
			},
			wantDecision: "rejected",
			wantReason:   string(errcommon.RequestDroppedReasonNoEndpoints),
			wantErr:      true,
		},
		{
			name: "admitter plugin denies after flow control admits",
			setup: func(d *Director) {
				d.admissionController = NewFlowControlAdmissionController(
					&mockFlowController{outcome: fctypes.QueueOutcomeDispatched}, "pool", poolWithEndpoint)
				d.requestControlPlugins = *NewConfig().WithAdmissionPlugins(newMockAdmissionPlugin("deny", errors.New("latency objective cannot be met")))
			},
			wantDecision: "rejected",
			wantReason:   errcommon.Internal,
			wantQueued:   true,
			wantErr:      true,
		},
		{
			name: "scheduling failure after admission stays admitted",
			setup: func(d *Director) {
				d.scheduler = &mockScheduler{scheduleErr: errors.New("no endpoint fits")}
			},
			wantDecision: "admitted",
			wantErr:      true,
		},
		{
			name: "failure before admission records no decision",
			setup: func(d *Director) {
				d.requestControlPlugins = *NewConfig().WithRequestHeaderPlugins(
					&mockRequestHeaderPlugin{name: "fail", err: errcommon.Error{Code: errcommon.BadRequest, Msg: "bad header"}})
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			previous := otel.GetTracerProvider()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
			otel.SetTracerProvider(provider)
			t.Cleanup(func() { otel.SetTracerProvider(previous); _ = provider.Shutdown(context.Background()) })

			dir, ctx := newSinglePodDirector(t, scheduled)
			tc.setup(dir)

			body := []byte(`{"model":"m","prompt":"p"}`)
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					ID: "admission-test",
					Headers: map[string]string{
						reqcommon.RequestIDHeaderKey: "admission-test",
						":path":                      "/v1/completions",
					},
					RawBody: body,
				},
				Parser: openai.NewOpenAIParser(),
			}
			parsed, err := reqCtx.Parser.ParseRequest(ctx, body, reqCtx.Request.Headers)
			require.NoError(t, err)

			_, err = dir.HandleRequest(ctx, reqCtx, parsed.Body)
			assert.Equal(t, tc.wantErr, err != nil, "HandleRequest error: %v", err)

			var attrs *attribute.Set
			for _, span := range recorder.Ended() {
				if span.Name() == "request_orchestration" {
					set := attribute.NewSet(span.Attributes()...)
					attrs = &set
				}
			}
			require.NotNil(t, attrs, "request_orchestration span must end")

			decision, ok := attrs.Value(semconv.LLMDEPPAdmissionDecisionKey)
			if tc.wantDecision == "" {
				assert.False(t, ok, "decision recorded before admission: %q", decision.AsString())
			} else {
				assert.Equal(t, tc.wantDecision, decision.AsString())
			}

			reason, ok := attrs.Value(semconv.LLMDEPPAdmissionReasonKey)
			if tc.wantReason == "" {
				assert.False(t, ok, "unexpected reason %q", reason.AsString())
			} else {
				assert.Equal(t, tc.wantReason, reason.AsString())
			}

			queued, ok := attrs.Value(semconv.LLMDEPPAdmissionQueueDurationMsKey)
			assert.Equal(t, tc.wantQueued, ok, "queue duration presence")
			if ok {
				assert.InDelta(t, reqCtx.FlowControlQueueDuration.Seconds()*1000, queued.AsFloat64(), 1e-9, "queue duration is in fractional milliseconds")
			}
		})
	}
}
