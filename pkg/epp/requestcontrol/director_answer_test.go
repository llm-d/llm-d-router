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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/reserveendpoint"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

// TestHandleRequest_AnswerSpanStatus pins the request orchestration span and
// the request context of a reservation: an answer ends the span without an
// error status and sets the answer headers, and a rejected reservation ends
// the span with an error status.
func TestHandleRequest_AnswerSpanStatus(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder), sdktrace.WithSampler(sdktrace.AlwaysSample())))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	md := &fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Namespace: "default", Name: "pod1"},
		Address: "10.0.3.7",
		Port:    "8000",
	}
	pod := fwkdl.NewEndpoint(md, nil)
	scheduled := singleEndpointResult("prefill", md)

	tests := []struct {
		name        string
		plugins     []fwkrc.PreRequest
		wantErrCode string
		wantStatus  codes.Code
	}{
		{name: "answer ends the span without an error status", plugins: []fwkrc.PreRequest{reserveendpoint.New()}, wantStatus: codes.Unset},
		{name: "rejection ends the span with an error status", wantErrCode: errcommon.Internal, wantStatus: codes.Error},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := logutil.NewTestLoggerIntoContext(context.Background())
			director := &Director{
				datastore:             &mockDatastore{},
				scheduler:             &mockScheduler{scheduleResults: scheduled},
				admissionController:   &mockAdmissionController{},
				endpointCandidates:    &mockEndpointCandidates{result: []fwkdl.Endpoint{pod}},
				requestControlPlugins: *NewConfig().WithPreRequestPlugins(tt.plugins...),
			}
			headers := map[string]string{
				reqcommon.RequestIDHeaderKey: "req-" + tt.name,
				routing.PreferHeader:         routing.PreferReserveEndpoint,
				":path":                      "/v1/completions",
			}
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{Headers: headers, RawBody: []byte(`{"model":"m","prompt":"p"}`)},
				Parser:  openai.NewOpenAIParser(),
			}
			parsed, err := reqCtx.Parser.ParseRequest(ctx, reqCtx.Request.RawBody, headers)
			require.NoError(t, err)
			before := len(recorder.Ended())

			got, err := director.HandleRequest(ctx, reqCtx, parsed.Body)

			if tt.wantErrCode == "" {
				require.NoError(t, err)
				require.NotNil(t, got.Answer)
				assert.Equal(t, "10.0.3.7:8000", got.Answer.Headers[routing.ReservedEndpointHeader])
			} else {
				assert.Equal(t, tt.wantErrCode, errcommon.CanonicalCode(err))
				assert.Nil(t, got.Answer)
			}
			ended := recorder.Ended()[before:]
			require.Len(t, ended, 1)
			assert.Equal(t, "request_orchestration", ended[0].Name())
			assert.Equal(t, tt.wantStatus, ended[0].Status().Code)
		})
	}
}
