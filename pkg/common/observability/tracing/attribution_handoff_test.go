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

package tracing

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// The sidecar wraps its otelhttp handler with WithAttributionHandoff, so the server
// span and every child span carry the identity EPP resolved, and the handoff headers
// never reach the model server.
func TestWithAttributionHandoff(t *testing.T) {
	for _, tc := range []struct {
		name               string
		headers            map[string]string
		wantID, wantSource string
	}{
		{
			name:    "agent identity resolved by EPP",
			headers: map[string]string{FairnessIDHandoffHeader: "agent-7", FairnessSourceHandoffHeader: AttributionSourceAgentIdentity},
			wantID:  "agent-7", wantSource: AttributionSourceAgentIdentity,
		},
		{
			name:    "header identity resolved by EPP",
			headers: map[string]string{FairnessIDHandoffHeader: "team-a", FairnessSourceHandoffHeader: AttributionSourceHeader},
			wantID:  "team-a", wantSource: AttributionSourceHeader,
		},
		{
			name:   "no handoff",
			wantID: reqcommon.DefaultFairnessID, wantSource: AttributionSourceDefault,
		},
		{
			name:    "id without source",
			headers: map[string]string{FairnessIDHandoffHeader: "team-a"},
			wantID:  reqcommon.DefaultFairnessID, wantSource: AttributionSourceDefault,
		},
		{
			name:    "source without id",
			headers: map[string]string{FairnessSourceHandoffHeader: AttributionSourceAgentIdentity},
			wantID:  reqcommon.DefaultFairnessID, wantSource: AttributionSourceDefault,
		},
		{
			name:    "unknown source",
			headers: map[string]string{FairnessIDHandoffHeader: "team-a", FairnessSourceHandoffHeader: "authenticated"},
			wantID:  reqcommon.DefaultFairnessID, wantSource: AttributionSourceDefault,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			provider := sdktrace.NewTracerProvider(
				sdktrace.WithSpanProcessor(NewRequestAttributionProcessor()),
				sdktrace.WithSpanProcessor(recorder),
			)
			t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

			var leaked []string
			inner := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				_, child := provider.Tracer("test").Start(r.Context(), "decode")
				child.End()
				for _, h := range []string{FairnessIDHandoffHeader, FairnessSourceHandoffHeader} {
					if r.Header.Get(h) != "" {
						leaked = append(leaked, h)
					}
				}
			})
			handler := WithAttributionHandoff(otelhttp.NewHandler(inner, "proxy", otelhttp.WithTracerProvider(provider)))

			req := httptest.NewRequest(http.MethodPost, "/v1/completions", nil)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			handler.ServeHTTP(httptest.NewRecorder(), req)

			require.Empty(t, leaked, "handoff headers must not reach the model server")
			spans := recorder.Ended()
			require.Len(t, spans, 2)
			for _, span := range spans {
				attrs := attribute.NewSet(span.Attributes()...)
				id, _ := attrs.Value(semconv.LLMDEPPFairnessIDKey)
				source, _ := attrs.Value(semconv.LLMDEPPFairnessSourceKey)
				require.Equal(t, tc.wantID, id.AsString(), span.Name())
				require.Equal(t, tc.wantSource, source.AsString(), span.Name())
			}
		})
	}
}
