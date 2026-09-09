/*
Copyright 2025 The llm-d Authors.

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

package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// useTracerProviderForTest installs tp for the duration of the test.
func useTracerProviderForTest(t *testing.T, tp trace.TracerProvider) {
	t.Helper()

	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
}

// runPrefillHandler drives disaggregatedPrefillHandler with the prefill header
// set, appending every logged line to logged and returning the logger the
// downstream connector received through the request context.
func runPrefillHandler(t *testing.T, logged *[]string) (downstream logr.Logger) {
	t.Helper()

	capture := funcr.New(func(prefix, args string) {
		*logged = append(*logged, prefix+" "+args)
	}, funcr.Options{Verbosity: 5})

	s := NewProxy(Config{Port: "8000"})
	s.logger = capture
	s.allowlistValidator = &AllowlistValidator{}
	s.dataParallelProxies = make(map[string]http.Handler)
	s.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ APIType) {
		downstream = log.FromContext(r.Context())
	}

	req := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, http.NoBody)
	req.Header.Set(routing.PrefillEndpointHeader, "prefill-pod:8000")
	s.disaggregatedPrefillHandler(APITypeChatCompletions)(httptest.NewRecorder(), req)

	return downstream
}

// The handler's own log lines carry the forward_request trace, and the request
// context handed to connectors carries the same correlated logger.
func TestPrefillHandlerCorrelatesLogsWithTrace(t *testing.T) {
	useTracerProviderForTest(t, sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))

	var logged []string
	downstream := runPrefillHandler(t, &logged)

	var handlerLine string
	for _, rec := range logged {
		if strings.Contains(rec, "using P/D protocol") {
			handlerLine = rec
		}
	}
	if handlerLine == "" {
		t.Fatalf("handler log line not found in %v", logged)
	}
	if !strings.Contains(handlerLine, tracing.LogKeyTraceID) || !strings.Contains(handlerLine, tracing.LogKeySpanID) {
		t.Errorf("handler line missing correlation fields: %q", handlerLine)
	}
	if got := strings.Count(handlerLine, tracing.LogKeyTraceID); got != 1 {
		t.Errorf("trace_id appears %d times, want 1: %q", got, handlerLine)
	}

	// A connector that adopts the request context gets the same trace.
	before := len(logged)
	downstream.Info("connector line")
	if len(logged) != before+1 {
		t.Fatalf("expected the downstream logger to emit one line, got %d", len(logged)-before)
	}
	if !strings.Contains(logged[before], tracing.LogKeyTraceID) {
		t.Errorf("downstream logger is not correlated: %q", logged[before])
	}
}

// With tracing off the span context is invalid, so nothing is added.
func TestPrefillHandlerOmitsCorrelationWhenTracingDisabled(t *testing.T) {
	useTracerProviderForTest(t, noop.NewTracerProvider())

	var logged []string
	runPrefillHandler(t, &logged)

	for _, rec := range logged {
		if strings.Contains(rec, tracing.LogKeyTraceID) || strings.Contains(rec, tracing.LogKeySpanID) {
			t.Errorf("unexpected correlation fields with tracing disabled: %q", rec)
		}
	}
}
