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

package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// openMetricsAccept is the Accept header Prometheus sends with its default
	// scrape_protocols, which lists OpenMetrics 1.0.0 first.
	openMetricsAccept = "application/openmetrics-text;version=1.0.0,text/plain;version=0.0.4;q=0.5"
	// classicAccept is what a scraper that does not understand OpenMetrics sends.
	classicAccept = "text/plain;version=0.0.4"

	testTraceID = "4bf92f3577b34da6a3ce929d0e0e4736"
	testSpanID  = "0102030405060708"
)

// exemplarLabels returns the label set of the first exemplar in an OpenMetrics
// payload. client_golang builds an exemplar by ranging over a Go map, so the
// order of the labels is not stable between runs and must not be asserted on.
func exemplarLabels(body string) (string, bool) {
	for _, line := range strings.Split(body, "\n") {
		open := strings.Index(line, "# {")
		if open < 0 {
			continue
		}
		rest := line[open+len("# {"):]
		close := strings.Index(rest, "}")
		if close < 0 {
			continue
		}
		return rest[:close], true
	}
	return "", false
}

// sampledContext returns a context carrying a sampled span, which is the only
// case in which a latency observation attaches an exemplar.
func sampledContext(t *testing.T) context.Context {
	t.Helper()

	traceID, err := trace.TraceIDFromHex(testTraceID)
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex(testSpanID)
	require.NoError(t, err)

	return trace.ContextWithSpanContext(context.Background(), trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
	}))
}

// scrapeMetrics serves the metrics endpoint the runner builds in production and
// returns the Content-Type and body a scraper sending accept would receive.
func scrapeMetrics(t *testing.T, accept string) (string, string) {
	t.Helper()

	// Auth disabled is the case that used to silently skip the filter entirely,
	// so it is the one worth exercising here.
	provider := openMetricsFilterProvider(false)
	filter, err := provider(nil, nil)
	require.NoError(t, err)

	// The handler passed in is the one controller-runtime built; the filter
	// substitutes its own, so nil is enough here.
	handler, err := filter(testr.New(t), nil)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set("Accept", accept)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	return rec.Header().Get("Content-Type"), rec.Body.String()
}

// TestMetricsEndpointServesExemplarsOnTheWire covers the step the unit tests in
// pkg/epp/metrics cannot: an exemplar recorded onto a collector is only useful
// if it also survives serialisation. Exemplars exist solely in the OpenMetrics
// exposition format, so a handler built without it drops them at encode time
// without erroring.
func TestMetricsEndpointServesExemplarsOnTheWire(t *testing.T) {
	eppmetrics.Register()

	received := time.Now()
	require.True(t, eppmetrics.RecordRequestLatencies(
		sampledContext(t), "model", "target-model", "fairness", "priority",
		received, received.Add(420*time.Millisecond),
	))

	contentType, body := scrapeMetrics(t, openMetricsAccept)

	require.True(t, strings.HasPrefix(contentType, "application/openmetrics-text"),
		"a scraper asking for OpenMetrics must be served OpenMetrics, got %q", contentType)

	labels, ok := exemplarLabels(body)
	require.True(t, ok, "the observation must carry an exemplar on the wire")
	require.Contains(t, labels, `trace_id="`+testTraceID+`"`,
		"the trace ID must reach the wire")
	require.Contains(t, labels, `span_id="`+testSpanID+`"`,
		"the span ID must reach the wire, so a backend can open the span that observed the latency")
}

// TestMetricsEndpointKeepsClassicFormatForClassicScrapers guards the other half:
// the handler negotiates per request, so enabling OpenMetrics must not change
// what a scraper that never asks for it receives.
func TestMetricsEndpointKeepsClassicFormatForClassicScrapers(t *testing.T) {
	eppmetrics.Register()

	contentType, body := scrapeMetrics(t, classicAccept)

	require.True(t, strings.HasPrefix(contentType, "text/plain"),
		"a classic scraper must keep receiving the classic format, got %q", contentType)
	require.NotContains(t, body, "# {trace_id=",
		"the classic exposition format has no representation for exemplars")
}
