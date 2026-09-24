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

package runner

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/pprof"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// openMetricsAccept is the Accept header Prometheus sends with its default
	// scrape_protocols, which lists OpenMetrics 1.0.0 first.
	openMetricsAccept = "application/openmetrics-text;version=1.0.0,text/plain;version=0.0.4;q=0.5"
	// classicAccept is what a scraper that does not understand OpenMetrics sends.
	classicAccept = "text/plain;version=0.0.4"

	// wireTestModel labels the observation this file records, so its exemplar can
	// be told apart from any other test's on the shared registry.
	wireTestModel = "openmetrics-wire-test"
)

// exemplarLabels returns the label set of the first exemplar in an OpenMetrics
// payload on a series carrying modelName. client_golang builds an exemplar by
// ranging over a Go map, so the order of the labels is not stable between runs
// and must not be asserted on.
func exemplarLabels(body, modelName string) (string, bool) {
	for _, line := range strings.Split(body, "\n") {
		if !strings.Contains(line, `model_name="`+modelName+`"`) {
			continue
		}
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

// recordingSpan returns a context with a real SDK span, like the one the EPP
// starts per request.
func recordingSpan(t *testing.T) (context.Context, trace.Span) {
	t.Helper()

	tp := sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample()))
	t.Cleanup(func() { require.NoError(t, tp.Shutdown(context.Background())) })

	return tp.Tracer("openmetrics-wire-test").Start(context.Background(), "request")
}

// scrapeMetrics serves the metrics endpoint the runner builds in production and
// returns the Content-Type and body a scraper sending accept would receive.
func scrapeMetrics(t *testing.T, accept string) (string, string) {
	t.Helper()

	// Auth off, since that used to skip the filter entirely.
	provider := openMetricsFilterProvider(false)
	filter, err := provider(nil, nil)
	require.NoError(t, err)

	// Stand-in for the handler controller-runtime builds. The filter should
	// replace it for /metrics, so seeing this text means the swap didn't happen.
	handler, err := filter(testr.New(t), http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "handler controller-runtime built")
	}))
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set("Accept", accept)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	return rec.Header().Get("Content-Type"), rec.Body.String()
}

// TestMetricsEndpointServesExemplarsOnTheWire checks the exemplar actually makes
// it into the scrape. A handler without OpenMetrics drops it silently.
func TestMetricsEndpointServesExemplarsOnTheWire(t *testing.T) {
	eppmetrics.Register()

	ctx, span := recordingSpan(t)
	defer span.End()

	received := time.Now()
	require.True(t, eppmetrics.RecordRequestLatencies(
		ctx, wireTestModel, "target-model", "fairness", "priority",
		received, received.Add(420*time.Millisecond),
	))

	contentType, body := scrapeMetrics(t, openMetricsAccept)

	require.True(t, strings.HasPrefix(contentType, "application/openmetrics-text"),
		"a scraper asking for OpenMetrics must be served OpenMetrics, got %q", contentType)

	labels, ok := exemplarLabels(body, wireTestModel)
	require.True(t, ok, "the observation must carry an exemplar on the wire")
	require.Contains(t, labels, `trace_id="`+span.SpanContext().TraceID().String()+`"`,
		"the trace ID must reach the wire")
	require.Contains(t, labels, `span_id="`+span.SpanContext().SpanID().String()+`"`,
		"the span ID must reach the wire, so a backend can open the span that observed the latency")
}

// TestMetricsEndpointKeepsClassicFormatForClassicScrapers checks that a scraper
// not asking for OpenMetrics still gets the classic format.
func TestMetricsEndpointKeepsClassicFormatForClassicScrapers(t *testing.T) {
	eppmetrics.Register()

	contentType, body := scrapeMetrics(t, classicAccept)

	require.True(t, strings.HasPrefix(contentType, "text/plain"),
		"a classic scraper must keep receiving the classic format, got %q", contentType)
	require.NotContains(t, body, "# {trace_id=",
		"the classic exposition format has no representation for exemplars")
}

// TestMetricsServerLeavesExtraHandlersAlone checks extra handlers still serve
// their own content. controller-runtime runs the filter over every handler, not
// just /metrics, so swapping unconditionally made pprof serve the metrics page.
func TestMetricsServerLeavesExtraHandlersAlone(t *testing.T) {
	eppmetrics.Register()

	srv, err := metricsserver.NewServer(metricsserver.Options{
		BindAddress:    "127.0.0.1:0",
		FilterProvider: openMetricsFilterProvider(false),
		ExtraHandlers: map[string]http.Handler{
			"/debug/pprof/cmdline": http.HandlerFunc(pprof.Cmdline),
		},
	}, nil, nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan error, 1)
	go func() {
		// Start blocks until the context is cancelled.
		started <- srv.Start(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-started)
	})

	addr := waitForBindAddr(t, srv)

	cmdline := get(t, "http://"+addr+"/debug/pprof/cmdline", classicAccept)
	require.NotContains(t, cmdline, "# HELP",
		"the pprof handler must serve its own output, not the metrics page")

	metrics := get(t, "http://"+addr+"/metrics", openMetricsAccept)
	require.Contains(t, metrics, "# HELP", "the metrics endpoint must still serve metrics")
}

// waitForBindAddr returns the address the server listened on. It is only set
// once Start has created its listener.
func waitForBindAddr(t *testing.T, srv metricsserver.Server) string {
	t.Helper()

	withAddr, ok := srv.(interface{ GetBindAddr() string })
	require.True(t, ok, "the metrics server must expose its bind address")

	require.Eventually(t, func() bool {
		return withAddr.GetBindAddr() != ""
	}, 10*time.Second, 10*time.Millisecond, "metrics server did not start")

	return withAddr.GetBindAddr()
}

// get returns the body served at url for a scraper sending accept.
func get(t *testing.T, url, accept string) string {
	t.Helper()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	require.NoError(t, err)
	req.Header.Set("Accept", accept)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { require.NoError(t, resp.Body.Close()) }()

	require.Equal(t, http.StatusOK, resp.StatusCode)
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return string(body)
}
