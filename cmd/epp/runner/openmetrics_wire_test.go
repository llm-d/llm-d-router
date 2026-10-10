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
	"io"
	"net/http"
	"net/http/pprof"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/client-go/rest"
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

// passThroughFilter stands in for the auth filter, which needs an API server to
// run TokenReviews. OpenMetrics must not depend on whether a filter is set.
func passThroughFilter(*rest.Config, *http.Client) (metricsserver.Filter, error) {
	return func(_ logr.Logger, next http.Handler) (http.Handler, error) { return next, nil }, nil
}

// startMetricsServer runs a metrics server built from opts on a loopback port
// and returns its address.
func startMetricsServer(t *testing.T, opts metricsserver.Options) string {
	t.Helper()

	opts.BindAddress = "127.0.0.1:0"
	// The auth filter needs a rest config to build its API clients. Requests
	// without a token are rejected before any API call, so the host is never dialed.
	srv, err := metricsserver.NewServer(opts, &rest.Config{Host: "https://127.0.0.1:1"}, &http.Client{})
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

	return waitForBindAddr(t, srv)
}

// scrapeMetrics serves the metrics endpoint the runner builds in production and
// returns the Content-Type and body a scraper sending accept would receive. A
// non-nil filterProvider builds the options with auth on, then stands in for
// the auth filter, which needs an API server to run TokenReviews.
func scrapeMetrics(t *testing.T, filterProvider func(*rest.Config, *http.Client) (metricsserver.Filter, error), accept string) (string, string) {
	t.Helper()

	opts := newMetricsServerOptions(0, filterProvider != nil)
	opts.FilterProvider = filterProvider
	return get(t, "http://"+startMetricsServer(t, opts)+"/metrics", accept)
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

	for name, filterProvider := range map[string]func(*rest.Config, *http.Client) (metricsserver.Filter, error){
		"auth off": nil,
		"auth on":  passThroughFilter,
	} {
		t.Run(name, func(t *testing.T) {
			contentType, body := scrapeMetrics(t, filterProvider, openMetricsAccept)

			require.True(t, strings.HasPrefix(contentType, "application/openmetrics-text"),
				"a scraper asking for OpenMetrics must be served OpenMetrics, got %q", contentType)

			labels, ok := exemplarLabels(body, wireTestModel)
			require.True(t, ok, "the observation must carry an exemplar on the wire")
			require.Contains(t, labels, `trace_id="`+span.SpanContext().TraceID().String()+`"`,
				"the trace ID must reach the wire")
			require.Contains(t, labels, `span_id="`+span.SpanContext().SpanID().String()+`"`,
				"the span ID must reach the wire, so a backend can open the span that observed the latency")
		})
	}
}

// TestMetricsEndpointKeepsClassicFormatForClassicScrapers checks that a scraper
// not asking for OpenMetrics still gets the classic format.
func TestMetricsEndpointKeepsClassicFormatForClassicScrapers(t *testing.T) {
	eppmetrics.Register()

	contentType, body := scrapeMetrics(t, nil, classicAccept)

	require.True(t, strings.HasPrefix(contentType, "text/plain"),
		"a classic scraper must keep receiving the classic format, got %q", contentType)
	require.NotContains(t, body, "# {trace_id=",
		"the classic exposition format has no representation for exemplars")
}

// TestMetricsServerLeavesExtraHandlersAlone guards against swapping the /metrics
// handler inside a filter again: controller-runtime filters every handler, so
// such a swap made pprof serve the metrics page.
func TestMetricsServerLeavesExtraHandlersAlone(t *testing.T) {
	eppmetrics.Register()

	opts := newMetricsServerOptions(0, false)
	opts.ExtraHandlers = map[string]http.Handler{
		"/debug/pprof/cmdline": http.HandlerFunc(pprof.Cmdline),
	}
	addr := startMetricsServer(t, opts)

	_, cmdline := get(t, "http://"+addr+"/debug/pprof/cmdline", classicAccept)
	require.NotContains(t, cmdline, "# HELP",
		"the pprof handler must serve its own output, not the metrics page")

	_, metrics := get(t, "http://"+addr+"/metrics", openMetricsAccept)
	require.Contains(t, metrics, "# HELP", "the metrics endpoint must still serve metrics")
}

// TestMetricsEndpointRejectsAnonymousScrapesWithAuth checks that enabling
// metrics auth actually installs the auth filter.
func TestMetricsEndpointRejectsAnonymousScrapesWithAuth(t *testing.T) {
	eppmetrics.Register()

	addr := startMetricsServer(t, newMetricsServerOptions(0, true))

	resp, err := http.Get("http://" + addr + "/metrics")
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.Equal(t, http.StatusUnauthorized, resp.StatusCode,
		"with metrics auth on, a scrape without a token must be rejected")
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

// get returns the Content-Type and body served at url for a scraper sending accept.
func get(t *testing.T, url, accept string) (string, string) {
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
	return resp.Header.Get("Content-Type"), string(body)
}
