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

package proxy

import (
	"context"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

const (
	// openMetricsAccept is what Prometheus sends by default.
	openMetricsAccept = "application/openmetrics-text;version=1.0.0,text/plain;version=0.0.4;q=0.5"
	// classicAccept is what a scraper that does not understand OpenMetrics sends.
	classicAccept = "text/plain;version=0.0.4"
)

// scrapeSidecarMetrics serves the metrics endpoint serveMetrics builds in
// production and returns the Content-Type and body a scraper sending accept
// would receive.
func scrapeSidecarMetrics(t *testing.T, accept string) (string, string) {
	t.Helper()

	ln := reserveListener(t)
	s := &Server{logger: logr.Discard(), MetricsListener: ln}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- s.serveMetrics(ctx) }()
	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-errCh)
	})

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet,
		"http://"+ln.Addr().String()+"/metrics", nil)
	require.NoError(t, err)
	req.Header.Set("Accept", accept)

	var contentType, body string
	require.Eventually(t, func() bool {
		resp, err := http.DefaultClient.Do(req.Clone(context.Background()))
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return false
		}
		payload, err := io.ReadAll(resp.Body)
		if err != nil {
			return false
		}
		contentType = resp.Header.Get("Content-Type")
		body = string(payload)
		return true
	}, 2*time.Second, 20*time.Millisecond, "metrics server did not serve a scrape")

	return contentType, body
}

// A scraper asking for OpenMetrics is served OpenMetrics. Without it exemplars
// have no representation on the wire and are dropped silently, so this is the
// prerequisite for attaching them to the sidecar's latency histograms.
func TestSidecarMetricsEndpointServesOpenMetrics(t *testing.T) {
	contentType, body := scrapeSidecarMetrics(t, openMetricsAccept)

	require.True(t, strings.HasPrefix(contentType, "application/openmetrics-text"),
		"a scraper asking for OpenMetrics must be served OpenMetrics, got %q", contentType)
	require.Contains(t, body, "# HELP", "the endpoint must still serve metrics")
	require.Contains(t, body, "# EOF", "OpenMetrics payloads terminate with # EOF")
}

// A scraper that does not understand OpenMetrics keeps receiving the classic
// format, so enabling negotiation does not break existing scrapers.
func TestSidecarMetricsEndpointKeepsClassicFormatForClassicScrapers(t *testing.T) {
	contentType, body := scrapeSidecarMetrics(t, classicAccept)

	require.True(t, strings.HasPrefix(contentType, "text/plain"),
		"a classic scraper must keep receiving the classic format, got %q", contentType)
	require.Contains(t, body, "# HELP", "the endpoint must still serve metrics")
	require.NotContains(t, body, "# EOF", "the classic format has no # EOF terminator")
}

// Enabling OpenMetrics must not rename any of the sidecar's series. OpenMetrics
// shortens a counter's TYPE line by dropping _total, but the sample keeps it, so
// the names a scraper stores are the same in both formats. A counter that did
// not already end in _total would gain the suffix and silently split dashboards.
func TestSidecarSeriesNamesAreTheSameInBothFormats(t *testing.T) {
	_, openMetrics := scrapeSidecarMetrics(t, openMetricsAccept)
	_, classic := scrapeSidecarMetrics(t, classicAccept)

	require.Equal(t, sidecarSeriesNames(classic), sidecarSeriesNames(openMetrics),
		"OpenMetrics must expose the same series names as the classic format")
}

// sidecarSeriesNames returns the sorted set of series names the payload carries
// for this component, ignoring the Go runtime and controller-runtime metrics
// that share the registry.
func sidecarSeriesNames(payload string) []string {
	seen := map[string]bool{}
	for _, line := range strings.Split(payload, "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name := line
		if i := strings.IndexAny(name, "{ "); i >= 0 {
			name = name[:i]
		}
		if strings.HasPrefix(name, "moriio_") || strings.HasPrefix(name, "llm_d_") {
			seen[name] = true
		}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// stageExemplars returns the labels of the exemplars on metric's bucket lines
// in an OpenMetrics payload that carry traceID. client_golang builds an
// exemplar by ranging over a Go map, so the label order is not stable and the
// labels are parsed rather than matched as a string.
func stageExemplars(payload, metric, traceID string) []map[string]string {
	var found []map[string]string
	for _, line := range strings.Split(payload, "\n") {
		if !strings.HasPrefix(line, metric+"_bucket{") {
			continue
		}
		_, exemplar, ok := strings.Cut(line, " # {")
		if !ok {
			continue
		}
		exemplar, _, _ = strings.Cut(exemplar, "}")
		labels := map[string]string{}
		for _, pair := range strings.Split(exemplar, ",") {
			name, value, _ := strings.Cut(pair, "=")
			labels[name] = strings.Trim(value, `"`)
		}
		if labels[metricsutil.ExemplarTraceIDLabel] == traceID {
			found = append(found, labels)
		}
	}
	return found
}

// A stage observed while the request span is open reaches the scrape with the
// span's trace and span IDs. A handler without OpenMetrics drops them silently.
func TestSidecarMetricsEndpointServesStageExemplarsOnTheWire(t *testing.T) {
	recordSpans(t)
	ctx, span := otel.Tracer("test").Start(context.Background(), "forward_request")
	defer span.End()
	sc := span.SpanContext()

	stages := map[string]string{
		routing.StageEncode:  metricEncodeDuration,
		routing.StagePrefill: metricPrefillDuration,
		routing.StageDecode:  metricDecodeDuration,
	}
	for stage := range stages {
		observeStageDuration(ctx, stage, 420*time.Millisecond)
	}

	_, openMetrics := scrapeSidecarMetrics(t, openMetricsAccept)
	_, classic := scrapeSidecarMetrics(t, classicAccept)
	for stage, metric := range stages {
		require.Equal(t, []map[string]string{{
			metricsutil.ExemplarTraceIDLabel: sc.TraceID().String(),
			metricsutil.ExemplarSpanIDLabel:  sc.SpanID().String(),
		}}, stageExemplars(openMetrics, metric, sc.TraceID().String()),
			"the %s duration must carry the request's trace on the wire", stage)
		require.Empty(t, stageExemplars(classic, metric, sc.TraceID().String()),
			"the classic format has no representation for exemplars")
	}
}

// Every connector records its stage durations with the request's context, so
// each stage it records links to the request's trace on the wire. A connector
// added to supportedKVConnectors or supportedECConnectors runs here without
// changes to this test.
func TestConnectorStageDurationsCarryRequestTraceOnTheWire(t *testing.T) {
	// One backend plays prefiller, decoder, encoder and Mooncake bootstrap.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			statusHandler(http.StatusOK, `{"0":{"engine_id":"engine"}}`).ServeHTTP(w, r)
			return
		}
		statusHandler(http.StatusOK, chatResponse("ok", "stop", 1, 1)).ServeHTTP(w, r)
	}))
	defer upstream.Close()
	upstreamURL, err := url.Parse(upstream.URL)
	require.NoError(t, err)
	bootstrapPort, err := strconv.Atoi(upstreamURL.Port())
	require.NoError(t, err)

	type connector struct{ kv, ec string }
	connectors := make([]connector, 0, len(supportedKVConnectors)+len(supportedECConnectors))
	for _, kv := range slices.Sorted(maps.Keys(supportedKVConnectors)) {
		connectors = append(connectors, connector{kv: kv})
	}
	for _, ec := range slices.Sorted(maps.Keys(supportedECConnectors)) {
		connectors = append(connectors, connector{ec: ec})
	}

	for _, c := range connectors {
		t.Run(c.kv+c.ec, func(t *testing.T) {
			sr := tracetest.NewSpanRecorder()
			useTracerProviderForTest(t, sdktrace.NewTracerProvider(
				sdktrace.WithSpanProcessor(sr),
				sdktrace.WithSampler(sdktrace.AlwaysSample()),
			))

			s := NewProxy(Config{Port: "8000", KVConnector: c.kv, ECConnector: c.ec,
				DecoderURL: upstreamURL, MooncakeBootstrapPort: bootstrapPort})
			s.logger = logr.Discard()
			s.allowlistValidator = &AllowlistValidator{}
			s.decoderProxy = s.createDecoderProxyHandler(upstreamURL, false)

			req := chatRequest(t, userMessageRequest(imageURLItem("https://example.com/img.jpg")))
			req.Header.Set(routing.PrefillEndpointHeader, upstreamURL.Host)
			if c.kv == constants.KVConnectorOffloading {
				req.Header.Set(routing.KVCachePrefillHeader, upstreamURL.Host)
			}
			if c.ec != "" {
				req.Header.Set(routing.EncoderEndpointsHeader, upstreamURL.Host)
			}
			before := snapshotStageMetrics(t)
			rw := httptest.NewRecorder()
			s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(rw, req)
			require.Equal(t, http.StatusOK, rw.Code, rw.Body.String())

			// Concurrent-dispatch prefill outlives the handler and records its
			// duration before its span ends.
			require.Eventually(t, func() bool { return len(sr.Started()) == len(sr.Ended()) },
				5*time.Second, 10*time.Millisecond, "spans still open")
			var forward trace.SpanContext
			for _, span := range sr.Ended() {
				if span.Name() == "forward_request" {
					forward = span.SpanContext()
				}
			}
			require.True(t, forward.IsValid(), "forward_request span not recorded")

			_, payload := scrapeSidecarMetrics(t, openMetricsAccept)
			delta := snapshotStageMetrics(t).delta(before)
			recorded := 0
			for metric, count := range map[string]uint64{
				metricEncodeDuration:  delta.encodeCount,
				metricPrefillDuration: delta.prefillCount,
				metricDecodeDuration:  delta.decodeCount,
			} {
				if count == 0 {
					continue
				}
				recorded++
				exemplars := stageExemplars(payload, metric, forward.TraceID().String())
				require.NotEmpty(t, exemplars, "%s was recorded without the request's trace", metric)
				for _, labels := range exemplars {
					// A concurrent prefill can finish after forward_request
					// ended, and then carries the trace ID alone.
					if spanID, ok := labels[metricsutil.ExemplarSpanIDLabel]; ok {
						require.Equal(t, forward.SpanID().String(), spanID,
							"%s must name the request span, not a stage span", metric)
					}
				}
			}
			require.NotZero(t, recorded, "the connector recorded no stage duration")
		})
	}
}

// With the sidecar's tracing off, a request still carries its caller's sampled
// trace, so each stage links to that trace by ID. The span ID is left out: it
// would name the caller's span, not one the sidecar started.
func TestStageExemplarsCarryTraceIDOnlyWithTracingOff(t *testing.T) {
	useTracerProviderForTest(t, noop.NewTracerProvider())

	upstream := httptest.NewServer(statusHandler(http.StatusOK, chatResponse("ok", "stop", 1, 1)))
	defer upstream.Close()
	upstreamURL, err := url.Parse(upstream.URL)
	require.NoError(t, err)

	s := NewProxy(Config{Port: "8000", KVConnector: constants.KVConnectorNIXLV2, DecoderURL: upstreamURL})
	s.logger = logr.Discard()
	s.allowlistValidator = &AllowlistValidator{}
	s.decoderProxy = s.createDecoderProxyHandler(upstreamURL, false)

	const callerTraceID = "c0ffeec0ffeec0ffeec0ffeec0ffee00"
	traceID, err := trace.TraceIDFromHex(callerTraceID)
	require.NoError(t, err)
	spanID, err := trace.SpanIDFromHex("0102030405060708")
	require.NoError(t, err)
	caller := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: traceID, SpanID: spanID, TraceFlags: trace.FlagsSampled, Remote: true,
	})

	// otelhttp extracts the caller's traceparent into the request context the
	// same way.
	req := chatRequest(t, textChatBody())
	req = req.WithContext(trace.ContextWithRemoteSpanContext(req.Context(), caller))
	req.Header.Set(routing.PrefillEndpointHeader, upstreamURL.Host)
	rw := httptest.NewRecorder()
	s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(rw, req)
	require.Equal(t, http.StatusOK, rw.Code, rw.Body.String())

	_, payload := scrapeSidecarMetrics(t, openMetricsAccept)
	for _, metric := range []string{metricPrefillDuration, metricDecodeDuration} {
		require.Equal(t, []map[string]string{{metricsutil.ExemplarTraceIDLabel: callerTraceID}},
			stageExemplars(payload, metric, callerTraceID),
			"%s must link to the caller's trace without a span ID", metric)
	}
}
