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
	"net/http"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
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
