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

package metrics

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/stretchr/testify/require"
)

// scrapeText registers every coordinator metric through the same Register
// call main.go uses, observes each of the five amplification-family metrics
// once, and scrapes the registry through promhttp -- the exact text
// Prometheus would pull from /metrics. It returns the response body.
//
// A never-observed HistogramVec produces no exposition output, so each
// family must be recorded at least once for the series to appear on the
// wire; that is also the point: this test fails if a collector is missing
// from allCollectors (registered list drift) or the subsystem/name assembly
// produces a wrong series name, both of which leave unit tests green.
func scrapeText(t *testing.T) string {
	t.Helper()
	Reset()
	reg := prometheus.NewRegistry()
	require.NoError(t, Register(reg))

	RecordEncodeSubrequests(2)
	RecordOrchestrationOverhead(RouteChatCompletions, 15*time.Millisecond)
	RecordMediaItems(MediaTypeImage, 1)
	RecordMediaDownloadDuration(DownloadResultSuccess, 10*time.Millisecond)
	RecordResponseSize(true, 128)

	srv := httptest.NewServer(promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	defer srv.Close()

	// Proxy must be bypassed explicitly: the scrape target is loopback, and
	// an ambient HTTP_PROXY (common on dev machines) would hijack the request.
	client := &http.Client{Transport: &http.Transport{Proxy: nil}}
	resp, err := client.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return string(body)
}

func TestScrape_ExposeFiveAmplificationSeries(t *testing.T) {
	text := scrapeText(t)

	for _, name := range []string{
		"llm_d_coordinator_encode_subrequests",
		"llm_d_coordinator_orchestration_overhead_seconds",
		"llm_d_coordinator_media_items",
		"llm_d_coordinator_media_download_duration_seconds",
		"llm_d_coordinator_response_size_bytes",
	} {
		require.Contains(t, text, "# TYPE "+name+" histogram",
			"series %s missing from the scrape output; is it registered in allCollectors and named correctly?", name)
	}
}

func TestScrape_SeriesCarryBoundedLabelsAndSamples(t *testing.T) {
	text := scrapeText(t)

	// Each observed label set must appear with real samples in the
	// exposition: the bucket line with the bounded label proves both the
	// label assembly and the observation reached the wire.
	require.Contains(t, text, `llm_d_coordinator_encode_subrequests_bucket{le="2"} 1`)
	require.Contains(t, text, `llm_d_coordinator_orchestration_overhead_seconds_bucket{route="chat_completions",le="0.025"} 1`)
	require.Contains(t, text, `llm_d_coordinator_media_items_bucket{media_type="image",le="1"} 1`)
	require.Contains(t, text, `llm_d_coordinator_media_download_duration_seconds_bucket{result="success",le="0.025"} 1`)
	require.Contains(t, text, `llm_d_coordinator_response_size_bytes_bucket{stream="true",le="128"} 1`)
}
