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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// sglangLoadsExposition is SGLang's /v1/loads?format=prometheus&include=core response for
// two data-parallel ranks, cut to the families the engine config of the test reads. SGLang
// writes each numeric field of a rank's load this way
// (python/sglang/srt/entrypoints/v1_loads.py).
const sglangLoadsExposition = `# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs{dp_rank="0"} 3
sglang_num_running_reqs{dp_rank="1"} 40
# TYPE sglang_num_waiting_reqs gauge
sglang_num_waiting_reqs{dp_rank="0"} 1
sglang_num_waiting_reqs{dp_rank="1"} 24
# TYPE sglang_token_usage gauge
sglang_token_usage{dp_rank="0"} 0.25
sglang_token_usage{dp_rank="1"} 0.91
`

// TestMetricsExtractionSGLangLoadsPath verifies the source path and the engine config that
// the metrics data source README gives for SGLang's /v1/loads. The path needs its query
// string, and the metric names are not the ones of /metrics.
func TestMetricsExtractionSGLangLoadsPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// SGLang answers with Prometheus text only for format=prometheus.
		if r.URL.Path != "/v1/loads" || r.URL.Query().Get("format") != "prometheus" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(sglangLoadsExposition))
	}))
	defer srv.Close()

	source, err := sourcemetrics.NewHTTPMetricsDataSource("http", "/v1/loads?format=prometheus&include=core", "metrics-data-source")
	require.NoError(t, err)
	plugin, err := CoreMetricsExtractorFactory("core-metrics-extractor", json.NewDecoder(strings.NewReader(`{
		"engineConfigs": [{
			"name": "sglang",
			"queuedRequestsSpec": "sglang_num_waiting_reqs",
			"runningRequestsSpec": "sglang_num_running_reqs",
			"kvUsageSpec": "sglang_token_usage"
		}]
	}`)), nil)
	require.NoError(t, err)
	ext, ok := plugin.(*Extractor)
	require.True(t, ok)
	require.NoError(t, source.AppendExtractor(ext))

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{DefaultEngineTypeLabelKey: "sglang"})
	data, err := source.Poll(ctx, ep)
	require.NoError(t, err)
	require.NoError(t, ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep}))

	// The endpoint is the whole server, so it reads every rank: the counts are sums and
	// the usage is the maximum.
	m := ep.GetMetrics()
	assert.Equal(t, 25, m.WaitingQueueSize, "WaitingQueueSize")
	assert.Equal(t, 43, m.RunningRequestsSize, "RunningRequestsSize")
	assert.InDelta(t, 0.91, m.KVCacheUsagePercent, 0.001, "KVCacheUsagePercent")
}
