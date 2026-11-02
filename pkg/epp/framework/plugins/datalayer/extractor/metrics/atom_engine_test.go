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

package metrics_test

import (
	"context"
	"strings"
	"testing"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// atomScrape mirrors the load and cache gauges ATOM's exporter serves
// (ROCm/ATOM atom/metrics/exporter.py at df86c48). ATOM emits no
// atom:cache_config_info family and no block-size gauge.
const atomScrape = `# HELP atom:requests_running Number of requests currently running across data-parallel ranks.
# TYPE atom:requests_running gauge
atom:requests_running 3.0
# HELP atom:requests_waiting Number of requests waiting across data-parallel ranks.
# TYPE atom:requests_waiting gauge
atom:requests_waiting 7.0
# HELP atom:kv_cache_blocks_used Number of allocated KV-cache blocks.
# TYPE atom:kv_cache_blocks_used gauge
atom:kv_cache_blocks_used 512.0
# HELP atom:kv_cache_blocks_free Number of free KV-cache blocks.
# TYPE atom:kv_cache_blocks_free gauge
atom:kv_cache_blocks_free 1536.0
# HELP atom:kv_cache_blocks_total Total number of KV-cache blocks.
# TYPE atom:kv_cache_blocks_total gauge
atom:kv_cache_blocks_total 2048.0
# HELP atom:kv_cache_usage_ratio Fraction of KV-cache blocks currently allocated.
# TYPE atom:kv_cache_usage_ratio gauge
atom:kv_cache_usage_ratio 0.25
`

func TestBuiltinAtomEngineExtractsWithoutErrors(t *testing.T) {
	ctx := context.Background()
	plugin, err := metrics.CoreMetricsExtractorFactory("core", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ext := plugin.(*metrics.Extractor)

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{metrics.DefaultEngineTypeLabelKey: "atom"},
	}, nil)
	for scrape := 1; scrape <= 3; scrape++ {
		parser := expfmt.NewTextParser(model.LegacyValidation)
		fams, err := parser.TextToMetricFamilies(strings.NewReader(atomScrape))
		if err != nil {
			t.Fatal(err)
		}
		err = ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: fams, Endpoint: ep})
		if err != nil {
			t.Errorf("scrape %d of a healthy ATOM endpoint returned an extraction error: %v", scrape, err)
		}
	}
	m := ep.GetMetrics()
	if m.WaitingQueueSize != 7 || m.RunningRequestsSize != 3 || m.KVCacheUsagePercent != 0.25 {
		t.Errorf("WaitingQueueSize=%d RunningRequestsSize=%d KVCacheUsagePercent=%v, want 7, 3, 0.25",
			m.WaitingQueueSize, m.RunningRequestsSize, m.KVCacheUsagePercent)
	}
}
