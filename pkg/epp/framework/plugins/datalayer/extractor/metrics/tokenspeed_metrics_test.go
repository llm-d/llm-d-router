/*
Copyright 2026 The Kubernetes Authors.
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

// TokenSpeed integration tests for the built-in "tokenspeed" engine mapping.
//
// The fixture file next to this test holds real Prometheus exposition text whose
// metric names, types, units and labels are transcribed from the pinned
// TokenSpeed source (lightseekorg/tokenspeed @ fb2d6bcff64bfeb99f2b8980714622ec5281b197,
// python/tokenspeed/runtime/metrics/collector.py) rather than from memory. It is
// served verbatim over HTTP so the parser under test is the router's real
// expfmt text parser, not a constructed MetricFamily map.
//
// See the fixture header for the provenance and for the one unit detail the
// mapping depends on.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// tokenSpeedFixturePath is the raw /metrics fixture relative to this package.
const tokenSpeedFixturePath = "testdata/tokenspeed_metrics.txt"

// tokenSpeedEngineName is the name the built-in TokenSpeed engine config and the
// engine-type label both use.
const tokenSpeedEngineName = "tokenspeed"

// serveMetricsText serves the given Prometheus exposition text over HTTP exactly
// as an engine's /metrics handler would, so the extractor exercises its real
// text parser.
func serveMetricsText(t *testing.T, exposition string) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		_, err := w.Write([]byte(exposition))
		require.NoError(t, err)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestTokenSpeedEngineConfigMatchesFixture pins the built-in engine mapping to
// the metric names the fixture serves. Renaming a spec without regenerating the
// fixture fails here instead of silently degrading to an unscrapeable endpoint.
func TestTokenSpeedEngineConfigMatchesFixture(t *testing.T) {
	var cfg *engineConfigParams
	for i := range defaultEngineConfigs {
		if defaultEngineConfigs[i].Name == tokenSpeedEngineName {
			cfg = &defaultEngineConfigs[i]
			break
		}
	}
	require.NotNil(t, cfg, "tokenspeed must be a built-in engine config")

	assert.Equal(t, "tokenspeed:num_requests_waiting", cfg.QueuedRequestsSpec)
	assert.Equal(t, "tokenspeed:num_requests_running", cfg.RunningRequestsSpec)
	assert.Equal(t, "tokenspeed:kv_cache_usage_perc", cfg.KVUsageSpec)

	// TokenSpeed exposes none of these, so they must stay unset rather than
	// guess at a family the engine never emits.
	assert.Empty(t, cfg.LoRASpec, "TokenSpeed exposes no LoRA metric")
	assert.Empty(t, cfg.CacheInfoSpec, "TokenSpeed exposes no cache_config_info family")
	assert.Empty(t, cfg.CacheBlockSizeSpec)
	assert.Empty(t, cfg.CacheNumBlocksSpec)
}

// TestTokenSpeedMetricsExtractionFromFixture runs the full source -> extractor
// pipeline against the fixture and asserts the endpoint attributes the scheduler
// reads.
func TestTokenSpeedMetricsExtractionFromFixture(t *testing.T) {
	exposition, err := os.ReadFile(tokenSpeedFixturePath)
	require.NoError(t, err, "read TokenSpeed metrics fixture")
	require.Contains(t, string(exposition), "tokenspeed:num_requests_waiting",
		"fixture must carry the token speed queue gauge")

	srv := serveMetricsText(t, string(exposition))

	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		// Pods carrying this label select the built-in tokenspeed mapping; the
		// router renders it from router.modelServers.type.
		DefaultEngineTypeLabelKey: tokenSpeedEngineName,
	})

	require.NoError(t, p.Poll(ctx, ep))

	m := ep.GetMetrics()
	assert.Equal(t, 5, m.WaitingQueueSize, "WaitingQueueSize")
	assert.Equal(t, 2, m.RunningRequestsSize, "RunningRequestsSize")
	// 0.375 is the ratio TokenSpeed reports, not a 0-100 percentage. The
	// kv-cache scorer computes 1 - KVCacheUsagePercent, so a 37.5 percentage
	// would drive its score negative.
	assert.InDelta(t, 0.375, m.KVCacheUsagePercent, 0.0001, "KVCacheUsagePercent")

	// No LoRA and no cache-config families exist for TokenSpeed; the extractor
	// must report no adapters and leave block size to the prefix plugin config.
	assert.Empty(t, m.ActiveModels)
	assert.Empty(t, m.WaitingModels)
	assert.Zero(t, m.MaxActiveModels)
	assert.Zero(t, m.CacheBlockSize, "block size must come from prefix plugin config")
	assert.Zero(t, m.CacheNumBlocks)
}

// TestTokenSpeedUnscrapeableEndpointKeepsLastKnownMetrics covers the failure
// mode the plan calls out for C1-U03: once a scrape fails, the endpoint must not
// look idle. If a failed or empty scrape reset the counters to zero, an endpoint
// that has stopped reporting would score as the least loaded and attract traffic.
func TestTokenSpeedUnscrapeableEndpointKeepsLastKnownMetrics(t *testing.T) {
	exposition, err := os.ReadFile(tokenSpeedFixturePath)
	require.NoError(t, err)

	srv := serveMetricsText(t, string(exposition))
	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: tokenSpeedEngineName,
	})

	require.NoError(t, p.Poll(ctx, ep))
	m := ep.GetMetrics()
	require.Equal(t, 5, m.WaitingQueueSize)
	require.InDelta(t, 0.375, m.KVCacheUsagePercent, 0.0001)

	// The endpoint stops answering.
	srv.Close()

	require.Error(t, p.Poll(ctx, ep), "a closed metrics endpoint must surface as a poll error")

	after := ep.GetMetrics()
	assert.Equal(t, 5, after.WaitingQueueSize, "stale queue size must survive a failed scrape")
	assert.InDelta(t, 0.375, after.KVCacheUsagePercent, 0.0001,
		"stale KV utilization must survive a failed scrape")
	assert.Equal(t, 1-m.KVCacheUsagePercent, 1-after.KVCacheUsagePercent,
		"scorer input must not change on a failed scrape")
}

// TestTokenSpeedKVCacheUsageIsARatioNotAPercentage guards the unit contract the
// kv-cache utilization scorer relies on: the router passes the engine value
// through unscaled (extractor.go), and the scorer scores 1 - value.
func TestTokenSpeedKVCacheUsageIsARatioNotAPercentage(t *testing.T) {
	srv := serveMetricsText(t, "# TYPE tokenspeed:kv_cache_usage_perc gauge\n"+
		"tokenspeed:kv_cache_usage_perc{model_name=\"m\",app_key=\"\"} 0.375\n")

	p, err := buildPipeline(t, srv.URL, &modelServerExtractorParams{
		EngineConfigs: []engineConfigParams{{
			Name:                tokenSpeedEngineName,
			QueuedRequestsSpec:  "tokenspeed:num_requests_waiting",
			RunningRequestsSpec: "tokenspeed:num_requests_running",
			KVUsageSpec:         "tokenspeed:kv_cache_usage_perc",
		}},
	})
	require.NoError(t, err)

	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: tokenSpeedEngineName,
	})

	// The queued/running families are absent from this fixture, so Poll reports
	// them; the KV value is still written by the successful spec.
	_ = p.Poll(context.Background(), ep)

	got := ep.GetMetrics()
	require.InDelta(t, 0.375, got.KVCacheUsagePercent, 0.0001)
	assert.GreaterOrEqual(t, 1-got.KVCacheUsagePercent, 0.0,
		"kv-cache scorer score must stay non-negative")
}

// TestTokenSpeedTypeResolvesWithoutExplicitEngineConfig is the payoff of making
// tokenspeed a built-in engine: setting router.modelServers.type to it renders
// defaultEngine: "tokenspeed" into the EPP config, and the extractor resolves
// that from its built-in table with no engineConfigs block at all.
//
// Before tokenspeed was built in, the same config failed plugin construction
// with `defaultEngine "tokenspeed" not found in engineConfigs`; the generic form
// of that rejection is already covered upstream by an "unknown-engine" case.
func TestTokenSpeedTypeResolvesWithoutExplicitEngineConfig(t *testing.T) {
	ext, err := newCoreMetricsExtractorPlugin(context.Background(), "core-metrics-extractor",
		&modelServerExtractorParams{DefaultEngine: tokenSpeedEngineName})
	require.NoError(t, err, "tokenspeed must resolve as a default engine")

	mapping, ok := ext.registry.Get(tokenSpeedEngineName)
	require.True(t, ok, "tokenspeed mapping must be registered")
	assert.Equal(t, []string{
		"tokenspeed:num_requests_waiting",
		"tokenspeed:num_requests_running",
		"tokenspeed:kv_cache_usage_perc",
	}, mapping.MetricNames())

	// Unlabeled pods fall back to the configured default, so the model server
	// pods do not strictly need llm-d.ai/engine-type when it is the only engine.
	fallback, ok := ext.registry.Get("")
	require.True(t, ok)
	assert.Equal(t, mapping.MetricNames(), fallback.MetricNames())
}
