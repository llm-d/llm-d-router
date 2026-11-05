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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// scrapeText serves exposition over HTTP, scrapes it with the default engine
// configs as the engine named by label, and returns the extraction error and
// the endpoint's metrics. Passing ep reuses an endpoint across scrapes.
func scrapeText(t *testing.T, engine, exposition string, ep fwkdl.Endpoint) (fwkdl.Endpoint, error) {
	t.Helper()
	srv := serveMetricsText(t, exposition)
	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)
	if ep == nil {
		ep = newEndpointAt(mustHost(t, srv.URL), map[string]string{DefaultEngineTypeLabelKey: engine})
	} else {
		ep.GetMetadata().MetricsHost = mustHost(t, srv.URL)
	}
	ctx := context.Background()
	data, err := p.source.Poll(ctx, ep)
	require.NoError(t, err)
	return ep, p.ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: data, Endpoint: ep})
}

const sglangSchedulingExposition = `# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="Qwen/Qwen3-0.6B"} 0.0
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="Qwen/Qwen3-0.6B"} 1.0
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="Qwen/Qwen3-0.6B"} 0.1
# TYPE sglang:page_size gauge
sglang:page_size{model_name="Qwen/Qwen3-0.6B"} 1.0
# TYPE sglang:num_pages gauge
sglang:num_pages{model_name="Qwen/Qwen3-0.6B"} 200000.0
`

const sglangHiCacheExposition = `# HELP sglang:hicache_host_total_tokens Total capacity of the host KV cache in tokens.
# TYPE sglang:hicache_host_total_tokens gauge
sglang:hicache_host_total_tokens{model_name="Qwen/Qwen3-0.6B",tp_rank="0"} 400001.0
`

// TestCacheTotalTokensSGLang checks that the built-in SGLang config reads the
// HiCache host pool capacity when present and extracts without error, and
// leaves the capacity at 0 without error, when SGLang runs without HiCache.
func TestCacheTotalTokensSGLang(t *testing.T) {
	t.Run("hicache enabled", func(t *testing.T) {
		ep, err := scrapeText(t, "sglang", sglangSchedulingExposition+sglangHiCacheExposition, nil)
		require.NoError(t, err)
		assert.Equal(t, 400001, ep.GetMetrics().KvCacheMaxTokenCapacity)
		assert.Equal(t, 200000, ep.GetMetrics().CacheNumBlocks)
		assert.False(t, ep.GetMetrics().KvCacheOffloadDetected, "sglang config has no offload detection spec")
	})
	t.Run("hicache disabled", func(t *testing.T) {
		ep, err := scrapeText(t, "sglang", sglangSchedulingExposition, nil)
		require.NoError(t, err, "an absent optional capacity metric is not an extraction error")
		assert.Equal(t, 0, ep.GetMetrics().KvCacheMaxTokenCapacity)
	})
	t.Run("one series per engine takes the largest", func(t *testing.T) {
		perEngine := `# TYPE sglang:hicache_host_total_tokens gauge
sglang:hicache_host_total_tokens{model_name="m",tp_rank="0"} 300000.0
sglang:hicache_host_total_tokens{model_name="m",tp_rank="1"} 400001.0
`
		ep, err := scrapeText(t, "sglang", sglangSchedulingExposition+perEngine, nil)
		require.NoError(t, err)
		assert.Equal(t, 400001, ep.GetMetrics().KvCacheMaxTokenCapacity)
	})
}

// vllmOffloadCountersExposition is the deprecated transfer counter that vLLM's
// OffloadingConnector registers at startup on every release, as captured in
// testdata/vllm-flag.prom.
const vllmOffloadCountersExposition = `# TYPE vllm:kv_offload_total_bytes_total counter
vllm:kv_offload_total_bytes_total{engine="0",model_name="Qwen/Qwen3-8B",transfer_type="CPU_to_GPU"} 0.0
vllm:kv_offload_total_bytes_total{engine="0",model_name="Qwen/Qwen3-8B",transfer_type="GPU_to_CPU"} 0.0
`

// vllmOffloadConfigInfoExposition is vllm:kv_offload_config_info as a
// standalone CPUOffloadingSpec reports it after the engine's first scheduler
// step. The label set follows vllm-project/vllm#53902; it is not in a vLLM
// release at the time of writing, so this exposition is written from that
// change rather than captured from a running engine.
const vllmOffloadConfigInfoExposition = `# TYPE vllm:kv_offload_config_info gauge
vllm:kv_offload_config_info{cpu_blocks_per_chunk="4",cpu_kv_cache_size_bytes="2147483648",cpu_kv_cache_size_tokens="1048576",cpu_num_chunks="16384",engine="0",model_name="Qwen/Qwen3-8B"} 1.0
`

// TestCacheTotalTokensVLLM checks that the built-in vLLM config reads the CPU
// tier capacity from the kv_offload_config_info label and reports offloading
// from the connector counters, across the engine's lifecycle.
func TestCacheTotalTokensVLLM(t *testing.T) {
	tests := []struct {
		name         string
		exposition   string
		wantTokens   int
		wantDetected bool
	}{
		{
			name:       "no offloading",
			exposition: vllmSchedulingExposition,
		},
		{
			name:         "offloading before the first scheduler step",
			exposition:   vllmSchedulingExposition + vllmOffloadCountersExposition,
			wantDetected: true,
		},
		{
			name:         "offloading with the capacity label",
			exposition:   vllmSchedulingExposition + vllmOffloadCountersExposition + vllmOffloadConfigInfoExposition,
			wantTokens:   1048576,
			wantDetected: true,
		},
		{
			name: "tiering puts the label on the primary tier series",
			exposition: vllmSchedulingExposition + vllmOffloadCountersExposition + `# TYPE vllm:kv_offload_config_info gauge
vllm:kv_offload_config_info{cpu_kv_cache_size_tokens="1048576",engine="0",fs_path="",model_name="m",tier="0:primary"} 1.0
vllm:kv_offload_config_info{cpu_kv_cache_size_tokens="",engine="0",fs_path="/mnt/kv",model_name="m",tier="1:fs"} 1.0
`,
			wantTokens:   1048576,
			wantDetected: true,
		},
		{
			name: "capacity the engine cannot compute reads None",
			exposition: vllmSchedulingExposition + vllmOffloadCountersExposition + `# TYPE vllm:kv_offload_config_info gauge
vllm:kv_offload_config_info{cpu_kv_cache_size_tokens="None",cpu_num_chunks="16384",engine="0",model_name="m"} 1.0
`,
			wantDetected: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ep, err := scrapeText(t, "vllm", tc.exposition, nil)
			require.NoError(t, err)
			m := ep.GetMetrics()
			assert.Equal(t, tc.wantTokens, m.KvCacheMaxTokenCapacity, "KvCacheMaxTokenCapacity")
			assert.Equal(t, tc.wantDetected, m.KvCacheOffloadDetected, "KvCacheOffloadDetected")
			assert.Equal(t, 1024, m.CacheNumBlocks, "CacheNumBlocks still comes from cache_config_info")
		})
	}
}

// TestCacheTotalTokensKeepsLastValue checks that a scrape without the optional
// capacity metric keeps the capacity a previous scrape reported, while the
// offload signal follows the current scrape.
func TestCacheTotalTokensKeepsLastValue(t *testing.T) {
	ep, err := scrapeText(t, "vllm", vllmSchedulingExposition+vllmOffloadCountersExposition+vllmOffloadConfigInfoExposition, nil)
	require.NoError(t, err)
	require.Equal(t, 1048576, ep.GetMetrics().KvCacheMaxTokenCapacity)
	require.True(t, ep.GetMetrics().KvCacheOffloadDetected)

	_, err = scrapeText(t, "vllm", vllmSchedulingExposition, ep)
	require.NoError(t, err)
	assert.Equal(t, 1048576, ep.GetMetrics().KvCacheMaxTokenCapacity, "capacity is kept across a scrape that lacks it")
	assert.False(t, ep.GetMetrics().KvCacheOffloadDetected, "offload detection follows the scrape")
}

func TestMappingCacheTotalTokensLabelRequiresSpec(t *testing.T) {
	_, err := NewMappingFromConfig(MappingConfig{CacheTotalTokensLabel: "cpu_kv_cache_size_tokens"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cacheTotalTokensLabelName requires cacheTotalTokensSpec")

	_, err = NewMappingFromConfig(MappingConfig{CacheTotalTokens: "vllm:kv_offload_config_info", CacheTotalTokensLabel: "cpu_kv_cache_size_tokens"})
	require.NoError(t, err)
}

// The .prom files under testdata are /metrics scrapes captured from running
// model servers, trimmed to the families the engine configs read, so the
// fixture tests below run the default engine configs against real exposition
// output:
//   - vllm-none.prom: vLLM v0.26.0, no offloading configured.
//   - vllm-flag.prom: vLLM v0.26.0 with --kv-offloading-size 4.
//   - vllm-conn.prom: vLLM v0.26.0 with --kv-transfer-config naming
//     OffloadingConnector directly.
//   - sglang-new.prom: SGLang v0.5.16 with --enable-hierarchical-cache
//     --hicache-ratio 2 --max-total-tokens 200000.
//   - sglang-old.prom: SGLang v0.5.9, same flags, which predates the host
//     capacity gauge.
//   - trtllm-legacy.prom: trtllm-serve v1.3.0rc23, default KV cache manager,
//     kv_cache_config.host_cache_size 4GiB.
//   - trtllm-nohost.prom: same without host_cache_size.
//   - trtllm-v2.prom: same host_cache_size with use_kv_cache_manager_v2: true.
func extractFixture(t *testing.T, engine, name string) *fwkdl.Metrics {
	t.Helper()
	exposition, err := os.ReadFile(filepath.Join("testdata", name))
	require.NoError(t, err)
	// The fixtures are trimmed, so required families can be absent and the
	// extraction error is not meaningful; the present families still extract.
	ep, _ := scrapeText(t, engine, string(exposition), nil)
	return ep.GetMetrics()
}

func TestVLLMFixtures(t *testing.T) {
	tests := []struct {
		fixture      string
		wantDetected bool
	}{
		{fixture: "vllm-none.prom", wantDetected: false},
		{fixture: "vllm-flag.prom", wantDetected: true},
		{fixture: "vllm-conn.prom", wantDetected: true},
	}
	for _, tc := range tests {
		t.Run(tc.fixture, func(t *testing.T) {
			m := extractFixture(t, "vllm", tc.fixture)
			assert.Equal(t, tc.wantDetected, m.KvCacheOffloadDetected, "KvCacheOffloadDetected")
			assert.Equal(t, 74272, m.CacheNumBlocks, "CacheNumBlocks")
			assert.Equal(t, 16, m.CacheBlockSize, "CacheBlockSize")
			// v0.26.0 predates kv_offload_config_info.
			assert.Equal(t, 0, m.KvCacheMaxTokenCapacity, "KvCacheMaxTokenCapacity")
		})
	}
}

func TestSGLangFixtures(t *testing.T) {
	tests := []struct {
		fixture    string
		wantTokens int
	}{
		// --hicache-ratio 2 on a 200000-token device pool.
		{fixture: "sglang-new.prom", wantTokens: 400001},
		// v0.5.9 reports the device pool only; the host tier is invisible.
		{fixture: "sglang-old.prom", wantTokens: 0},
	}
	for _, tc := range tests {
		t.Run(tc.fixture, func(t *testing.T) {
			m := extractFixture(t, "sglang", tc.fixture)
			assert.Equal(t, tc.wantTokens, m.KvCacheMaxTokenCapacity, "KvCacheMaxTokenCapacity")
			assert.False(t, m.KvCacheOffloadDetected, "KvCacheOffloadDetected")
			// Both versions were launched with a 200000-token pool and SGLang's
			// default 1-token pages. v0.5.16 reports them as the page_size and
			// num_pages gauges; v0.5.9 only as cache_config_info labels, which the
			// built-in config no longer reads.
			if tc.fixture == "sglang-new.prom" {
				assert.Equal(t, 200000, m.CacheNumBlocks, "CacheNumBlocks")
				assert.Equal(t, 1, m.CacheBlockSize, "CacheBlockSize")
			}
		})
	}
}

// TestTRTLLMFixtures documents the trtllm-serve semantics the producer relies
// on: the default KV cache manager folds host offload blocks into
// trtllm_kv_cache_max_blocks, so CacheNumBlocks already covers the
// offload-extended cache; the V2 manager reports GPU blocks only. The 4GiB host
// cache is about 1170 blocks at about 3.6MiB per 32-token block, which is the
// legacy-to-V2 gap.
func TestTRTLLMFixtures(t *testing.T) {
	tests := []struct {
		fixture       string
		wantNumBlocks int
	}{
		{fixture: "trtllm-nohost.prom", wantNumBlocks: 19960},
		{fixture: "trtllm-legacy.prom", wantNumBlocks: 21129},
		{fixture: "trtllm-v2.prom", wantNumBlocks: 19958},
	}
	for _, tc := range tests {
		t.Run(tc.fixture, func(t *testing.T) {
			m := extractFixture(t, "trtllm-serve", tc.fixture)
			assert.Equal(t, tc.wantNumBlocks, m.CacheNumBlocks, "CacheNumBlocks")
			assert.Equal(t, 32, m.CacheBlockSize, "CacheBlockSize")
			assert.Equal(t, 0, m.KvCacheMaxTokenCapacity, "KvCacheMaxTokenCapacity")
		})
	}
}
