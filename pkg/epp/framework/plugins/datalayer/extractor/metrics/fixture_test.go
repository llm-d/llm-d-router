/*
Copyright 2026 The Kubernetes Authors.

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

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// The .prom files under testdata are real /metrics scrapes captured from model
// servers (families irrelevant to KV cache trimmed), so these tests exercise
// the default engine mappings against actual exposition output:
//   - vllm-none.prom: vLLM v0.26.0, no offloading configured.
//   - vllm-flag.prom: vLLM v0.26.0 with --kv-offloading-size 4
//     (kv_offloading_size label set, OffloadingConnector metrics present).
//   - vllm-conn.prom: vLLM v0.26.0 with --kv-transfer-config naming
//     OffloadingConnector directly (kv_offloading_size label is "None";
//     only the connector runtime metrics reveal offloading).
//   - sglang-new.prom: SGLang v0.5.16 with --enable-hierarchical-cache
//     --hicache-ratio 2 --max-total-tokens 200000 (hicache_host_total_tokens
//     gauge present; no cache_config_info, page_size/num_pages are plain gauges).
//   - sglang-old.prom: SGLang v0.5.9, same flags (predates the host capacity
//     gauge and the plain page gauges; cache_config_info carries the labels).
//   - trtllm-legacy.prom: trtllm-serve v1.3.0rc23, default (legacy) KV cache
//     manager with kv_cache_config.host_cache_size 4GiB: max_blocks counts
//     GPU plus host blocks.
//   - trtllm-nohost.prom: same, without host_cache_size (GPU-only baseline).
//   - trtllm-v2.prom: same host_cache_size with use_kv_cache_manager_v2: true:
//     max_blocks is GPU-only, the host tier is invisible on Prometheus.
func loadFixture(t *testing.T, name string) sourcemetrics.PrometheusMetricMap {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Skipf("fixture %s not available: %v", name, err)
	}
	defer f.Close() //nolint:errcheck

	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(f)
	if err != nil {
		t.Fatalf("failed to parse fixture %s: %v", name, err)
	}
	return families
}

func extractFixture(t *testing.T, engine, fixture string) *fwkdl.Metrics {
	t.Helper()
	extractor, err := newCoreMetricsExtractorPlugin(context.Background(), "fixture-test", nil)
	if err != nil {
		t.Fatalf("failed to build extractor with default engine configs: %v", err)
	}
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		Labels: map[string]string{DefaultEngineTypeLabelKey: engine},
	}, nil)
	// Absent optional families produce errors while present ones still extract.
	_ = extractor.Extract(context.Background(), fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{
		Payload:  loadFixture(t, fixture),
		Endpoint: ep,
	})
	return ep.GetMetrics()
}

func TestVLLMFixtures(t *testing.T) {
	tests := []struct {
		fixture       string
		wantDetected  bool
		wantNumBlocks int
	}{
		{fixture: "vllm-none.prom", wantDetected: false, wantNumBlocks: 74272},
		{fixture: "vllm-flag.prom", wantDetected: true, wantNumBlocks: 74272},
		{fixture: "vllm-conn.prom", wantDetected: true, wantNumBlocks: 74272},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			m := extractFixture(t, "vllm", tt.fixture)
			if m.KvCacheOffloadDetected != tt.wantDetected {
				t.Errorf("KvCacheOffloadDetected = %v, want %v", m.KvCacheOffloadDetected, tt.wantDetected)
			}
			if m.CacheNumBlocks != tt.wantNumBlocks {
				t.Errorf("CacheNumBlocks = %d, want %d", m.CacheNumBlocks, tt.wantNumBlocks)
			}
			if m.CacheBlockSize != 16 {
				t.Errorf("CacheBlockSize = %d, want 16", m.CacheBlockSize)
			}
			// vLLM has no total-capacity metric; capacity must stay unset so the
			// producer falls back to GPU blocks (plus multiplier if configured).
			if m.KvCacheMaxTokenCapacity != 0 {
				t.Errorf("KvCacheMaxTokenCapacity = %d, want 0", m.KvCacheMaxTokenCapacity)
			}
		})
	}
}

func TestSGLangFixtures(t *testing.T) {
	tests := []struct {
		fixture           string
		wantTokenCapacity int
	}{
		// v0.5.16 with hicache-ratio 2: host pool gauge = 2x device tokens and
		// wins the max. v0.5.9 predates the gauge: device tokens only.
		{fixture: "sglang-new.prom", wantTokenCapacity: -1}, // filled from device gauge at runtime; see body
		{fixture: "sglang-old.prom", wantTokenCapacity: -1},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			m := extractFixture(t, "sglang", tt.fixture)
			families := loadFixture(t, tt.fixture)

			device := 0
			if fam, ok := families["sglang:max_total_num_tokens"]; ok && len(fam.GetMetric()) > 0 {
				device = int(fam.GetMetric()[0].GetGauge().GetValue())
			}
			host := 0
			if fam, ok := families["sglang:hicache_host_total_tokens"]; ok && len(fam.GetMetric()) > 0 {
				host = int(fam.GetMetric()[0].GetGauge().GetValue())
			}
			want := max(host, device)
			if want == 0 {
				t.Fatalf("fixture %s has no capacity gauges", tt.fixture)
			}
			if m.KvCacheMaxTokenCapacity != want {
				t.Errorf("KvCacheMaxTokenCapacity = %d, want %d (host=%d device=%d)",
					m.KvCacheMaxTokenCapacity, want, host, device)
			}
			if tt.fixture == "sglang-new.prom" {
				if host == 0 {
					t.Error("expected hicache_host_total_tokens gauge in v0.5.16 fixture")
				}
				if host <= device {
					t.Errorf("expected host pool (%d) > device pool (%d) with hicache-ratio 2", host, device)
				}
			}
			if tt.fixture == "sglang-old.prom" && host != 0 {
				t.Error("v0.5.9 fixture unexpectedly has hicache_host_total_tokens; update version notes")
			}
			if m.KvCacheOffloadDetected {
				t.Error("sglang mapping should not set KvCacheOffloadDetected")
			}
			// Both fixture versions were launched with a 200000-token pool and
			// SGLang's default 1-token pages: v0.5.9 exposes them as
			// cache_config_info labels, v0.5.16 as plain gauges.
			if m.CacheNumBlocks != 200000 {
				t.Errorf("CacheNumBlocks = %d, want 200000", m.CacheNumBlocks)
			}
			if m.CacheBlockSize != 1 {
				t.Errorf("CacheBlockSize = %d, want 1", m.CacheBlockSize)
			}
		})
	}
}

// TestTRTLLMFixtures documents the trtllm-serve tiering semantics the router
// relies on: the legacy (default) KV cache manager folds host offload blocks
// into trtllm_kv_cache_max_blocks, so auto-tuning from CacheNumBlocks already
// covers the offload-extended cache; the V2 manager reports GPU only and needs
// an explicit lruCapacityPerServer override when host offload is enabled.
// Captured on trtllm-serve v1.3.0rc23 with Qwen3-0.6B: the 4GiB host cache is
// ~1169 blocks at ~3.6MiB per 32-token block.
func TestTRTLLMFixtures(t *testing.T) {
	tests := []struct {
		fixture       string
		wantNumBlocks int
	}{
		{fixture: "trtllm-nohost.prom", wantNumBlocks: 19960}, // GPU-only baseline
		{fixture: "trtllm-legacy.prom", wantNumBlocks: 21129}, // GPU + 4GiB host tier
		{fixture: "trtllm-v2.prom", wantNumBlocks: 19958},     // host tier configured but not counted
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			m := extractFixture(t, "trtllm-serve", tt.fixture)
			if m.CacheNumBlocks != tt.wantNumBlocks {
				t.Errorf("CacheNumBlocks = %d, want %d", m.CacheNumBlocks, tt.wantNumBlocks)
			}
			if m.CacheBlockSize != 32 {
				t.Errorf("CacheBlockSize = %d, want 32", m.CacheBlockSize)
			}
		})
	}
}
