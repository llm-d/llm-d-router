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
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
)

const familyFilterPage = `# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 0.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
# HELP vllm:num_requests_waiting_by_reason Requests waiting, by reason.
# TYPE vllm:num_requests_waiting_by_reason gauge
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 1.0
# a free-form comment

# HELP vllm:prompt_tokens Number of prefill tokens processed.
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 1234.0
vllm:prompt_tokens_created{engine="0",model_name="m"} 1.7e+09
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="m"} 5.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 5.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="m"} 9.5
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="64",engine="0",num_gpu_blocks="5330"} 1.0`

func sortedKeys(m PrometheusMetricMap) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func TestFamilyFilterParsesOnlyListedFamilies(t *testing.T) {
	// The page ends without a newline; the filter completes the last line, the plain parser needs it.
	full, err := parseMetrics(strings.NewReader(familyFilterPage + "\n"))
	if err != nil {
		t.Fatalf("parse full page: %v", err)
	}
	listed := []string{
		"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens",
		"vllm:e2e_request_latency_seconds", "vllm:cache_config_info", "vllm:not_exposed",
	}
	got, err := newFamilyFilter(listed).parse(strings.NewReader(familyFilterPage))
	if err != nil {
		t.Fatalf("parse filtered page: %v", err)
	}

	// The text parser files counter samples under their sample names (_total, _created).
	want := []string{
		"vllm:cache_config_info", "vllm:e2e_request_latency_seconds", "vllm:num_requests_running",
		"vllm:num_requests_waiting", "vllm:prompt_tokens_created", "vllm:prompt_tokens_total",
	}
	if keys := sortedKeys(got); strings.Join(keys, ",") != strings.Join(want, ",") {
		t.Fatalf("families = %v, want %v", keys, want)
	}
	for _, name := range want {
		if !proto.Equal(got[name], full[name]) {
			t.Errorf("family %s differs from the unfiltered parse:\n got  %v\n want %v", name, got[name], full[name])
		}
	}
}

func TestFamilyFilterKeepsLinesLongerThanTheReadBuffer(t *testing.T) {
	long := strings.Repeat("x", 200*1024)
	page := "# TYPE vllm:cache_config_info gauge\n" +
		`vllm:cache_config_info{engine="0",note="` + long + `"} 1.0` + "\n" +
		"# TYPE vllm:other gauge\n" +
		`vllm:other{note="` + long + `"} 2.0` + "\n"

	got, err := newFamilyFilter([]string{"vllm:cache_config_info"}).parse(strings.NewReader(page))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if keys := sortedKeys(got); len(keys) != 1 || keys[0] != "vllm:cache_config_info" {
		t.Fatalf("families = %v, want [vllm:cache_config_info]", keys)
	}
	if v := got["vllm:cache_config_info"].GetMetric()[0].GetLabel()[1].GetValue(); v != long {
		t.Fatalf("long label value truncated to %d bytes", len(v))
	}
}

func TestFamilyFilterKeepsTabSeparatedMetadata(t *testing.T) {
	page := "# HELP\tselected\tSelected gauge\n# TYPE\tselected\tgauge\nselected 3\n"
	full, err := parseMetrics(strings.NewReader(page))
	if err != nil {
		t.Fatalf("parse full page: %v", err)
	}
	filtered, err := newFamilyFilter([]string{"selected"}).parse(strings.NewReader(page))
	if err != nil {
		t.Fatalf("parse filtered page: %v", err)
	}
	if !proto.Equal(full["selected"], filtered["selected"]) {
		t.Fatalf("selected family differs: full %v, filtered %v", full["selected"], filtered["selected"])
	}
}

func TestFamilyFilterDropsSeparateSuffixFamily(t *testing.T) {
	page := "# TYPE selected gauge\nselected 1\n# TYPE selected_total gauge\nselected_total 2\n"
	filtered, err := newFamilyFilter([]string{"selected"}).parse(strings.NewReader(page))
	if err != nil {
		t.Fatalf("parse filtered page: %v", err)
	}
	if keys := sortedKeys(filtered); len(keys) != 1 || keys[0] != "selected" {
		t.Fatalf("families = %v, want [selected]", keys)
	}
}

func TestMetricsDataSourceFactoryAcceptsFamilies(t *testing.T) {
	raw, err := json.Marshal(map[string]any{"families": []string{"vllm:num_requests_running"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := MetricsDataSourceFactory("metrics", json.NewDecoder(strings.NewReader(string(raw))), nil); err != nil {
		t.Fatalf("factory with families: %v", err)
	}
}
