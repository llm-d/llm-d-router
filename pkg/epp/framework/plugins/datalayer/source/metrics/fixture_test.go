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
	"fmt"
	"strings"
	"sync"
)

// vllmConsumedFamilies is what the vLLM mapping in the metrics extractor reads.
// The families a scrape exposes beyond these are parsed today and discarded.
var vllmConsumedFamilies = []string{
	"vllm:num_requests_waiting",
	"vllm:num_requests_running",
	"vllm:kv_cache_usage_perc",
	"vllm:lora_requests_info",
	"vllm:cache_config_info",
}

var (
	fixtureOnce sync.Once
	fixture     string
)

// vllmScrapeFixture approximates a vLLM /metrics response: the five families the
// endpoint picker reads, plus the counters, histograms and process metrics it
// does not. Shape and proportions follow a single-model vLLM server; a server
// hosting several models or LoRA adapters emits proportionally more of the
// families that are discarded.
func vllmScrapeFixture() string {
	fixtureOnce.Do(func() { fixture = buildVLLMScrape() })
	return fixture
}

func buildVLLMScrape() string {
	var b strings.Builder
	const labels = `{model_name="meta-llama/Llama-3.1-8B-Instruct",engine="0"}`

	gauge := func(name, value, lbl string) {
		fmt.Fprintf(&b, "# HELP %s help text.\n# TYPE %s gauge\n%s%s %s\n", name, name, name, lbl, value)
	}
	gauge("vllm:num_requests_running", "12.0", labels)
	gauge("vllm:num_requests_waiting", "3.0", labels)
	gauge("vllm:kv_cache_usage_perc", "0.4213", labels)
	gauge("vllm:lora_requests_info", "1.0",
		`{max_lora="4",running_lora_adapters="adapter-a,adapter-b",waiting_lora_adapters=""}`)
	gauge("vllm:cache_config_info", "1.0", `{block_size="16",num_gpu_blocks="32768"}`)

	for _, name := range []string{
		"vllm:prompt_tokens_total", "vllm:generation_tokens_total",
		"vllm:num_preemptions_total", "vllm:request_success_total",
		"vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total",
		"vllm:num_requests_swapped", "vllm:cpu_cache_usage_perc",
		"vllm:gpu_cache_usage_perc", "vllm:iteration_tokens_total_sum",
	} {
		fmt.Fprintf(&b, "# HELP %s help text.\n# TYPE %s counter\n%s%s 123456.0\n", name, name, name, labels)
	}

	buckets := []string{
		"0.001", "0.005", "0.01", "0.02", "0.04", "0.06", "0.08", "0.1",
		"0.25", "0.5", "0.75", "1.0", "2.5", "5.0", "7.5", "10.0", "20.0",
		"40.0", "80.0", "+Inf",
	}
	for _, name := range []string{
		"vllm:time_to_first_token_seconds", "vllm:time_per_output_token_seconds",
		"vllm:e2e_request_latency_seconds", "vllm:request_queue_time_seconds",
		"vllm:request_inference_time_seconds", "vllm:request_prefill_time_seconds",
		"vllm:request_decode_time_seconds", "vllm:request_prompt_tokens",
		"vllm:request_generation_tokens", "vllm:request_max_num_generation_tokens",
		"vllm:request_params_n", "vllm:request_params_max_tokens",
		"vllm:iteration_tokens_total", "vllm:time_in_queue_requests",
		"vllm:model_forward_time_milliseconds",
	} {
		fmt.Fprintf(&b, "# HELP %s help text.\n# TYPE %s histogram\n", name, name)
		for i, le := range buckets {
			fmt.Fprintf(&b, "%s_bucket{model_name=%q,engine=\"0\",le=%q} %d.0\n",
				name, "meta-llama/Llama-3.1-8B-Instruct", le, (i+1)*37)
		}
		fmt.Fprintf(&b, "%s_sum%s 4321.5\n%s_count%s 740.0\n", name, labels, name, labels)
	}

	for _, name := range []string{
		"python_gc_objects_collected_total", "python_gc_objects_uncollectable_total",
		"python_gc_collections_total", "process_virtual_memory_bytes",
		"process_resident_memory_bytes", "process_start_time_seconds",
		"process_cpu_seconds_total", "process_open_fds", "process_max_fds",
	} {
		fmt.Fprintf(&b, "# HELP %s help.\n# TYPE %s counter\n%s{generation=\"0\"} 99.0\n", name, name, name)
	}

	return b.String()
}
