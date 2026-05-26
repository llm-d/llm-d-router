# Plugin Metric Protocol

This document describes the contract the EPP expects from model servers
it routes traffic to. Because the EPP uses a pluggable architecture, the
requirements below describe what is needed to use the built-in plugins;
individual plugins may relax or extend these requirements.

## Metrics Reporting

The inference extension scrapes metrics from the model servers to make optimal request scheduling
decisions. The model servers MUST provide the following metrics via a Prometheus endpoint. The exact
metric names don't necessarily need to be the same as the recommended names here, however the
metric types and semantics MUST follow this doc.

The **Required by** column lists the plugins that need each metric to function. You only need
to expose the metrics for plugins you actually enable.

Note the requirements here are aligned with the
[model server metrics standardization](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
effort.


| Metric | Required by | Type | Description | vLLM metric | Triton TensorRT-LLM | trtllm-serve | SGLang |
| ----- | ---- | ---- | ------------ | ---- | ---- | ---- | ---- |
| TotalQueuedRequests         | `queue-scorer`, `load-aware-scorer`, `latency-scorer` (via `predicted-latency`) | Gauge     | The current total number of requests in the queue.| `vllm:num_requests_waiting`| `nv_trt_llm_request_metrics{request_type=waiting}`| `trtllm_num_requests_waiting` | `sglang:num_queue_reqs`
| TotalRunningRequests         | `running-requests-size-scorer`, `latency-scorer` (via `predicted-latency`) | Gauge     | The current total number of requests actively being served on the model server.| `vllm:num_requests_running`| `nv_trt_llm_request_metrics{request_type=scheduled}`| `trtllm_num_requests_running` | `sglang:num_running_reqs`
| KVCacheUtilization| `kv-cache-utilization-scorer`, `latency-scorer` (via `predicted-latency`) | Gauge     | The current KV cache utilization in percentage.| `vllm:kv_cache_usage_perc`| `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}`| `trtllm_kv_cache_utilization` | `sglang:token_usage`
| [Optional] BlockSize         | `prefix-cache-scorer`, `prefix-cache-affinity-filter` (via `approximate-prefix` when `AutoTune` is enabled) | Labeled/Gauge     | The block size in tokens to allocate memory. Used to auto-tune the approximate prefix cache; otherwise the value is taken from the `approximate-prefix` plugin's `BlockSizeTokens` config.| name: `vllm:cache_config_info`, label name: `block_size`| `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=tokens_per}` | `trtllm_kv_cache_tokens_per_block` | name: `sglang:cache_config_info`, label name: `page_size`
| [Optional] NumGPUBlocks| `prefix-cache-scorer`, `prefix-cache-affinity-filter` (via `approximate-prefix` when `AutoTune` is enabled) | Labeled/Gauge     | The total number of blocks in the HBM KV cache. Used to auto-tune the approximate prefix cache.| name: `vllm:cache_config_info`, label name: `num_gpu_blocks`| `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=max}` | `trtllm_kv_cache_max_blocks` | name: `sglang:cache_config_info`, label name: `num_pages`


## LoRA Adapter Serving

**Required by:** `lora-affinity-scorer`

Model servers that support dynamic LoRA serving can benefit from the LoRA affinity algorithm. Note
the current LoRA affinity algorithm in this EPP is highly biased towards vLLM's current
dynamic LoRA implementation.

The model servers MUST support serving a LoRA adapter specified in the `model` argument of the
request, provided the requested adapter is valid.

The model server MUST expose the following LoRA adapter metrics via the same Prometheus endpoint:

* Metric name implemented in vLLM: `vllm:lora_requests_info` 
* Metric type: Gauge
* Metric value: The last updated timestamp (so the EPP can find the latest).
* Metric labels: 
  * `max_lora`: The maximum number of adapters that can be loaded to GPU memory to serve a batch.
  Requests will be queued if the model server has reached MaxActiveAdapter and cannot load the
  requested adapter. Example: `"max_lora": "8"`.
  * `running_lora_adapters`: A comma separated list of adapters that are currently loaded in GPU
    memory and ready to serve requests. Example: `"running_lora_adapters": "adapter1, adapter2"`
  * `waiting_lora_adapters`: A comma separated list of adapters that are waiting to be served. Example: `"waiting_lora_adapters": "adapter1, adapter2"`

## Prefix Cache Reuse

**Required by:** `precise-prefix-cache-scorer`, `prefix-cache-scorer`, `prefix-cache-affinity-filter`

The EPP supports prefix cache optimized request scheduling via the
[precise prefix cache plugin](../pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache/README.md).
To benefit from optimal prefix-aware request scheduling, model servers SHOULD support prefix
cache reuse, such as the [vllm automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) feature.
