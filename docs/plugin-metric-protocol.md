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

Each metric below lists the plugins that need it under **Required by**. You only need to expose metrics for plugins you are using.

### TotalQueuedRequests

The current total number of requests in the queue.

- **Type:** Gauge
- **Required by:** `queue-scorer`, `load-aware-scorer`, `latency-scorer` (via `predicted-latency`)

| Model server | Metric |
| --- | --- |
| vLLM | `vllm:num_requests_waiting` |
| SGLang | `sglang:num_queue_reqs` |
| Triton TensorRT-LLM | `nv_trt_llm_request_metrics{request_type=waiting}` |
| trtllm-serve | `trtllm_num_requests_waiting` |

### TotalRunningRequests

The current total number of requests actively being served on the model server.

- **Type:** Gauge
- **Required by:** `running-requests-size-scorer`, `latency-scorer` (via `predicted-latency`)

| Model server | Metric |
| --- | --- |
| vLLM | `vllm:num_requests_running` |
| SGLang | `sglang:num_running_reqs` |
| Triton TensorRT-LLM | `nv_trt_llm_request_metrics{request_type=scheduled}` |
| trtllm-serve | `trtllm_num_requests_running` |

### KVCacheUtilization

The current KV cache utilization in percentage.

- **Type:** Gauge
- **Required by:** `kv-cache-utilization-scorer`, `latency-scorer` (via `predicted-latency`)

| Model server | Metric |
| --- | --- |
| vLLM | `vllm:kv_cache_usage_perc` |
| SGLang | `sglang:token_usage` |
| Triton TensorRT-LLM | `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=fraction}` |
| trtllm-serve | `trtllm_kv_cache_utilization` |

### BlockSize (optional)

The block size in tokens to allocate memory. Used to auto-tune the approximate prefix cache.
If absent, the value is taken from the `approximate-prefix` plugin's `BlockSizeTokens` config.

- **Type:** Labeled/Gauge
- **Required by:** `prefix-cache-scorer`, `prefix-cache-affinity-filter` (via `approximate-prefix` when `AutoTune` is enabled)

| Model server | Metric | Label |
| --- | --- | --- |
| vLLM | `vllm:cache_config_info` | `block_size` |
| SGLang | `sglang:cache_config_info` | `page_size` |
| Triton TensorRT-LLM | `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=tokens_per}` | — |
| trtllm-serve | `trtllm_kv_cache_tokens_per_block` | — |

### NumGPUBlocks (optional)

The total number of blocks in the HBM KV cache. Used to auto-tune the approximate prefix cache.
If absent, the value is taken from the `approximate-prefix` plugin's `LRUCapacityPerServer` config.

- **Type:** Labeled/Gauge
- **Required by:** `prefix-cache-scorer`, `prefix-cache-affinity-filter` (via `approximate-prefix` when `AutoTune` is enabled)

| Model server | Metric | Label |
| --- | --- | --- |
| vLLM | `vllm:cache_config_info` | `num_gpu_blocks` |
| SGLang | `sglang:cache_config_info` | `num_pages` |
| Triton TensorRT-LLM | `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=max}` | — |
| trtllm-serve | `trtllm_kv_cache_max_blocks` | — |

Note on trtllm-serve with host offloading (`kv_cache_config.host_cache_size`): under the default
(legacy) KV cache manager, `trtllm_kv_cache_max_blocks` already counts GPU plus host blocks, so
the auto-tuned capacity covers the offload-extended cache with no further configuration. Under
the opt-in `use_kv_cache_manager_v2` (also the default for some hybrid-Mamba model families),
the same gauge is GPU-only and the host tier is not observable via Prometheus; on such
deployments disable the `approx-prefix-cache-producer`'s `autoTune` and set
`lruCapacityPerServer` explicitly to avoid under-reporting prefix matches.

### TotalKVCacheTokens (optional)

The total effective KV cache capacity in tokens across all memory tiers (GPU HBM plus host/CPU
offload). When reported, it takes precedence over NumGPUBlocks for auto-tuning the approximate
prefix cache, so prefix-match estimates cover the offload-extended cache.

- **Type:** Gauge (the maximum across the configured per-tier gauges is used)
- **Required by:** `prefix-cache-scorer`, `prefix-cache-affinity-filter` (via `approximate-prefix` when `AutoTune` is enabled)

| Model server | Metric | Notes |
| --- | --- | --- |
| SGLang | `sglang:hicache_host_total_tokens`, `sglang:max_total_num_tokens` | The hicache host pool (SGLang v0.5.10+, hierarchical cache enabled) is an inclusive superset of the device pool under the default `write_through` policy, so the effective total is the max of the two gauges, not their sum. On older SGLang versions only the device gauge exists; disable `autoTune` and set `lruCapacityPerServer` to `hicache-ratio x device tokens / page size` to account for the host tier. |
| vLLM | — | Not reported. vLLM exposes no offload-tier capacity metric (`num_cpu_blocks` is always `None` on V1). See KVCacheOffloadDetection below. |
| trtllm-serve | — | Not needed under the legacy KV cache manager (NumGPUBlocks already includes host blocks); not available under the V2 manager. |

### KVCacheOffloadDetection (optional)

Metrics whose presence indicates the model server offloads KV cache to a tier whose capacity is
not reported. Detection does not change auto-tuned capacity by itself; it drives an operator
warning that prefix matches will be under-reported unless `autoTune` is disabled and
`lruCapacityPerServer` is set.

| Model server | Signal |
| --- | --- |
| vLLM | Presence of any `vllm:kv_offload_*` OffloadingConnector series (registered at startup, before traffic), or a numeric `kv_offloading_size` label (GiB) on `vllm:cache_config_info` (only populated by the built-in `--kv-offloading-size` flag; hand-written `--kv-transfer-config` deployments leave it `None`). `vllm:external_prefix_cache_*` is deliberately not used: it is registered even without a connector. |

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
  * `waiting_lora_adapters`: A comma separated list of adapters that are waiting to be served.
    Example: `"waiting_lora_adapters": "adapter1, adapter2"`

## Prefix Cache Reuse

**Required by:** `precise-prefix-cache-producer`, `prefix-cache-scorer`, `prefix-cache-affinity-filter`

The EPP supports prefix cache optimized request scheduling via the
[precise prefix cache producer](../pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache/README.md).
To benefit from optimal prefix-aware request scheduling, model servers SHOULD support prefix
cache reuse, such as the [vllm automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) feature.
