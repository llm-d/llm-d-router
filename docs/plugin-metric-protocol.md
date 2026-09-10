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
| SGLang | `sglang:page_size` | — |
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
| SGLang | `sglang:num_pages` | — |
| Triton TensorRT-LLM | `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=max}` | — |
| trtllm-serve | `trtllm_kv_cache_max_blocks` | — |

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

## LoRA Adapter Residency

**Required by:** `lora-load-state-scorer`, `lora-residency-filter`

Model servers that keep a cache of loaded LoRA adapters can report which adapters are resident and
where, so the EPP can route an adapter's requests to servers that already hold it. Unlike the
`vllm:lora_requests_info` metric above, which lists adapters with in-flight requests, these metrics
describe the cache contents and so include idle adapters and adapters loaded at startup.

The model server SHOULD expose the following metrics via the same Prometheus endpoint:

* Metric name implemented in vLLM: `vllm:lora_adapter_loaded`
* Metric type: Gauge
* Metric value: `1` while the adapter is resident. A series at `0` is treated as absent.
* Metric labels:
  * `adapter_name`: The adapter's public model name, as used in the request's `model` argument.
  * `level`: `gpu` when the adapter occupies a GPU slot and can serve immediately, `cpu` when it is
    held only in the host cache.
  * `pinned`: `true` when the adapter is exempt from eviction.

and

* Metric name implemented in vLLM: `vllm:num_gpu_loaded_lora_adapters`
* Metric type: Gauge
* Metric value: The number of adapters occupying GPU slots. This gauge exists from startup, so its
  presence tells the EPP the server reports residency even when no adapter is loaded yet.
* Metric labels:
  * `model_name`: The served base model. A request whose `model` equals it needs no adapter, and
    the residency plugins treat it as not applicable.

and, for the GPU slot capacity,

* Metric name implemented in vLLM: `vllm:max_gpu_lora_adapters`
* Metric type: Gauge
* Metric value: The number of GPU adapter slots (`max_loras`). Present from startup.

When this gauge is absent, capacity falls back to the `max_lora` label of
`vllm:lora_requests_info`, which only appears once an adapter has served a request.

Availability: vLLM adds these gauges in [vllm-project/vllm#51433](https://github.com/vllm-project/vllm/pull/51433)
and [vllm-project/vllm#54830](https://github.com/vllm-project/vllm/pull/54830) (not in a tagged
release yet). Against a server without them, `LoadedModels` stays nil and the residency plugins
fall through to the capacity tiers, which score every such endpoint the same. The gauges update
when the model server finishes loading an adapter, so requests routed during the load itself
(hundreds of milliseconds for a 1 GB adapter) still see it as absent; the scorer's placement bonus
sends them to the same endpoint anyway.

## Prefix Cache Reuse

**Required by:** `precise-prefix-cache-producer`, `prefix-cache-scorer`, `prefix-cache-affinity-filter`

The EPP supports prefix cache optimized request scheduling via the
[precise prefix cache producer](../pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache/README.md).
To benefit from optimal prefix-aware request scheduling, model servers SHOULD support prefix
cache reuse, such as the [vllm automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) feature.
