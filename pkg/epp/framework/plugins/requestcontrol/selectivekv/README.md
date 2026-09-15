# Selective KV Policy

**Type:** `selective-kv-policy`

The plugin controls KV loading and offloading independently immediately before EPP dispatches an inference request to the selected endpoint. The loading policy can preserve the request, disable external KV loading, or compare the reusable prefix outside GPU memory with a static deployment-specific threshold. The offloading policy can preserve the request or disable offloading.

## Status and vLLM requirement

This Alpha plugin uses a deployment-specific reusable-token threshold. It can also stop loading when the selected endpoint's smoothed vLLM waiting queue reaches a configured limit.

The load control depends on [vLLM PR #55885](https://github.com/vllm-project/vllm/pull/55885) or a vLLM build with equivalent behavior. That API defines `kv_transfer_params.max_load_tokens: 0` as disabling external loads from the CPU primary tier and every secondary tier while preserving local GPU prefix-cache reuse and the independent store path. Omitting `max_load_tokens` preserves uncapped external loading.

## Configuration

Because the plugin is Alpha, the EPP process must include `--allow-experimental-plugins`. EPP initialization fails if the configuration contains this plugin without the flag.

```sh
epp \
  --allow-experimental-plugins \
  --config-file /etc/epp/config.yaml
```

The minimal pressure-aware threshold configuration is:

```yaml
- type: selective-kv-policy
  name: selective-kv-policy
  parameters:
    loadPolicy: threshold
    minExternalReusableTokens: 1024
    maxWaitingRequests: 8
```

For Qwen3-32B with tensor parallelism 2, four replicas on one node with eight H100 GPUs, and the tested CPU and NVMe offloading configuration, `maxWaitingRequests: 8` is the current experimental queue threshold. This value is specific to that deployment and is not a general default. The example combines it with the deployment's `minExternalReusableTokens: 1024` token threshold.

`prefixMatchInfoProducerName` defaults to `precise-prefix-cache-producer`, and `offloadPolicy` defaults to `preserve`, so neither field is needed in the common configuration. A precise prefix cache producer with its default name must still be present in the plugin chain.

The threshold is inclusive. If the selected endpoint has at least `minExternalReusableTokens` reusable tokens outside GPU memory, the plugin allows uncapped loading by removing `max_load_tokens`. If fewer external tokens are reusable, the plugin sets `max_load_tokens: 0` and the compatible vLLM backend recomputes the portion that is not already resident in GPU memory. Missing or invalid tier evidence fails open and preserves loading. A non-nil empty tier map is valid evidence of zero reusable blocks, so it falls below every valid threshold and disables loading.

The plugin derives external reusable tokens from the selected endpoint's precise prefix match. It subtracts the GPU-resident matched prefix from the longest matched prefix reported by a non-GPU tier and converts the remaining blocks to tokens. The comparison therefore represents reusable prefix length outside GPU memory, not transfer bytes, queue depth, or measured latency.

When `maxWaitingRequests` is greater than zero, the plugin applies an endpoint-local EWMA to `vllm:num_requests_waiting`. Loading closes when the EWMA reaches the configured value and reopens after it falls to half that value. The EWMA has a fixed two-second half-life. A missing endpoint identity or metrics update timestamp disables the queue veto for that request. Omitting `maxWaitingRequests` preserves the token-only behavior.

At DEBUG log verbosity, threshold mode records the external reusable tokens, configured threshold, and resulting load action. Missing endpoints and missing tier evidence have separate messages. Unsupported request shapes are skipped with a DEBUG message.

## Policies

The load policy accepts:

- `preserve` or omitted: leave `max_load_tokens` unchanged.
- `disable`: always set `max_load_tokens: 0`.
- `threshold`: load when the selected endpoint's external reusable tokens meet `minExternalReusableTokens` and its optional waiting-queue gate is open; otherwise set `max_load_tokens: 0`.

The offload policy accepts:

- `preserve` or omitted: leave `max_offload_tokens` unchanged.
- `disable`: set `max_offload_tokens: 0`.

At least one direction must use an active policy. The plugin rejects configurations where both policies are `preserve`, including an empty `parameters` object.

Loading and offloading decisions are independent. For example, explicitly disabling both directions produces:

```json
{
  "kv_transfer_params": {
    "max_load_tokens": 0,
    "max_offload_tokens": 0
  }
}
```

The plugin does not modify `kv_load_tiers`. That field continues to filter secondary tiers, while CPU remains available as a direct source and as the required staging tier for secondary-to-GPU promotion. `max_load_tokens: 0` disables loading regardless of the tier filter.

## Calibrating the static threshold (WIP)

Calibrate the threshold on the same model, accelerator and CPU topology, tensor parallelism, KV dtype, cache block and offload chunk sizes, connector configuration, and concurrency range used by the deployment. A threshold measured for another deployment is not portable because both recomputation cost and KV restoration cost change with those parameters.

Use paired trials that differ only in the load decision. Populate the external cache with a known prefix, clear or displace its GPU-resident KV while retaining its CPU or secondary copy, and issue one request that permits loading and another with `max_load_tokens: 0`. Keep the uncached suffix and generated-token count fixed. Verify the permitted arm reports external reuse and the disabled arm reports recomputation before comparing latency.

Sweep the reusable prefix length across a range that includes short prefixes where recomputation should win and long prefixes where restoration should win. Repeat each point enough times to obtain stable TTFT percentiles, and repeat the sweep at the deployment's expected concurrency levels. Record errors, cancellations, preemptions, local and external cached tokens, request TTFT, total latency, prefill throughput, and any available KV lookup and transfer latency or byte counters.

A useful first-order estimate is:

```
L_crossover ~= external restore latency at the target percentile / measured prefill time per token
```

This estimate treats restore latency as representative and is only a starting point. The more accurate decision compares the measured curves: loading is beneficial when lookup time, queueing, promotion, and CPU-to-GPU transfer together cost less than recomputing the reusable tokens. Choose `minExternalReusableTokens` as the smallest tested prefix length for which loading consistently improves the target TTFT percentile, then add a safety margin if measurements near the crossing are noisy.

Calibrate CPU-resident and secondary-resident cases separately when both matter. A secondary hit includes secondary-to-CPU promotion before CPU-to-GPU transfer, while a CPU-resident hit does not. The current policy receives a reusable-token count without source-specific restore cost, so one static threshold must be conservative enough for the sources it is allowed to load.

Recalibrate after changes to the model, hardware topology, parallelism, KV representation, offload chunking, storage backend, transfer implementation, or expected concurrency. Keep the calibration script and its raw results with the deployment configuration so the threshold can be reproduced.

## Potential improvements

The token threshold and waiting-queue gate do not measure load-specific transfer congestion.

Possible follow-up policies include:

- Consume pending load bytes and completed transfer throughput, then adjust the crossover estimate using load-specific transfer conditions.
- Estimate recomputation cost from model-specific prefill time per token and compare it directly with predicted lookup, promotion, queueing, and CPU-to-GPU transfer time.
- Require enough load-specific samples before allowing transfer telemetry to affect decisions.
- Track separate restore models for CPU-resident and secondary-resident prefixes once vLLM and the EPP data path expose reliable source provenance and source-specific cost.
- Include endpoint health and load in the decision so the gate does not restore KV through a saturated CPU or transfer queue merely because the prefix exceeds the static threshold.
- Periodically recalibrate from production observations with bounded defaults and fail-open behavior when measurements are missing or stale.

Dynamic gating should predict the cost of the action before changing the request. Metrics that only report completed transfers describe prior load conditions and need smoothing, freshness checks, and sufficient sample volume before they can safely influence routing.

## Request handling limitations

The plugin overwrites the fields controlled by its configured policies and preserves other existing `kv_transfer_params` fields. It supports parsed OpenAI-compatible JSON requests on the direct EPP path. Unsupported payloads, including native-generate and unparsed requests, fail open: the plugin logs the skip and leaves the request unchanged.

The coordinator prefill path currently replaces `kv_transfer_params` after this hook runs, so it discards this plugin's mutation. Selective KV policy is therefore not effective on that path until coordinator propagation is implemented.

The threshold decision uses the first endpoint in the scheduling result's primary profile. In a disaggregated deployment the primary profile is normally decode even though KV loading happens at prefill, so this implementation must not be used there unless the selected primary endpoint is also the endpoint whose tier evidence and load decision apply. Supporting disaggregated serving requires selecting the prefill profile explicitly.
