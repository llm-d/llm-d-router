# In-Flight Load Producer Plugin

**Type:** `inflight-load-producer`

Tracks real-time in-flight request and token counts per endpoint by hooking into the request lifecycle. Writes an `InFlightLoad` attribute onto each endpoint in the `Produce` phase, consumed by the following plugins:
- `token-load-scorer`: Scores endpoints based on in-flight tokens.
- `active-request-scorer`: Scores endpoints based on in-flight requests.
- `concurrency-detector`: Provides admission control based on in-flight requests/tokens.
- `prefix-cache-affinity-filter`: Uses in-flight tokens as a load gate to break stickiness.

## Behavior

- **Prefix Cache Discounting**: Automatically detects if an endpoint has a prefix cache hit (via `PrefixCacheMatchInfo`). Only the **uncached** portion of the prompt is added to the in-flight token counter, providing a more accurate estimate of the actual compute load. The discount uses the cached block count, so a block cached in CPU RAM counts the same as one in GPU memory.
- **Token Release Timing**:
    - If `addEstimatedOutputTokens` is `false` (default): All tokens are released at prefill completion, signaled by a matching KV event or the first response chunk (`StartOfStream`). `EndOfStream` releases any remaining tokens when neither earlier signal arrives.
    - If `addEstimatedOutputTokens` is `true`: Prefill targets' tokens are released at prefill completion. Other profiles retain their full charged amount until `EndOfStream`, including input on a primary target that has no selected prefiller.
- **Request Release**: Prefill targets' request counters are released at prefill completion. Other profiles' request counters are released at `EndOfStream`. Contributions are tracked separately by profile even when profiles select the same endpoint.
- **Prefill Completion from KV Events**: With `kvEventsSourceRef` the producer consumes the engines' KV cache events from that `kv-events-source`. It stamps each dispatched request with a session id (`X-Session-ID` header set to the request id, unless the client supplied `session_id`, that header, or `vllm_xargs.session_id`), which vLLM echoes on the `BlockStored` events for the blocks the request stores. The event that carries the prompt's last full block from the engine that decodes the request releases what `StartOfStream` releases: the prefill profile's entry in full and, when output tokens are excluded, the other profiles' tokens. In disaggregated serving that engine is the decode target, which stores the blocks once it has pulled them from the prefill engine, so the prefill entry covers the prefill compute and the time the prefill engine holds the blocks for the transfer; a single engine serving both phases stores the blocks as it computes them. For a request the producer stamped, a store of a block outside the prompt (the prompt's tail with generated tokens) releases as well, since the engine generates only after reading the prompt; this covers prompts the decode engine already held in full and stored nothing new for. A client-supplied session id may be shared by concurrent requests, so those release only on their own prompt's last full block. The release is the same for streaming and non-streaming requests. `StartOfStream` remains the fallback for prompts shorter than one block, for fully cached prompts under a client-supplied session id, and for endpoints whose event stream was reset. The counter `inflight_prefill_completions_total{signal="kv_event"|"first_byte"}` reports which signal completed each request, and the gauge `inflight_prefill_pending` the requests currently awaiting an event; a request leaves it at either signal or when its in-flight entries are evicted, so the gauge returns to zero when the pool is idle.

The producer hooks three lifecycle phases:
- **Produce**: Writes current in-flight counts to each endpoint's attributes.
- **PreRequest**: Increments counters when a request is dispatched to an endpoint.
- **ResponseBody**: Decrements counters when a response completes or the request is aborted.

Endpoint departure events (pod removed from the pool) are handled via the `EndpointExtractor` interface to clean up stale counters.

When the scheduling result leaves `PrefillProfileName` empty, targets labeled `llm-d.ai/role=prefill` or `llm-d.ai/role=encode-prefill` use prefill completion for load release and are excluded from non-prefill KV-event sources. An explicit prefill profile name takes precedence over role labels. Unlabeled and combined-role targets require an explicit prefill profile name to receive prefill treatment.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addEstimatedOutputTokens` | `bool` | No | `false` | If true, adds an estimate of the generated output work to the in-flight counter. Text estimates are read from the output-length bucket published by the `outlen-bucket` plugin; enable that plugin and order it before this producer so requests are classified. Image requests use diffusion work instead. |
| `maxEstimatedOutputTokens` | `int` | No | _(none)_ | Optional upper bound on the text output estimate added per request when `addEstimatedOutputTokens` is true. Must be non-negative. Unset means no cap. Image diffusion work is not capped. |
| `prefixMatchInfoProducerName` | `string` | No | _(none)_ | Optional `prefix-cache producer` name to read to find cached prefix discount. Unset defaults to approximate-prefix producer. |
| `syncCrossReplicaState` | `bool` | No | `true` | When a `CrossReplicaSyncer` is configured, publishes this producer's per-endpoint load and adds the other replicas' published load to the value its consumers read. Set `false` to count only this replica's requests. See [Multiple active replicas](../../../../../flowcontrol/README.md#multiple-active-replicas). |
| `kvEventsSourceRef` | `string` | No | _(none)_ | Name of the `kv-events-source` covering the engines that decode the requests (the decode engines in disaggregated serving). With it set, the producer stamps requests with a session id and completes their prefill accounting from those engines' `BlockStored` events. Requires a `token-producer` that emits exact engine tokens (`exact_tokens: "true"`, provided by `backend: vllm` or `backend: passthrough`): the match compares the request's token ids with the event's, so the producer requires that capability on the tokenized request and startup rejects `backend: estimate`. The engines have to run prefix caching with KV cache events enabled and echo the session id on `BlockStored`; requests on engines that do not complete at the first response byte. |

When `addEstimatedOutputTokens` is true, a text request's estimated output is determined
by the output-length bucket published by the `outlen-bucket` plugin:

| Output-Length Bucket | Estimated output tokens |
|------------|------------------------|
| `LONG` (reasoning chains) | 4 096 |
| `SHORT` (tool-call JSON) | 100 |
| `UNKNOWN` (no reliable signal) | running average of the completion tokens of finished UNKNOWN requests, weighted toward roughly the last 100 (1 000 until the first one) |

For text requests, the estimate is then bounded by the client-requested cap
(`max_output_tokens` / `max_tokens`) and `maxEstimatedOutputTokens`. The UNKNOWN
average is kept within [100, 4 096], so SHORT <= UNKNOWN <= LONG.
Image requests are estimated as diffusion work (`num_inference_steps` x output
pixels x `n`, in token-equivalent units); they are not read from the bucket and
are not bounded by these caps. Layered image requests denoise `layers + 1`
frames and use `resolution`, or the 640px layered default, when `size` is
absent or `auto`.
When the `outlen-bucket` plugin is not enabled, every request reads as UNKNOWN and the producer
logs a one-time warning.

---

## Related Documentation
- [Token Load Scorer](../../../scheduling/scorer/tokenload/README.md)
- [Active Request Scorer](../../../scheduling/scorer/activerequest/README.md)
- [Concurrency Detector](../../../flowcontrol/saturationdetector/concurrency/README.md)
- [Prefix Cache Affinity Filter](../../../scheduling/filter/prefixcacheaffinity/README.md)
- [Concurrency Attributes](../../../datalayer/attribute/concurrency/README.md)

**Configuration Example:**
```yaml
plugins:
  - type: inflight-load-producer
    name: inflight-load
    parameters:
      addEstimatedOutputTokens: true
```

**Prefill completion from KV events** in disaggregated serving, sharing the decode engines' event stream with a precise prefix-cache producer:
```yaml
plugins:
  - type: kv-events-source
    name: decode-events
    parameters:
      podDiscoveryConfig:
        podLabelSelector: llm-d.ai/role=decode
        socketPort: 5557
        snapshotSocketPort: 6557
  - type: precise-prefix-cache-producer
    name: decode-prefix-cache
    parameters:
      kvEventsSourceRef: decode-events
  - type: inflight-load-producer
    name: inflight-load
    parameters:
      kvEventsSourceRef: decode-events
```
A deployment without a prefill profile names the source covering its serving engines instead.
