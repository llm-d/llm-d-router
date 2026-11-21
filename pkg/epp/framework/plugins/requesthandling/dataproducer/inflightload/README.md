# In-Flight Load Producer Plugin

**Type:** `inflight-load-producer`

Tracks real-time in-flight request and token counts per endpoint by hooking into the request lifecycle. Writes an `InFlightLoad` attribute onto each endpoint in the `Produce` phase, consumed by the following plugins:
- `token-load-scorer`: Scores endpoints based on in-flight tokens.
- `active-request-scorer`: Scores endpoints based on in-flight requests.
- `concurrency-detector`: Provides admission control based on in-flight requests/tokens.
- `prefix-cache-affinity-filter`: Uses in-flight tokens as a load gate to break stickiness.
- `utilization-detector`: Corrects its queue depth for scrape lag with the in-flight requests, and counts the projected KV cache growth toward saturation.

## Behavior

- **Prefix Cache Discounting**: Automatically detects if an endpoint has a prefix cache hit (via `PrefixCacheMatchInfo`). Only the **uncached** portion of the prompt is added to the in-flight token counter, providing a more accurate estimate of the actual compute load. The discount uses the cached block count, so a block cached in CPU RAM counts the same as one in GPU memory.
- **Token Release Timing**: 
    - If `addEstimatedOutputTokens` is `false` (default): For streaming requests, all tokens are released as soon as the first chunk of the response is received (`StartOfStream`), as the prefill compute is complete. For non-streaming requests (or as a safety net), tokens are released when the response completes (`EndOfStream`).
    - If `addEstimatedOutputTokens` is `true`: The prompt portion is released at `StartOfStream` (for streaming) or `EndOfStream`, and the estimated output portion is released only when the response completes (`EndOfStream`).
- **Request Release**: In-flight request counters are always released when the response completes (`EndOfStream`).
- **Prefill Completion from KV Events**: With `kvEventsSourceRef` the producer consumes the engines' KV cache events from that `kv-events-source`. It stamps each dispatched request with a session id (`X-Session-ID` header set to the request id, unless the client supplied `session_id`, that header, or `vllm_xargs.session_id`), which vLLM echoes on the `BlockStored` events for the blocks the request stores. The event that carries the prompt's last full block from the engine that decodes the request releases what `StartOfStream` releases: the prefill profile's entry in full and, when output tokens are excluded, the other profiles' tokens. In disaggregated serving that engine is the decode target, which stores the blocks once it has pulled them from the prefill engine, so the prefill entry covers the prefill compute and the time the prefill engine holds the blocks for the transfer; a single engine serving both phases stores the blocks as it computes them. For a request the producer stamped, a store of a block outside the prompt (the prompt's tail with generated tokens) releases as well, since the engine generates only after reading the prompt; this covers prompts the decode engine already held in full and stored nothing new for. A client-supplied session id may be shared by concurrent requests, so those release only on their own prompt's last full block. The release is the same for streaming and non-streaming requests. `StartOfStream` remains the fallback for prompts shorter than one block, for fully cached prompts under a client-supplied session id, and for endpoints whose event stream was reset. The counter `inflight_prefill_completions_total{signal="kv_event"|"first_byte"}` reports which signal completed each request, and the gauge `inflight_prefill_pending` the requests currently awaiting an event; a request leaves it at either signal or when its in-flight entries are evicted, so the gauge returns to zero when the pool is idle.
- **KV Cache Growth Projection**: With `kvGrowthProjection` the producer also publishes `KVGrowthTokens`, the peak of the KV cache tokens the endpoint is projected to need beyond its scraped usage, for the requests in flight and one request dispatched next.
    - A request reserves its whole prompt, cached or not, until the projection has read the endpoint's KV cache usage from two scrapes after the request's first response chunk. vLLM counts cached blocks as free until a request takes them, and the first scrape to complete after the chunk may have read the usage before it. The tokens a request streams stay reserved in the same way until the scraped usage includes them, so a usage that is not scraped for a while does not read as free room.
    - A running request holds the prompt tokens the endpoint had not cached plus the tokens it has streamed, keeps growing while it decodes and frees them when it ends. A request with an output limit of its own (`max_tokens`) below `maxOutputTokens` is assumed to stop there. A request generates one sequence for each choice (`n`) of each prompt it carries (`/v1/completions` takes a list of prompts): its output limit counts once for each sequence, and its prompt tokens once for each choice. `n` is read from a JSON request body. A request in another encoding, such as vLLM gRPC, counts as one choice.
    - How long a request still runs is read from the output lengths of the most recently dispatched requests (Kaplan-Meier): the last 200, or 2.5 for every running request when that is more. A response the model server completed counts at its completion tokens, or at its stream events when it carries no usage block. A running request, a request cut off by an output limit of its own, and a stream that ended early (client disconnect, eviction, error) count as longer than what they streamed. A request that generated several sequences counts as longer than their mean length, since only the total over its sequences is reported. A pool whose streams all end early, or whose requests all generate several sequences, therefore reserves every sequence its output up to the cap. A request that ends without output, such as an error status, is not sampled.
    - Assumed requests that run to `maxOutputTokens` are part of the sample: twenty, or one for every ten sampled requests when that is more. Before any request completes, every request is reserved its output up to the cap.
    - The projection is the largest value, over the tokens still to be generated, of the expected change in occupancy plus `z` standard deviations, with the requests taken as independent. It does not exceed what the running requests and the next one can still generate under their output limits.
    - The projection is charged to the endpoint of the primary scheduling profile, unless its role has no decode stage (`prefill`, `encode`, `encode-prefill`). A request that generates no tokens (embeddings) is not charged. It is zero while no request is running on the endpoint. It is recomputed when a request is dispatched to the endpoint or ends on it, and otherwise every 100 ms. A request stays in the projection of its endpoint until its stream ends, also when the endpoint leaves the pool and returns, as a pod that fails its readiness probe does.
    - Progress is the count of stream events, or the completion tokens a chunk reports when that is more. A non-streaming request reports neither until it completes: for as long as it runs its prompt stays reserved, and its remaining output is taken from the whole sample. A model server that sends several tokens per stream event is under-counted until its usage block arrives.
    - Prompt tokens are those of the tokenized request. A request that was not tokenized reserves no prompt.
    - With a `CrossReplicaSyncer` the replicas' projections are summed, which reserves more than one projection over all their requests would.

The producer hooks three lifecycle phases:
- **Produce**: Writes current in-flight counts to each endpoint's attributes.
- **PreRequest**: Increments counters when a request is dispatched to an endpoint.
- **ResponseBody**: Decrements counters when a response completes or the request is aborted.

Endpoint departure events (pod removed from the pool) are handled via the `EndpointExtractor` interface to clean up stale counters.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addEstimatedOutputTokens` | `bool` | No | `false` | If true, adds an estimate of the generated output tokens to the in-flight counter. The estimate is read from the output-length bucket published by the `outlen-bucket` plugin; enable that plugin and order it before this producer so requests are classified. |
| `maxEstimatedOutputTokens` | `int` | No | _(none)_ | Optional upper bound on the estimated output tokens added per request when `addEstimatedOutputTokens` is true. Must be non-negative. Unset means no cap. |
| `prefixMatchInfoProducerName` | `string` | No | _(none)_ | Optional `prefix-cache producer` name to read to find cached prefix discount. Unset defaults to approximate-prefix producer. |
| `syncCrossReplicaState` | `bool` | No | `true` | When a `CrossReplicaSyncer` is configured, publishes this producer's per-endpoint load and adds the other replicas' published load to the value its consumers read. Set `false` to count only this replica's requests. See [Multiple active replicas](../../../../../flowcontrol/README.md#multiple-active-replicas). |
| `kvEventsSourceRef` | `string` | No | _(none)_ | Name of the `kv-events-source` covering the engines that decode the requests (the decode engines in disaggregated serving). With it set, the producer stamps requests with a session id and completes their prefill accounting from those engines' `BlockStored` events. Requires a `token-producer` that emits exact engine tokens (`exact_tokens: "true"`, provided by `backend: vllm` or `backend: passthrough`): the match compares the request's token ids with the event's, so the producer requires that capability on the tokenized request and startup rejects `backend: estimate`. The engines have to run prefix caching with KV cache events enabled and echo the session id on `BlockStored`; requests on engines that do not complete at the first response byte. |
| `kvGrowthProjection` | `object` | No | _(none)_ | Enables the KV cache growth projection. |
| `kvGrowthProjection.maxOutputTokens` | `int` | With `kvGrowthProjection` | _(none)_ | Output length a request is assumed to stop at when it sets no smaller limit of its own. Set it to the longest output the model server generates: output past it is not reserved for, and a value far above the real outputs reserves more than needed, because the assumed requests run to it. Must be positive. |
| `kvGrowthProjection.z` | `float` | No | `3` | Standard deviations added to the projected mean. Must be non-negative. |

When `addEstimatedOutputTokens` is true, the estimated output per request is determined
by the output-length bucket published by the `outlen-bucket` plugin:

| Output-Length Bucket | Estimated output tokens |
|------------|------------------------|
| `LONG` (reasoning chains) | 4 096 |
| `SHORT` (tool-call JSON) | 100 |
| `UNKNOWN` (no reliable signal) | running average of the completion tokens of finished UNKNOWN requests, weighted toward roughly the last 100 (1 000 until the first one) |

The estimate is then bounded by the client-requested cap (`max_output_tokens` / `max_tokens`)
and `maxEstimatedOutputTokens`. The UNKNOWN average is kept within [100, 4 096], so
SHORT <= UNKNOWN <= LONG.
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

**KV cache growth projection**, which the `utilization-detector` counts toward saturation. The detector reads it from the producer its `inFlightLoadProducerName` names, and requests wait for the projected need only with [flow control](../../../../../flowcontrol/README.md) enabled:
```yaml
plugins:
  - type: inflight-load-producer
    parameters:
      kvGrowthProjection:
        maxOutputTokens: 4096
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
