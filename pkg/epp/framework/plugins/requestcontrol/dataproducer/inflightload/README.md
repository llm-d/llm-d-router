# In-Flight Load Producer Plugin

**Type:** `inflight-load-producer`

Tracks real-time in-flight request and token counts per endpoint by hooking into the request lifecycle. Writes an `InFlightLoad` attribute onto each endpoint in the `Produce` phase, consumed by the following plugins:
- `token-load-scorer`: Scores endpoints based on in-flight tokens.
- `active-request-scorer`: Scores endpoints based on in-flight requests.
- `concurrency-detector`: Provides admission control based on in-flight requests/tokens.
- `prefix-cache-affinity-filter`: Uses in-flight tokens as a load gate to break stickiness.

## Behavior

- **Prefix Cache Discounting**: Automatically detects if an endpoint has a prefix cache hit (via `PrefixCacheMatchInfo`). Only the **uncached** portion of the prompt is added to the in-flight token counter, providing a more accurate estimate of the actual compute load.
- **Token Release Timing**: 
    - If `addEstimatedOutputTokens` is `false` (default): For streaming requests, all tokens are released as soon as the first chunk of the response is received (`StartOfStream`), as the prefill compute is complete. For non-streaming requests (or as a safety net), tokens are released when the response completes (`EndOfStream`).
    - If `addEstimatedOutputTokens` is `true`: The prompt portion is released at `StartOfStream` (for streaming) or `EndOfStream`, and the estimated output portion is released only when the response completes (`EndOfStream`).
- **Request Release**: In-flight request counters are always released when the response completes (`EndOfStream`).

The producer hooks three lifecycle phases:
- **Produce**: Writes current in-flight counts to each endpoint's attributes.
- **PreRequest**: Increments counters when a request is dispatched to an endpoint.
- **ResponseBody**: Decrements counters when a response completes or the request is aborted.

Endpoint departure events (pod removed from the pool) are handled via the `EndpointExtractor` interface to clean up stale counters.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addEstimatedOutputTokens` | `bool` | No | `false` | If true, adds an estimate of the generated output tokens to the in-flight counter. The estimate comes from the configured output estimator (see `outputEstimator`). |
| `maxEstimatedOutputTokens` | `int` | No | _(none)_ | Optional upper bound on the estimated output tokens added per request when `addEstimatedOutputTokens` is true. Must be non-negative. Unset means no cap. |
| `prefixMatchInfoProducerName` | `string` | No | _(none)_ | Optional `prefix-cache producer` name to read to find cached prefix discount. Unset defaults to approximate-prefix producer. |
| `outputEstimator` | `string` | No | `static` | Selects how output tokens are estimated: `static` maps the `outlen-bucket` signal rules to a flat estimate; `sketch` uses the learned output-length sketch (more compute, more precise, self-adapting). The two are mutually exclusive. |
| `sketch` | `object` | When `outputEstimator: sketch` | _(none)_ | Learned-estimator configuration (see below). |

### `outputEstimator: static` vs `sketch`

The output-token estimate feeds the in-flight token load. Two estimators are available; an operator chooses one:

- **`static`** (default) maps the three-way `outlen-bucket` signal (LONG/SHORT/UNKNOWN) to a flat token estimate (4096 / 100 / 1000). Negligible compute; abstains (UNKNOWN) whenever the request carries no explicit length signal. Enable the `outlen-bucket` plugin for it to classify.
- **`sketch`** predicts a per-request output-token magnitude from the prompt itself with the learned [output-length sketch](outlensketch/README.md): window the prompt, embed it with a static table-lookup embedding (CPU-only, no model server), assign it to an online k-means centroid, fold that with the request signals into a key, and read the key's count-decaying output-length histogram. It needs no training and keeps adapting to the live workload. A key that has not yet seen enough traffic abstains and falls back to the static estimate, so it self-heals from a cold start.

| `sketch` field | Type | Required | Default | Description |
|----------------|------|----------|---------|-------------|
| `modelDir` | `string` | Yes | _(none)_ | Mounted directory holding the static-embedding model (`vocab.txt` + `model.safetensors`). |
| `quantile` | `float` | No | `0.85` | Over-provision quantile read from each key's histogram. Higher is more conservative about long requests. |
| `decayHalfLifeRequests` | `int` | No | `200` | Per-key count-decay half-life, in observations of that key. |
| `centroids` | `int` | No | `64` | Number of AdaptiveKey k-means centroids. |
| `maxKeys` | `int` | No | `100000` | LRU safety cap on the histogram store. |

When `addEstimatedOutputTokens` is true, the estimated output per request is a flat
value determined by the output-length bucket published by the `outlen-bucket` plugin:

| Output-Length Bucket | Estimated output tokens |
|------------|------------------------|
| `LONG` (reasoning chains) | 4 096 |
| `SHORT` (tool-call JSON) | 100 |
| `UNKNOWN` (no reliable signal) | 1 000 |

The estimate is then bounded by the client-requested cap (`max_output_tokens` / `max_tokens`)
and `maxEstimatedOutputTokens`. Ranking invariant: SHORT (100) < UNKNOWN (1 000) < LONG (4 096).
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
