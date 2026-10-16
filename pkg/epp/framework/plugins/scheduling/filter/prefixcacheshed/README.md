# Prefix Cache Shed Filter

`prefix-cache-shed-filter` sheds cache-cold requests while the pool is near saturation.

Stability: Alpha.

## Rule

Let `S` be the pool saturation reported by the referenced saturation detector, `T` the shed threshold and `H` the minimum hit ratio.

- `S < T`: pass-through. The filter returns its input unchanged.
- `S >= T`: keep only endpoints that have room (the detector's `HasCapacity`) and a prefix cache hit ratio `>= H`. If none remains, the request is rejected.

A rejection carries a typed error. The HTTP status is `shedStatusCode` (503 or 429). The `x-llm-d-request-dropped-reason` header is:

| Reason | Meaning |
| --- | --- |
| `rejected-shed-cold` | No endpoint has a hit ratio `>= H`. |
| `rejected-shed-displaced-hot` | Some endpoint has a hit ratio `>= H`, but every such endpoint is full. |

Shedding has hysteresis: it starts at `S >= T` and stops at `S < T - hysteresis`.

## Placement

The filter must be the **first** filter of its profile. It reads `S` from the endpoint list it receives, and only the first filter receives the full pool. Placed after `concurrency-detector` it would compute `S` over the endpoints that still have room, which is lower than the pool saturation.

Below `T` the filter changes nothing, so the rest of the profile (including `concurrency-detector` and the affinity filters) behaves as before.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `saturationDetectorRef` | required | Name of a `concurrency-detector` instance. It supplies `S` and the per-endpoint capacity check. |
| `shedThreshold` | `0.85` | `T`, in `(0, 1]`. |
| `minHitRatio` | `0.6` | `H`, in `[0, 1]`. |
| `hysteresis` | `0.05` | Shedding stops at `S < T - hysteresis`. In `[0, T)`. |
| `minUncachedTokens` | `0` | A request that adds fewer uncached tokens than this on its cheapest endpoint with room is never shed. `0` disables. |
| `shedStatusCode` | `503` | `503` or `429`. |
| `prefixMatchInfoProducerName` | approx producer | Producer of `PrefixCacheMatchInfo`. |
| `inFlightLoadProducerName` | in-flight load producer | Producer of `InFlightLoad` and `UncachedRequestTokens`. Must match the detector's producer. |

## Example

```yaml
plugins:
  - type: approx-prefix-cache-producer
  - type: concurrency-detector
    name: concurrency-detector
    parameters:
      concurrencyMode: hybrid
      maxConcurrency: 32
      maxTokenConcurrency: 40000
  - type: inflight-load-producer
    name: inflight-load-producer
    parameters:
      prefixMatchInfoProducerName: approx-prefix-cache-producer
  - type: prefix-cache-shed-filter
    name: prefix-cache-shed-filter
    parameters:
      saturationDetectorRef: concurrency-detector
      shedThreshold: 0.85
      minHitRatio: 0.6
      hysteresis: 0.05
      prefixMatchInfoProducerName: approx-prefix-cache-producer
  - type: prefix-cache-affinity-filter
    parameters:
      affinityThreshold: 0.8
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-cache-shed-filter
      - pluginRef: concurrency-detector
      - pluginRef: prefix-cache-affinity-filter
      - pluginRef: max-score-picker
```

## Metrics

| Metric | Labels | Description |
| --- | --- | --- |
| `llm_d_router_endpoint_picker_prefix_cache_shed_filter_decisions_total` | `plugin_name`, `outcome` | One per `Filter` call. Outcomes: `pass_below_threshold`, `admitted_hot`, `shed_cold`, `shed_displaced_hot`, `bypass_small_request`, `not_applicable`. |
| `llm_d_router_endpoint_picker_prefix_cache_shed_filter_active` | `plugin_name` | `1` while shedding. |
| `llm_d_router_endpoint_picker_prefix_cache_shed_filter_saturation` | `plugin_name` | `S` read on the last decision. |

## Relation to `failOpen: false`

Both an empty result from this filter and an empty result from `concurrency-detector` with `failOpen: false` take the same path: `SchedulerProfile.Run` returns a typed error. The difference is the predicate. `failOpen: false` rejects when every endpoint is over its own limit for this request, at any pool saturation. This filter rejects on a pool-wide threshold and a per-request hit ratio, and never below `T`.

## Notes

- `S` is computed over the endpoints the scheduler received. Flow control computes its `S` over the same list before screeners run. With screeners configured, the two can differ.
- With more than one EPP replica, each replica sees only the in-flight load it dispatched. Share the detector limits across replicas as you do for flow control.
