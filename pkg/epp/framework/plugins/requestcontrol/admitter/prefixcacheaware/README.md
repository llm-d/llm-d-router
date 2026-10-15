# Prefix Cache Aware Admitter

`prefix-cache-aware-admitter` implements cache-aware load shedding: while the pool is near saturation it admits only requests that have a prefix cache hit on some endpoint, and denies the rest with the same response flow control uses for a saturated pool.

Stability: Alpha.

## Rule

Let `S` be the pool saturation reported by the referenced saturation detector, `T` the shed threshold and `H` the minimum hit ratio.

- `S < T`: every request is admitted.
- `S >= T`: a request is admitted only if some endpoint has a prefix cache hit ratio `>= H` for it. Otherwise the request is denied.

A denial looks to the client exactly like flow control's own saturation backpressure: HTTP `shedStatusCode` (429 by default, 503 optional), header `x-llm-d-request-dropped-reason: rejected-saturated`, body `pool saturated, request rejected`. Shedding is a platform decision; nothing about the request's cache state is exposed to the client. The decision inputs (`saturation`, `bestHitRatio`) are in the debug log, and the outcome is in the plugin's metrics.

The decision is stateless: it depends only on `S` and the request's hit ratios at that moment. There is no hysteresis; the threshold is a plain comparison, so the plugin never rejects while `S` is below `T`, and nothing is lost on an EPP restart.

`S` is evaluated the way flow control evaluates it: per pipeline stage by the `llm-d.ai/role` label (prefill, decode; interleaved endpoints count in both), taking the maximum, so an idle decode tier cannot hide a saturated prefill tier. Endpoints without a role label are decode, so a monolithic pool is one stage. One difference remains: the admitter sees the candidate endpoints after screeners, flow control the pool before them. With screeners configured the two can differ, and `T` should be read against this plugin's `..._saturation` gauge rather than flow control's.

## Placement

Admitters run after the data producers and before scheduling, so the admitter sees the full candidate pool and the request's prefix match on every endpoint. It does not change the candidate set. An admitted request is routed by the scheduling profile as usual; pair it with `prefix-cache-affinity-filter` so that admitted hot requests land on their warm endpoint. Whether that endpoint currently has room is the profile's decision (for example `concurrency-detector` as a filter), the same as for any admitted request.

Below `T` the admitter changes nothing.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `saturationDetectorRef` | required | Name of a saturation detector instance (any plugin implementing `flowcontrol.SaturationDetector`, typically the `concurrency-detector` flow control already uses). It supplies `S`. |
| `shedThreshold` | `0.85` | `T`, in `(0, 1]`. |
| `minHitRatio` | `0.6` | `H`, in `[0, 1]`. |
| `minUncachedTokens` | `0` | A request that adds fewer uncached tokens than this on its cheapest endpoint is never shed. `0` disables. When set, `UncachedRequestTokens` becomes a required attribute; an endpoint without it is ignored, and a request with the count unknown on every endpoint is not exempted. A value larger than any prompt (for example `1000000000`) exempts everything and so acts as a kill switch without removing the plugin. |
| `shedStatusCode` | `429` | `429` or `503`. |
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
  - type: prefix-cache-aware-admitter
    name: prefix-cache-aware-admitter
    parameters:
      saturationDetectorRef: concurrency-detector
      shedThreshold: 0.85
      minHitRatio: 0.6
      prefixMatchInfoProducerName: approx-prefix-cache-producer
  - type: prefix-cache-affinity-filter
    parameters:
      affinityThreshold: 0.8
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: concurrency-detector
      - pluginRef: prefix-cache-affinity-filter
      - pluginRef: max-score-picker
```

Admitters are not listed in a scheduling profile; declaring the plugin is enough.

## Metrics

| Metric | Labels | Description |
| --- | --- | --- |
| `llm_d_epp_prefix_cache_aware_admitter_decisions_total` | `plugin_name`, `outcome` | One per `Admit` call. Outcomes: `admitted_below_threshold`, `admitted_hot`, `shed_cold`, `admitted_small_request`, `not_applicable`. The last one is the empty-pool branch; the Director rejects an empty pool before admitters run, so it should stay at zero. |
| `llm_d_epp_prefix_cache_aware_admitter_saturation` | `plugin_name` | `S` read on the last decision. |

## Relation to `failOpen: false` and to flow control

`concurrency-detector` with `failOpen: false` rejects when every endpoint is over its own limit for the request, at any pool saturation. Flow control rejects when `S >= 1.0` and the queue is full. Neither looks at the request's cache state. This admitter rejects on a pool-wide threshold below `1.0` and a per-request hit ratio, never below `T`, and only requests without a usable prefix hit. Because its denial reuses flow control's status and reason, a client cannot tell the two apart; operators can, through `llm_d_epp_prefix_cache_aware_admitter_decisions_total{outcome="shed_cold"}`.

## Notes

- With more than one EPP replica, each replica sees only the in-flight load it dispatched. Share the detector limits across replicas as you do for flow control.
