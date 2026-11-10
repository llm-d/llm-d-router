# Prefix Cache Aware Admitter

`prefix-cache-aware-admitter` implements cache-aware load shedding: while the pool is near saturation it admits sheddable requests that have a prefix cache hit on some endpoint, and denies the rest with flow control's saturation status and reason header.

Stability: Alpha.

## Rule

Let `S` be the pool saturation reported by the referenced saturation detector, `T` the shed threshold and `H` the minimum hit ratio.

Requests with priority `>= 0` are admitted. Only priority `< 0` can be shed, the same split `probabilistic-admitter` and `latency-slo-admitter` use.

- `S < T`: every sheddable request is admitted.
- `S >= T`: a sheddable request is admitted when some endpoint has a prefix cache hit ratio `>= H` for it, when the prefix attribute is present and the prompt is shorter than one KV block (`TotalBlocks == 0`), or when it adds fewer than `minUncachedTokens` uncached tokens on some endpoint (`minUncachedTokens` off at `0`). Otherwise the request is denied. A missing prefix attribute counts as cold.

The hit ratio is the fraction of the prompt the cache already holds, so it ranks requests by how much of their prefill is free rather than by absolute size: a 20K-token turn with 16K cached (80 %) is a better use of a slot than a 1K prompt with 600 cached (60 %), and a shed of the former risks a 20K recompute on retry. `minUncachedTokens` is the absolute complement: a request whose prefill is small is not worth shedding whatever its ratio.

A denial uses flow control's saturation status, HTTP `shedStatusCode` (429 by default, 503 optional), and header `x-llm-d-request-dropped-reason: rejected-saturated`. The body is the fixed string `pool saturated, request rejected`. Shedding is a platform decision; nothing about the request's cache state is exposed to the client. The decision inputs (`saturation`, `bestHitRatio`) are in the debug log, and the outcome is in the plugin's metrics.

The decision is stateless: it depends only on `S` and the request's hit ratios at that moment. There is no hysteresis; the threshold is a plain comparison, so the plugin never rejects while `S` is below `T`, and nothing is lost on an EPP restart.

`S` is evaluated the way flow control evaluates it: per pipeline stage by the `llm-d.ai/role` label (prefill, decode; interleaved endpoints count in both), taking the maximum, so an idle decode tier cannot hide a saturated prefill tier. Endpoints without a role label are decode, so a monolithic pool is one stage. One difference remains: the admitter sees the candidate endpoints after screeners, flow control the pool before them. With screeners configured the two can differ, and `T` should be read against this plugin's `..._saturation` gauge rather than flow control's.

## Placement

Admitters run after the data producers and before scheduling, so the admitter sees the full candidate pool and the request's prefix match on every endpoint. It does not change the candidate set. An admitted request is routed by the scheduling profile as usual; pair it with `prefix-cache-affinity-filter` so that admitted hot requests land on their warm endpoint. Whether that endpoint currently has room is the profile's decision (for example `concurrency-detector` as a filter), the same as for any admitted request.

Below `T` the admitter changes nothing.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `saturationDetectorRef` | required | Name of a saturation detector instance (any plugin implementing `flowcontrol.SaturationDetector`, typically the `concurrency-detector` flow control already uses). It supplies `S`. |
| `shedThreshold` | required | `T`, in `(0, 1]`. |
| `minHitRatio` | required | `H`, in `[0, 1]`. An explicit `0` admits every request that has a prefix attribute. |
| `minUncachedTokens` | `0` | Cost floor: a request that adds fewer uncached tokens than this on its cheapest endpoint is never shed, because its prefill is too small to matter for the TTFT or ITL of other requests. A fraction of the model server's prefill chunk (for example a quarter of `max_num_batched_tokens`) is a reasonable value. `0` disables. When set, `UncachedRequestTokens` becomes a required attribute; an endpoint without it is ignored, and a request with the count unknown on every endpoint is not exempted. A value larger than any prompt (for example `1000000000`) exempts everything and so acts as a kill switch without removing the plugin. |
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
| `llm_d_epp_prefix_cache_aware_admitter_decisions_total` | `plugin_name`, `outcome` | One per sheddable `Admit` call. Outcomes: `admitted_below_threshold`, `admitted_hot`, `admitted_sub_block`, `shed_cold`, `admitted_small_request`, `not_applicable`. Priority `>= 0` and a nil request return before a decision is recorded. `not_applicable` is the empty-pool branch; the Director rejects an empty pool before admitters run, so it should stay at zero. |
| `llm_d_epp_prefix_cache_aware_admitter_saturation` | `plugin_name` | `S` read on the last decision. |

## Relation to `failOpen: false` and to flow control

`concurrency-detector` with `failOpen: false` rejects when every endpoint is over its own limit for the request, at any pool saturation. Flow control rejects when `S >= 1.0` and the queue is full. Neither looks at the request's cache state. This admitter rejects on a pool-wide threshold below `1.0` and a per-request hit ratio, never below `T`, and only requests without a usable prefix hit. Because its denial reuses flow control's status and reason, a client cannot tell the two apart; operators can, through `llm_d_epp_prefix_cache_aware_admitter_decisions_total{outcome="shed_cold"}`.

## Notes

- With more than one EPP replica, each replica sees only the in-flight load it dispatched. Share the detector limits across replicas as you do for flow control.
- A prefix shared by every request, such as a common system prompt, counts toward the hit ratio on every endpoint. A 500-token system prompt in a 700-token first-turn request is a 71 % hit and is admitted at `H = 0.6`. That request is cheap (200 uncached tokens), so admitting it costs little; a first-turn request with substantial new content behind the same prompt (500 shared of 1800) is 28 % and is still shed. Where request slots are the binding limit and the flood consists of short first-turn requests behind a shared prompt, the plugin sheds little. Set `H` above the shared-prefix fraction in that case (for example `0.9` with the prompt above), and a hot turn with a long new message (3K cached, 1K new = 75 %) is shed too. A non-zero `minUncachedTokens` exempts those short requests; when slots are the limit, that exemption keeps the flood.
- With `utilization-detector`, `T` is relative to that detector's own thresholds, and the same detector sets flow control's limit and the routing filter, so tune them together. At its defaults (waiting queue 5, KV 0.8) and an empty queue, `T = 0.85` starts shedding around 68% average KV use (`0.85 * 0.8`).
- `utilization-detector` defaults `stalenessPolicy` to `saturated`. A metrics outage scores those endpoints as fully saturated, so this admitter sheds every cold sheddable request. `stalenessPolicy: ignore` leaves stale endpoints out of the average.
- When this admitter and flow control share a `concurrency-detector`, flow control reserves the request at dispatch and holds that reservation until after scheduling. `Admit` therefore reads an `S` that includes the request itself, about `+1/(pods × maxConcurrency)` (3 pods at `maxConcurrency` 4 is about `+0.083`, so 0.80 reads as 0.88). `utilization-detector` does not track reservations.
