# Prefix Cache Affinity Filter (`prefix-cache-affinity-filter`)

**Type:** `prefix-cache-affinity-filter`

## When to use this filter

Enable this filter when your workload has repeated or similar prompts across requests (e.g.,
shared system prompts, multi-turn conversations, or RAG pipelines with overlapping context).
In these scenarios, vLLM's automatic prefix caching keeps KV cache blocks from previous
requests in GPU memory. Without this filter, the default load-balancing strategy may route a
request to any endpoint, forcing a full prefill even when another endpoint already has most
of the prompt cached. This filter steers requests toward endpoints that hold a cache hit,
converting expensive prefill into a near-free cache lookup and significantly reducing
time-to-first-token (TTFT).

If your workload consists of unique, non-overlapping prompts, this filter has no effect
because no endpoint will accumulate cache hits, and the filter falls through to keeping all
candidates (no-op).

## Difference from `prefix-cache-scorer`

The `prefix-cache-scorer` plugin scores endpoints by prefix cache hit ratio. It works with
any picker, but the choice of picker creates a trade-off:

- **With `max-score-picker`** (the default): the scorer consistently picks the single
  highest-scoring endpoint, which maximizes cache hits but causes **hot-spotting** — many
  concurrent requests with similar prompts all land on the same endpoint, overloading it and
  degrading TTFT.
- **With `weighted-random-picker`**: requests spread across endpoints proportional to their
  cache scores. This avoids hot-spotting but dilutes cache affinity — requests are frequently
  sent to endpoints with low or zero cache hits, losing the prefill savings that prefix
  caching provides.

This filter resolves the trade-off by operating as a **pre-filter** rather than a scorer.
It narrows the candidate set to only the sticky endpoints (those above `affinityThreshold`),
then passes them to downstream plugins. When paired with `weighted-random-picker`, requests
are spread across the sticky set — maintaining cache affinity while distributing load. The
TTFT load gate (`maxTTFTPenaltyMs`) adds automatic back-off: if sticky endpoints become
overloaded and their TTFT exceeds non-sticky endpoints by more than the configured penalty,
the filter breaks stickiness and opens up all endpoints, preventing the hot-spotting problem.
The exploration mechanism (`explorationProbability`) seeds cache state on other
endpoints over time, preventing permanent stickiness to a fixed subset.

## Overview

Probabilistic filter that narrows candidates to "sticky" endpoints. An endpoint is sticky
when it has a high prefix cache score for the current request, meaning the request's prompt
(or most of it) is already cached on that endpoint from a previous request with the same or
similar prompt. Routing to a sticky endpoint avoids redundant prefill computation, reducing
TTFT.

Can be instantiated multiple times with different thresholds (e.g., 0.99 for global gate,
0.80 for within-tier gate).

## Behavior

- Keep only endpoints with prefix cache score >= `affinityThreshold`
- If no endpoints pass, all are kept (no-op)
- With probability `explorationProbability` (default 0, disabled), skip the gate entirely for exploration
- TTFT load gate: if best sticky endpoint's TTFT exceeds best non-sticky by more than
  `maxTTFTPenaltyMs`, break stickiness and keep all endpoints (0 = always stick). The
  per-endpoint TTFT is estimated from in-flight tokens as
  `inFlightTokens / peakPrefillThroughput * 1000` (ms) when `ttftSource` is
  `prefillThroughput` (default), or comes from the latency predictor when `ttftSource`
  is `latencyPredictor`
- If no endpoints have the TTFT source attribute (`LatencyPredictionInfo` or `InFlightLoad`),
  the TTFT load gate is skipped. If no endpoints have `PrefixCacheMatchInfo`, all prefix
  scores default to 0 and no endpoints pass the affinity threshold, so all are kept (no-op)

## Composition requirements

The filter narrows the candidate set or leaves it unchanged; the routing decision is made
by the scorers and picker that run after it. When the TTFT load gate breaks stickiness, the
filter keeps all endpoints, and the request moves off the sticky set only if downstream
scoring prefers the less-loaded endpoints. Nothing tells the scorers that a break happened;
they apply the same weights to whatever candidates arrive.

Scorers running after this filter must not include a separately weighted prefix-affinity
scorer. Each scorer's output is clamped to [0, 1] before weighting, so the other scorers can
favor a non-sticky endpoint by at most the sum of their weights (loadWeightSum), while a
prefix-affinity scorer penalizes it by its weight (prefixWeight) times the prefix cache
score difference. The break can therefore move a request only to endpoints whose prefix
cache score is within loadWeightSum / prefixWeight of the sticky endpoint's. A prefix weight
at or above loadWeightSum lets a zero-match endpoint at best tie a fully matched sticky one;
at or above loadWeightSum / affinityThreshold no zero-match endpoint can beat any sticky
endpoint outright (at exact equality a tie, resolved at random, remains possible). The break
still moves requests to partially warm endpoints inside the bound, so a weighted
`prefix-cache-scorer` with `max-score-picker` attenuates the gate in proportion to the prefix
weight, up to full inertness for cold endpoints. Breaks are counted by
`llm_d_epp_prefix_cache_affinity_filter_decisions_total` with `outcome="load_override"`. A
break that changed no routing decision shows as that counter climbing while the sticky
endpoint's queue depth and TTFT hold; there is no counter for a break that left the request
on the same endpoint.

Excluding a weighted prefix scorer does not remove affinity from the post-break decision.
The recommended load scorers count uncached tokens (the endpoint's uncached tokens in flight
plus the tokens of this request the endpoint has not cached), so cache warmth lowers a
request's cost on a warm endpoint in the same units as the load it is compared against.

Recommended compositions pair the filter with a single load scorer and `max-score-picker`:
`token-load-scorer` for prefill-bound workloads, `active-request-scorer` for decode-bound
workloads. `weighted-random-picker` spreads requests across the sticky set rather than
concentrating them on the top-scored endpoint; the picker choice does not affect the
scorer requirement. The
[optimized-baseline guide](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline)
ships the prefill-bound composition, and the
[Sticky Until Saturated post](https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing)
documents the strategy and its calibration.

## Config

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `affinityThreshold` | `float64` | No | `0.80` | Prefix cache score threshold for stickiness |
| `explorationProbability` | `float64` | No | `0` | Probability of skipping the gate |
| `maxTTFTPenaltyMs` | `float64` | No | `18000` | Max TTFT penalty (ms) before breaking stickiness. 0 = always stick |
| `ttftSource` | `string` | No | `prefillThroughput` | TTFT source for the load gate: `prefillThroughput` or `latencyPredictor` |
| `peakPrefillThroughput` | `float64` | No | `15928` | Peak prefill throughput (tokens/sec), used to estimate TTFT when `ttftSource` is `prefillThroughput` |

The `peakPrefillThroughput` default of `15928` tokens/sec is calibrated for Qwen 32B on
2x H100 80GB (TP=2) with vLLM 0.19, measured as the prefill throughput of a single
unloaded chunk of `max_num_batched_tokens` (8192) tokens. It is hardware-, model-, and
serving-stack-specific; retune it for a different deployment, or set `ttftSource`
to `latencyPredictor` to source TTFT from the latency predictor instead.

## Dependencies

- Reads `PrefixCacheMatchInfo` from endpoint attributes (from `prefix-cache-scorer`)
- Reads `InFlightLoad` for the TTFT load gate when `ttftSource` is `prefillThroughput` (from `in-flight-load-producer`)
- Reads `LatencyPredictionInfo` for the TTFT load gate when `ttftSource` is `latencyPredictor` (from `predicted-latency-producer`)

**Configuration Example:**
```yaml
plugins:
  - type: prefix-cache-affinity-filter
    name: prefix-affinity
    parameters:
      affinityThreshold: 0.80
      explorationProbability: 0.01
      maxTTFTPenaltyMs: 5000
      ttftSource: prefillThroughput
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-affinity
```
