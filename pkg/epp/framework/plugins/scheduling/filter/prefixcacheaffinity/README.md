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

- **With `max-picker`** (the default): the scorer consistently picks the single
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
- With probability `explorationProbability` (default 1%), skip the gate entirely for exploration
- TTFT load gate: if best sticky endpoint's TTFT exceeds best non-sticky by more than
  `maxTTFTPenaltyMs`, break stickiness and keep all endpoints (0 = always stick). The
  per-endpoint TTFT comes from the latency predictor when `ttftSource` is
  `latencyPredictor` (default), or is estimated from in-flight tokens as
  `inFlightTokens / peakPrefillThroughput * 1000` (ms) when `ttftSource` is `prefillThroughput`
- If no endpoints have the TTFT source attribute (`LatencyPredictionInfo` or `InFlightLoad`),
  the TTFT load gate is skipped. If no endpoints have `PrefixCacheMatchInfo`, all prefix
  scores default to 0 and no endpoints pass the affinity threshold, so all are kept (no-op)

## Config

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `affinityThreshold` | `float64` | No | `0.80` | Prefix cache score threshold for stickiness |
| `explorationProbability` | `float64` | No | `0.01` | Probability of skipping the gate |
| `maxTTFTPenaltyMs` | `float64` | No | `5000` | Max TTFT penalty (ms) before breaking stickiness. 0 = always stick |
| `ttftSource` | `string` | No | `latencyPredictor` | TTFT source for the load gate: `latencyPredictor` or `prefillThroughput` |
| `peakPrefillThroughput` | `float64` | No | `15928` | Peak prefill throughput (tokens/sec), used to estimate TTFT when `ttftSource` is `prefillThroughput` |

The `peakPrefillThroughput` default of `15928` tokens/sec is calibrated for Qwen 32B on
2x H100 80GB (TP=2) with vLLM 0.19, measured as the prefill throughput of a single
unloaded chunk of `max_num_batched_tokens` (8192) tokens. It is hardware-, model-, and
serving-stack-specific; retune it for a different deployment, or set `ttftSource`
to `latencyPredictor` to source TTFT from the latency predictor instead.

## Dependencies

- Reads `PrefixCacheMatchInfo` from endpoint attributes (from `prefix-cache-scorer`)
- Reads `LatencyPredictionInfo` for the TTFT load gate when `ttftSource` is `latencyPredictor` (from `predicted-latency-producer`)
- Reads `InFlightLoad` for the TTFT load gate when `ttftSource` is `prefillThroughput` (from `in-flight-load-producer`)

**Configuration Example:**
```yaml
plugins:
  - type: prefix-cache-affinity-filter
    name: prefix-affinity
    parameters:
      affinityThreshold: 0.80
      explorationProbability: 0.01
      maxTTFTPenaltyMs: 5000
      ttftSource: latencyPredictor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: prefix-affinity
```
