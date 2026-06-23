# Load-Aware Picker

**Type:** `load-aware-picker`

Serializing picker that aims to mitigate thundering-herd endpoint concentration by making each pick
visible to the next before it chooses.

## Why this exists

All other pickers are stateless: concurrent requests score and pick from the same (or very
similar) snapshot of endpoint metrics. When a burst of requests arrives simultaneously, all see
identical scores and converges on the same endpoint. The resulting concentration is a
scheduler artifact, not a reflection of actual endpoint capacity.

This picker addresses the root cause: the final selection step is serialized behind a mutex
so each goroutine in a burst observes the picks made by all preceding goroutines.

## How it works

1. **Sample** up to 32 candidates at random from the scored pool (avoids O(n) work under the mutex for large pools; 32 is sufficient to find a good endpoint in practice).
2. **Acquire the mutex.**
3. **Score each candidate** with two multiplicative adjustments applied to the scorer-pipeline score:
   - *Concentration factor* `min(1.0, expectedShare/actual)`: penalizes endpoints that have received more than their fair share of picks in the last 10 seconds.
   - *Capacity factor* `max(0, 1 - load/maxConcurrency)`: penalizes endpoints approaching their estimated throughput ceiling, derived via Little's Law from a per-endpoint exponential moving average of throughput, latency, and tokens per request.
4. **Select** the highest adjusted-score candidate and increment its pending-request counter before releasing the mutex. The pending counter bridges the gap between `Pick` returning and `PreRequest` being called, so the next request in the same burst sees it immediately.
5. **Decrement** the pending counter at `PreRequest` and record the dispatch timestamp in a short-lived cache.
6. **Update** the per-endpoint EMA values at `ResponseBody` on request completion (end-of-stream only, zero-output-token responses excluded).

## When to use

Use this picker when:
- Request bursts cause multiple concurrent scheduling decisions (common under high QPS or
  bursty traffic patterns, such as RL steps).
- The pool contains endpoints with similar scores where small differences matter for load
  distribution.
- The capacity factor provides value: the plugin is most effective when `ResponseBody` data
  is flowing (the EMA is populated). On a cold start it falls back to concentration-only
  adjustment.

Pair with any scorer pipeline. The adjusted score is multiplicative, so scorers that produce
zero for an endpoint (e.g., a filter expressed as a scorer) are preserved.

## Inputs consumed

- `ScoredEndpoint.Score` from the scorer pipeline.
- `ScoredEndpoint.GetMetrics().RunningRequestsSize` (committed in-flight requests) for the
  capacity factor.
- `InferenceRequest.RequestID` and `Body.TokenizedPrompt.TokenIDs` for EMA updates.
- `Response.Usage.CompletionTokens` and `Response.EndOfStream` for EMA updates.

## Configuration

This plugin accepts no configuration parameters. The internal constants (candidate sample
size, concentration window, EMA decay) are tuned to the expected timescale / load and
require understanding the algorithm internals to adjust safely; they are not exposed as
operator knobs.

```yaml
plugins:
  - type: load-aware-picker
    name: load-aware
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: load-aware
```
