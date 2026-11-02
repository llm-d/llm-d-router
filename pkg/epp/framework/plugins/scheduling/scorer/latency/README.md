# Latency Scorer Plugin (`latency-scorer`)

**Type:** `latency-scorer`

Scores endpoints based on predicted latency headroom, defined as the gap between the predicted
request latency and the user's SLO. Endpoints with more favorable headroom get higher
scores and are more likely to be selected by the picker. For negative headroom
(all endpoints violate SLO), idle endpoints are preferred.

## Inputs

- `LatencyPredictionInfo` endpoint attribute:
  - `TTFTHeadroom` / `TPOTHeadroom` - `SLO - predicted` (positive = meets SLO, negative = violates)
  - `DispatchedRequestCount` - in-flight requests tracked by the EPP
- `PrefixCacheMatchInfo` endpoint attribute:
  - Prefix cache score (0.0-1.0), used in composite fallback

## Output

A score in [0, 1] per endpoint. Higher = better candidate.

## Scoring

### Positive vs negative headroom

If both positive and negative headroom endpoints are present, only positive
endpoints are scored. Negative endpoints receive score 0. This should not normally happen when SLO filters (e.g. `slo-headroom-tier-filter`)
are configured upstream, but provides safe fallback behavior.

### Idle preference (negative headroom only)

When all endpoints have negative headroom, idle preference is applied: if any
endpoint has zero dispatched requests, only idle endpoints are scored. This
ensures idle pods absorb traffic before adding load to already-struggling pods.

### Deficit bucketing (negative headroom only)

Among non-idle negative endpoints, the scorer groups by which SLOs are violated
and scores only the best (least severe) non-empty bucket:
1. Only TPOT negative (most preferred - TTFT is met)
2. Only TTFT negative (TTFT impacts perceived responsiveness most)
3. Both negative (least preferred - violates both SLOs)

### Normalization and blending

TTFT and TPOT headroom values are normalized to [0, 1] and blended:
`combined = ttftWeight * nTTFT + tpotWeight * nTPOT`.

### No SLO behavior

When no SLO headers are set, headroom = `0 - predicted`, which is always negative
for both dimensions. All endpoints are already homogeneous (all negative), so no
tier filter is needed. The scorer differentiates by relative deficit magnitude,
effectively routing to the endpoint with the lowest predicted latency.

### Strategies

| Strategy | Behavior |
|----------|----------|
| `least` (default) | Prefer endpoints closest to SLO (bin-packing to preserve capacity) |
| `most` | Prefer endpoints with most headroom (conservative, max safety margin) |

`least` applies to both positive and negative headroom (closest to SLO boundary in
either direction). `most` only applies to positive headroom endpoints. For negative
headroom, `least` is always used regardless of config, since `most` would incorrectly
prefer the most overloaded endpoint.

### Range-based weight re-normalization

If all endpoints in a bucket have identical TTFT headroom (range = 0), the TTFT
weight is set to 0 and TPOT weight to 1 only when TPOT headroom varies and its
configured weight is positive (and vice versa). This keeps the varying dimension
fully weighted without enabling a dimension whose configured weight is 0.

### Composite fallback

When no predictions are available (sidecar down or timed out), falls back to a weighted
combination of KV cache utilization, queue depth, and prefix cache score.

## Config

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ttftWeight` | `float64` | No | `0.8` | TTFT blending weight. Higher = favor lower TTFT. Range: [0, ∞) |
| `tpotWeight` | `float64` | No | `0.2` | TPOT blending weight. Set to 0 for non-streaming. Range: [0, ∞) |
| `headroomSelectionStrategy` | `string` | No | `"least"` | Scoring strategy. Options: `least` / `most` |
| `compositeKVWeight` | `float64` | No | `1` | KV cache weight in composite fallback. Range: [0, ∞) |
| `compositeQueueWeight` | `float64` | No | `1` | Queue depth weight in composite fallback. Range: [0, ∞) |
| `compositePrefixWeight` | `float64` | No | `1` | Prefix cache weight in composite fallback. Range: [0, ∞) |
