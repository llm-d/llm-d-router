# Heavy Hitter Tracker

**Type:** `heavy-hitter-tracker`
**Interfaces:** `requesthandling.ResponseBodyProcessor`
**Stability:** Alpha

Reports the users carrying the highest load over a sliding time window. Load per completed request is token-weighted:

```
cost = costPerRequest + costPerInputToken*promptTokens + costPerOutputToken*completionTokens
```

The plugin rejects, delays, and reroutes nothing. It reports the heaviest users as metrics and through a Go read API, `Top(k)`, `Lookup(key)` and `IsHeavy(key)`, for a future enforcement consumer.

## What It Does

At end of stream, the plugin charges the request's cost to its user in a windowed Space-Saving summary: a ring of time slices, each holding at most `slotsPerSlice` keys, plus a merged view of each key's sum across the retained slices. Data older than the window is deleted when its slice rotates. Every termination cause is charged: a disconnected, evicted, or errored stream consumed capacity too.

Token counts come from the model server's usage block, which a streaming client must ask for with `stream_options.include_usage`. When completion tokens are absent and the response streamed, the observed stream events stand in for them and the tokenized prompt for the prompt tokens.

**Identity**: the tracked key is the user ID from the `x-llm-d-user-id` header inside the tenant that `x-llm-d-inference-fairness-id` names (its deprecated `x-gateway-inference-fairness-id` alias is accepted), joined as `<tenant>/<user>` with the tenant percent-escaped so the first `/`-separated segment of a key is exactly the tenant. A request with no user header keys on the tenant alone; one with a user but no tenant composes onto the default tenant, as the director does; one with neither identity falls back to the default flow. Keys longer than 256 bytes are truncated at a rune boundary; distinct keys sharing the truncated prefix are tracked as one, and a key whose whole prefix is invalid UTF-8 folds to `other`. The key is independent of `requestHandler.composeFairnessIDs`: the tracker reads the headers as sent, not the composed fairness ID.

## Heavy users and table sizing

`heavyThreshold` is the windowed load at or above which a user is heavy, in cost units: with request counting (below) and a threshold of 200, a user sending 200 requests in the window is heavy. When a threshold is set, the per-ID metrics report the heavy users only, and `IsHeavy` classifies a key using the lower bound `weight - error`, so a user is never called heavy on load it may not have sent.

The Space-Saving table size follows from the threshold: every user whose windowed load exceeds the window total divided by the slot count is guaranteed tracked, so a table of `2 * expectedWindowLoad / heavyThreshold` slots keeps every heavy user tracked with headroom for total load above the estimate. `expectedWindowLoad` is the operator's estimate of the peak total load of all users in one window, in the same cost units. Set `slotsPerSlice` to bypass the derivation.

Worked example: 10 requests/second over a 5-minute window is 3000 requests per window; with a threshold of 200 requests, the derived table is `2 * 3000 / 200 = 30` slots per slice. Every user sending more than 200 requests in the window is guaranteed tracked even at twice the estimated load.

## Accuracy

- While the distinct active keys per slice stay at or below the table size, nothing is evicted and every reported weight is exact.
- Above that, `weight` is within `window total / slots per slice` of a key's true windowed load, and `weight - error` is a lower bound on it. `error` counts only load inherited at slot takeover: load lost to an eviction carries no `error`, so an `error` of 0 does not by itself mean the value is exact.
- The window boundary moves in slice steps: the retained span is between `window - slice` and `window` (the window must be a multiple of the slice).
- A nonzero rate of `llm_d_epp_heavy_hitter_evictions_total` means the reported loads are approximate and the table size is below the active key count per slice.
- Each request's cost is rounded up to a whole unit, so with fractional coefficients many cheap requests can outweigh one expensive one.

## Configuration

```yaml
plugins:
- type: heavy-hitter-tracker
  name: heavy-hitters
  parameters:
    windowSeconds: 300
    sliceSeconds: 15
    heavyThreshold: 200
    expectedWindowLoad: 3000
    costPerRequest: 1
    costPerInputToken: 0
    costPerOutputToken: 0
```

| Field | Default | Description |
|---|---|---|
| `windowSeconds` | `300` | Retained span of the sliding window; must be a multiple of `sliceSeconds`. |
| `sliceSeconds` | `15` | Slice length. `windowSeconds/sliceSeconds` must yield 2 to 3600 slices. |
| `heavyThreshold` | `0` | Windowed load at or above which a user is heavy. `0` reports the top `metricTopK` users instead of the heavy ones. |
| `expectedWindowLoad` | `0` | Peak total load of all users in one window, in cost units; with `heavyThreshold` it sizes the table. Not needed when `slotsPerSlice` is set. |
| `slotsPerSlice` | derived | Space-Saving table size per slice. `0` derives it as `2 * expectedWindowLoad / heavyThreshold`, or 256 when no threshold is set. `slices * slotsPerSlice` is capped at 2^20. |
| `costPerRequest` | `0` | Fixed cost of one completed request. |
| `costPerInputToken` | `1` | Cost per prompt token. |
| `costPerOutputToken` | `2` | Cost per completion token. |
| `metricTopK` | `10` | Bounds the per-ID series the collector emits. `0` reports the totals only. |

An explicit `0` overrides a default, so `metricTopK: 0` disables the per-ID series while `costPerRequest: 0` keeps the default zero.

### Counting requests instead of tokens

Set the request coefficient to 1 and the token coefficients to 0. Every completed request then weighs 1, whatever its token counts, so the reported load is the request count per user:

```yaml
parameters:
  costPerRequest: 1
  costPerInputToken: 0
  costPerOutputToken: 0
```

A user sending 200 requests in the window reads 200; a user sending 2 reads 2. With request counting, `heavyThreshold` and `expectedWindowLoad` are in requests.

## Sizing

Choose `expectedWindowLoad` as the peak total load you expect in one window and let the table derive from the threshold, or set `slotsPerSlice` directly above the number of distinct active keys per slice. Watch `llm_d_epp_heavy_hitter_evictions_total` to confirm the sizing holds. At the defaults with a set threshold (5-minute window, 15-second slices, derived slots m):

- 20 slices of m slots at 32 bytes each.
- A merged-map entry of roughly 100 bytes per distinct key in the window, bounded by `20 * m` entries.
- Key bytes: at most `20 * m` distinct keys of at most 256 bytes.

## Observability

| Metric | Type | Labels |
|---|---|---|
| `llm_d_epp_heavy_hitter_weight` | Gauge | `plugin_name`, `fairness_id` |
| `llm_d_epp_heavy_hitter_weight_error` | Gauge | `plugin_name`, `fairness_id` |
| `llm_d_epp_heavy_hitter_window_weight` | Gauge | `plugin_name` |
| `llm_d_epp_heavy_hitter_evictions_total` | Counter | `plugin_name` |

The `fairness_id` label carries the tracking key: `<tenant>/<user>` when the user header is present, the tenant alone otherwise.

The per-ID series do not pass through the shared fairness-ID label limiter that other `fairness_id` metrics use: they are rebuilt from scratch on every scrape, hold no state between scrapes, and are bounded by `metricTopK`, so admitting top-k keys into the process-wide cap would spend it (roughly a hundred scrapes of a changing top-10 would fill the default cap of 1000, after which flow-control and request metrics would fold new tenants into `other`). A key that is not valid UTF-8 is skipped; key length is already bounded by truncation. This departs from the shared-cap rule in `pkg/common/observability/metrics/cardinality.go`.

Share of window load per user:

```promql
llm_d_epp_heavy_hitter_weight
/
on(plugin_name) group_left llm_d_epp_heavy_hitter_window_weight
```

Series for users that leave the top `metricTopK` disappear on the next scrape.

## Limits

- Each EPP replica tracks only the traffic it serves; there is no cross-replica aggregation.
- The tracker sees the raw `x-llm-d-user-id` and `x-llm-d-inference-fairness-id` headers. A client presenting unstable identities (one per request, or split load) reads as many light keys; defending the identity itself is an enforcement-layer concern.
- With the disaggregation coordinator, each phase is a separate EPP request, so prompt tokens are charged per phase and a client request using P/D is charged once for its prefill phase and once for its decode phase.
