# Endpoint Attribute Filter Plugin

**Type:** `endpoint-attribute-filter`

This plugin filters candidate endpoints by a single configured numeric endpoint attribute.

## What it does

For each scheduling cycle, the plugin reads the configured attribute (`attribute`) from each candidate endpoint and keeps only the endpoints whose value satisfies the configured algorithm. Four algorithms are supported:

| Algorithm | Description |
|-----------|-------------|
| `threshold` | Compare each endpoint against a fixed value using a comparison operator. |
| `range` | Keep endpoints whose value falls within an inclusive `[min, max]` range. |
| `topK` | Keep the K endpoints with the highest (or lowest) attribute value. |
| `percentile` | Keep endpoints at or above (or at or below) a computed percentile of the candidate values. |

### `threshold`

An endpoint is kept when

\[
\text{value(endpoint)} \ \langle op \rangle\ \text{threshold.value}
\]

is true for the configured `threshold.operator`.

### `range`

An endpoint is kept when

\[
\text{range.min} \leq \text{value(endpoint)} \leq \text{range.max}
\]

This is more convenient than two separate `threshold` filters (`GreaterThan` + `LessThan`) and correctly expresses "within range" as a single condition. Useful for mixed-model deployments where a request must only be routed to endpoints whose context-length capacity is within a required window.

### `topK`

Endpoints are ranked by their attribute value and the K highest (when `higherIsBetter: true`) or lowest (when `higherIsBetter: false`) are kept. When K exceeds the number of endpoints with the attribute, all of them are kept.

Useful at fleet scale to narrow the candidate set before expensive scorers run, reducing per-request scorer compute.

### `percentile`

Endpoints are sorted ascending by attribute value, and the nearest-rank percentile cutoff is computed as the value at position `ceil(P/100 * N)` (1-based) in the sorted array. When `higherIsBetter: true`, endpoints with value >= cutoff are kept; when `higherIsBetter: false`, endpoints with value <= cutoff are kept.

Unlike a fixed `threshold`, the percentile cutoff adapts to the current candidate distribution—useful under dynamic load where absolute thresholds become ineffective.

### Policies for missing attributes and empty results

These policies apply to all four algorithms:

- **`onMissing`** — what happens to an endpoint that does not have the attribute: `Pass` keeps it (the default), `Fail` drops it. For `topK` and `percentile`, missing-attribute endpoints are excluded from ranking and then kept or dropped according to this policy.
- **`fallbackOnEmpty`** — when every endpoint is filtered out and this is `true`, the original candidate list is returned unchanged, so the request can still be routed somewhere. Default `false`.

The attribute is a numeric `ScalarMetricValue` endpoint attribute. With `producer` unset it is a custom metric of the core metrics extractor (see the [metrics extractor](../../../datalayer/extractor/metrics/README.md)); set `producer` to read the attribute of another plugin, such as the [DCGM extractor](../../../datalayer/extractor/dcgm/README.md).

## Inputs consumed

The plugin consumes:

- the configured `attribute` (`ScalarMetricValue`)

## Configuration

| Parameter                       | Required | Description                                                                              |
|---------------------------------|----------|------------------------------------------------------------------------------------------|
| `attribute`                     | yes      | Endpoint attribute to read, e.g. `num_requests_running`.                                  |
| `producer`                      | no       | Plugin publishing the attribute. Omitted defaults to the core metrics extractor; set to e.g. `dcgm-extractor` to read another producer's attribute, or to `""` for a producer-agnostic attribute. An `attribute` containing `/` is rejected unless `producer` is set explicitly. |
| `onMissing`                     | no       | `Pass` (default) or `Fail` — keep or drop endpoints missing the attribute.                |
| `fallbackOnEmpty`               | no       | When `true`, return the unfiltered candidates if every endpoint was dropped. Default `false`. |
| `algorithm.type`                | yes      | One of `threshold`, `range`, `topK`, or `percentile`.                                     |

### `threshold` parameters

| Parameter                       | Required | Description                                                                              |
|---------------------------------|----------|------------------------------------------------------------------------------------------|
| `algorithm.threshold.operator`  | yes      | `LessThan`, `LessThanOrEqual`, `GreaterThan`, `GreaterThanOrEqual`, `Equal` or `NotEqual`. |
| `algorithm.threshold.value`     | yes      | The value compared against the endpoint's attribute.                                      |

### `range` parameters

| Parameter                | Required | Description                                                                              |
|--------------------------|----------|------------------------------------------------------------------------------------------|
| `algorithm.range.min`    | yes      | Minimum value (inclusive). Must be <= `max`.                                              |
| `algorithm.range.max`    | yes      | Maximum value (inclusive). Must be >= `min`.                                              |

### `topK` parameters

| Parameter                       | Required | Description                                                                              |
|---------------------------------|----------|------------------------------------------------------------------------------------------|
| `algorithm.topK.k`              | yes      | Number of endpoints to keep. Must be a positive integer.                                 |
| `algorithm.topK.higherIsBetter` | no      | `true` (default) keeps the K highest values; `false` keeps the K lowest.                |

### `percentile` parameters

| Parameter                          | Required | Description                                                                              |
|------------------------------------|----------|------------------------------------------------------------------------------------------|
| `algorithm.percentile.percentile`   | yes      | Percentile cutoff in `[0, 100]`.                                                          |
| `algorithm.percentile.higherIsBetter` | no     | `true` (default) keeps endpoints >= the percentile cutoff; `false` keeps those <= it.   |

**Configuration Example (threshold):**
```yaml
plugins:
  - type: endpoint-attribute-filter
    name: drop-loaded-endpoints
    parameters:
      attribute: "num_requests_running"
      onMissing: "Pass"
      fallbackOnEmpty: true
      algorithm:
        type: "threshold"
        threshold:
          operator: "LessThan"
          value: 10
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: drop-loaded-endpoints
```

**Configuration Example (range):**
```yaml
plugins:
  - type: endpoint-attribute-filter
    name: context-length-window
    parameters:
      attribute: "max_context_length"
      onMissing: "Fail"
      fallbackOnEmpty: true
      algorithm:
        type: "range"
        range:
          min: 4096
          max: 131072
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: context-length-window
```

**Configuration Example (topK):**
```yaml
plugins:
  - type: endpoint-attribute-filter
    name: top-cache-hits
    parameters:
      attribute: "cache_hit_rate"
      onMissing: "Fail"
      algorithm:
        type: "topK"
        topK:
          k: 5
          higherIsBetter: true
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: top-cache-hits
```

**Configuration Example (percentile):**
```yaml
plugins:
  - type: endpoint-attribute-filter
    name: least-loaded-quartile
    parameters:
      attribute: "waiting_queue"
      onMissing: "Pass"
      fallbackOnEmpty: true
      algorithm:
        type: "percentile"
        percentile:
          percentile: 25
          higherIsBetter: false
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: least-loaded-quartile
```

## See also

The `endpoint-attribute-scorer` plugin ([llm-d/llm-d-router#1620](https://github.com/llm-d/llm-d-router/pull/1620)) is the scoring counterpart of this filter: instead of dropping endpoints, it ranks them by the same kind of configured attribute.
