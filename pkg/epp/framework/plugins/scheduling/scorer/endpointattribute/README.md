# Endpoint Attribute Scorer Plugin

**Type:** `endpoint-attribute-scorer`

This plugin scores candidate endpoints by a single configured numeric endpoint attribute.

## What it does

For each scheduling cycle, the plugin reads the configured attribute (`attributeKey`) from each candidate endpoint and computes a linearly normalized score:

\[
\text{score(endpoint)} = \frac{\text{value(endpoint)} - \min}{\max - \min}
\]

With `algorithm.type: linear_lower_is_better` the result is inverted, so the lowest value gets score `1.0` and the highest gets `0.0`. With `linear_higher_is_better` the highest value gets `1.0`.

The `[min, max]` range comes from the configured normalization strategy:

- **`adaptiveRange`** (the default) — the range is computed across the candidate endpoints each scheduling cycle. Suited to open-ended attributes such as queue depth. If all endpoints that have the attribute share the same value, they all receive a neutral score of `1.0`.
- **`fixedRange`** — the range is the configured `min`/`max`. Suited to attributes with known bounds such as kv-cache utilization (`[0, 1]`). Values outside the range are clamped.

In both strategies, an endpoint missing the attribute receives score `0.0` (and, for `adaptiveRange`, does not participate in the range computation).

The attribute is expected to be a numeric custom metric produced by the core metrics extractor (see the [metrics extractor](../../../datalayer/extractor/metrics/README.md)), stored as a `ScalarMetricValue` endpoint attribute.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests according to the configured attribute.

## Inputs consumed

The plugin consumes:

- the configured `attributeKey` (`ScalarMetricValue`)

## Configuration

| Parameter                                    | Required | Description                                                                    |
|----------------------------------------------|----------|--------------------------------------------------------------------------------|
| `attributeKey`                               | yes      | Endpoint attribute to read, e.g. `custom.queue_depth`.                          |
| `algorithm.type`                             | yes      | `linear_lower_is_better` or `linear_higher_is_better`.                          |
| `algorithm.normalization`                    | no       | At most one strategy: `adaptiveRange` (the default) or `fixedRange`.            |
| `algorithm.normalization.fixedRange.min/max` | no       | Bounds for `fixedRange` normalization; `min` must be less than `max`.           |

**Configuration Example (adaptive range):**
```yaml
plugins:
  - type: endpoint-attribute-scorer
    name: queue-depth-attribute
    parameters:
      attributeKey: "custom.queue_depth"
      algorithm:
        type: "linear_lower_is_better"
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-depth-attribute
        weight: 1
```

**Configuration Example (fixed range):**
```yaml
plugins:
  - type: endpoint-attribute-scorer
    name: kv-cache-attribute
    parameters:
      attributeKey: "custom.kv_cache_utilization"
      algorithm:
        type: "linear_lower_is_better"
        normalization:
          fixedRange:
            min: 0.0
            max: 1.0
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: kv-cache-attribute
        weight: 1
```
