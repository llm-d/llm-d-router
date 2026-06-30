# Metric Scorer Plugin

**Type:** `metric-scorer`

This plugin scores candidate endpoints by a single configured numeric endpoint attribute.

## What it does

For each scheduling cycle, the plugin reads the configured attribute (`attributeKey`) from each candidate endpoint and computes a linearly normalized score across the candidates:

\[
\text{score(endpoint)} = \frac{\text{value(endpoint)} - \min}{\max - \min}
\]

With `direction: lower_is_better` the result is inverted, so the lowest value gets score `1.0` and the highest gets `0.0`.

Special cases:

- If all endpoints that have the attribute share the same value, they all receive a neutral score of `1.0`.
- An endpoint missing the attribute receives score `0.0` and does not participate in normalization.

The attribute is expected to be a numeric custom metric produced by the core metrics extractor (see the [metrics extractor](../../../datalayer/extractor/metrics/README.md)), stored as a `ScalarMetricValue` endpoint attribute.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests according to the configured metric.

## Inputs consumed

The plugin consumes:

- the configured `attributeKey` (`ScalarMetricValue`)

## Configuration

| Parameter                 | Required | Description                                                          |
|---------------------------|----------|----------------------------------------------------------------------|
| `attributeKey`            | yes      | Endpoint attribute to read, e.g. `custom.queue_depth`.               |
| `normalization.type`      | no       | Normalization strategy. Only `linear` is supported (the default).    |
| `normalization.direction` | yes      | `lower_is_better` or `higher_is_better`.                             |

**Configuration Example:**
```yaml
plugins:
  - type: metric-scorer
    name: queue-depth-metric
    parameters:
      attributeKey: "custom.queue_depth"
      normalization:
        type: "linear"
        direction: "lower_is_better"
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-depth-metric
        weight: 1
```
