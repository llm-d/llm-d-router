# Endpoint Attribute Filter Plugin

**Type:** `endpoint-attribute-filter`

This plugin filters candidate endpoints by a single configured numeric endpoint attribute.

## What it does

For each scheduling cycle, the plugin reads the configured attribute (`attribute`) from each candidate endpoint and keeps only the endpoints whose value satisfies the configured algorithm. This PR supports the `threshold` algorithm; `percentile`, `topK` and `range` are planned follow-ups.

With `threshold`, an endpoint is kept when

\[
\text{value(endpoint)} \ \langle op \rangle\ \text{threshold.value}
\]

is true for the configured `threshold.operator`.

Policies for the awkward cases:

- **`onMissing`** — what happens to an endpoint that does not have the attribute: `Pass` keeps it (the default), `Fail` drops it.
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
| `algorithm.type`                | yes      | Only `threshold` is currently supported.                                                  |
| `algorithm.threshold.operator`  | yes      | `LessThan`, `LessThanOrEqual`, `GreaterThan`, `GreaterThanOrEqual`, `Equal` or `NotEqual`. |
| `algorithm.threshold.value`     | yes      | The value compared against the endpoint's attribute.                                      |

**Configuration Example:**
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

## See also

The `endpoint-attribute-scorer` plugin ([llm-d/llm-d-router#1620](https://github.com/llm-d/llm-d-router/pull/1620)) is the scoring counterpart of this filter: instead of dropping endpoints, it ranks them by the same kind of configured attribute.
