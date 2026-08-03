# Endpoint Attribute Weight Scorer

**Type:** `endpoint-attribute-weight-scorer`

Scores candidate endpoints by mapping a configured string endpoint attribute to a static, configurable weight. The attribute can be populated from a Pod label by `label-producer` or by another data-layer producer.

**Use Cases:**
- Prefer faster GPU types (e.g. H100 over A100 over L40S) in a heterogeneous InferencePool.
- Prefer endpoints in a given zone or rack.
- Any other case where a discrete endpoint attribute should map to a routing preference.

## Important: picker choice determines traffic concentration

This scorer only expresses a static preference; it has no notion of current load. If a scheduling profile has no picker configured, the EPP auto-injects `max-score-picker` (see `ensureSchedulingLayer` in the config loader). With its default `maxNumOfEndpoints: 1`, the picker selects an endpoint with the highest aggregate score. When this is the only scorer, ties are randomized but lower-scoring attribute groups are not selected.

If the profile explicitly configures another picker, the EPP does not inject `max-score-picker`, even when this is the only scorer. For traffic distribution, compose this scorer with a load-aware scorer such as `running-requests-size-scorer` or `queue-scorer`, and use a picker such as `weighted-random-picker`. See the configuration example below.

## What it does

Weights are normalized at configuration time: the highest configured weight scores `1.0`, and the rest scale proportionally (`weight / maxConfiguredWeight`). No configured value ever scores `0`.

Endpoints missing the configured attribute, or carrying a value with no configured weight, fall back to the **lowest configured, normalized weight** — the same score as your worst configured tier. This keeps endpoints eligible during incremental attribute rollout without giving missing values an advantage over a known weaker tier.

## Scheduling intent

The scorer returns category `Affinity`, indicating a stable preference for endpoints with higher-weighted attribute values. The category is descriptive; the scheduler still combines each scorer's result using its profile weight. Use a load-aware scorer and an appropriate picker when the goal is to avoid concentrating traffic on one endpoint group.

## Relationship to other plugins

- **`label-producer`**: copies a Pod label into a string or numeric endpoint attribute. Use it when the source is a Pod label and the consumer should remain independent of Kubernetes metadata.
- **`label-selector-filter`**: a hard filter (include/exclude by label value). Use it when an endpoint must never receive certain traffic. The legacy `by-label` and `by-label-selector` types are deprecated. This scorer never excludes — it only ranks.
- **`accelerator-capability-aware`** (tracks [#1868](https://github.com/llm-d/llm-d-router/issues/1868)): matches a *numeric* request-size range encoded in a label value (e.g. `"0-2048"`) against the request's token count. This scorer maps a *discrete* label value (e.g. a GPU model string) to a fixed weight — there is no range matching and no dependency on request size.
- **`endpoint-attribute-scorer`**: scores a custom numeric endpoint attribute via linear normalization. Use it when the attribute is produced by `core-metrics-extractor` through `customMetrics`; it does not read standard fields such as `RunningRequestsSize` or `WaitingQueueSize`. Use `running-requests-size-scorer` or `queue-scorer` for those standard metrics.

## Configuration

| Parameter | Required | Description                                                           |
|-----------|----------|-----------------------------------------------------------------------|
| `attributeKey` | yes      | String endpoint attribute to read, e.g. `gpu.product`.                 |
| `weights`     | yes      | Map of attribute value to positive weight, e.g. `{"H100": 4, "A100": 2}`. |

**Configuration Example (static label preference composed with standard load scoring):**
```yaml
plugins:
- type: label-producer
  name: gpu-product
  parameters:
    label: nvidia.com/gpu.product
    attributeKey: gpu.product
    valueType: string
- type: endpoint-attribute-weight-scorer
  name: accelerator-weight
  parameters:
    attributeKey: gpu.product
    weights:
      NVIDIA-H100: 4
      NVIDIA-A100: 2
      NVIDIA-L40S: 1
- type: running-requests-size-scorer
  name: running-requests
- type: weighted-random-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: accelerator-weight
        weight: 3
      - pluginRef: running-requests
        weight: 5
      - pluginRef: weighted-random-picker
```

**Example Pod Labels:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-h100
  labels:
    nvidia.com/gpu.product: NVIDIA-H100
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-l40s
  labels:
    nvidia.com/gpu.product: NVIDIA-L40S
```
