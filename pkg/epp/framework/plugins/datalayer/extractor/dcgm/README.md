# DCGM Extractor

**Type:** `dcgm-extractor`

The DCGM Extractor converts the Prometheus metrics response from a
`dcgm-data-source` into a per-endpoint GPU utilization attribute consumed by
the generic `endpoint-attribute-filter` and `endpoint-attribute-scorer`
(configurable for any scalar attribute, not hardwired to GPU).

## What it does

1. Receives the parsed Prometheus metric families forwarded by `dcgm-data-source`.
2. Looks up the `DCGM_FI_DEV_GPU_UTIL` metric family.
3. Keeps samples that belong to this endpoint:
   - If a sample has a `pod` label and the endpoint has a `Name`, only
     matching samples are kept (DaemonSet multi-pod payloads).
   - If the sample has no `pod` label, it is kept (sidecar / unlabeled payloads).
4. Aggregates matching samples using `max` (highest-utilized GPU for the pod).
5. Normalizes the value from 0-100 to [0.0, 1.0].
6. Stores the result as `ScalarMetricValue` on the corresponding endpoint.

## Attributes produced

- **Type:** `ScalarMetricValue` (from `attribute/metrics`)
- **Attribute:** `GPUUtilization`
- **Producer:** `dcgm-extractor`

## Configuration

No configuration parameters.

## Cluster prerequisites

- **DCGM Exporter** must be running on the cluster and accessible from the
  EPP. See [`source/dcgm/README.md`](../../source/dcgm/README.md) for
  data source configuration (port, DaemonSet vs sidecar, TLS).

## EPP config example

Use the generic `endpoint-attribute-filter` and `endpoint-attribute-scorer`
to consume the produced attribute. Their `attribute`/`attributeKey` and
`producer` parameters must match the pair above; `producer` defaults to the
core metrics extractor, so it must be set explicitly here.

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: dcgm-data-source
  name: dcgm-source
  parameters:
    port: 9400
- type: dcgm-extractor
  name: dcgm-extractor
- type: endpoint-attribute-filter
  name: gpu-utilization-filter
  parameters:
    attribute: "GPUUtilization"
    producer: "dcgm-extractor"
    onMissing: "Pass"
    fallbackOnEmpty: true
    algorithm:
      type: "threshold"
      threshold:
        operator: "LessThanOrEqual"
        value: 0.90
- type: endpoint-attribute-scorer
  name: gpu-utilization-scorer
  parameters:
    attributeKey: "GPUUtilization"
    producer: "dcgm-extractor"
    algorithm:
      type: "linear_lower_is_better"
      normalization:
        fixedRange:
          min: 0.0
          max: 1.0
- type: kv-cache-utilization-scorer
- type: max-score-picker
- type: single-profile-handler
- type: decode-filter
dataLayer:
  sources:
  - pluginRef: dcgm-source
    extractors:
    - pluginRef: dcgm-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: gpu-utilization-filter
  - pluginRef: gpu-utilization-scorer
    weight: 2.0
  - pluginRef: kv-cache-utilization-scorer
    weight: 1.0
  - pluginRef: max-score-picker
```
