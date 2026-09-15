# Endpoint Attribute Weight Scorer

**Type:** `endpoint-attribute-weight-scorer`

Scores string endpoint attributes, such as region or GPU model, using static
weights. Enable the plugins with `--allow-experimental-plugins=true`.

## Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `attributeKey` | yes | String endpoint attribute to read. |
| `producer` | yes | Producer instance name; `""` selects the empty namespace. |
| `weights` | yes | Non-empty map of attribute values to finite, positive weights. |

Scores are `weight / max(weights)`. Missing or unknown values receive the lowest
normalized score. Weights that normalize to zero are rejected.
Match `attributeKey` and `producer` to an `attribute/string.Value` output.
If no producer publishes the matching attribute, startup logs a warning and all Pods fall back.

## Region and GPU preference

Set `topology.kubernetes.io/region` and `nvidia.com/gpu.product` on serving Pods.
Configure one producer and two scorers:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: label-producer
  name: endpoint-labels
  parameters:
    labels:
    - label: topology.kubernetes.io/region
      attributeKey: region
    - label: nvidia.com/gpu.product
      attributeKey: gpu.product
- type: endpoint-attribute-weight-scorer
  name: region-preference
  parameters:
    attributeKey: region
    producer: endpoint-labels
    weights:
      region-1: 2
      region-2: 1
- type: endpoint-attribute-weight-scorer
  name: gpu-preference
  parameters:
    attributeKey: gpu.product
    producer: endpoint-labels
    weights:
      NVIDIA-H100: 2
      NVIDIA-A100: 1
- type: weighted-random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: region-preference
    weight: 3  # 3 * normalized region score
  - pluginRef: gpu-preference
    weight: 1  # 1 * normalized GPU score
  - pluginRef: weighted-random-picker
```

| Region / GPU | Total score |
| --- | ---: |
| region-1 / H100 | 4 |
| region-1 / A100 | 3.5 |
| region-2 / H100 | 2.5 |
| region-2 / A100 | 2 |

With one Pod per row, weighted random sends an expected `(4 + 3.5) / 12 = 62.5%` to region-1;
`max-score-picker` selects region-1/H100. Replica counts and other scorers change
the distribution. These weights express preferences; calibrate them for the workload.

## Combine with length, load and sessions

Define inclusive input-token ranges from the deployment's validated context limit,
reserving room for output:

```yaml
# Serving Deployment excerpt
spec:
  template:
    metadata:
      labels:
        # Example: 32768 total tokens minus a 4096-token output budget
        llm-d.ai/context-length-range: "0-28672"
```

Run [context-length-aware](../contextlengthaware/README.md),
[utilization-filter](../../filter/utilization/README.md), and
[session-affinity-filter](../../filter/sessionaffinity/README.md) in that order,
then score and pick. For length filtering alone, set `enableFiltering: true` and
profile weight `0`; missing range labels pass. Use the model's tokenizer via
[token-producer](../../../requestcontrol/dataproducer/tokenizer/README.md).

A replacement session binding persists after the original Pod recovers. With
`fallbackOnEmpty: false`, load filtering can reject requests when all candidates
exceed the configured limits.
