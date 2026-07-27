# Label Producer

**Type:** `label-producer`

Publishes Pod labels as string attributes that follow label updates.
Enable this Alpha plugin with `--allow-experimental-plugins=true`.

Configure one producer with a non-empty `labels` list. Each `label` maps to a
unique output `attributeKey`:

```yaml
plugins:
- type: label-producer
  name: endpoint-labels
  parameters:
    labels:
    - label: topology.kubernetes.io/region
      attributeKey: region
    - label: nvidia.com/gpu.product
      attributeKey: gpu.product
```

Consumers match the output `attributeKey` and `producer: endpoint-labels`.
See [endpoint-attribute-weight-scorer](../../../scheduling/scorer/attributeweight/README.md)
for a complete scheduling configuration.
