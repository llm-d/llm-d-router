# Accelerator Capability Aware

**Type:** `accelerator-capability-aware`

Routes inference requests based on estimated request size and static accelerator
capability ranges declared on endpoint labels. This is intended for
heterogeneous GPU or MIG pools where larger multimodal requests should prefer
larger accelerator profiles without depending on runtime DCGM metrics.

Each pod declares its supported range via the `label` parameter. The default is
`llm-d.ai/accelerator-capability-range`, formatted as `min-max`, where the values
are estimated request tokens.

```yaml
metadata:
  labels:
    llm-d.ai/accelerator-capability-range: "0-2048"
```

The request size is read from `TokenizedPrompt.TokenCount()`, produced by
`token-producer`. For multimodal requests, the tokenizer estimate backend can
include image placeholder tokens, so image-heavy requests can be matched to
larger accelerator profiles.

## Parameters

- `label` (string, optional, default: `llm-d.ai/accelerator-capability-range`):
  Pod label key carrying the `min-max` range.
- `enableFiltering` (bool, optional, default: `false`): When true, endpoints
  with a labeled range that does not contain the request size are filtered out.

Unlabeled endpoints pass through filtering and receive a neutral score, allowing
incremental rollout.

## Example

```yaml
plugins:
  - type: accelerator-capability-aware
    parameters:
      label: llm-d.ai/accelerator-capability-range
      enableFiltering: true

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: accelerator-capability-aware
        weight: 3
```
