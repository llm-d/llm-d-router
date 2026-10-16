# Label-Based Filter Plugins

**Interfaces**: `scheduling.Filter`

Label-based filters that retain or remove candidate pods based on Kubernetes label values.

---

## LabelSelectorFilter

**Type:** `label-selector-filter`

### What it does

Retains only candidate pods that match a standard Kubernetes label selector. Supports both `matchLabels` (all key-value pairs must match, AND logic) and `matchExpressions` (operators: `In`, `NotIn`, `Exists`, `DoesNotExist`).

### Inputs consumed

- Pod Kubernetes labels (read from the candidate pod's metadata).

### Configuration

#### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `matchLabels` | `map[string]string` | No | — | Map of `{key: value}` pairs. All pairs must match (AND logic). |
| `matchExpressions` | `[]LabelSelectorRequirement` | No | — | List of label selector requirements. Each specifies a `key`, an `operator`, and optionally `values`. |

#### Example
```yaml
plugins:
  - type: label-selector-filter
    parameters:
      matchLabels:
        inference-role: decode
        hardware-type: H100
```

---

## Role-Based Filters

Pre-configured filters for disaggregated inference architectures. Each checks the `llm-d.ai/role` label on candidate pods.

### Dynamic P/D layouts

The EPP admin server supports runtime P/D/M role assignment when both `prefill-filter`
and `decode-filter` are configured and the default-off `dynamicPD` feature gate is
explicitly enabled:

```yaml
featureGates:
- dynamicPD
```

It can also be enabled with `--feature-gates=dynamicPD=true`. To disable the feature,
remove the gate, set `dynamicPD=false` in the config, or pass
`--feature-gates=dynamicPD=false`. When disabled, no layout manager is injected into
role filters and `/v1/layout/show` and `/v1/layout/change` are not registered.

Each participating Pod must provide one transfer group and its hard worker-shape fields:

```yaml
metadata:
  labels:
    llm-d.ai/role: prefill
    llm-d.ai/role-transfer-group: tinyllama-simulator
    llm-d.ai/tensor-parallel-size: "1"
    llm-d.ai/model-identity: tinyllama-v1
    llm-d.ai/accelerator-class: simulator
```

The admin routes are served on the EPP metrics port. A layout update keeps endpoints
in their current role where possible and only moves endpoints between roles when their
transfer group matches. The current flat layout API accepts exactly one transfer group
and rejects inventories containing multiple groups. Pods in one group must also agree
on model identity and accelerator class. A group may contain multiple TP buckets, but
an endpoint keeps its TP size when its role changes; each TP bucket's endpoint count is
preserved independently. Pod labels are not modified. Requests scheduled after the
update use the new assignment.

Optional per-endpoint soft-limit labels are `llm-d.ai/max-model-len`,
`llm-d.ai/max-num-batched-tokens`, and `llm-d.ai/max-num-seqs`. They do not split a
transfer group, but must be enforced by request eligibility and capacity estimation.

```bash
curl http://localhost:9090/v1/layout/show

curl -X POST http://localhost:9090/v1/layout/change \
  -H 'Content-Type: application/json' \
  -d '{"layout":"p_1tp1_d_3tp1_m_none"}'
```

Layouts use `p_<components>_d_<components>_m_<components>`. The `m` role maps to
`prefill-decode`. Roles may contain multiple TP components, for example
`p_2tp4_d_4tp1_m_none` can change to `p_1tp4_d_4tp1-1tp4_m_none`. This changes one
TP4 endpoint from prefill to decode; it does not change any endpoint's TP topology.
Encode-only and encode-prefill endpoints keep their Pod roles and do not participate
in dynamic layouts.

**Example Target Pod:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    llm-d.ai/role: "decode"
spec:
  # ... pod specification
```

#### Inference Roles

| Role | Description |
|------|-------------|
| `encode` | Encode stage only |
| `prefill` | Prefill stage only |
| `decode` | Decode stage only |
| `encode-prefill` | Encode + Prefill |
| `prefill-decode` | Prefill + Decode |
| `encode-prefill-decode` | All stages (monolithic) |
| `both` | Prefill + Decode (alias for `prefill-decode`) — **Deprecated**, use `prefill-decode` instead |

### EncodeRole Filter

**Type:** `encode-filter`

#### What it does

Retains pods whose `llm-d.ai/role` value is `encode`, `encode-prefill`, or `encode-prefill-decode`; all other pods are filtered out.

#### Inputs consumed

- `llm-d.ai/role` pod label.

#### Configuration

##### Parameters

None.

---

### PrefillRole Filter

**Type:** `prefill-filter`

#### What it does

Retains pods whose `llm-d.ai/role` value is `prefill`, `encode-prefill`, `prefill-decode`, `both`, or `encode-prefill-decode`; all other pods are filtered out.

#### Inputs consumed

- `llm-d.ai/role` pod label.

#### Configuration

##### Parameters

None.

---

### DecodeRole Filter

**Type:** `decode-filter`

#### What it does

Retains pods whose `llm-d.ai/role` value is `decode`, `prefill-decode`, `both`, or `encode-prefill-decode`; pods that completely lack the `llm-d.ai/role` label are also retained.

#### Inputs consumed

- `llm-d.ai/role` pod label.

#### Configuration

##### Parameters

None.

#### Limitations

- Pods without the `llm-d.ai/role` label are passed through (not filtered out), unlike `encode-filter` and `prefill-filter` which exclude unlabeled pods.

---

## Related Documentation
- [Creating a Custom Filter](../../../../../../../docs/create_new_filter.md)
- [Disaggregated Inference Serving in llm-d](../../../../../../../docs/disaggregation.md)
