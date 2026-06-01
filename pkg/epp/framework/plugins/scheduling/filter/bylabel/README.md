# Label-Based Filter Plugins

**Interfaces**: `scheduling.Filter`

Label-based filters that retain or remove candidate pods based on Kubernetes label values.

---

## LabelSelectorFilter

**Type:** `label-selector-filter`

> [!NOTE]
> The previous type name `by-label-selector` is deprecated but still accepted for backward compatibility.

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

## ByLabel (Deprecated)

**Type:** `by-label`

> [!WARNING]
> The `by-label` filter is deprecated. Use `label-selector-filter` for generic label-based filtering, or the role-specific filters (`decode-filter`, `prefill-filter`, `encode-filter`) for role-based filtering.

### What it does

Retains only candidate pods that carry a specific Kubernetes label with one of a set of allowed values; pods missing the label are included or excluded based on `allowsNoLabel`.

### Inputs consumed

- Pod Kubernetes labels (read from the candidate pod's metadata).

### Configuration

#### Parameters
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `label` | `string` | Yes | — | The name of the Kubernetes label to inspect on each pod. |
| `validValues` | `[]string` | Yes (unless `allowsNoLabel=true`) | — | List of acceptable label values. A pod is kept if its value matches any entry. |
| `allowsNoLabel` | `bool` | No | `false` | If `true`, pods that lack the label entirely are included. If `false`, they are filtered out. |

#### Example
```yaml
plugins:
  - type: by-label
    parameters:
      label: "gpu.type"
      validValues: ["a100"]
      allowsNoLabel: false
```

### Limitations

- Only exact string equality is checked — no wildcards, prefix matching, or regular expressions.

---

## Role-Based Filters

Pre-configured `by-label` filters for disaggregated inference architectures. Each checks the `llm-d.ai/role` label on candidate pods.

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
