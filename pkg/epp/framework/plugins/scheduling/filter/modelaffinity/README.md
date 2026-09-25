# Model Affinity Filter

**Type:** `model-affinity-filter`

**Interface:** `scheduling.Filter`

Retains only candidate endpoints whose configured label value matches the
request's target model. Designed for multi-cluster hub deployments where
endpoints are discovered via `multicluster-file-discovery` and each endpoint
entry is labelled with the model it serves.

---

## What It Does

In a hub EPP topology, the EPP routes inference requests to downstream clusters
(spokes). Each spoke typically serves a specific model. By labelling
`multicluster-file-discovery` entries with their served model, this filter
ensures that only spokes serving the requested model are considered as
candidates - enabling model-aware routing without requiring per-model scheduler
profiles.

## Inputs Consumed

- **Request header** (configurable, default: `x-gateway-model-name`) - set by
  IPP's `body-field-to-header` plugin in the ext_proc chain before EPP.
- **Fallback: request body `model` field** - parsed by the EPP director from
  the JSON body (e.g., `{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", ...}`).
- Endpoint label matching the configured `labelKey` (default: `model`).

## Configuration

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `labelKey` | `string` | No | `model` | The endpoint label key whose value is compared against the resolved model name. |
| `modelHeader` | `string` | No | `x-gateway-model-name` | The request header from which the target model name is read. Set by IPP's `body-field-to-header` plugin. When not present, falls back to the body's `model` field. Set to empty string (`""`) to always use the body. |

### Example

```yaml
plugins:
  - type: model-affinity-filter
    name: model-filter
    parameters:
      labelKey: model
      modelHeader: x-gateway-model-name
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: model-filter
```

### Endpoints File (multicluster-file-discovery)

```yaml
endpoints:
  - name: spoke1
    address: inference-gateway.apps.spoke1.example.com
    port: "443"
    labels:
      model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
      metricsAddress: epp-metrics.apps.spoke1.example.com
      metricsPort: "443"
  - name: spoke2
    address: inference-gateway.apps.spoke2.example.com
    port: "443"
    labels:
      model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
      metricsAddress: epp-metrics.apps.spoke2.example.com
      metricsPort: "443"
  - name: spoke3
    address: inference-gateway.apps.spoke3.example.com
    port: "443"
    labels:
      model: Qwen/Qwen2.5-0.5B-Instruct
      metricsAddress: epp-metrics.apps.spoke3.example.com
      metricsPort: "443"
```

With the above configuration, a request targeting `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
is routed only to `spoke1` and `spoke2`; a request for `Qwen/Qwen2.5-0.5B-Instruct`
is routed only to `spoke3`.

## Behaviour Details

- **Model resolution order**: Header (`modelHeader`) -> body `model` field.
- **No model resolved** (nil request, empty header, empty body): All
  endpoints pass through unfiltered.
- **Endpoints without the label**: Filtered out (they cannot match any model).
- **No matching endpoints**: Returns an empty list - the scheduler reports a
  routing error rather than sending to the wrong model server.

## Limitations

- Only exact string equality is checked against the label value - no wildcards,
  prefix matching, or regular expressions.
- Each endpoint can only carry a single value per label key. Endpoints serving
  multiple models require one entry per model (with different names but the same
  address).

