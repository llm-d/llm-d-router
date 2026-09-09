# LoRA Residency Filter (`lora-residency-filter`)

**Type:** `lora-residency-filter`

## When to use this filter

Enable this filter on a fleet serving many LoRA adapters when you want to **control how far each
adapter spreads**. `lora-load-state-scorer` prefers the endpoints where an adapter is already
resident, but a scorer can only prefer: under load the token-load scorer outvotes it, the
request lands on a pod without the adapter, that pod loads a new copy, and the copy takes a
GPU slot another adapter needed. In a mixed workload popular adapters spread until they have
evicted everyone else and the long tail thrashes.

This filter decides *whether* a new copy is allowed; the scorer decides *where*.

## What it does

For each request, candidates are split into **homes** (endpoints where the adapter occupies a GPU
slot, from the residency metrics), **warm** endpoints (host-cache copy only) and **cold** ones. A
host-cache copy is not a home: activating it evicts a GPU resident and, measured on Qwen3-32B,
costs about as much as a load from disk. Treating it as a home made popular adapters' traffic
spread over every pod and thrash their GPU slots.

| situation | kept | outcome label |
|---|---|---|
| no home anywhere | everything (the scorer's placement bonus picks the first home) | `no_home` |
| at least one home has room | the homes | `sticky` |
| every home saturated, `maxReplicas` reached | the homes | `cap_blocked` |
| every home saturated, another endpoint has room | the non-saturated warm endpoints, else the non-saturated cold ones (one new copy) | `spread` |
| every home saturated, nothing has room | the homes | `fleet_saturated` |

An endpoint is **saturated** when its waiting queue exceeds `queueThreshold` or its KV cache
utilization exceeds `kvCacheThreshold`. Set either to `0` to disable that check; with both disabled
homes are always sticky.

Decisions are counted in `llm_d_router_endpoint_picker_lora_residency_filter_decisions_total{plugin_name,outcome}`.

## Inputs consumed

- `metrics.LoadedModelsKey` (`map[string]datalayer.LoraLoadState`)
- `metrics.WaitingQueueSizeKey` (`int`)
- `metrics.KVCacheUsagePercentKey` (`float64`)

Residency arrives by scrape, so for one refresh interval after a load the new home is not visible
yet and requests keep hitting the old set.

## Configuration

**Location:** `plugins[N].parameters`

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `maxReplicas` | `int` | No | `0` (unbounded) | Maximum endpoints that may hold one adapter. |
| `queueThreshold` | `int` | No | `8` | Waiting-queue depth above which an endpoint is saturated. `0` disables. |
| `kvCacheThreshold` | `float` | No | `0.8` | KV cache utilization above which an endpoint is saturated. `0` disables. |

### Example

```yaml
plugins:
  - type: lora-residency-filter
    name: lora-homes
    parameters:
      maxReplicas: 2
  - type: lora-load-state-scorer
    name: lora-load-state
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-homes
      - pluginRef: lora-load-state
        weight: 1
```
