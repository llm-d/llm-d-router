# LoRA Load State Scorer Plugin

**Type:** `lora-load-state-scorer`

This plugin scores candidate endpoints by where the requested LoRA adapter's weights currently
live on each model server. It reads adapter **residency** (what is loaded) rather than request
activity (what is being served), so an adapter that was loaded at startup or has gone idle still
attracts its own traffic instead of being reloaded elsewhere.

Compared with `lora-affinity-scorer`, which reads `vllm:lora_requests_info` and therefore only
sees adapters with in-flight requests, this scorer reads the residency gauges vLLM publishes from
its worker adapter caches.

## What it does

For each candidate endpoint, the plugin looks up the request's `targetModel` in the endpoint's
resident adapter set and assigns:

- `1.0`: adapter occupies a GPU slot and can serve immediately
- `0.8`: adapter is in the host (CPU) cache; serving costs a device copy and possibly an eviction
- `0.6`: adapter is not resident but the endpoint has a free GPU slot
- `0.0`: adapter is not resident and every GPU slot is taken

Endpoints whose model server does not report residency score by the capacity tiers only, which
is the same for all such endpoints and leaves the choice to the other scorers in the profile.

## Scheduling intent

The scorer returns category `Affinity`, preferring endpoints with the lowest adapter-load cost.

## Inputs consumed

The plugin consumes:

- `metrics.LoadedModelsKey` (`map[string]datalayer.LoraLoadState`)
- `metrics.GPULoadedModelsKey` (`int`)

It also relies on endpoint metric `MaxActiveModels` to determine remaining GPU slot capacity.

## Model server requirements

See [LoRA Adapter Residency](../../../../../../../docs/plugin-metric-protocol.md#lora-adapter-residency)
for the metrics the model server must expose. The core metrics extractor reads them for vLLM by
default; other engines set `loraLoadedSpec` and `loraGPULoadedSpec` in their `engineConfigs` entry.

## Configuration

**Location:** `plugins[N].parameters`

### Parameters

The gaps between tiers encode the relative cost of serving the adapter from each state, and
that cost grows with adapter size. A small adapter loads in tens of milliseconds and the
defaults are generous; a large one makes a miss expensive, so pull `freeSlotScore` toward
`saturatedScore`. Use the profile `weight` to scale the whole scorer against the load and
prefix-cache scorers.

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `gpuResidentScore` | `float` | No | `1.0` | Adapter occupies a GPU slot. |
| `cpuResidentScore` | `float` | No | `0.8` | Adapter is only in the host cache. |
| `freeSlotScore` | `float` | No | `0.6` | Adapter not resident, a GPU slot is free. |
| `saturatedScore` | `float` | No | `0.0` | Adapter not resident, every GPU slot taken. |

Each score must be in `[0, 1]` and they must satisfy `gpuResidentScore >= cpuResidentScore >=
freeSlotScore >= saturatedScore`; otherwise the whole set is ignored, logged, and the defaults
are used.

### Example

```yaml
plugins:
  - type: lora-load-state-scorer
    name: lora-load-state
    parameters:
      cpuResidentScore: 0.9
      freeSlotScore: 0.2
      saturatedScore: 0.1
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-load-state
        weight: 2
```
