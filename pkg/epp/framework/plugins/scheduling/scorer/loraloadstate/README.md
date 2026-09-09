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

This scorer currently has no runtime parameters.

**Configuration Example:**
```yaml
plugins:
  - type: lora-load-state-scorer
    name: lora-load-state
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-load-state
        weight: 1
```
