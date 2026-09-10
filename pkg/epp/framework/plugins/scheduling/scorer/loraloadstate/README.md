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
resident adapter set and assigns a tier:

- `1.0`: adapter occupies a GPU slot and can serve immediately
- `0.7`: adapter is only in the host (CPU) cache; activating it evicts a GPU resident and, measured on
  Qwen3-32B, costs about as much as a load from disk
- `0.6`: adapter is not resident but the endpoint has a free GPU slot
- `0.3`: adapter is not resident, every slot is taken, but an unpinned resident has no request in
  flight, so loading evicts an adapter nobody is waiting on
- `0.0`: adapter is not resident and every slot holds a busy or pinned adapter

Two small bonuses then order endpoints within a tier without ever crossing one (the tiers are
scaled into the range left over by the bonuses):

- **placement**: a rendezvous hash of the adapter name over the candidate endpoints picks one
  preferred home, which gets the bonus while the adapter does not occupy a GPU slot there. All of an
  adapter's first misses then land on the same pod instead of scattering by load.
- **headroom**: proportional to the endpoint's share of free GPU slots, so among equals the
  endpoint with the most room wins.

Endpoints whose model server does not report residency score by the capacity tiers only, which
is the same for all such endpoints and leaves the choice to the other scorers in the profile.

A request for the base model itself (the `model_name` the server stamps on its metrics) needs no
adapter. It scores by the endpoint's share of free GPU slots, so base-model traffic drifts away
from the pods serving as adapter homes; once every endpoint is full the term is the same
everywhere and the other scorers decide.

## Scheduling intent

The scorer returns category `Affinity`, preferring endpoints with the lowest adapter-load cost.

## Inputs consumed

The plugin consumes:

- `metrics.LoadedModelsKey` (`map[string]datalayer.LoraLoadState`)
- `metrics.GPULoadedModelsKey` (`int`)
- `metrics.ActiveModelsKey` (`map[string]int`), to tell idle residents from busy ones

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
| `cpuResidentScore` | `float` | No | `0.7` | Adapter is only in the host cache. |
| `freeSlotScore` | `float` | No | `0.6` | Adapter not resident, a GPU slot is free. |
| `evictableScore` | `float` | No | `0.3` | Adapter not resident, slots full, an unpinned resident is idle. |
| `saturatedScore` | `float` | No | `0.0` | Adapter not resident, every slot busy or pinned. |
| `placementBonus` | `float` | No | `0.03` | Bonus for the rendezvous-hash home while the adapter is not resident there. |
| `headroomBonus` | `float` | No | `0.03` | Bonus scaled by the share of free GPU slots. |

Each score must be in `[0, 1]`, the tiers must satisfy `gpuResidentScore >= cpuResidentScore >=
freeSlotScore >= evictableScore >= saturatedScore`, and the two bonuses together must be smaller
than every non-zero gap between consecutive tiers after scaling; otherwise the whole set is
ignored, logged, and the defaults are used.

### Example

```yaml
plugins:
  - type: lora-load-state-scorer
    name: lora-load-state
    parameters:
      cpuResidentScore: 0.9
      freeSlotScore: 0.2
      evictableScore: 0.1
      saturatedScore: 0.0
      placementBonus: 0.02
      headroomBonus: 0.02
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-load-state
        weight: 2
```
