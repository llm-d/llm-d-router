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

Residency is polled with the other metrics (every `--refresh-metrics-interval`, 50 ms by default)
and the gauges flip when the load completes, so an adapter's first requests can arrive while it
is still loading and see it as absent. The placement bonus exists for that window: they all go to
the same endpoint, which becomes the home once the load finishes, instead of each triggering a
load somewhere else.

A request for the base model itself (the `model_name` the server stamps on its metrics) needs no
adapter. It scores by the share of GPU slots not serving an adapter right now (a resident adapter
with nothing in flight does not count), so base-model traffic drifts away from pods that are
batching LoRA work; once every endpoint is busy the term is the same everywhere and the other
scorers decide.

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
for the metrics the model server must expose and which vLLM version carries them. The core
metrics extractor reads them for vLLM by default; other engines set `loraLoadedSpec`,
`loraGPULoadedSpec` and `loraGPUSlotsSpec` in their `engineConfigs` entry. Servers without the
gauges are scored by the capacity tiers alone, so the scorer is safe to enable on a mixed fleet.

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
| `loadHorizonSeconds` | `float` | No | `0` | Price misses from observed transition times instead of the fixed tiers (see below). `0` disables. |

Each score must be in `[0, 1]`, the tiers must satisfy `gpuResidentScore >= cpuResidentScore >=
freeSlotScore >= evictableScore >= saturatedScore`, and the two bonuses together must be smaller
than every non-zero gap between consecutive tiers after scaling; otherwise the whole set is
ignored, logged, and the defaults are used.

### Pricing misses from observed load times

With `loadHorizonSeconds` set, the three miss tiers stop being fixed numbers. Each model server
reports how long its adapter transitions took (`vllm:lora_adapter_load_seconds`, by transition),
and an endpoint that would need `t` seconds to make the adapter servable scores
`gpuResidentScore * (1 - t / loadHorizonSeconds)`, floored at `saturatedScore`:

- host-cache resident: `t` is the endpoint's mean activation time
- free GPU slot: `t` is mean load plus mean activation
- evictable: the same, scaled down by `evictableScore / freeSlotScore`

An endpoint that has not reported a transition yet is priced at the fleet mean; with no reports
anywhere the fixed tiers apply, so the parameter is safe to leave on against servers that lack the
histogram. The horizon is the TTFT you are willing to trade for adapter locality: a 1 GB adapter
that takes 0.4 s to load scores 0.8 against a 2 s horizon and 0.2 against a 0.5 s one. The
bonus-versus-tier-gap guarantee above holds for the fixed tiers only; priced tiers can sit closer
together than the bonus budget.

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
