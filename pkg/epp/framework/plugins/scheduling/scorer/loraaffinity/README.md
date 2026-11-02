# LoRA Affinity Scorer Plugin

**Type:** `lora-affinity-scorer`

This plugin scores candidate endpoints based on whether a requested LoRA adapter is already loaded (or likely to load quickly) on each model server.

## What it does

For each candidate endpoint, the plugin checks endpoint metrics for the request's `targetModel` and assigns:

- `1.0`: target model is already active on endpoint (`ActiveModels` contains target)
- `0.8`: target model is not active, but endpoint still has capacity to load more models, or has not reported its capacity yet
- `0.6`: target model is already waiting to be loaded (`WaitingModels` contains target)
- `0.0`: endpoint is at capacity and target model is neither active nor waiting

## Scheduling intent

The scorer returns category `Affinity`, preferring endpoints with higher probability of immediate adapter reuse and lower model-load latency.

## Inputs consumed

The plugin consumes:

- `metrics.ActiveModelsKey` (`map[string]int`)
- `metrics.WaitingModelsKey` (`map[string]int`)

It also relies on endpoint metric `MaxActiveModels` to determine remaining adapter capacity.

`MaxActiveModels` is `0` until the model server has reported `max_lora`. vLLM only publishes `vllm:lora_requests_info` after its first engine step, so an endpoint that has not served a request yet reports nothing. The scorer treats that as unknown capacity and scores it `0.8`, the same as an endpoint with a free slot. Scoring it `0.0` would keep requests away from it, and it would never report.

A model server that never reports `max_lora` at all, for example a frontend that omits the label, keeps scoring `0.8` for adapters it is not running. The scorer cannot tell when such an endpoint is full.

## Configuration

This scorer currently has no runtime parameters.

**Configuration Example:**
```yaml
plugins:
  - type: lora-affinity-scorer
    name: lora-affinity
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-affinity
        weight: 1
```
