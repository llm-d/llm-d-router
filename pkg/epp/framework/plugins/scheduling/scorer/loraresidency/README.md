# LoRA Residency Scorer Plugin

**Type:** `lora-residency-scorer`

Scores candidate endpoints for a LoRA request by the estimated cost of serving the adapter there. The estimate is a router-side model of which adapters each vLLM pod holds on GPU and in its CPU cache, plus a per-adapter demand rate used to price the evictions a placement would cause.

## What it does

vLLM keeps adapters in two LRU tiers: up to `max_loras` active on GPU and up to `max_cpu_loras` in CPU memory. Every engine step touches each adapter in the scheduled batch. An adapter in the CPU cache only needs a CPU-to-GPU copy. An adapter in neither tier is loaded from storage, which costs much more.

The plugin keeps one recency order per pod, built from evidence that an adapter executed on it:

- **Routed** (PreRequest): the request is pending. Recency and residency do not change, because vLLM can leave a waiting request queued while its adapter does not fit.
- **Started** (first successful response chunk): the adapter executed and is touched.
- **Finished** (end of stream): a started request touches the adapter again. A request that ends before it started leaves no trace.

The `max_lora` label of `vllm:lora_requests_info`, scraped as `MaxActiveModels`, sets each pod's GPU slots. Adapters reported active in `ActiveModels` (running or queued) are never chosen as GPU victims, and they count toward a pod being blocked. They do not make an adapter resident, because a queued adapter need not be loaded.

Endpoint lifecycle events keep the estimate per endpoint object. A deleted or replaced pod starts empty, and callbacks for requests scheduled on an older object are ignored.

### Scoring formula

Per candidate pod, in units of one load from storage:

```text
cost = tierCost
     + blockedCost                                   if every GPU slot is running or reported active
     + evictionWeight * (share(gpuVictim) * copyCost + share(cpuVictim) * loadCost)
```

- **`tierCost`**: `0` on GPU, `pendingCost` if an earlier request for the adapter is routed to the pod but has not started, `copyCost` if only in the CPU cache, `loadCost` if in neither tier.
- **Victims** follow vLLM's load order. An absent adapter is first loaded into the CPU cache, dropping its oldest idle entry to storage when the cache is full (`cpuVictim`). It is then activated, demoting the oldest idle GPU adapter to the CPU cache when no slot is free (`gpuVictim`). With `max_cpu_loras == max_loras` the two coincide and the victim costs a load.
- **`share(a)`** is adapter `a`'s fraction of the total estimated request rate, an exponentially decaying count with half-life `demandHalfLife`.
- No eviction is charged when the request is pending, because the earlier request already caused it.

Scores are min-max normalized over the candidates: the cheapest pod scores `1.0`, the costliest `0.0`, and equal costs score every pod `1.0`. A request whose target model is listed in `baseModels` scores every pod `1.0`, leaving the choice to the other scorers.

## Inputs consumed

- `metrics.ActiveModelsKey` (`map[string]int`)
- endpoint metric `MaxActiveModels`

The plugin also implements PreRequest, ResponseHeader and ResponseBody, and registers an endpoint-lifecycle extractor. The runner wires a configured plugin into each of these, so declaring it once as a scorer is enough.

## Configuration

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `baseModels` | list of strings | The pool's base model names. Required: the target model of a base-model request cannot be told apart from a cold adapter by the metrics. | none |
| `maxCpuLoras` | integer | The pods' `--max-cpu-loras`. `0` means vLLM's default, `max_cpu_loras = max_loras`. | `0` |
| `demandHalfLife` | duration | Half-life of the per-adapter demand estimate. | `60s` |
| `copyCost` | float | Cost of a CPU-to-GPU copy. | `0.2` |
| `loadCost` | float | Cost of a load from storage. | `1.0` |
| `blockedCost` | float | Added when every GPU slot is running or reported active. | `1.0` |
| `pendingCost` | float | Tier cost when a load for the adapter is already under way on the pod. | `0.1` |
| `evictionWeight` | float | Scale of the eviction term. | `1.0` |

All costs must be finite and non-negative.

### Example

```yaml
plugins:
  - type: lora-residency-scorer
    name: lora-residency
    parameters:
      baseModels: ["meta-llama/Llama-3.1-8B-Instruct"]
      maxCpuLoras: 16
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lora-residency
        weight: 4
```

The plugin is alpha and needs `--allow-experimental-plugins`.

## Operational notes

- The estimate covers only traffic this EPP routes. Requests sent to the pods another way, and adapters loaded through the vLLM API without a request, are not seen.
- With more than one EPP replica, only the leader routes, so the estimate is that replica's. A newly elected leader starts empty.
- Until a pod reports `max_lora`, it is treated as having one GPU slot. vLLM publishes the gauge only after a step, so an idle pod reports no value (see llm-d/llm-d-router#3361).
- The default costs were not tuned. On H100 with vLLM v0.31.0, a CPU-to-GPU copy measured 0.03 to 0.24 of a load from storage, falling with rank.
