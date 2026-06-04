# Program-Aware Fairness Plugin

**Type:** `program-aware-fairness`
**Interfaces:** `flowcontrol.FairnessPolicy`, `requestcontrol.DataProducer`, `requestcontrol.PreRequest`, `requestcontrol.ResponseBodyProcessor`

Program-level fairness for agentic workloads: requests are grouped by program ID, and dispatch decisions are made on aggregated per-program metrics rather than per-request attributes.

## What It Does

Agentic workloads (coding agents, research pipelines, multi-step reasoning chains) generate sequences of LLM inference requests that form a logical program. Scheduling those requests individually ignores the program-level context: one program may have consumed far more compute than another, or a program's requests may be consistently starved while others proceed.

This plugin recognizes that requests belong to higher-level programs and:

- **Identifies programs** via the `x-gateway-inference-fairness-id` HTTP header (defined by the Gateway API Inference Extension).
- **Tracks program-level metrics** across the full request lifecycle — accumulated token usage, queue wait times, dispatch counts, and service rates.
- **Selects which program to dispatch next** using a configurable scoring strategy that compares per-program metrics.

Use it when distinct workflows or tenants share the same inference pool and you want fair allocation at the workflow level — not just per-request fairness.

## How It Works

The plugin participates in the request lifecycle at four points:

```
Incoming Request
       |
       v
  Flow Control ─── FairnessPolicy.Pick()
       |             Picks which program's queue to service next.
       v
  Produce ──────── DataProducer.Produce()
       |             Associates the request with its program; updates counters.
       v
  Scheduling ───── (queue-scorer + max-score-picker select the endpoint)
       |
       v
  PreRequest ───── PreRequest.PreRequest()
       |             Records flow-control queue wait time.
       v
  Model Server
       |
       v
  ResponseBody ──── ResponseBodyProcessor.ResponseBody()
                       On EndOfStream: records token usage and updates
                       per-program EWMA / attained service.
```

Each program is identified by its `x-gateway-inference-fairness-id` header and gets its own flow queue.

### Picking a queue

`Pick()` builds a `map[string]QueueInfo` from all program queues in the priority band and delegates to the configured `ScoringStrategy`:

- **LAS** and **DRR** use a two-pass algorithm with adaptive normalization:
  1. *Pass 1* — Bookkeeping (decay inactive queues / allocate quantum to non-empty queues), then collect raw metric dimensions for non-empty queues, tracking per-dimension min/max.
  2. *Pass 2* — Normalize each dimension to `[0, 1]` using the observed range, compute a weighted score, and select the highest-scoring queue.
- **RR** walks a sorted cursor through program IDs and picks the next non-empty queue.

The selected item's enqueue time is stashed on the `*scheduling.InferenceRequest` itself so `PreRequest` can compute the actual flow-control wait without a side map.

### Token weighting

Token costs are weighted to reflect compute:

| Token type | Weight |
|---|---|
| Prompt (input) | 1 |
| Completion (output) | 2 |

Generation is roughly twice as expensive as prompt processing; the plugin accounts for that when comparing how much compute a program has consumed.

### Strategies

The `strategy` config field selects the scoring algorithm. All strategies operate on per-program aggregated metrics.

| Strategy | `strategy` value | When to use |
|---|---|---|
| Round-Robin | `rr` | Baseline; equal turns regardless of usage |
| Least-Attained Service | `las` (default) | Equitable resource allocation; promotes underserved programs |
| Deficit Round Robin | `drr` | Highly variable request sizes; proportional bandwidth |

**Round-Robin.** Programs are sorted by ID for deterministic ordering, and a cursor walks forward picking the next non-empty queue. Each program gets an equal turn regardless of how many tokens it has consumed or how long its requests have been waiting. Appropriate when all programs have roughly equal workloads.

**Least-Attained Service.** Tracks a time-decayed accumulator of weighted tokens consumed per program. Programs with **lower** attained service receive **higher** scores.

| Dimension | Signal | Effect |
|---|---|---|
| Attained service (inverted) | Time-decayed weighted tokens consumed | Lower service = higher priority |
| Head-of-queue wait | Age of oldest queued request | Tiebreaker for cold start |

Active programs accumulate service without decay so persistent heavy users stay deprioritized; idle programs lose stale service so they can compete on return. On each completion the weighted token cost is added to the program's attained service. Decay is skipped while a request is in flight to preserve the upcoming `OnCompleted` `AddService`.

**Deficit Round Robin.** Adapted from Shreedhar & Varghese 1995 for token-based scheduling. Each non-empty program earns a fixed token quantum per `Pick()` cycle; actual token cost is deducted at response completion. Provides provably proportional fairness regardless of request rate or size.

| Dimension | Signal | Effect |
|---|---|---|
| Deficit counter | Quantum allocated minus tokens consumed | Positive = owed service, negative = overserved |
| Head-of-queue wait | Age of oldest queued request | Prevents starvation of new programs |

Inactive programs (queue empty and no in-flight request) have their deficit decayed so stale credit shrinks toward zero. The program with the highest deficit (most owed service) is selected next.

### Eviction

The per-program metrics map is bounded by a periodic sweeper. A program with no completed requests inside `evictionTtlSeconds` (default 1 hour) and no in-flight work is dropped. The next request from that program reallocates fresh metrics. Default deficit half-life of 60 s means an hour-idle program's accumulators are already near zero, so eviction is observationally indistinguishable from natural decay. Set `evictionTtlSeconds: 0` to disable eviction entirely.

## Inputs Consumed

| Input | Source | Required | Notes |
|---|---|---|---|
| `x-gateway-inference-fairness-id` header | HTTP request | No | Falls back to a default fairness ID when absent; all unidentified requests share one queue |
| Token usage | Model-server response (`Usage.PromptTokens`, `Usage.CompletionTokens`) | Yes | Read on the final stream chunk (`response.EndOfStream == true`) |
| Flow-control enqueue time | `flowcontrol.QueueItemAccessor.EnqueueTime()` | Yes | Stashed on the request's attribute store at `Pick()`, read at `PreRequest()` |

## Configuration

**Location:** plugin entry under top-level `plugins`; referenced from `flowControl.defaultPriorityBand.fairnessPolicyRef`.
**Enabled by default:** No. Both `flowControl` and `prepareDataPlugins` feature gates must be enabled.

### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `strategy` | string | no | `las` | One of `las`, `drr`, `rr` |
| `weightDeficit` | float | no | `0.8` | DRR: weight for the deficit signal |
| `weightDrrHeadWait` | float | no | `0.2` | DRR: weight for head-of-queue wait |
| `quantumTokens` | int | no | `1000` | DRR: token budget added per Pick() per non-empty queue |
| `deficitHalfLifeSeconds` | float | no | `60` | DRR: half-life of inactive-program deficit decay; `0` disables time-based decay |
| `deficitDecayFactor` | float | no | `0` | DRR: per-Pick decay factor when time-based decay is disabled. Must be in `[0, 1)`; `0` disables |
| `weightService` | float | no | `0.8` | LAS: weight for the inverted attained-service signal |
| `weightServiceHeadWait` | float | no | `0.2` | LAS: weight for head-of-queue wait |
| `serviceDecayFactor` | float | no | `0.995` | LAS: per-cycle multiplicative decay. Must be in `(0, 1]`. Ignored when `serviceHalfLifeSeconds` is set |
| `serviceHalfLifeSeconds` | float | no | `0` | LAS: half-life of attained-service decay when set; overrides `serviceDecayFactor` |
| `evictionTtlSeconds` | float | no | `3600` | Programs with no completions in this window are evicted from the metrics map. `0` disables eviction |
| `evictionSweepSeconds` | float | no | `300` | Eviction sweep cadence. Must be `> 0` |

### Examples

Minimal config (default LAS strategy):

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: program-aware-fairness
  - type: queue-scorer
  - type: max-score-picker
  - type: single-profile-handler

featureGates:
  - flowControl
  - prepareDataPlugins

flowControl:
  defaultPriorityBand:
    fairnessPolicyRef: program-aware-fairness

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: max-score-picker
```

Round-robin:

```yaml
plugins:
  - type: program-aware-fairness
    config:
      strategy: rr
```

LAS with time-based decay:

```yaml
plugins:
  - type: program-aware-fairness
    config:
      strategy: las
      weightService: 0.9
      weightServiceHeadWait: 0.1
      serviceHalfLifeSeconds: 30
```

DRR with custom quantum:

```yaml
plugins:
  - type: program-aware-fairness
    config:
      strategy: drr
      weightDeficit: 0.7
      weightDrrHeadWait: 0.3
      quantumTokens: 2000
```

### Observability

| Metric | Type | Description |
|---|---|---|
| `program_aware_requests_total` | Counter | Total requests per program |
| `program_aware_dispatched_total` | Counter | Total dispatched per program |
| `program_aware_ewma_wait_time_milliseconds` | Gauge | EWMA of queue wait time per program |
| `program_aware_input_tokens_total` | Counter | Prompt tokens per program |
| `program_aware_output_tokens_total` | Counter | Completion tokens per program |
| `program_aware_pick_latency_microseconds` | Histogram | `Pick()` call latency |
| `program_aware_jains_fairness_index` | Gauge | Jain's fairness index over service rates (1.0 = perfect) |
| `program_aware_attained_service_tokens` | Gauge | Current attained service per program |
| `program_aware_service_rate_tokens_per_second` | Gauge | EWMA of weighted tokens/sec per program |
| `program_aware_queue_score` | Gauge | Score computed per program during `Pick()` |
| `program_aware_deficit_tokens` | Gauge | DRR deficit counter per program |

## Limitations

- Requests without the `x-gateway-inference-fairness-id` header fall into a single shared default queue and receive no per-program isolation.
- `rr` does not track tokens or service; programs are equal turns regardless of consumed compute.
- Eviction reset: a program whose metrics are dropped (idle past `evictionTtlSeconds`) starts with zero deficit / attained service on its next request. With default 60 s deficit half-life and 1-hour TTL, accumulated state is already near zero by the time eviction fires, so this is observationally indistinguishable from natural decay.
- The plugin requires the `flowControl` and `prepareDataPlugins` feature gates; it cannot run without them.
- Token weighting (input=1, output=2) is hard-coded; there is no config knob today.

## Related Documentation

- Gateway API Inference Extension fairness contract: <https://gateway-api-inference-extension.sigs.k8s.io/>
- Sibling fairness plugins in this repo: `roundrobin`, `globalstrict` under `pkg/epp/framework/plugins/flowcontrol/fairness/`
- DRR paper — Shreedhar & Varghese, 1995: <https://dl.acm.org/doi/pdf/10.1145/217391.217453>
- Example config: `deploy/config/sim-program-aware-config.yaml`
