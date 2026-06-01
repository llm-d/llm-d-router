# Program-Aware Fairness Plugin

The program-aware fairness plugin introduces program-level scheduling to `llm-d-router`. Rather than treating each inference request independently, this plugin recognizes that requests belong to higher-level **agentic programs** — workflows composed of multiple LLM calls — and makes scheduling decisions based on aggregated program-level metrics.

## Overview

Agentic workloads (coding agents, research pipelines, multi-step reasoning chains) generate sequences of LLM inference requests that form a logical program. Scheduling these requests individually ignores the program-level context: one program may have consumed far more compute than another, or a program's requests may be consistently starved while others proceed.

This plugin addresses that by:

- **Identifying agentic programs** via the standard `x-gateway-inference-fairness-id` HTTP header (defined by the [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/))
- **Tracking program-level metrics** across the full request lifecycle — accumulated token usage, queue wait times, dispatch counts, and service rates
- **Making dispatch decisions at the program level** using configurable scoring strategies that consider each program's accumulated metrics rather than individual request attributes

The plugin type is `program-aware-fairness`.

## Architecture

The plugin implements four interfaces from the Gateway API Inference Extension framework, giving it visibility into the full request lifecycle:

```
Incoming Request
       |
       v
  Flow Control ─── FairnessPolicy.Pick()
       |             Evaluates program-level metrics to decide which
       |             program's queue to service next
       v
  PrepareData ──── DataProducer.PrepareRequestData()
       |             Associates request with its program, updates counters
       v
  Scheduling ───── (queue-scorer + max-score-picker select the best endpoint)
       |
       v
  PreRequest ───── PreRequest.PreRequest()
       |             Records wait time from queue entry to dispatch
       v
  Model Server
       |
       v
  ResponseComplete ─ ResponseComplete.ResponseComplete()
                       Records token usage, updates program-level state
```

Each program is identified by its `x-gateway-inference-fairness-id` header and automatically gets its own flow queue. The `Pick()` function evaluates all program queues using accumulated metrics to decide which program should be serviced next.

### How Pick() Works

The plugin builds a `map[string]QueueInfo` from all program queues and delegates to the configured `ScoringStrategy.Pick()`. Each strategy owns its own selection logic:

- **LAS and DRR** use a two-pass algorithm with adaptive normalization:
  1. **Pass 1** — Bookkeeping (decay service / allocate quantum) for all queues, then collect raw metric dimensions for non-empty queues, tracking per-dimension min/max.
  2. **Pass 2** — Normalize each dimension to `[0, 1]` using the observed min/max range, compute a weighted score, and select the highest-scoring queue.
- **RR** walks a sorted cursor through program IDs and picks the next non-empty queue.

### Token Weighting

Token costs are weighted to reflect actual compute:
- Prompt (input) tokens: weight = 1
- Completion (output) tokens: weight = 2

This means the plugin accounts for the fact that generation is roughly twice as expensive as prompt processing when evaluating how much compute a program has consumed.

## Scheduling Strategies

The plugin supports multiple scoring strategies, selected via the `strategy` config field. All strategies operate on program-level aggregated metrics rather than individual request attributes.

| Strategy | `strategy` value | Description |
|----------|-----------------|-------------|
| Round-Robin | `rr` | Cycles through programs in order, equal turns regardless of usage |
| Least-Attained Service | `las` (default) | Promotes programs that have consumed the least compute |
| Deficit Round Robin | `drr` | Token-budget based proportional fairness |

### Round-Robin

The simplest scheduling strategy. Programs are sorted by ID for deterministic ordering, and a cursor walks forward picking the next non-empty queue. Each program gets an equal turn regardless of how many tokens it has consumed or how long its requests have been waiting.

Round-robin is appropriate as a baseline or when all programs have roughly equal workloads and no program-level fairness adjustment is needed. It does not account for differences in token consumption or queue depth across programs.

**Configuration parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deferRRCursor` | `false` | When `true`, the cursor only advances in `OnPreRequest()` (after a real dispatch) rather than in `Pick()`. This prevents the cursor from advancing when `Pick()` is called but the request is not actually dispatched. |

### Least-Attained Service (default)

Tracks a time-decayed accumulator of weighted tokens consumed per program. Programs with **lower** attained service receive **higher** scores, promoting programs that have received less compute.

**Dimensions:**
| Dimension | Signal | Effect |
|-----------|--------|--------|
| Attained service (inverted) | Time-decayed weighted tokens consumed | Lower service = higher priority |
| Head-of-queue wait | Age of oldest queued request | Tiebreaker for cold start |

**How it works:**
- Each `Pick()` cycle decays every program's attained service (forgetting old usage over time)
- When a response completes, the weighted token cost is added to the program's attained service
- Scoring inverts the service dimension so programs that have consumed less compute are promoted

This is the default and recommended strategy. It directly targets equitable resource allocation — programs that have received less service are promoted, and the decay mechanism ensures that historical usage is gradually forgotten so programs are not permanently penalized for past bursts.

**Configuration parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weightService` | 0.8 | Weight for the inverted attained-service signal |
| `weightServiceHeadWait` | 0.2 | Weight for head-of-queue wait time |
| `serviceDecayFactor` | 0.995 | Per-cycle multiplicative decay (ignored if `serviceHalfLifeSeconds` is set) |
| `serviceHalfLifeSeconds` | — | Time-based decay: service halves every N seconds (overrides `serviceDecayFactor`) |

### Deficit Round Robin (DRR)

Adapted from [Shreedhar & Varghese 1995](https://dl.acm.org/doi/pdf/10.1145/217391.217453) for token-based LLM scheduling. Each active program earns a fixed token quantum per round; actual token usage is deducted at response completion. Provides provably proportional fairness regardless of request rate or size.

**Dimensions:**
| Dimension | Signal | Effect |
|-----------|--------|--------|
| Deficit counter | Quantum allocated minus tokens consumed | Positive = owed service, negative = overserved |
| Head-of-queue wait | Age of oldest queued request | Prevents starvation of new programs |

**How it works:**
- The first `Pick()` in a dispatch cycle allocates `quantumTokens` to all non-empty queues; subsequent `Pick()` calls in the same cycle only allocate to newly arrived programs (cycle-aware quantum prevents double allocation)
- Idle programs have their deficit reset to 0 (standard DRR: no credit accumulation while inactive)
- When a response completes, actual token cost is deducted from the deficit
- The program with the highest deficit (most owed service) is selected next

DRR is suited for workloads where programs have highly variable request sizes (token counts). Unlike attained-service fairness which equalizes total consumption, DRR guarantees proportional bandwidth allocation through its quantum mechanism — programs that submit large requests are naturally throttled by deficit deduction without needing explicit rate limiting.

**Configuration parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weightDeficit` | 0.8 | Weight for the deficit counter signal |
| `weightDrrHeadWait` | 0.2 | Weight for head-of-queue wait time |
| `quantumTokens` | 1000 | Token budget added per program per `Pick()` cycle |

## Configuration

The plugin is configured via an `EndpointPickerConfig` YAML file. See [`deploy/config/sim-program-aware-config.yaml`](../../../deploy/config/sim-program-aware-config.yaml) for a complete working example.

### Minimal Configuration

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

Key points:
- **`featureGates`** — Both `flowControl` and `prepareDataPlugins` must be enabled
- **`fairnessPolicyRef`** — Must reference the plugin type `program-aware-fairness` to wire it as the flow control fairness policy
- **`schedulingProfiles`** — Defines the endpoint scoring pipeline that runs after flow control dispatches a request from the selected program's queue

### Declaring a Strategy with Custom Parameters

To select a strategy and tune its weights, add a `config` block to the plugin entry:

**Round-robin:**
```yaml
plugins:
- type: program-aware-fairness
  config:
    strategy: rr
```

**LAS strategy with time-based decay:**
```yaml
plugins:
- type: program-aware-fairness
  config:
    strategy: las
    weightService: 0.9
    weightServiceHeadWait: 0.1
    serviceHalfLifeSeconds: 30
```

**DRR strategy with custom quantum:**
```yaml
plugins:
- type: program-aware-fairness
  config:
    strategy: drr
    weightDeficit: 0.7
    weightDrrHeadWait: 0.3
    quantumTokens: 2000
```

If no `config` block is provided, the plugin defaults to the **las** strategy with default weights.

## Observability

The plugin exports Prometheus metrics for monitoring program-level scheduling behavior:

| Metric | Type | Description |
|--------|------|-------------|
| `program_aware_requests_total` | Counter | Total requests per program |
| `program_aware_dispatched_total` | Counter | Total dispatched per program |
| `program_aware_wait_time_milliseconds` | Histogram | Flow-control queue wait time |
| `program_aware_ewma_wait_time_milliseconds` | Gauge | EWMA of queue wait time per program |
| `program_aware_input_tokens_total` | Counter | Prompt tokens per program |
| `program_aware_output_tokens_total` | Counter | Completion tokens per program |
| `program_aware_pick_latency_microseconds` | Histogram | `Pick()` call latency |
| `program_aware_jains_fairness_index` | Gauge | Jain's fairness index over service rates (1.0 = perfect) |
| `program_aware_attained_service_tokens` | Gauge | Current attained service per program |
| `program_aware_service_rate_tokens_per_second` | Gauge | EWMA of weighted tokens/sec per program |
| `program_aware_queue_score` | Gauge | Score computed per program during `Pick()` |

## Source Files

| File | Purpose |
|------|---------|
| `plugin.go` | Plugin struct, factory, `FairnessPolicy.Pick()` |
| `strategy.go` | `ScoringStrategy` interface, LAS, DRR, and RR implementations |
| `programmetrics.go` | Per-program metrics aggregation (EWMA, atomic counters) |
| `request_hooks.go` | Lifecycle hooks: `PrepareData`, `PreRequest`, `ResponseComplete` |
| `prometheus.go` | Prometheus metric definitions |
