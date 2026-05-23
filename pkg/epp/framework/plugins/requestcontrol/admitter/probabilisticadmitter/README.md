# Probabilistic Admitter (`probabilistic-admitter`)

Probabilistically sheds sheddable requests under load while always admitting critical requests.

## Interface

Admitter

## When to Use

Use this plugin when you need gradual, probabilistic load shedding for lower-priority traffic
rather than a hard binary gate. It complements flow-control by shedding requests *before* they
enter the scheduling pipeline, reducing queuing pressure on the cluster. Flow-control operates
after scheduling; this plugin operates before.

## Behavior

Critical and standard requests (priority >= 0) are always admitted.

For sheddable requests (priority < 0), the plugin computes cluster saturation using the
roofline formula and rejects with probability `p = min(sat^power * k, 1.0)`:

```
saturation = avg over pods of: max(WaitingQueueSize / queueDepthThreshold, KVCacheUsagePercent / kvCacheUtilThreshold)
```

| Condition | Behavior |
|-----------|----------|
| `priority >= 0` | Always admit |
| `len(pods) == 0` | Admit (safe default) |
| `request == nil` | Admit |
| Pod has nil metrics | Treated as fully saturated (score = 1.0) |

With defaults (`power=5, k=300`), shedding is ~2.3% at saturation 0.15 and reaches 100% at
saturation ≈ 0.34. This creates a dead zone at low load with a steep ramp at moderate overload.

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `queueDepthThreshold` | `5` | Per-pod waiting queue depth at which saturation contribution = 1.0 |
| `kvCacheUtilThreshold` | `0.8` | Per-pod KV cache fraction (0.0–1.0) at which saturation contribution = 1.0 |
| `power` | `5.0` | Controls the steepness of the shedding curve. Higher values create a longer dead zone at low saturation with a sharper ramp-up. Must be > 0. |
| `k` | `300.0` | Controls where the shedding gate kicks in on the saturation axis. 100% shed occurs at `saturation = (1/k)^(1/power)`. With defaults: `(1/300)^(1/5) ≈ 0.34`. Must be > 0. |

### Tuning Guidance

- **`power`**: Start with 5 (quintic). Lower values (e.g., 2–3) create a more gradual onset;
  higher values (e.g., 7–10) sharpen the cliff. Values below 1 create a concave curve that
  sheds aggressively even at low saturation.
- **`k`**: Determines the saturation point at which all sheddable traffic is dropped. The
  relationship is `saturation_100pct = (1/k)^(1/power)`. For the default power=5:
  - `k=100` → 100% shed at saturation ≈ 0.40
  - `k=300` → 100% shed at saturation ≈ 0.34
  - `k=1000` → 100% shed at saturation ≈ 0.25
- **`queueDepthThreshold` / `kvCacheUtilThreshold`**: Set these to the point at which a single
  pod should be considered "fully loaded." The defaults (QD=5, KV=0.8) work for typical vLLM
  deployments; adjust if your model has unusually deep prefill queues or low KV headroom.

### Example Configuration

See [`deploy/config/probabilistic-admitter-epp-config.yaml`](../../../../../../../deploy/config/probabilistic-admitter-epp-config.yaml) for the full EPP config. The
key snippet:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: probabilistic-admitter
- type: random-picker  # feel free to plug in other scorers/pickers here
- type: utilization-detector
  parameters:
    queueDepthThreshold: 999999999
    kvCacheUtilThreshold: 1.0
saturationDetector:
  pluginRef: utilization-detector
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
```

The `utilization-detector` is the system's built-in saturation gate - it normally rejects
all sheddable requests once queue depth or KV cache crosses its thresholds (a hard binary
cutoff). Setting `queueDepthThreshold: 999999999` and `kvCacheUtilThreshold: 1.0` makes
that gate effectively unreachable so it never fires, letting the `probabilistic-admitter` be
the sole decision-maker for shedding using its gradual probability curve instead of the
system's all-or-nothing behavior. The `saturationDetector: pluginRef: utilization-detector`
line is still required because the EPP framework expects a saturation detector to be
registered - you can't omit it, only neuter it.

## Deploying

### 1. Create InferenceObjective resources

Define one `InferenceObjective` per traffic tier pointing to the same model and pool.

**Sheddable tier** - batch jobs, background processing, non-interactive workloads:

```yaml
apiVersion: llm-d.ai/v1alpha2
kind: InferenceObjective
metadata:
  name: batch-workload
  namespace: default
spec:
  modelName: my-model
  poolRef:
    name: my-inference-pool
  priority: -1
```

**Critical tier** - interactive workloads, SLO-bound traffic:

```yaml
apiVersion: llm-d.ai/v1alpha2
kind: InferenceObjective
metadata:
  name: interactive-workload
  namespace: default
spec:
  modelName: my-model
  poolRef:
    name: my-inference-pool
  priority: 1
```

### 2. Deploy the InferencePool

Deploy your `InferencePool` resource. The pool must exist before the EPP can connect to it.

### 3. Apply the EPP config

Use `deploy/config/probabilistic-admitter-epp-config.yaml` as the `EndpointPickerConfig`
for your EPP deployment.

The config disables the system-level saturation gate (by setting the `utilization-detector`
to effectively infinite thresholds) so that all admission decisions are made by the
`probabilistic-admitter`.

## Benchmark Results

### Shedding Curve

With the default parameters (`power=5`, `k=300`, `queueDepthThreshold=5`,
`kvCacheUtilThreshold=0.8`):

| Cluster Saturation | Shedding Probability | Effect |
|--------------------|----------------------|--------|
| 0.00 – 0.08 | < 0.01% | Dead zone - no observable shedding |
| 0.15 | ~2.3% | Onset - occasional shed, negligible impact on throughput |
| 0.20 | ~9.6% | Moderate shedding begins |
| 0.30 | ~73% | Heavy shedding - most sheddable traffic rejected |
| 0.34+ | 100% | Full shed - all sheddable traffic rejected |

This curve is intentionally steep: it preserves throughput for sheddable traffic under normal
load and ramps aggressively only when the cluster is approaching capacity, leaving headroom
for critical traffic.

### Empirical Results

Benchmarked against the baseline binary saturation gate (`utilization-detector` at default
thresholds) on Qwen3-14B with five workload profiles (chatbot, w1, w2, codecompletion,
blindspot) at two load levels (under, mid).

**Critical traffic protection:** 0% error rate in 9/10 scenarios (one had 0.01% - 1 error
out of 8,013 requests). Baseline also achieves 0% for critical traffic, so the probabilistic
admitter maintains parity.

**Critical traffic latency (P99 TTFT):**

| Workload | Baseline | Treatment | Change |
|----------|----------|-----------|--------|
| w1_mid | 5,573 ms | 5,208 ms | -7% |
| w2_mid | 6,069 ms | 5,271 ms | -13% |
| codecompletion_mid | 14,422 ms | 4,540 ms | -69% |
| blindspot_mid | 23,204 ms | 696 ms | -97% |

**Critical traffic throughput (P50 TPS):**

| Workload | Baseline | Treatment | Change |
|----------|----------|-----------|--------|
| w1_mid | 16.23 | 25.92 | +60% |
| w2_mid | 30.35 | 40.19 | +32% |
| codecompletion_mid | 10.68 | 22.81 | +114% |
| codecompletion_under | 13.47 | 30.64 | +127% |

**Total goodput trade-off:** The admitter trades 5–20% of aggregate throughput (8–10% in
typical scenarios, up to 40% in the blindspot workload) to achieve the latency improvements
above. This is the intended behavior - shedding sheddable traffic earlier to protect critical
request quality.

### Full Data

The complete benchmark dataset is available at the llm-d-benchmark
[Google Drive](https://drive.google.com/drive/u/3/folders/12OOkXBEmGCobRmjILxC1KqZx7F3pGJJN).
It contains:

- **`generated/`** - The EPP configs and plugin code used for the benchmark run.
  `baseline_config.yaml` is a plain `random-picker` scheduler (no admission control);
  `treatment_config.yaml` adds the `probabilistic-admitter` with the neutered
  `utilization-detector`.
- **`results/baseline/`** and **`results/treatment/`** - Per-workload trace data. Each
  subdirectory (e.g., `w1_mid`, `codecompletion_under`) contains:
  - `trace_header.yaml` - Run metadata (model, workload spec, warm-up count).
  - `trace_data.csv` - One row per request with columns for `client_id` (critical/sheddable),
    `status` (ok/error), `arrival_time_us`, `first_chunk_time_us`, `last_chunk_time_us`,
    `output_tokens`, and `error_message`. Compute TTFT as
    `first_chunk_time_us - send_time_us` and TPS as
    `output_tokens / ((last_chunk_time_us - first_chunk_time_us) / 1e6)`.
- **`deploy_comparison_table.txt`** - Pre-computed side-by-side comparison of TTFT, TPOT, and
  E2E latency (mean / P50 / P99) for critical traffic across all workloads, with absolute
  deltas and percentage changes.

## Relationship to Flow Control

This plugin is designed to **complement** flow control, not replace it.

**Today:** Flow control in llm-d is still being finalized and formalized. The probabilistic
admitter gives operators an immediate, deployable mechanism for probabilistic load shedding
that protects premium traffic classes. By assigning negative priorities to sheddable tiers
and positive priorities to critical tiers, you get graceful degradation under load right now, without waiting for a full flow-control implementation.

**Future:** Once flow control is formalized, the probabilistic shedding algorithm here is
intended to be adapted as a shedding behavior *within* the flow-control framework. The core
math (roofline saturation → rejection probability) remains the same; what changes is the
integration point, moving from a standalone admitter to a flow-control-managed component
that participates in a broader admission and rate-limiting pipeline.

### Recommended Use Cases

| Scenario | Example | Recommendation |
|----------|---------|----------------|
| Interactive + batch on a shared cluster | A chatbot serving live users (`priority=1`) shares a GPU pool with nightly summarization jobs (`priority=-1`). During peak hours the cluster saturates. | Use now. The admitter sheds summarization requests as saturation rises, keeping chatbot P99 TTFT low without manual scaling. |
| Code completion + bulk embedding | An IDE code-completion endpoint (`priority=1`) coexists with a bulk document-embedding pipeline (`priority=-1`) on the same inference pool. | Use now. Embedding requests are shed first under load, protecting the latency-sensitive completion path that directly affects developer experience. |
| Single-tier API with no priority split | A single model serves one class of traffic with no sheddable tier, but you want a safety valve against traffic spikes. | Consider using with aggressive thresholds (`k >= 1000`) so shedding only kicks in near full saturation. This provides a backstop until flow control is available. |
| Flow control is deployed | The flow-control framework is finalized and managing admission and rate-limiting across the pipeline. | Migrate this plugin's shedding behavior into the flow-control configuration. The probabilistic curve parameters (`power`, `k`) carry over directly. |

## Dependencies

None. The plugin reads `WaitingQueueSize` and `KVCacheUsagePercent` directly from endpoint
metrics - no data producer prerequisite.
