# EPP tracing overhead analysis

## Decision and conclusions

1. Across the clean, regular workloads, enabling standard EPP tracing produced
   no major end-to-end throughput or latency impact. This result supports using
   a 10% sampling ratio when tracing is enabled.
2. Tracing is not free at high sampling ratios. At 100% sampling, tracing frames
   accounted for 3.15% to 4.36% of sampled EPP CPU, and EPP-internal scheduler
   p99 increased in both focused workloads. A 100% ratio is appropriate for
   diagnosis rather than routine operation.
3. The 800 requests/s and concurrency-128 stages are corner cases in this data.
   They failed at every tracing ratio, including tracing off, because the load
   generator reached its connection limit. They do not show a tracing-specific
   regression, and tracing impact under sustained extreme concurrency requires
   a separate clean measurement.

Keep tracing disabled unless an OTLP collector is configured. Every measured
run used a collector, so this study does not support enabling tracing by
default with the chart's localhost exporter endpoint.

## Primary experiment

The primary workload used GuideLLM at a fixed 400 requests/s with 200 input
tokens and 100 output tokens. Responses were not streamed. Each run lasted 300
seconds and targeted one EPP replica. The matrix compared tracing disabled with
standard tracing at 1%, 10%, and 100%. A separate 100% run collected a 30-second
CPU profile and heap profile after a 60-second delay. Profiled runs are excluded
from request-latency comparisons because CPU profiling changes the measurement.
Each final configuration was run once.

The simulator deployment removes model execution as the dominant source of
latency. The live deployment uses Qwen/Qwen3-0.6B with eight model-serving
replicas and checks that the result remains representative with real inference.

### EPP-only simulator

| Tracing | Throughput (requests/s) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | EPP CPU (cores) | EPP memory (MiB) | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | 401.513 | 138.343 | 138.079 | 142.063 | 148.062 | 0.818 | 116.04 | 0 |
| 1% | 401.510 | 138.404 | 138.119 | 141.964 | 148.077 | 0.842 | 107.83 | 0 |
| 10% | 401.523 | 138.348 | 138.051 | 141.988 | 148.505 | 0.762 | 108.97 | 0 |
| 100% | 401.527 | 138.570 | 138.166 | 142.262 | 150.479 | 0.839 | 115.28 | 0 |

At 100%, throughput changed by less than 0.01%, mean latency increased 0.16%,
p95 increased 0.14%, and p99 increased 1.63%. Average EPP CPU increased 2.56%.
The 10% run used 6.8% less EPP CPU than tracing off. These one-run CPU values do
not establish a tracing-related CPU delta.

The simulator configured 20 ms to first token and 1 ms between 100 output
tokens. This accounts for about 120 ms of the observed 138 ms end-to-end
latency and makes the end-to-end metric insensitive to sub-millisecond EPP
changes.

### Live inference

| Tracing | Throughput (requests/s) | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | EPP CPU (cores) | EPP memory (MiB) | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | 401.393 | 274.723 | 268.751 | 336.185 | 397.598 | 1.195 | 336.36 | 0 |
| 1% | 401.373 | 276.427 | 270.321 | 338.093 | 400.795 | 1.216 | 329.69 | 2 |
| 10% | 401.393 | 252.600 | 248.228 | 299.125 | 365.210 | 1.083 | 330.04 | 0 |
| 100% | 401.390 | 275.117 | 269.123 | 333.458 | 409.277 | 1.239 | 338.02 | 0 |

The live result sustained the offered rate at every ratio. The 100% run changed
mean latency by 0.14%, p95 by -0.81%, and p99 by 2.94% relative to tracing off.
The 10% run was about 8% faster than tracing off, and the profiled 100% run was
faster than the unprofiled 100% run. These non-monotonic results prevent a
request-latency attribution from one run per cell. The 1% cell had two 503
responses and is not a clean comparison cell.

### EPP-internal latency

The following values are means of the 10-second Prometheus histogram-quantile
samples. Durations are in microseconds.

| Target | Tracing | Scheduler p50 | Scheduler p95 | Scheduler p99 | Request processing p99 | Response processing p99 |
|---|---:|---:|---:|---:|---:|---:|
| Simulator | Off | 50.66 | 96.26 | 138.80 | 1443.50 | 234.35 |
| Simulator | 1% | 50.66 | 96.26 | 141.39 | 1482.78 | 236.37 |
| Simulator | 10% | 50.72 | 96.37 | 152.15 | 1455.23 | 234.43 |
| Simulator | 100% | 51.34 | 97.55 | 258.63 | 1605.64 | 238.77 |
| Live | Off | 50.65 | 96.23 | 125.18 | 1799.18 | 242.69 |
| Live | 1% | 50.78 | 96.47 | 138.82 | 1810.55 | 243.39 |
| Live | 10% | 50.50 | 95.94 | 109.30 | 1379.12 | 234.52 |
| Live | 100% | 52.68 | 103.36 | 187.11 | 1808.30 | 242.85 |

At 100%, the simulator scheduler p99 increased from 138.80 to 258.63
microseconds while p50 increased by less than one microsecond. The live
scheduler p99 increased from 125.18 to 187.11 microseconds. The non-monotonic
intermediate ratios and single run per cell limit comparison of smaller
changes. The CPU-throttling query returned no series for all eight runs, so
throttling was not measured as zero and cannot be assessed from this cohort.

## CPU profile attribution

Both 100% diagnostic runs captured every ready EPP replica successfully. The
[pprof instructions](epp_tracing_overhead/pprof) describe how to collect and
inspect profiles during a diagnostic run.

| Target | Profile samples | Tracer handle lookup cumulative CPU | Batch processor cumulative CPU | Exporter cumulative CPU | Samples with any tracing frame |
|---|---:|---:|---:|---:|---:|
| Simulator | 26.81 CPU-s / 30 s | 0.28 s (1.04%) | 0.40 s (1.49%) | 0.36 s (1.34%) | 1.17 s (4.36%) |
| Live | 36.80 CPU-s / 30 s | 0.35 s (0.95%) | 0.48 s (1.30%) | 0.40 s (1.09%) | 1.16 s (3.15%) |

The tracer handle lookup obtains and configures a tracer. It does not represent
span creation. The final column counts each sample once when its stack contains
either a router tracing frame or an OpenTelemetry frame. It includes span
creation and completion, propagation, attribute construction, batching, and
export without double-counting overlapping stacks. This is a cumulative-stack
attribution, not a comparison against an unprofiled CPU baseline.

## Workload sensitivity

The broad EPP-only matrix covered 25, 100, 400, and 800 offered requests/s;
concurrency 1, 32, and 128; 200/1, 200/100, 200/1000, and 8000/100 token
shapes; tracing disabled; and 0%, 1%, 10%, 50%, and 100% sampling. Clean
non-streaming runs recorded no request errors. Across the complete matrix, the
median 100%-versus-off change was -0.002% for throughput, 0.015% for mean
latency, and -0.091% for p99 latency.

The equivalent streaming matrix produced similar clean-run medians: about 0%
throughput, 0.079% mean latency, and 0.181% p99 latency at 100%. The 800
requests/s and concurrency-128 stages are excluded. All sampling ratios failed
at those stages with `httpx.ConnectError: All connection attempts failed`. The
failure was reproducible while the EPP, gateway, and simulator remained
healthy. The failed stages are excluded from the tracing comparison. An AIPerf
sweep is also excluded because every run had request errors and model execution
dominated its latency and throughput.

## Interpretation and limits

The 0% control enables the tracing SDK with a configured root sampling ratio of
zero. The sampler follows an incoming parent's decision, so the configured
ratio does not prove the effective ratio. The saved metrics do not contain
collector span counts. The configured 1%, 10%, and 100% ratios therefore were
not independently verified. The reproduction procedure compares accepted span
counts with EPP request counts to detect parent-sampling overrides.

Increasing the configured ratio did not produce a monotonic end-to-end request
latency change. CPU profiles show that samples with tracing frames account for
roughly 3% to 4.5% of sampled EPP CPU at 100% sampling.

The final fixed-load simulator and live cells each contain one run. They do not
measure run-to-run variation, and sub-percent request differences are not
treated as effect estimates. The EPP-internal latency and CPU profiles establish
that tracing executes measurable work. Three matched repetitions of the fixed
matrix are required to estimate its request-level effect with confidence.

## Reproduction

Native Kubernetes and Helm deployment definitions, GuideLLM commands, pprof
collection, and Prometheus queries are stored in
[`epp_tracing_overhead`](epp_tracing_overhead).
