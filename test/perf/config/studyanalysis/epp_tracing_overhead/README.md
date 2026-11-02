# Reproducing the EPP tracing overhead study

This directory contains the deployment definitions, native GuideLLM commands,
Prometheus queries, and pprof collection script used by the study.
The study decision and conclusions are in the
[`EPP tracing overhead analysis`](../epp_tracing_overhead_analysis.md).

## Software versions

- llm-d `v0.9.0`
- llm-d router chart and EPP `v0.10.0`
- GuideLLM `0.7.3`
- inference simulator `v0.11.2`
- vLLM `v0.27.0`
- model `Qwen/Qwen3-0.6B`

The deployment instructions are in [`deployments`](deployments). They create
the EPP, an OTLP collector and Jaeger sink, and either eight simulator replicas
or eight live vLLM replicas. The EPP uses one ready replica, matching the
measured configuration.

## Primary matrix

For both the simulator and live deployment, we run the fixed workload once for
each of these EPP configurations:

| Configuration | Values files |
|---|---|
| Tracing off | `router-base.values.yaml` |
| Standard tracing, 1% | base + `tracing-standard.values.yaml` + `ratio-1.values.yaml` |
| Standard tracing, 10% | base + `tracing-standard.values.yaml` + `ratio-10.values.yaml` |
| Standard tracing, 100% | base + `tracing-standard.values.yaml` + `ratio-100.values.yaml` |
| Standard tracing, 100% with pprof | previous values + `pprof.values.yaml` |

We use a fresh EPP deployment for every configuration and wait for it to become
ready before starting GuideLLM and then we run the primary workload with:

```sh
TARGET=http://GATEWAY_ADDRESS \
  guidellm/run.sh fixed non-streaming
```

This study runs the pprof configuration separately, starts the same GuideLLM command in the
background, waits 60 seconds from its invocation, and then runs
`pprof/collect.sh`. CPU profiling adds latency and its GuideLLM result is not
included in the request-performance comparison.

We run the primary matrix three times for matched confidence intervals. The
recorded final matrix contains one run per cell, so sub-percent differences
is not treated as precise estimates.

## Workload sensitivity

The broader non-streaming matrix is:

```sh
TARGET=http://GATEWAY_ADDRESS guidellm/run.sh all non-streaming
```

The streaming matrix is:

```sh
TARGET=http://GATEWAY_ADDRESS guidellm/run.sh all streaming
```

This study runs each command under tracing off, 0%, 1%, 10%, 50%, and 100%. The 0% case
enables the tracing SDK and exporter while sampling no spans. It is distinct
from tracing off.

It inspects every GuideLLM stage separately. Rejecting a stage if it reports request
errors. A failed final stage does not invalidate clean earlier stages from the
same invocation. The recorded streaming 800 requests/s and concurrency-128
stages reached a load-generator connection limit and are excluded.

## Measurements

GuideLLM supplies request throughput, latency, and error counts and collects the
queries in [`metrics/prometheus.md`](metrics/prometheus.md) every 10 seconds for
the benchmark window. This study reports mean EPP CPU cores and memory working set together
with request p50, p95, and p99.

For each run, we verify that:

1. GuideLLM completed and produced JSON output
2. Every included stage has zero request errors
3. The EPP, gateway, simulator, and model-server pods did not restart
4. The EPP tracing ratio matches the selected values file and the collector
   span count, normalized by EPP request count and the 100% baseline, is close
   to the configured ratio
5. Prometheus returned EPP CPU and memory samples
6. A profiling run captured non-empty CPU and heap profiles from every ready
   EPP replica
