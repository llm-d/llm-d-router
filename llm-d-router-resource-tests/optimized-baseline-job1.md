# EPP Router Performance Benchmarking Results: optimized-baseline-job1

| Timestamp | Namespace | Router Config | Perf Job | Machine Family | Sim Replicas | EPP Images | Container | Idle CPU (m) | Idle Mem (MiB) | Peak CPU (m) | Peak Mem (MiB) | P50 Latency (ms) | P95 Latency (ms) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-25 00:52:00 | llm-d-perf-1782348720 | optimized-baseline | shared_prefix_job1.yaml | - | 10 | docker.io/envoyproxy/envoy:distroless-v1.33.2<br>ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main | TOTAL | 146 | 105 | 4858 | 210 | 0.07 | 0.18 | SUCCESS |
| 2026-06-25 00:52:00 | llm-d-perf-1782348720 | optimized-baseline | shared_prefix_job1.yaml | - | 10 | docker.io/envoyproxy/envoy:distroless-v1.33.2<br>ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main | envoy-proxy | 27 | 85 | 2368 | 103 | 0.07 | 0.18 | SUCCESS |
| 2026-06-25 00:52:00 | llm-d-perf-1782348720 | optimized-baseline | shared_prefix_job1.yaml | - | 10 | docker.io/envoyproxy/envoy:distroless-v1.33.2<br>ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main | epp | 119 | 20 | 2523 | 109 | 0.07 | 0.18 | SUCCESS |
