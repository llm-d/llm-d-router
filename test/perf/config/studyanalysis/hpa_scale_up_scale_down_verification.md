# Horizontal Pod Autoscaling (HPA v2) Scale-Up and Scale-Down Verification

This report evaluates Horizontal Pod Autoscaling (`autoscaling/v2`) of the `llm-d-router` in both EPP sidecar mode (`router.epp.autoscaling`) and standalone proxy service mode (`router.proxy.mode: service`, `router.proxy.autoscaling`) across a multi-stage `inference-perf` workload on GKE Autopilot (`llm-d-ap-usc1-router-perf`, `e2` machine family).

---

## 1. Benchmark Configuration and Methodology

- **EPP HPA Router Configuration (`proxy.mode: sidecar`)**: [`test/perf/config/router-configs/load-aware-session-affinity-hpa.yaml`](../router-configs/load-aware-session-affinity-hpa.yaml)
  - Active-active mode (`ha-enable-leader-election: "false"`) with `passthrough-parser`, `session-affinity-filter`, `load-aware-scorer`, and `weighted-random-picker`.
  - HPA target: `Deployment/<release>-epp` (`minReplicas: 1`, `maxReplicas: 4`, `targetCPUUtilizationPercentage: 80`, `behavior.scaleDown.stabilizationWindowSeconds: 60`).
  - Container CPU requests: `epp: 700m` (limit `4`), `envoy-proxy: 200m` (`900m` total pod CPU request).
  - Envoy drain timeout: `--drain-time-s 2` with `--drain-strategy immediate`.
- **Standalone Proxy Service HPA Router Configuration (`proxy.mode: service`)**: [`test/perf/config/router-configs/load-aware-session-affinity-proxy-hpa.yaml`](../router-configs/load-aware-session-affinity-proxy-hpa.yaml)
  - Standalone `envoy-proxy` service (`Deployment/<release>-proxy`, `Service/<release>-proxy:80`) communicating over `ext_proc` gRPC (`9002`) with `Deployment/<release>-epp` (`health-checking: true`).
  - HPA target: `Deployment/<release>-proxy` (`minReplicas: 1`, `maxReplicas: 4`, `targetCPUUtilizationPercentage: 80`, `behavior.scaleDown.stabilizationWindowSeconds: 60`).
  - Container CPU requests: `envoy-proxy: 250m` (limit `2`).
  - Envoy drain timeout: `--drain-time-s 2` with `--drain-strategy immediate`.
- **Workload Configuration**: [`test/perf/config/shared_prefix_hpa_scale_up_down.yaml`](../shared_prefix_hpa_scale_up_down.yaml)
  - Multi-turn shared-prefix streaming completion workload (`25` prefix groups, `5` prompts per group, `2,000` system prompt tokens, `500` question tokens, `200` output tokens) against 5 `llm-d-sim` replicas:
    - **Stage 1 (Warm-up)**: `1 QPS` for `20s`
    - **Stage 2 (Scale-up burst)**: `12 QPS` for `90s`
    - **Stage 3 (Cooldown)**: `1 QPS` for `30s`, followed by post-stage scale-down observation to `minReplicas: 1`.
- **Verification Script**: [`test/perf/verify_epp_autoscaling.py`](../../verify_epp_autoscaling.py) (unit tests in [`test/perf/test_verify_epp_autoscaling.py`](../../test_verify_epp_autoscaling.py)).

---

## 2. Verification Summary

| Metric / Verification Gate | EPP HPA (`proxy.mode: sidecar`) | Standalone Proxy HPA (`proxy.mode: service`) |
|---|---|---|
| **Overall Verdict** | **PASS** (`llm-d-hpa-1791326916`) | **PASS** (`llm-d-hpa-1791403439`) |
| **Scale-Up Verified** | `True` (`1 -> 2 -> 3 -> 4` ready replicas at `12 QPS`) | `True` (`1 -> 2 -> 3 -> 4` ready replicas at `12 QPS`) |
| **Scale-Down Verified** | `True` (`4 -> 3 -> 2 -> 1` ready replica after cooldown) | `True` (`4 -> 1` ready replica after cooldown) |
| **Target CPU Metrics Verified** | `True` (Peak EPP CPU: `2,463m`, Peak HPA CPU: `210%`) | `True` (Peak Proxy CPU: `837m`, Peak HPA CPU: `113%`) |
| **Inference-Perf Request Audit** | `3/3` stages completed, `0` failed requests | `3/3` stages completed, `0` failed requests |
| **EPP and Envoy Log Audit** | `0` error/panic lines, `0` container restarts | `0` error/panic lines, `0` container restarts |
| **Scheduler E2E Latency** | P50 = `0.91 ms`, P95 = `4.25 ms`, P99 = `8.97 ms` | P50 = `0.75 ms`, P95 = `1.18 ms`, P99 = `1.85 ms` |

---

## 3. EPP HPA (`proxy.mode: sidecar`) Time-Series and Events

### Resource Time-Series

| Timestamp | HPA Current | HPA Desired | Ready Pods | HPA CPU (%) | Total EPP CPU (m) | Avg EPP CPU/Pod (m) | Total Envoy CPU (m) | Total Pod CPU (m) | Total EPP Mem (MiB) | Per-Pod EPP Breakdown |
|---|---|---|---|---|---|---|---|---|---|---|
| 22:55:28 | 1 | 1 | 1 | 9% | 66 | 66 | 15 | 81 | 24 | `x896n`: epp=66m, envoy=15m |
| 22:56:03 | 1 | 1 | 1 | 9% | 76 | 76 | 19 | 95 | 24 | `x896n`: epp=76m, envoy=19m |
| 22:56:26 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:57:25 | 1 | 1 | 1 | 10% | 228 | 228 | 53 | 281 | 26 | `x896n`: epp=228m, envoy=53m |
| 22:57:36 | 1 | 1 | 1 | 46% | 228 | 228 | 53 | 281 | 26 | `x896n`: epp=228m, envoy=53m |
| 22:57:59 | 3 | 3 | 3 | 210% | 745 | 745 | 533 | 1278 | 37 | `x896n`: epp=745m, envoy=533m |
| 22:58:12 | 3 | 4 | 3 | 106% | 745 | 745 | 533 | 1278 | 37 | `x896n`: epp=745m, envoy=533m |
| 22:58:26 | 4 | 4 | 4 | 106% | 2463 | 616 | 903 | 3366 | 124 | `2jgdw`: epp=578m, envoy=40m<br>`vvxnr`: epp=445m, envoy=76m<br>`wd627`: epp=393m, envoy=47m<br>`x896n`: epp=1047m, envoy=740m |
| 22:58:53 | 4 | 4 | 4 | 106% | 1896 | 474 | 868 | 2764 | 155 | `2jgdw`: epp=303m, envoy=57m<br>`vvxnr`: epp=363m, envoy=75m<br>`wd627`: epp=271m, envoy=13m<br>`x896n`: epp=959m, envoy=723m |
| 22:59:35 | 4 | 4 | 4 | 76% | 1893 | 473 | 795 | 2688 | 208 | `2jgdw`: epp=363m, envoy=81m<br>`vvxnr`: epp=448m, envoy=157m<br>`wd627`: epp=295m, envoy=39m<br>`x896n`: epp=787m, envoy=518m |
| 23:00:02 | 4 | 4 | 4 | 76% | 1991 | 498 | 874 | 2865 | 247 | `2jgdw`: epp=438m, envoy=151m<br>`vvxnr`: epp=465m, envoy=194m<br>`wd627`: epp=361m, envoy=73m<br>`x896n`: epp=727m, envoy=456m |
| 23:00:58 | 4 | 4 | 4 | 56% | 1930 | 482 | 840 | 2770 | 366 | `2jgdw`: epp=431m, envoy=156m<br>`vvxnr`: epp=444m, envoy=177m<br>`wd627`: epp=403m, envoy=115m<br>`x896n`: epp=652m, envoy=392m |
| 23:01:26 | 4 | 4 | 4 | 36% | 1174 | 294 | 196 | 1370 | 115 | `2jgdw`: epp=271m, envoy=36m<br>`vvxnr`: epp=299m, envoy=40m<br>`wd627`: epp=290m, envoy=38m<br>`x896n`: epp=314m, envoy=82m |
| 23:01:57 | 4 | 3 | 3 | 36% | 1076 | 269 | 119 | 1195 | 114 | `2jgdw`: epp=252m, envoy=23m<br>`vvxnr`: epp=284m, envoy=48m<br>`wd627`: epp=259m, envoy=0m<br>`x896n`: epp=281m, envoy=48m |
| 23:02:25 | 2 | 2 | 2 | 35% | 788 | 263 | 89 | 877 | 90 | `2jgdw`: epp=246m, envoy=0m<br>`vvxnr`: epp=268m, envoy=40m<br>`x896n`: epp=274m, envoy=49m |
| 23:02:51 | 2 | 2 | 2 | 35% | 580 | 290 | 99 | 679 | 61 | `vvxnr`: epp=295m, envoy=49m<br>`x896n`: epp=285m, envoy=50m |
| 23:03:17 | 1 | 1 | 1 | 33% | 580 | 290 | 99 | 679 | 61 | `vvxnr`: epp=295m, envoy=49m<br>`x896n`: epp=285m, envoy=50m |

### Kubernetes HPA Events

```text
TIME                   REASON                           MESSAGE
2026-10-06T22:52:57Z   ADD                              llm-d-hpa-1791326916/load-aware-session-affinity-hpa-epp
2026-10-06T22:52:58Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 0 to 1
2026-10-06T22:52:58Z   DNSRecordProvisioningSucceeded   DNS records updated
2026-10-06T22:53:13Z   FailedGetResourceMetric          No recommendation
2026-10-06T22:55:13Z   FailedGetResourceMetric          did not receive metrics for targeted pods (pods might be unready)
2026-10-06T22:57:49Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 1 to 2
2026-10-06T22:57:49Z   SuccessfulRescale                New size: 2; reason: cpu resource utilization (percentage of request) above target
2026-10-06T22:57:54Z   SuccessfulRescale                New size: 3; reason: cpu resource utilization (percentage of request) above target
2026-10-06T22:57:54Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 2 to 3
2026-10-06T22:58:09Z   SuccessfulRescale                New size: 4; reason: cpu resource utilization (percentage of request) above target
2026-10-06T22:58:09Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 3 to 4
2026-10-06T23:01:43Z   SuccessfulRescale                New size: 3; reason: cpu resource utilization (percentage of request) below target
2026-10-06T23:01:43Z   ScalingReplicaSet                Scaled down replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 4 to 3
2026-10-06T23:02:13Z   SuccessfulRescale                New size: 2; reason: cpu resource utilization (percentage of request) below target
2026-10-06T23:02:13Z   ScalingReplicaSet                Scaled down replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 3 to 2
2026-10-06T23:03:13Z   HpaProfilePerformance            The HPA rescaled target based on performance profile
2026-10-06T23:03:13Z   SuccessfulRescale                New size: 1; reason: cpu resource utilization (percentage of request) below target
2026-10-06T23:03:13Z   ScalingReplicaSet                Scaled down replica set load-aware-session-affinity-hpa-epp-866d757fc9 from 2 to 1
```

---

## 4. Standalone Proxy Service HPA (`proxy.mode: service`) Time-Series and Events

### Resource Time-Series

| Timestamp | HPA Current | HPA Desired | Ready Pods | HPA CPU (%) | Total Envoy CPU (m) | Per-Pod Proxy Breakdown |
|---|---|---|---|---|---|---|
| 20:15:15 | 1 | 1 | 1 | 4% | 13 | `9v5zf`: envoy=13m |
| 20:16:51 | 1 | 1 | 1 | 5% | 13 | `9v5zf`: envoy=13m |
| 20:17:03 | 1 | 1 | 1 | 15% | 29 | `9v5zf`: envoy=29m |
| 20:17:29 | 1 | 2 | 2 | 113% | 240 | `9v5zf`: envoy=240m |
| 20:17:41 | 2 | 3 | 3 | 111% | 240 | `9v5zf`: envoy=240m |
| 20:17:54 | 3 | 3 | 3 | 111% | 682 | `9v5zf`: envoy=635m<br>`glbd4`: envoy=23m<br>`svqrg`: envoy=24m |
| 20:18:32 | 3 | 4 | 4 | 91% | 735 | `9v5zf`: envoy=634m<br>`glbd4`: envoy=58m<br>`svqrg`: envoy=43m |
| 20:18:58 | 4 | 4 | 4 | 91% | 837 | `9v5zf`: envoy=645m<br>`c6ggz`: envoy=63m<br>`glbd4`: envoy=129m |
| 20:19:11 | 4 | 4 | 4 | 19% | 837 | `9v5zf`: envoy=645m<br>`c6ggz`: envoy=63m<br>`glbd4`: envoy=129m |
| 20:19:25 | 4 | 4 | 4 | 19% | 122 | `247hg`: envoy=10m<br>`9v5zf`: envoy=76m<br>`c6ggz`: envoy=12m<br>`glbd4`: envoy=24m |
| 20:19:52 | 4 | 1 | 1 | 19% | 12 | `9v5zf`: envoy=12m |
| 20:20:05 | 1 | 1 | 1 | 4% | 12 | `9v5zf`: envoy=12m |

### Kubernetes HPA Events

```text
TIME                   REASON                           MESSAGE
2026-10-07T20:12:02Z   ADD                              llm-d-hpa-1791403439/load-aware-session-affinity-proxy-hpa-proxy
2026-10-07T20:12:02Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-proxy-hpa-proxy-6fdc9f857c from 0 to 1
2026-10-07T20:12:03Z   DNSRecordProvisioningSucceeded   DNS records updated
2026-10-07T20:12:21Z   FailedGetResourceMetric          No recommendation
2026-10-07T20:17:20Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-proxy-hpa-proxy-6fdc9f857c from 1 to 2
2026-10-07T20:17:20Z   SuccessfulRescale                New size: 2; reason: cpu resource utilization (percentage of request) above target
2026-10-07T20:17:33Z   SuccessfulRescale                New size: 3; reason: cpu resource utilization (percentage of request) above target
2026-10-07T20:17:33Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-proxy-hpa-proxy-6fdc9f857c from 2 to 3
2026-10-07T20:18:20Z   SuccessfulRescale                New size: 4; reason: cpu resource utilization (percentage of request) above target
2026-10-07T20:18:20Z   ScalingReplicaSet                Scaled up replica set load-aware-session-affinity-proxy-hpa-proxy-6fdc9f857c from 3 to 4
2026-10-07T20:19:47Z   HpaProfilePerformance            The HPA rescaled target based on performance profile
2026-10-07T20:19:47Z   SuccessfulRescale                New size: 1; reason: cpu resource utilization (percentage of request) below target
2026-10-07T20:19:47Z   ScalingReplicaSet                Scaled down replica set load-aware-session-affinity-proxy-hpa-proxy-6fdc9f857c from 4 to 1
```

---

## 5. Key Operational Findings

1. **Multi-Container Pod HPA Utilization vs. Standalone Proxy Service HPA**:
   - In `proxy.mode: sidecar`, each EPP pod runs `epp` and `envoy-proxy`. Kubernetes HPA (`type: Resource`, `name: cpu`) computes utilization as total pod CPU usage (`epp + envoy-proxy`) divided by total pod CPU request (`epp + envoy-proxy`). Because each active-active EPP replica scrapes `/metrics` from all model-server pods, sizing `epp` CPU requests (`700m`) and `envoy-proxy` CPU requests (`200m`) keeps low-rate HPA utilization at `9-36%` while `12 QPS` traffic drives utilization to `210%`.
   - In `proxy.mode: service`, `Deployment/<release>-proxy` runs only `envoy-proxy`, which has `12-13m` idle CPU usage (`4-5%` of a `250m` CPU request). Under `12 QPS` streaming traffic, proxy CPU climbs to `837m` across `4` replicas (`113%` HPA utilization) and drops back to `19%` (`4` pods) and `4%` (`1` pod) during cooldown.

2. **gRPC Health Checking Between Standalone Proxy and EPP**:
   - In `config/charts/llm-d-router-standalone/values.yaml`, Envoy's `ext_proc` cluster configures active `grpc_health_check` on port `9002` (`service_name: "envoy.service.ext_proc.v3.ExternalProcessor"`). Enabling `router.epp.flags.health-checking: true` registers the gRPC health service on port `9002` so standalone proxy replicas mark the `ext_proc` upstream healthy.

3. **Drain Ordering During Scale-Down**:
   - In `config/charts/routerlib/templates/_deployment.yaml`, the `epp` container defines a `preStop: sleep: seconds: 5` hook before receiving `SIGTERM`, whereas `envoy-proxy` begins draining immediately on pod termination.
   - Setting `--drain-time-s 2` on `envoy-proxy` ensures that `envoy-proxy` drains and closes idle HTTP/1.1 keep-alive connections before `epp`'s 5-second `preStop` window expires.
