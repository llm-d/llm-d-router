# EPP Horizontal Pod Autoscaling (HPA v2) Scale-Up and Scale-Down Verification

This report evaluates Horizontal Pod Autoscaling (`autoscaling/v2`) of the Endpoint Picker (EPP) router across a multi-stage `inference-perf` workload on GKE Autopilot (`llm-d-ap-usc1-router-perf`, `e2` machine family).

---

## 1. Benchmark Configuration and Methodology

- **Router Configuration**: [`test/perf/config/router-configs/load-aware-session-affinity-hpa.yaml`](../router-configs/load-aware-session-affinity-hpa.yaml)
  - Active-active mode (`ha-enable-leader-election: "false"`) with `passthrough-parser`, `session-affinity-filter`, `load-aware-scorer`, and `weighted-random-picker`.
  - HPA configuration: `minReplicas: 1`, `maxReplicas: 4`, `targetCPUUtilizationPercentage: 80`, `behavior.scaleDown.stabilizationWindowSeconds: 60`.
  - Container CPU requests: `epp: 700m` (limit `4`), `envoy-proxy: 200m` (`900m` total pod CPU request).
  - Envoy drain timeout: `--drain-time-s 2` with `--drain-strategy immediate`.
- **Workload Configuration**: [`test/perf/config/shared_prefix_hpa_scale_up_down.yaml`](../shared_prefix_hpa_scale_up_down.yaml)
  - Multi-turn shared-prefix streaming completion workload (`2,000` system prompt tokens, `500` question tokens, `200` output tokens) against 5 `llm-d-sim` replicas:
    - **Stage 1 (Warm-up)**: `1 QPS` for `30s`
    - **Stage 2 (Scale-up burst)**: `12 QPS` for `180s`
    - **Stage 3 (Cooldown and scale-down)**: `1 QPS` for `120s`, followed by post-stage scale-down observation to `minReplicas: 1`.
- **Verification Script**: [`test/perf/verify_epp_autoscaling.py`](../../verify_epp_autoscaling.py) (unit tests in [`test/perf/test_verify_epp_autoscaling.py`](../../test_verify_epp_autoscaling.py)).

---

## 2. Verification Summary

| Metric / Verification Gate | Result | Details |
|---|---|---|
| **Overall Verdict** | **PASS** | Namespace `llm-d-hpa-1791326916` |
| **Scale-Up Verified** | `True` | Scaled `1 -> 2 -> 3 -> 4` ready replicas during Stage 2 (`12 QPS`) |
| **Scale-Down Verified** | `True` | Scaled down `4 -> 3 -> 2 -> 1` ready replica during Stage 3 (`1 QPS`) and cooldown |
| **EPP CPU Metrics Verified** | `True` | Peak total EPP CPU: `2,463m` (`1,047m` on initial pod), Peak HPA CPU utilization: `210%` (target: `80%`) |
| **Inference-Perf Request Audit** | `0` failed requests | `3/3` stages completed, `0` failed requests across scale-up and scale-down transitions |
| **EPP and Envoy Log Audit** | `0` errors | `0` error/panic lines across all 4 pods (`x896n`, `2jgdw`, `vvxnr`, `wd627`), `0` container restarts |
| **Scheduler E2E Latency** | Sub-10ms | P50 = `0.91 ms`, P95 = `4.25 ms`, P99 = `8.97 ms` |

---

## 3. HPA and EPP Resource Time-Series

| Timestamp | HPA Current | HPA Desired | Ready Pods | HPA CPU (%) | Total EPP CPU (m) | Avg EPP CPU/Pod (m) | Total Envoy CPU (m) | Total Pod CPU (m) | Total EPP Mem (MiB) | Per-Pod EPP Breakdown |
|---|---|---|---|---|---|---|---|---|---|---|
| 22:55:28 | 1 | 1 | 1 | 9% | 66 | 66 | 15 | 81 | 24 | `x896n`: epp=66m, envoy=15m |
| 22:55:39 | 1 | 1 | 1 | 9% | 66 | 66 | 15 | 81 | 24 | `x896n`: epp=66m, envoy=15m |
| 22:55:51 | 1 | 1 | 1 | 9% | 66 | 66 | 15 | 81 | 24 | `x896n`: epp=66m, envoy=15m |
| 22:56:03 | 1 | 1 | 1 | 9% | 76 | 76 | 19 | 95 | 24 | `x896n`: epp=76m, envoy=19m |
| 22:56:14 | 1 | 1 | 1 | 9% | 76 | 76 | 19 | 95 | 24 | `x896n`: epp=76m, envoy=19m |
| 22:56:26 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:56:38 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:56:49 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:57:01 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:57:13 | 1 | 1 | 1 | 10% | 75 | 75 | 18 | 93 | 24 | `x896n`: epp=75m, envoy=18m |
| 22:57:25 | 1 | 1 | 1 | 10% | 228 | 228 | 53 | 281 | 26 | `x896n`: epp=228m, envoy=53m |
| 22:57:36 | 1 | 1 | 1 | 46% | 228 | 228 | 53 | 281 | 26 | `x896n`: epp=228m, envoy=53m |
| 22:57:48 | 1 | 1 | 1 | 46% | 228 | 228 | 53 | 281 | 26 | `x896n`: epp=228m, envoy=53m |
| 22:57:59 | 3 | 3 | 3 | 210% | 745 | 745 | 533 | 1278 | 37 | `x896n`: epp=745m, envoy=533m |
| 22:58:12 | 3 | 4 | 3 | 106% | 745 | 745 | 533 | 1278 | 37 | `x896n`: epp=745m, envoy=533m |
| 22:58:26 | 4 | 4 | 4 | 106% | 2463 | 616 | 903 | 3366 | 124 | `2jgdw`: epp=578m, envoy=40m<br>`vvxnr`: epp=445m, envoy=76m<br>`wd627`: epp=393m, envoy=47m<br>`x896n`: epp=1047m, envoy=740m |
| 22:58:40 | 4 | 4 | 4 | 106% | 2463 | 616 | 903 | 3366 | 124 | `2jgdw`: epp=578m, envoy=40m<br>`vvxnr`: epp=445m, envoy=76m<br>`wd627`: epp=393m, envoy=47m<br>`x896n`: epp=1047m, envoy=740m |
| 22:58:53 | 4 | 4 | 4 | 106% | 1896 | 474 | 868 | 2764 | 155 | `2jgdw`: epp=303m, envoy=57m<br>`vvxnr`: epp=363m, envoy=75m<br>`wd627`: epp=271m, envoy=13m<br>`x896n`: epp=959m, envoy=723m |
| 22:59:07 | 4 | 4 | 4 | 106% | 1896 | 474 | 868 | 2764 | 155 | `2jgdw`: epp=303m, envoy=57m<br>`vvxnr`: epp=363m, envoy=75m<br>`wd627`: epp=271m, envoy=13m<br>`x896n`: epp=959m, envoy=723m |
| 22:59:21 | 4 | 4 | 4 | 76% | 1896 | 474 | 868 | 2764 | 155 | `2jgdw`: epp=303m, envoy=57m<br>`vvxnr`: epp=363m, envoy=75m<br>`wd627`: epp=271m, envoy=13m<br>`x896n`: epp=959m, envoy=723m |
| 22:59:35 | 4 | 4 | 4 | 76% | 1893 | 473 | 795 | 2688 | 208 | `2jgdw`: epp=363m, envoy=81m<br>`vvxnr`: epp=448m, envoy=157m<br>`wd627`: epp=295m, envoy=39m<br>`x896n`: epp=787m, envoy=518m |
| 22:59:48 | 4 | 4 | 4 | 76% | 1893 | 473 | 795 | 2688 | 208 | `2jgdw`: epp=363m, envoy=81m<br>`vvxnr`: epp=448m, envoy=157m<br>`wd627`: epp=295m, envoy=39m<br>`x896n`: epp=787m, envoy=518m |
| 23:00:02 | 4 | 4 | 4 | 76% | 1991 | 498 | 874 | 2865 | 247 | `2jgdw`: epp=438m, envoy=151m<br>`vvxnr`: epp=465m, envoy=194m<br>`wd627`: epp=361m, envoy=73m<br>`x896n`: epp=727m, envoy=456m |
| 23:00:16 | 4 | 4 | 4 | 76% | 1991 | 498 | 874 | 2865 | 247 | `2jgdw`: epp=438m, envoy=151m<br>`vvxnr`: epp=465m, envoy=194m<br>`wd627`: epp=361m, envoy=73m<br>`x896n`: epp=727m, envoy=456m |
| 23:00:30 | 4 | 4 | 4 | 76% | 1930 | 482 | 840 | 2770 | 366 | `2jgdw`: epp=431m, envoy=156m<br>`vvxnr`: epp=444m, envoy=177m<br>`wd627`: epp=403m, envoy=115m<br>`x896n`: epp=652m, envoy=392m |
| 23:00:44 | 4 | 4 | 4 | 76% | 1930 | 482 | 840 | 2770 | 366 | `2jgdw`: epp=431m, envoy=156m<br>`vvxnr`: epp=444m, envoy=177m<br>`wd627`: epp=403m, envoy=115m<br>`x896n`: epp=652m, envoy=392m |
| 23:00:58 | 4 | 4 | 4 | 56% | 1930 | 482 | 840 | 2770 | 366 | `2jgdw`: epp=431m, envoy=156m<br>`vvxnr`: epp=444m, envoy=177m<br>`wd627`: epp=403m, envoy=115m<br>`x896n`: epp=652m, envoy=392m |
| 23:01:12 | 4 | 4 | 4 | 57% | 1665 | 416 | 658 | 2323 | 236 | `2jgdw`: epp=374m, envoy=129m<br>`vvxnr`: epp=392m, envoy=119m<br>`wd627`: epp=363m, envoy=79m<br>`x896n`: epp=536m, envoy=331m |
| 23:01:26 | 4 | 4 | 4 | 36% | 1174 | 294 | 196 | 1370 | 115 | `2jgdw`: epp=271m, envoy=36m<br>`vvxnr`: epp=299m, envoy=40m<br>`wd627`: epp=290m, envoy=38m<br>`x896n`: epp=314m, envoy=82m |
| 23:01:40 | 4 | 4 | 4 | 36% | 1174 | 294 | 196 | 1370 | 115 | `2jgdw`: epp=271m, envoy=36m<br>`vvxnr`: epp=299m, envoy=40m<br>`wd627`: epp=290m, envoy=38m<br>`x896n`: epp=314m, envoy=82m |
| 23:01:57 | 4 | 3 | 3 | 36% | 1076 | 269 | 119 | 1195 | 114 | `2jgdw`: epp=252m, envoy=23m<br>`vvxnr`: epp=284m, envoy=48m<br>`wd627`: epp=259m, envoy=0m<br>`x896n`: epp=281m, envoy=48m |
| 23:02:11 | 3 | 3 | 3 | 36% | 1076 | 269 | 119 | 1195 | 114 | `2jgdw`: epp=252m, envoy=23m<br>`vvxnr`: epp=284m, envoy=48m<br>`wd627`: epp=259m, envoy=0m<br>`x896n`: epp=281m, envoy=48m |
| 23:02:25 | 2 | 2 | 2 | 35% | 788 | 263 | 89 | 877 | 90 | `2jgdw`: epp=246m, envoy=0m<br>`vvxnr`: epp=268m, envoy=40m<br>`x896n`: epp=274m, envoy=49m |
| 23:02:38 | 2 | 2 | 2 | 35% | 788 | 263 | 89 | 877 | 90 | `2jgdw`: epp=246m, envoy=0m<br>`vvxnr`: epp=268m, envoy=40m<br>`x896n`: epp=274m, envoy=49m |
| 23:02:51 | 2 | 2 | 2 | 35% | 580 | 290 | 99 | 679 | 61 | `vvxnr`: epp=295m, envoy=49m<br>`x896n`: epp=285m, envoy=50m |
| 23:03:05 | 2 | 2 | 2 | 35% | 580 | 290 | 99 | 679 | 61 | `vvxnr`: epp=295m, envoy=49m<br>`x896n`: epp=285m, envoy=50m |
| 23:03:17 | 1 | 1 | 1 | 33% | 580 | 290 | 99 | 679 | 61 | `vvxnr`: epp=295m, envoy=49m<br>`x896n`: epp=285m, envoy=50m |

---

## 4. Kubernetes HPA Events

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

## 5. Key Operational Findings

1. **Multi-Container Pod HPA Utilization and Background Metric Scraping**:
   - In `llm-d-router-standalone` (`proxy.mode: sidecar`), each EPP pod runs `epp` and `envoy-proxy`. Kubernetes HPA (`type: Resource`, `name: cpu`) computes utilization as total pod CPU usage (`epp + envoy-proxy`) divided by total pod CPU request (`epp + envoy-proxy`).
   - Each active-active EPP replica independently scrapes `/metrics` from all model-server pods. Sizing `epp` CPU requests (`700m`) and `envoy-proxy` CPU requests (`200m`) keeps low-rate HPA utilization at `9-36%` (below the `80%` target) while `12 QPS` traffic drives utilization to `210%` and scales the deployment to `4` replicas.

2. **Sidecar Drain Ordering During Scale-Down**:
   - In `config/charts/routerlib/templates/_deployment.yaml`, the `epp` container defines a `preStop: sleep: seconds: 5` hook before receiving `SIGTERM`, whereas `envoy-proxy` begins draining immediately on pod termination.
   - Setting `--drain-time-s 2` on `envoy-proxy` ensures that `envoy-proxy` drains and closes idle HTTP/1.1 keep-alive connections before `epp`'s 5-second `preStop` window expires, preventing requests on pooled connections from reaching `envoy-proxy` after `epp` exits.
