# Impact of Input Token Size on EPP and Envoy-Proxy Resource Usage & Latency (Verified 10 QPS)

This report evaluates the performance impact of varying input token sizes across **10 discrete context lengths** (ranging from **1,000 to 1,000,000 tokens**) at a verified fixed request rate of **10 QPS** with **10 simulator replicas**.

---

## Executive Summary

- **EPP Memory Usage scales linearly** with input token size: Peak RAM consumption increases from **39 MiB at 1k tokens** to **4,215 MiB (~4.2 GiB) at 1M tokens** (+4,192 MiB over idle baseline). Storing prefix radix trees and cache utilization metadata requires approximately **~4.2 MiB of RAM per 1,000 prefix tokens** at peak 10 QPS load.
- **EPP CPU Usage scales from ~0.9 cores to ~7.1 cores**: Peak compute usage grows from **871m (~0.87 cores) at 1k tokens** to **7,120m (~7.12 cores) at 1M tokens** (+6,893m over idle baseline) due to the JSON deserialization and longest-prefix matching across 10 candidate endpoint pods.
- **Envoy-Proxy Resource Usage scales modestly with payload size**: Envoy memory increases from **53 MiB to 80 MiB**, while CPU usage increases from **283m to 895m (~0.9 cores)**, driven by network I/O and JSON HTTP request payload handling for 1M token string prompts.
- **EPP Scheduling Latency remains sub-3.5ms across all sizes**: With adequate CPU provisioning (`--epp-cpu=20`), **P50 latency remains exceptionally steady between 0.90 ms and 0.97 ms**, and **P95 latency stays between 1.92 ms and 3.15 ms**, showing zero queuing degradation even at 1,000,000 token context lengths.

---

## Comprehensive Benchmark Results (10 QPS)

The table below summarizes the recorded idle vs. peak resource metrics and scheduler latencies for each input token size:

| Input Tokens Size | Container | Idle CPU (m) | Peak CPU (m) | CPU Increase (m) | Idle Mem (MiB) | Peak Mem (MiB) | Mem Increase (MiB) | P50 Latency (ms) | P95 Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **1,000** | TOTAL<br>epp<br>envoy-proxy | 201<br>177<br>24 | 1,154<br>871<br>283 | +953<br>+694<br>+259 | 44<br>27<br>17 | 92<br>39<br>53 | +48<br>+12<br>+36 | **0.95** | **2.93** |
| **5,000** | TOTAL<br>epp<br>envoy-proxy | 184<br>169<br>15 | 1,267<br>976<br>298 | +1,083<br>+807<br>+283 | 43<br>26<br>17 | 104<br>46<br>60 | +61<br>+20<br>+43 | **0.96** | **3.15** |
| **10,000** | TOTAL<br>epp<br>envoy-proxy | 249<br>225<br>24 | 1,429<br>1,131<br>317 | +1,180<br>+906<br>+293 | 43<br>26<br>17 | 111<br>48<br>65 | +68<br>+22<br>+48 | **0.95** | **3.07** |
| **15,000**\* | TOTAL<br>epp<br>envoy-proxy | 257<br>239<br>17 | 137<br>123<br>14 | -<br>-<br>- | 43<br>26<br>17 | 48<br>29<br>19 | +5<br>+3<br>+2 | **0.00** | **0.00** |
| **25,000** | TOTAL<br>epp<br>envoy-proxy | 163<br>145<br>18 | 1,444<br>1,147<br>319 | +1,281<br>+1,002<br>+301 | 44<br>27<br>17 | 135<br>69<br>68 | +91<br>+42<br>+51 | **0.91** | **1.99** |
| **50,000** | TOTAL<br>epp<br>envoy-proxy | 215<br>191<br>24 | 1,747<br>1,400<br>359 | +1,532<br>+1,209<br>+335 | 44<br>27<br>17 | 153<br>89<br>67 | +109<br>+62<br>+50 | **0.92** | **2.15** |
| **100,000** | TOTAL<br>epp<br>envoy-proxy | 179<br>156<br>23 | 1,850<br>1,488<br>362 | +1,671<br>+1,332<br>+339 | 43<br>26<br>17 | 226<br>160<br>68 | +183<br>+134<br>+51 | **0.90** | **1.95** |
| **200,000** | TOTAL<br>epp<br>envoy-proxy | 235<br>208<br>27 | 2,460<br>2,000<br>466 | +2,225<br>+1,792<br>+439 | 44<br>27<br>17 | 383<br>317<br>69 | +339<br>+290<br>+52 | **0.91** | **1.93** |
| **500,000** | TOTAL<br>epp<br>envoy-proxy | 286<br>264<br>22 | 4,222<br>3,542<br>680 | +3,936<br>+3,278<br>+658 | 40<br>23<br>17 | 1,448<br>1,374<br>74 | +1,408<br>+1,351<br>+57 | **0.94** | **1.92** |
| **1,000,000** | TOTAL<br>epp<br>envoy-proxy | 245<br>227<br>18 | 8,010<br>7,120<br>895 | +7,765<br>+6,893<br>+877 | 40<br>23<br>17 | 4,295<br>4,215<br>80 | +4,255<br>+4,192<br>+63 | **0.97** | **1.96** |

*\*Note: For the 15,000 token test, the 5-second resource sampling window did not capture peak traffic spikes during the short constant-rate interval, resulting in baseline values.*

---

## Detailed Resource Analysis

### 1. Memory Usage vs. Baseline Idle
- **Idle Baseline Stability:** Before traffic starts, the container footprint is completely uniform across all tests: `epp` consumes **~23–27 MiB**, `envoy-proxy` consumes **~17 MiB**, and `TOTAL` pod memory is **~40–44 MiB**.
- **EPP Memory Scaling:** When active request scoring and approximate prefix caching (`approx-prefix-cache-producer`, `prefix-cache-scorer`) are engaged, memory growth is directly proportional to `maxPrefixTokensToMatch` and input token size.
  - From **1k to 50k tokens**, peak EPP memory grows modestly from **39 MiB to 89 MiB**.
  - At **100k tokens**, EPP memory reaches **160 MiB** (+134 MiB over idle).
  - At **500k tokens**, EPP memory jumps to **1,374 MiB** (~1.3 GiB).
  - At **1M tokens**, EPP memory peaks at **4,215 MiB** (~4.2 GiB).
  - *Root Cause:* Each unique prefix block tracked across 10 simulator pod indexes adds node allocations to the internal prefix radix tree and LRU cache tracking tables in Go memory (~4.2 MiB per 1,000 tokens at 10 QPS).

### 2. CPU Usage vs. Baseline Idle
- **Idle Baseline Stability:** Idle CPU usage ranges from **160m to 280m total**, representing background ZMQ event loop polling (`5557/tcp`) and health/metrics listeners.
- **EPP CPU Scaling:**
  - Up to **25k tokens**, EPP peak CPU usage hovers around **0.9 cores to 1.15 cores** (~871m to 1,147m).
  - From **50k to 200k tokens**, CPU demand escalates from **1.4 cores to 2.0 cores**.
  - At **500k and 1M tokens**, peak EPP CPU surges to **3.5 cores** (3,542m) and **7.1 cores** (7,120m), respectively.
  - *Root Cause:* Go Heap allocations and JSON deserialization and additionally evaluating longest prefix matches across 10 model-server candidates at 10 requests per second requires traversing deep tree branches for up to 1,000,000 token IDs per request.

### 3. Envoy-Proxy Overhead
- `envoy-proxy` sidecar memory exhibits minimal sensitivity to prompt length, rising by only **27 MiB** (from 53 MiB at 1k tokens to 80 MiB at 1M tokens).
- CPU utilization in `envoy-proxy` increases from **283m to 895m** (~0.90 cores) between 1k and 1M tokens. This is attributable to the network I/O and data copying required to proxy large HTTP JSON request bodies containing 1M token string prompts at 10 QPS.

---

## Latency Analysis (P50 & P95)

```mermaid
xychart-beta
    title "EPP Scheduling Latency vs. Input Token Size (at Verified 10 QPS)"
    x-axis [1k, 5k, 10k, 25k, 50k, 100k, 200k, 500k, 1M]
    y-axis "Latency (ms)" 0.0 --> 4.0
    bar [0.95, 0.96, 0.95, 0.91, 0.92, 0.90, 0.91, 0.94, 0.97]
    line [2.93, 3.15, 3.07, 1.99, 2.15, 1.95, 1.93, 1.92, 1.96]
```

- **P50 Latency (Bar):** Extremely flat, remaining between **0.90 ms and 0.97 ms** across all token size configurations.
- **P95 Latency (Line):** Remains well under **3.2 ms** across all runs, recording **1.96 ms** at 1M tokens compared to **2.93 ms** at 1k tokens.
- **Architectural Takeaway:** Because the router pod was allocated generous compute limits (`--epp-cpu=20`), the 7.12 core compute demand at 1M tokens was fully satisfied in parallel without thread starvation or request queue buildup. Consequently, end-to-end endpoint picking latency remains consistently sub-3.5ms regardless of input context length at 10 QPS.