# Multi-Turn Chat & Session Affinity Performance Evaluation Report

This report evaluates the performance, latency, and resource utilization of the **Endpoint Picker (EPP)** router when handling multi-turn conversational chat workloads. It compares three distinct routing architectures—**`session-affinity-filter`**, **`random-passthrough-multiturn`**, and **`optimized-baseline-multiturn`**—across the exact same 10 QPS multi-turn chat benchmark.

---

## 1. Inference-Perf Multi-Turn Chat Job Explained

### A. Why `enable_multi_turn_chat: true` is Required for Session Affinity
Testing session affinity requires stateful request sessions that persist across conversational turns:
- **Stateless Single-Turn Mode (`enable_multi_turn_chat: false`)**: By default, `shared_prefix` generates independent prompts with `session_id = None`. Without a session identifier, `inference-perf` never captures or re-sends session token headers.
- **Stateful Multi-Turn Mode (`enable_multi_turn_chat: true`)**: Enabling multi-turn chat instructs `shared_prefix` to generate stateful `LocalUserSession` objects (`user_session_id = "user_session_0"`, `"user_session_1"`, etc.).
- **Token Capture and Replay (PR 596)**:
  1. **Turn 1 (Session Initiation)**: `inference-perf` sends a request without a session token header. EPP selects an endpoint and returns `x-session-token: <encoded-pod>` in the response HTTP headers.
  2. **Turns 2+ (Affinity Pinning)**: `inference-perf` stores the received token and automatically injects `x-session-token` into every subsequent request belonging to that `session_id`, testing EPP's `session-affinity-filter` routing path.

### B. Why Prompt Sizing was Adjusted (`2k / 500 / 200`)
In multi-turn chat mode, `inference-perf` **appends** each turn's user question and assistant response to the session's ongoing conversation history (`context`):
- **Why large prompt sizes fail**: When configured with `95,000` system prompt tokens + `5,000` question tokens + `1,000` output tokens, a session's input prompt reaches **100,000 tokens on Turn 1** and grows by **6,000 tokens per turn**. By Turn 6, requests exceed the vLLM simulator pod's maximum context length (`max_model_len = 131072`), causing HTTP 400 Bad Request errors and multi-megabyte JSON body timeouts (`"body size exceeds the given limit"`).
- **Why `2,000 / 500 / 200` sizing is optimal**:
  - Configured in [shared_prefix_100k-1k-10qps_session-affinity.yaml](../config/shared_prefix_100k-1k-10qps_session-affinity.yaml) with `system_prompt_len: 2000`, `question_len: 500`, and `output_len: 200`, each turn adds **700 tokens** to the conversation.
  - Across a 300-second benchmark run (~60 turns per session at 10 QPS across 50 sessions), a session's total history reaches **~44,000 tokens**—safely below the 131,072 context limit.
  - This cleanly isolates control-plane session affinity overhead (header inspection, token decoding, and pod pinning) without network I/O saturation.

---

## 2. Multi-Turn Routing Performance Comparison Table

All tests were executed on GKE (`e2` machine family) with 10 simulated vLLM backend replicas (`--sim-replicas 10`) at **10 QPS** for **300 seconds** after a 15-second warmup stage.

| Metric / Attribute | 🔗 **`session-affinity`** (Random init) | ⚖️ **`load-aware-session-affinity`** | 🔀 **`pd-session-affinity`** (P/D Disagg) | 🎲 **`random-passthrough`** | ⚙️ **`optimized-baseline`** | Key Architectural Takeaway |
|---|---|---|---|---|---|---|
| **P50 Latency (ms)** | **0.39 ms** | **0.45 ms** | **1.01 ms** | **0.69 ms** | **1.20 ms** | Two-stage P/D scheduling adds ~0.62 ms P50 latency for dual profile execution |
| **P95 Tail Latency (ms)** | **0.96 ms** | **1.00 ms** | **2.61 ms** | **1.22 ms** | **1.96 ms** | Dual affinity filter evaluation (`prefill` + `decode`) per request |
| **EPP Peak CPU (m)** | **1,227m** (1.23 cores) | **1,315m** (1.32 cores) | **1,355m** (1.36 cores) | **1,266m** (1.27 cores) | **1,626m** (1.63 cores) | Minimal CPU overhead (~10%) for running two scheduling profile passes |
| **Envoy Peak CPU (m)** | **669m** (0.67 cores) | **751m** (0.75 cores) | **770m** (0.77 cores) | **679m** (0.68 cores) | **735m** (0.74 cores) | Consistent sidecar proxy CPU overhead across workloads |
| **Total System CPU (m)** | **1,896m** (1.90 cores) | **2,061m** (2.06 cores) | **2,125m** (2.13 cores) | **1,937m** (1.94 cores) | **2,317m** (2.32 cores) | **~8% lower total CPU** for two-stage P/D affinity vs. baseline scoring |
| **EPP Peak Heap Memory** | **104 MiB** | **177 MiB** | **135 MiB** | **183 MiB** | **310 MiB** | **~2.3x lower EPP heap memory** without prefix/KV cache tracking |
| **Total System Memory** | **164 MiB** | **236 MiB** | **188 MiB** | **244 MiB** | **365 MiB** | **~1.9x lower total system memory** footprint |
| **Router Configuration** | [session-affinity-filter.yaml](../config/router-configs/session-affinity-filter.yaml) | [load-aware-session-affinity.yaml](../config/router-configs/load-aware-session-affinity.yaml) | [pd-session-affinity.yaml](../config/router-configs/pd-session-affinity.yaml) | [random-passthrough-parser.yaml](../config/router-configs/random-passthrough-parser.yaml) | [optimized-baseline.yaml](../config/router-configs/optimized-baseline.yaml) | Configured plugins & request parsers |

---

## 3. Key Architectural Insights: Why `session-affinity-filter` Outperforms

1. 🟢 **Elimination of JSON Parsing & Hashing Overhead vs. `optimized-baseline`**:
   - In `optimized-baseline-multiturn`, parsing the JSON body and computing approximate prefix hashes across conversation turns adds **~0.81 ms** of median latency (`0.39 ms` vs `1.20 ms`) and increases EPP heap memory from **104 MiB** to **310 MiB** (a **~3x memory footprint increase**).
   - Bypassing scoring evaluation on pinned requests avoids executing `queue-scorer`, `kv-cache-utilization-scorer`, and `prefix-cache-scorer`.

2. 🟢 **Lowest Resource Footprint**:
   - `session-affinity-filter` achieves the lowest overall CPU utilization (**1.90 total cores** saturated at 10 QPS) and the smallest memory footprint (**104 MiB** EPP heap).

3. 🟢 **Intelligent Initial Placement with `load-aware-session-affinity`**:
   - Combining `load-aware-scorer` (which reads live `WaitingQueueSize` pod metrics to assign scores in `[0, 0.5]`) with `session-affinity-filter` delivers the best of both worlds:
     - **Turn 1 (Session Initiation)**: New sessions are intelligently balanced away from pods with queued requests via weighted-random reservoir sampling (`A-Res`).
     - **Turns 2+ (Affinity Pinning)**: Once placed, ongoing turns are pinned to the session pod and bypass scoring evaluation.
   - Even with `load-aware-scorer` active on initial turns, the median latency remains **0.45 ms** with a P95 tail latency of **1.00 ms** and **2.06 active cores**—only ~0.06 ms P50 latency and ~0.16 cores higher than pure random initial placement.

4. 🟢 **Two-Stage Prefill/Decode (P/D) Disaggregation with Dual Session Affinity (`pd-session-affinity`)**:
   - In P/D disaggregation (`always-disagg-pd-decider`), `disagg-profile-handler` executes **two complete scheduling cycles per request**—once for the `prefill` stage and once for the `decode` stage.
   - Each stage evaluates its own session affinity filter (`session-affinity-prefill` looking for `x-session-token-prefill` and `session-affinity-decode` looking for `x-session-token`).
   - Because EPP executes two scheduling profile passes per request, P50 latency is **1.01 ms** (vs. `0.39 ms` for a single stage) and P95 tail latency is **2.61 ms** (vs. `0.96 ms`), while EPP CPU usage increases modestly from `1,227m` to `1,355m` (**~10% CPU overhead** for the second scheduling pass) and EPP heap memory remains extremely lean at **135 MiB** (since no prefix/KV cache tracking is required).
