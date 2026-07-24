# EPP Control Plane Performance & Profiling Evaluation Report

## Executive Summary

This report synthesizes performance benchmarks, resource utilization metrics, and `pprof` CPU/memory profiling captured for the **Endpoint Picker (EPP)** router service under stress testing workloads (10 vLLM simulator replicas, high QPS, initial 10k baselines, 100k-token prompt inputs, passthrough parsing, and precise prefix ZMQ notification routing).

---

## 1. Comparative Benchmark & Router Sizing Matrix

Cross-test performance, latencies, and resource consumption across all evaluated router configurations:

| Benchmark Scenario | Prompt Sizing | Active Cores (CPU %) | EPP Peak Mem | Total System Mem | P50 Latency | P95 Latency | Summary Architecture |
|---|---|---|---|---|---|---|---|
| 🚀 **`random-passthrough-parser`** | **95k System + 5k Quest** | 3.22 cores (321.8%) | 665 MiB | 745 MiB | **0.94 ms** | **4.85 ms** | **Bypasses JSON parsing & unmarshalling** (sub-ms P50 latency) |
| 🎯 **`precise-prefix`** | **95k System + 5k Quest** | 3.38 cores (288.1%) | 1,407 MiB | 9,717 MiB | **2.05 ms** | **4.82 ms** | **Static ZMQ event stream + vLLM Tokenizer Sidecar** |
| ⚡ **100k Tokens (`maxTokens: 100`)** | **95k System + 5k Quest** | 3.62 cores (362.4%) | 2,924 MiB | 3,105 MiB | **1.64 ms** | **6.42 ms** | Approximate prefix matching capped at 100 tokens |
| ⚡ **100k Tokens (`maxTokens: 100000`)** | **95k System + 5k Quest** | 3.16 cores (316.5%) | 1,699 MiB | 1,755 MiB | **1.80 ms** | **30.06 ms** | Full approximate prefix matching across 100,000 tokens |
| 🏁 **Initial 10k Baseline (`job1`)** | **10k System + 1k Quest** | 2.13 cores (213.3%) | 109 MiB | 139 MiB | **2.04 ms** | **6.08 ms** | Standard baseline (2 CPU allocation) |
| ⏱️ **Baseline 180s Wait Run** | **10k System + 1k Quest** | 2.12 cores (212.0%) | 107 MiB | 136 MiB | **2.86 ms** | **8.56 ms** | Standard baseline (steady-state profile window) |

---

## 2. High-Level CPU Usage Breakdown (100% Accounting)

Under peak 100k-token prompt workload at 10 QPS (**~3.33 active CPU cores** / 100.02 CPU sample seconds), CPU execution time is distributed across the following core subsystems:

| Subsystem / Category | Flat CPU Time | % of Total CPU | Operational Description |
|---|---|---|---|
| 🌐 **1. Network I/O & Linux Kernel Syscalls** | **19.31s** | **19.31%** | Kernel socket operations (`Syscall6`, `netpoll`, `crypto/tls`) servicing streaming HTTP/gRPC data packets. |
| 📄 **2. JSON Request Payload & Stream Decoding** | **18.60s** | **18.60%** | Unmarshalling 100k-token prompt JSON request bodies (`encoding/json.checkValid`, `unquoteBytes`, `OpenAIParser.ParseRequest`). |
| 🧹 **3. Go GC & Runtime Heap Memory Management** | **13.11s** | **13.11%** | Go allocator & Garbage Collection routines (`mallocgc`, `nextFreeFast`, `memclr`, `memmove`) servicing short-lived JSON & buffer allocations. |
| ⚙️ **4. Goroutine Scheduling & Concurrency Sync** | **12.77s** | **12.77%** | Multithreaded OS thread coordination (`futex`, `selectgo`, `stealWork`, `schedule`) managing goroutines across active CPU cores. |
| 🛠️ **5. Go Runtime Low-level Internal Helpers** | **11.85s** | **11.85%** | Stack management, Go map operations (`internal/runtime/maps`), and assembly byte routines (`indexbytebody`, `aeshashbody`). |
| 📦 **6. gRPC Protocol Framing & Protobuf IPC** | **5.91s** | **5.91%** | Protobuf encoding/decoding and HTTP/2 framing for gRPC `ext_proc` communication between Envoy proxy and the EPP router. |
| 🔍 **7. Low-Level Crypto, Encoding & UTF-8 Helpers** | **5.91s** | **5.91%** | UTF-8 string validation (`unicode/utf8`), TLS AES encryption, gzip stream decompression, and `reflect` type assertions. |
| 📝 **8. Zap Structured Logging & Field Formatting** | **2.91s** | **2.91%** | JSON log line encoding (`zapcore.EncodeEntry`) and caller path formatting under active request traffic. |
| 📊 **9. Prometheus Metrics Scraper & Text Parser** | **1.53s** | **1.53%** | Flat token parsing time inside `prometheus/common/expfmt` (accumulates **12.5%** cumulative time when including subroutines). |
| 🎯 **10. Prefix Hashing Engine & EPP Scheduler** | **0.73s** | **0.73%** | Block hash generation (`matchLongestPrefix`), LRU cache lookups (`indexer.Get`), and scheduler scoring profile execution. |
| **Total Accounted CPU** | **100.02s** | **100.00%** | **~3.33 Active CPU Cores Saturated** |

---

## 3. Executive Takeaways: What Is Optimized

1. 🟢 **`passthrough-parser` Delivers Sub-Millisecond Latency (P50 = 0.94 ms)**
   - Bypassing full JSON unmarshalling with `passthrough-parser` eliminates request payload decoding overhead, dropping median routing latency to **0.94 ms** and P95 latency to **4.85 ms**.

2. 🟢 **`precise-prefix` Ensures Ultra-Stable Tail Latency (P95 = 4.82 ms)**
   - Utilizing static ZMQ event notifications (`kv@` event stream) for KV cache state updates and tokenizer sidecar preprocessing yields highly predictable routing latencies (**P50 = 2.05 ms, P95 = 4.82 ms**).

3. 🟢 **Prefix Matching Engine is Extremely Efficient ($< 0.3\%$ CPU)**
   - Computing 64-bit non-cryptographic `xxhash` block hashes across **95,000 system prompt tokens** (5,937 blocks per request) takes only **190ms of total CPU time** across an entire 30-second benchmark run ($< 0.3\%$ of total EPP CPU).

4. 🟢 **Linear Multi-Core CPU Scaling**
   - EPP scales cleanly from 2.13 cores (on 2 CPU allocation) to 3.38+ active CPU cores when allocated additional CPU quota (`10 CPU`), maintaining low mutex locking overhead (`futex` accounts for only ~3.5–4.9% of CPU).
