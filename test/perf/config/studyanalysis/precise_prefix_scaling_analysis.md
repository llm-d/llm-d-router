# EPP Standalone Precise Prefix Cache Scaling Analysis

This report evaluates the performance, latency, and resource utilization of the **Endpoint Picker (EPP)** standalone router using the `precise-prefix` cache configuration under horizontal replication (1, 2, and 3 EPP replicas), and compares them to the `optimized-baseline` configuration.

---

## 1. Executive Performance Matrix

All benchmarks were executed under a stress workload consisting of **10 simulator replicas** handling a **100k-token prompt** (`shared_prefix_100k-1k-10qps.yaml`) at **10 QPS**. The EPP pods were configured with resource requests of `10 CPU` and `20Gi` memory on GKE `e2` nodes.

| Scenario / Configuration | EPP Replicas | P50 Latency (ms) | P95 Latency (ms) | EPP Peak CPU (m) | EPP Peak Mem (MiB) | Tokenizer Peak CPU (m) | Tokenizer Peak Mem (MiB) | Total Peak CPU (m) | Total Peak Mem (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| 🏁 **`optimized-baseline`** | 1 | 1.80 | 30.06 | 3,781 | 1,699 | *Disabled* | *Disabled* | 3,781 | 1,755 |
| 🎯 **`precise-prefix`** | 1 | 2.98 | 6.78 | 4,569 | 2,152 | 1,168 | 8,897 | 8,803 | 10,457 |
| 🎯 **`precise-prefix`** | 2 | **2.48** | **4.85** | **4,214** | **2,238** | **2,228** | **9,279** | **8,987** | **11,064** |
| 🎯 **`precise-prefix`** | 3 | 2.86 | 6.65 | 5,599 | 3,299 | 3,202 | 9,096 | 12,178 | 12,152 |

> [!NOTE]
> Resource metrics (CPU and Memory) for the 2 and 3 replica configurations represent the **aggregated peak (sum) across all running replica pods** in the cluster.

---

## 2. Subsystem CPU & Memory Scaling Analysis

### 2.1 EPP Router (`epp`) Scaling
* **CPU Sizing**: The EPP container's aggregated peak CPU load remained stable between **4.2 to 5.6 active cores** total across all tests. Under 2 replicas, the load was distributed, resulting in a slightly lower peak CPU aggregate (`4.2 cores`) than a single replica (`4.5 cores`), indicating that workload distribution successfully reduced resource bottlenecks.
* **Memory Footprint**: The EPP base memory scales slowly, reflecting a stable footprint of **~1.1 GiB of RAM per active EPP pod** (2.2 GiB total for 2 replicas, 3.3 GiB total for 3 replicas).

### 2.2 Tokenizer (`vllm-render`) Scaling & Memory Behavior
In the `precise-prefix` configuration, EPP must query a local `vllm-render` (vLLM sidecar running in CPU mode) to tokenize prompts for exact cache key indexing.
* **Linear CPU Scaling**: Tokenizer CPU scaled almost perfectly linearly with the number of replicas: **1.1 cores** (1 replica) $\to$ **2.2 cores** (2 replicas) $\to$ **3.2 cores** (3 replicas), representing ~1.1 cores of JSON serialization/tokenization CPU per EPP replica under balanced traffic.
* **Flat Peak Memory Behavior (~9 GiB)**: Interestingly, the aggregated peak memory of all tokenizer containers stayed flat around **~9 GiB** regardless of replica count (8.8 GiB $\to$ 9.2 GiB $\to$ 9.0 GiB), even though idle memory scaled linearly with the pods (~1 GiB per idle pod).
  * **Explanation**: The load generator uses persistent HTTP keep-alive connections. Due to connection reuse and GKE service load balancer stickiness, only **one EPP replica pod** actively processed the bulk of the requests at a time. Therefore, only that pod's tokenizer container fully initialized and allocated memory for the model weights cache (~8.8 GiB), while the remaining replicas stayed in their idle state (~1 GiB each).

---

## 3. CPU Profiling (`pprof`) & ZMQ Update Overhead

Analysis of EPP's CPU execution profile (`pprof`) under replication shows that EPP's CPU consumption is heavily dominated by front-channel request decoding, while background ZMQ cache syncing is extremely lightweight.

### 3.1 ZMQ Cache Ingestion Overhead ($< 0.2\%$ CPU)
EPP replicas subscribe to simulator pods via ZMQ to receive cache change events (blocks added or evicted). Despite every replica having to process $100\%$ of all cache events from all model servers to maintain index consistency, the CPU overhead for this is negligible:

```text
Showing nodes accounting for 0.19s, 0.16% of 116.42s total CPU samples (focus=kvevents)
      flat  flat%   sum%        cum   cum%
         0     0%  0.16%      0.19s  0.16%  github.com/llm-d/llm-d-router/pkg/kvevents.(*Pool).worker
         0     0%  0.16%      0.19s  0.16%  github.com/llm-d/llm-d-router/pkg/kvevents.(*Pool).processRawMessage
         0     0%  0.16%      0.10s 0.086%  .../engineadapter.(*VLLMAdapter).ParseMessage
         0     0%  0.16%      0.08s 0.069%  github.com/vmihailenco/msgpack/v5.Unmarshal
         0     0%  0.16%      0.05s 0.043%  .../kvcache/kvblock.(*InMemoryIndex).Add
```

* **ZMQ Event Processing**: The entire ZMQ event ingestion loop (`kvevents.worker`) took just **0.16% of total CPU time** (0.19s out of 116s).
* **Msgpack Serialization**: Messages are encoded in Msgpack, which is highly efficient; parsing Msgpack bytes (`msgpack.Unmarshal`) took only **0.069% CPU**.
* **Index Insertion**: Updating EPP's local database (`InMemoryIndex.Add`) took only **0.043% CPU**.

### 3.2 Tokenizer JSON Decoding Overhead (~10% CPU)
In contrast, decoding the massive arrays of token IDs returned by the tokenizer over the HTTP client consumes a substantial fraction of EPP's CPU:
* `encoding/json.checkValid` (2.71% flat CPU)
* `encoding/json.unquoteBytes` (1.67% flat CPU)
* `encoding/json.rescanLiteral` (1.43% flat CPU)
* `encoding/json.appendString` (1.12% flat CPU)
* `runtime.mallocgc` (7.28% cumulative CPU) - garbage collection overhead from short-lived 100k-integer slice allocations.

---

## 4. Architectural Scaling Sizing Guidelines

When scaling EPP horizontally to handle larger workloads, the following characteristics should guide architecture design:

1. **Request Tokenization and Routing (Scales Horizontally)**:
   * Scaling EPP replicas linearly scales the capacity to handle incoming HTTP request volume and tokenization. 
   * **Recommendation**: If client load exceeds 10 QPS of large prompts, scale EPP replicas to distribute the tokenization JSON unmarshalling overhead.

2. **ZMQ Event Ingestion (Does Not Scale Horizontally)**:
   * Because EPP does not partition the cache index, every EPP replica must process $100\%$ of all cache events generated by all model servers. ZMQ replication CPU is currently $<0.2\%$, but at massive cluster sizes (e.g. 100+ model servers under high request volume), the event loop CPU will rise on *all* replica pods equally, regardless of EPP replica count.

3. **Memory Footprint**:
   * EPP memory footprint scales as $O(M \times N)$ where $M$ is the EPP replica count and $N$ is the model server count, because every EPP pod duplicates the entire index cache. Allocate at least **2 GiB of memory request per EPP container** and **16 GiB of memory limit per Tokenizer container** to ensure GKE node stability.
