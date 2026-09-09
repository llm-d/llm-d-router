# llm-d Router Container Sizing Guide

This guide provides resource sizing recommendations for both the Endpoint Picker (EPP) and the Envoy Proxy containers in the llm-d Router. Sizing recommendations are based on empirical benchmark results under various agentic and high-throughput workloads.

---

## 1. Endpoint Picker (EPP) Sizing

The EPP acts as the routing intelligence engine. Its resource usage scales primarily with the total request rate (throughput), the complexity of prefix cache matching configuration, and the number of model-serving pods.

### Sizing Recommendations

#### CPU Allocation
- **Rule of Thumb**: Allocate **0.5 to 1.0 CPU cores per request/second** of expected throughput for large agentic workloads (approximately 100k input / 1k output tokens).
- **Scaling Behavior**: CPU utilization scales linearly with the request rate, and increases with both the input prompt size and output token length.
- **Prefix Matching Overhead**: Increasing the `maxPrefixTokensToMatch` parameter increases EPP CPU utilization. At lower throughputs, a large prefix limit (such as 400,000 tokens / 6,250 blocks with effective `blockSizeTokens: 64`) can increase EPP CPU utilization by over 100% compared to a small limit (16,384 tokens / 256 blocks) due to the overhead of searching and matching prefix blocks.
- **Idle CPU Scaling**: Idle CPU usage of the EPP container scales with the number of model-serving pods in the cluster due to continuous metric scraping. For example, in a cluster with 100 model-serving pods, the idle CPU usage of the EPP container grows to approximately **7.5 cores**.

#### Memory Allocation
- **Base Memory**: EPP memory usage is relatively low and stable with small output token requests, but scales with the number of concurrent inflight requests.
- **Inflight Requests Impact**: Memory usage increases with the number of concurrent inflight requests and the output (decode) token length.
- **Flow Control Queues**: With flow control enabled, requests that cannot dispatch
  under saturation are buffered in EPP memory, including their request bodies. The buffered volume
  is bounded per priority band by `priorityBands[].maxRequests` (default 5000) and `maxBytes`
  (default 1G), which `defaultPriorityBand` sets as a template for bands you do not list; budget for
  the sum of the per-band `maxBytes` limits of the priority levels your traffic actually uses, on top
  of the inflight-request sizing above. The global `flowControl.maxRequests` / `maxBytes` caps
  default to unlimited, so set a global `maxBytes` under the container memory limit: at the per-band
  default, a handful of bands clears the sizing guidance below before any band cap engages. Lower
  these limits (or set a shorter `defaultRequestTTL`) to trade queueing for earlier shedding. A
  `noEndpointRequestTTL` sized for a cold start holds bodies for that whole budget while the pool is
  empty, so the band caps, not the budget, become what bounds queue memory during a scale-from-zero.
- **Sizing Guidelines**:
  - For a request rate of 50 to 100 requests/second with 1k output tokens, EPP requires between **4 GiB and 6 GiB** of memory.
  - For workloads with longer output lengths (such as 5k output tokens), memory usage can reach **20+ GiB** due to the accumulation of state for concurrent inflight requests.

#### Scaling Modes (Active-Active vs. Active-Passive)
The EPP's scaling behavior and effectiveness are highly dependent on the configured high availability (HA) mode:

- **Active-Passive Mode**: Only one EPP replica actively serves Envoy external processing (`ext-proc`) requests at a time, while the others remain in standby. 
  - **Sizing Impact**: Scaling the replica count does **not** increase the overall EPP throughput capacity or impact resource sizing, as only the active replica handles requests.
- **Active-Active Mode**: Multiple EPP replicas actively share and load-balance incoming requests, providing **near-linear throughput scaling**:

  | Replicas | Scaling Factor |
  | :--- | :--- |
  | 1 | 1.0x |
  | 2 | 2.0x |
  | 3 | 2.7x |
  | 4 | 3.5x |

  - **Note (Flow Control)**: Flow control state (queues, fairness accounting, and the saturation
    view) is per replica and not shared. In Active-Active mode, priority and fairness are enforced
    only within each replica's share of the traffic, and per-band capacity limits apply per
    replica, so the fleet-wide queued volume scales with the replica count.
  - **Warning (Prefix Routing)**: **Active-Active mode should be avoided when using approximate prefix routing.** Because EPP replicas do not share prefix state, each replica only has visibility into the prefix state of the requests it has individually handled. This partition of state significantly degrades prefix cache hit rates, making prefix caching highly inefficient.
  - For more technical details and context on EPP replica state sync and scaling limitations, see [Issue #1290](https://github.com/llm-d/llm-d-router/issues/1290).

### Performance Reference Data

The following tables present empirical benchmark results for EPP running with llm-d-simulator simulating Qwen/Qwen3-8B.

#### Throughput and Prefix Block Sizing
This table shows peak CPU and memory utilization for EPP under a 100k token workload (95k system prompt, 5k question prompt, and 1k output tokens) when using approximate prefix caching across 100 model-serving pods.

| Configuration | Request Rate (Req/s) | maxPrefixTokensToMatch | Peak CPU (Cores) | Peak Memory (GiB) | Scheduler P50 Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Small Prefix Match | 5.0 | 4096 | 1.19 | 0.26 | 0.00010 |
| Large Prefix Match | 5.0 | 100000 | 3.82 | 0.65 | 0.00010 |
| Small Prefix Match | 98.7 | 4096 | 35.17 | 2.46 | 0.00014 |
| Large Prefix Match | 98.8 | 100000 | 46.50 | 3.41 | 0.00020 |

Configuration used: [#1287](https://github.com/llm-d/llm-d-router/issues/1287#issuecomment-4666058475).
These were run against 0.9.0 EPP container image.

#### Output Length and Prefix Matching Complexity
This table shows EPP peak resource usage at a constant request rate of 50 requests/second with a 100k input token workload, varying the output token length and the `maxPrefixTokensToMatch` configuration.

| Input Tokens | Output Tokens | maxPrefixTokensToMatch | Peak CPU (Cores) | Peak Memory (GiB) |
| :--- | :--- | :--- | :--- | :--- |
| 100k | 500 | 4096 | 15.13 | 2.27 |
| 100k | 500 | 32768 | 17.14 | 3.76 |
| 100k | 1000 | 4096 | 17.51 | 3.66 |
| 100k | 1000 | 32768 | 20.28 | 5.23 |
| 100k | 5000 | 16384 | 30.95 | 12.54 |
| 100k | 10000 | 8192 | 32.53 | 12.54 |

Configuration used: [#1287](https://github.com/llm-d/llm-d-router/issues/1287#issuecomment-4619775397)
These were run against 0.9.0 EPP container image.

---

## 2. Envoy Proxy Sizing (Standalone Mode)

When running the llm-d Router in **Standalone Mode**, the Envoy proxy container runs in the same pod alongside the EPP container. Sizing the Envoy proxy container depends primarily on the request throughput (requests/second) and the request/response payload size (concurrency of streaming data).

### Sizing Recommendations

#### CPU Allocation
- **Scaling Behavior**: Envoy's CPU usage scales linearly with the total throughput (requests/second).
- **Sizing Guidelines**:
  - For lower throughput (e.g., < 10 requests/second), **1.2 to 2.0 CPU cores** is sufficient.
  - For higher throughput of large contexts (e.g., 100 requests/second with 100k/1k tokens), allocate at least **8 CPU cores** (peak usage observed at **7.27 cores**).
  - For very high throughput of smaller contexts (e.g., 892 requests/second with 10k/1k tokens), allocate at least **10 CPU cores** (peak usage observed at **8.78 cores**).

#### Memory Allocation
- **Sizing Guidelines**: Envoy's memory footprint remains extremely stable and is primarily influenced by the number of concurrent active connections and buffer sizes. Allocate at least **2 GiB of memory** (peak memory usage is stable between **1.3 and 1.4 GiB** across all tested throughputs and context lengths).

### Performance Reference Data

The following table presents empirical benchmark results for the Envoy proxy container in Standalone Mode under different workloads:

| Input Tokens | Output Tokens | Throughput (Req/s) | Peak CPU (Cores) | Peak Memory (GiB) |
| :--- | :--- | :--- | :--- | :--- |
| 100k | 1k | 10.0 | 1.20 | 1.30 |
| 100k | 1k | 100.0 | 7.27 | < 1.40 |
| 10k | 1k | 892.0 | 8.78 | 1.40 |

---

## 3. Helm Configuration Example

For deployments managed via Helm (such as using the `llm-d-router-standalone` chart), both the EPP and the Envoy proxy container resource requests and limits can be configured in a custom values file, such as `resource_overrides.yaml`.

Below is an example `resource_overrides.yaml` snippet configured to support a throughput of up to 50 requests/second for 100k/1k agentic requests in Standalone Mode:

```yaml
router:
  # Endpoint Picker (EPP) Container Resources
  epp:
    resources:
      requests:
        cpu: "32"
        memory: "64Gi"
      limits:
        memory: "128Gi"

  # Envoy Proxy Container Resources
  proxy:
    resources:
      requests:
        cpu: "8"
        memory: "2Gi"
      limits:
        memory: "4Gi"
```

To apply these values during deployment, run the Helm install or upgrade command with your custom values file:

```bash
helm install optimize-baseline ./config/charts/llm-d-router-standalone -f resource_overrides.yaml
```

---

## 4. High Availability (HA)

The router supports multiple High Availability (HA) modes:

1. **Fully Active-Active**: Multiple EPP replicas run concurrently and share load across all instances. Suitable when scheduling algorithms and plugins do not require unified state across pods or there is a synchronization mechanism in place.
2. **Active-Passive**: Traffic routes to a single primary replica set while standby replicas remain available for failover.
   - **Priority Routing**: Available only when proxy mode is set to service (`router.proxy.mode: service`). Uses Envoy Priority Routing and outlier detection to route traffic to Primary EPP replicas (Priority 0) and shift traffic to Standby EPP replicas (Priority 1) upon primary failure.
   - **Leader Election with Fail-Open**: Uses Kubernetes `coordination.k8s.io/Lease` coordination so only the elected leader serves inference extension requests. Standby pods remain idle until acquiring the lease. If the active leader fails, the proxy operates in fail-open mode, routing traffic directly to model servers until a standby acquires leadership.

### Priority Routing

Priority Routing is only available in standalone service mode (`router.proxy.mode: service`). When priority routing is enabled (`router.proxy.priorityRouting.enabled: true`), the router uses [Envoy Priority Routing](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/priority) to organize EPP endpoints into distinct priority tiers:
* **Priority 0 (Primary / Active)**: Handles 100% of steady-state scheduling traffic.
* **Priority 1 (Standby / Passive)**: Warm standby pods ready to accept failover traffic upon primary pod failure.

#### Architecture and Failover Mechanics

1. **Deterministic Endpoint Discovery**: EPP pods run as a StatefulSet with a headless Service (`publishNotReadyAddresses: true`). Envoy targets individual pod DNS entries (`<release>-epp-0`, `<release>-epp-1`, etc.) mapped to distinct priority levels.
2. **Active Health Probing**: Envoy actively probes EPP Port 9002 via gRPC health check (`grpc.health.v1.Health`).
3. **Outlier Detection Failover**: When priority routing is enabled, if a primary pod fails or crashes, Envoy's Outlier Detection detects TCP connection failure and ejects the primary host, shifting traffic to Priority 1 standbys in sub-second time without lease expiration delays.
4. **Graceful Pod Termination**: EPP pods include a `lifecycle.preStop` hook (`sleep 5`) during planned deletion or rollout. This gives Envoy active health checks time to detect pod shutdown and redirect new traffic to standby endpoints before SIGTERM, allowing in-flight gRPC streams to drain.
5. **Safe Failback**: When a replacement primary pod is rescheduled, the health check `healthy_threshold` requires consecutive passing health probes before Envoy restores traffic to Priority 0, ensuring the new EPP pod has finished syncing model server state and inference pools.

#### Helm Configuration

```yaml
router:
  proxy:
    mode: service
    priorityRouting:
      enabled: true
      primaryReplicas: 1
      standbyReplicas: 1
```

#### Tuning Parameters

| Parameter | Default | Description |
|---|---|---|
| `router.proxy.priorityRouting.healthyPanicThreshold` | `10.0` | Threshold percentage to prevent panic routing during primary ejection. |
| `router.proxy.priorityRouting.dnsRefreshRate` | `5s` | DNS resolution refresh rate for headless EPP endpoints. |
| `router.proxy.priorityRouting.connectTimeout` | `0.250s` | Connection timeout to detect unreachable primary pods. |
| `router.proxy.healthCheckInterval` | `10s` | Active gRPC health check probe interval. |
| `router.proxy.healthCheckTimeout` | `2s` | Health check probe timeout. |
| `router.proxy.healthCheckUnhealthyThreshold` | `3` | Number of failed probes before marking an endpoint unhealthy. |
| `router.proxy.healthCheckHealthyThreshold` | `2` | Number of passing probes required before admitting recreated pods. |
| `router.epp.terminationGracePeriodSeconds` | `130` | Grace period (seconds) before SIGKILL on pod teardown. |

### Leader Election and Fail-Open

In multi-replica deployments without priority routing (`router.epp.replicas > 1`), the router coordinates active-passive replicas using Kubernetes lease-based leader election:

- **Leader Coordination**: EPP replicas contend for a `coordination.k8s.io/Lease`. The `--ha-enable-leader-election` flag enables leader election in EPP (automatically injected by Helm when `router.epp.replicas > 1`). The elected leader responds to active gRPC extension requests on Port 9002, while standby replicas run idle. (To run multi-replica in Fully Active-Active mode instead, set `router.epp.flags.ha-enable-leader-election: false`).
- **Fail-Open Resiliency**: With `router.proxy.failOpen: true` (the default in standalone mode) or `router.inferencePool.failureMode: FailOpen`, if the active leader crashes or restarts, the proxy passes requests directly to backend model servers without dropping traffic during the lease transition period.

```yaml
router:
  epp:
    replicas: 2
    flags:
      ha-enable-leader-election: true
  proxy:
    failOpen: true
```

