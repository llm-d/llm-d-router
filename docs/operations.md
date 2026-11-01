# llm-d Router Container Sizing Guide

This guide provides resource sizing recommendations and high availability (HA) operational guidance for both the Endpoint Picker (EPP) and the Envoy Proxy containers in the llm-d Router. Sizing recommendations are based on empirical benchmark results under various agentic and high-throughput workloads.

---

## 1. Endpoint Picker (EPP) Sizing

The EPP acts as the routing intelligence engine. Its resource usage scales primarily with the total request rate (throughput), the complexity of prefix cache matching configuration, and the number of model-serving pods.

### Sizing Recommendations

#### CPU Allocation
- **Rule of Thumb**: Allocate **0.5 to 1.0 CPU cores per request/second** of expected throughput for large agentic workloads (approximately 100k input / 1k output tokens).
- **Scaling Behavior**: CPU utilization scales linearly with the request rate, and increases with both the input prompt size and output token length.
- **Prefix Matching Overhead**: Increasing the `maxPrefixTokensToMatch` parameter significantly increases EPP CPU utilization. At lower throughputs, a large prefix limit (such as 400,000 tokens / 6,250 blocks with effective `blockSizeTokens: 64`) can increase EPP CPU utilization by over 100% compared to a small limit (16,384 tokens / 256 blocks) due to the overhead of searching and matching prefix blocks.
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
The EPP's scaling behavior and effectiveness are highly dependent on the configured high availability (HA) mode (see [Section 4. High Availability (HA)](#4-high-availability-ha) for configuration details):

- **Active-Passive Mode**: Traffic routes to primary EPP replica(s) serving Envoy external processing (`ext-proc`) requests while standby replicas remain available for failover.
  - **Sizing Impact**: Scaling standby replicas (or total replica count under single-leader election) does **not** increase the overall EPP throughput capacity or impact resource sizing, as only the active primary replica(s) handle requests.
  - **Failover Trade-off**: [Priority Routing](#priority-routing-standalone-service-mode) and [GKE Preferred Backends](#gke-preferred-backends-gateway-mode) are recommended over lease-based leader election because they keep standby EPP pods warm and synced; see [Leader Election and Fail-Open](#leader-election-and-fail-open) for switchover timing and fail-open behavior details.
- **Active-Active Mode**: Multiple EPP replicas actively share and load-balance incoming requests, providing **near-linear throughput scaling**:

  | Replicas | Scaling Factor |
  | :--- | :--- |
  | 1 | 1.0x |
  | 2 | 2.0x |
  | 3 | 2.7x |
  | 4 | 3.5x |

  - **Note (Flow Control)**: Flow control queues and fairness accounting are per replica.
    In Active-Active mode, priority and fairness are enforced
    only within each replica's share of the traffic, and per-band capacity limits apply per
    replica, so the fleet-wide queued volume scales with the replica count.
  - **Plugin Compatibility**: EPP replicas do not share routing state by default. Active-Active mode works with stateless schedulers (`random-picker`), session affinity (`session-affinity-filter`), or plugins that read metrics from backend model servers. The optional Redis syncer shares in-flight load. Approximate prefix routing (`approx-prefix-cache-producer`) is supported in Active-Active when a `CrossReplicaSyncer` (for example `redis-syncer`) is configured and `syncCrossReplicaState` is left enabled (the default); without that sync path, each replica only learns prefixes from traffic it routed, so approximate hit rate drops under Active-Active.
  - For more technical details and context on EPP replica state sync and scaling limitations, see [Issue #1290](https://github.com/llm-d/llm-d-router/issues/1290).

##### Sharing in-flight load with Redis (Alpha)

Choose active-active with the syncer when several EPP replicas must handle traffic
concurrently and the routing plugins need peer in-flight load or shared approximate
prefix state. Use leader election when one active EPP has sufficient capacity and
the scheduling profile relies on local state that the syncer does not cover, such
as flow-control queues.

In the official charts, set `router.epp.crossReplica.enabled: true` and configure
an external Redis address. This disables leader election, enables the Alpha
`redis-syncer` plugin, and supports both `router.epp.replicas > 1` and
`router.epp.autoscaling`. Redis **7.4 or newer** must be provisioned separately;
the chart does not deploy it. For values and Secret-based authentication, see
[Active-active with Redis synchronization](../config/charts/README.md#active-active-with-redis-synchronization-alpha).

The `inflight-load-producer` shares its per-endpoint load when
`syncCrossReplicaState` is true (the default). Consumers such as
`active-request-scorer` and `token-load-scorer` can use that shared view.
The `approx-prefix-cache-producer` shares a capped recent window of prefix
hashes the same way when `syncCrossReplicaState` is true (the default).
Configuring a syncer does not change the scheduling profile or synchronize
flow-control queues or fairness accounting.

Load sharing is eventually consistent. Replicas can make simultaneous decisions
before receiving each other's updates; this does not provide an atomic
cluster-wide admission quota. Request-level `GetOrSet` coordination does not
reserve capacity across different requests.

During a Redis outage, EPP can start, and it keeps scheduling with the last peer
state it read, plus its own live load, until Redis is reachable again. An EPP
that has not read peer state yet uses local load only. Request-level
coordination calls return Redis errors. Set `stateTTL` above the observed interval between
successful endpoint refreshes, including latency variation; see the
[Redis syncer README](../pkg/epp/framework/plugins/datalayer/cross_plugin/redis/README.md)
for transport restrictions and publisher timing.

##### Approximate prefix cross-replica sync cost (microbenchmarks)

These numbers are in-process Go benchmarks (`go test -bench`), not full-cluster
llm-d-simulator runs. They isolate CPU on the request path and Redis publish
latency against miniredis. Production Valkey/Redis adds network RTT to publish
and read misses; the publisher runs off the request path on the cross-replica
sync interval (default `200ms`, `publishTimeout` default `1s`).

**Request-path `Produce` (96 decode pods, `sharedPrefix` layout, Apple M5 Pro):**

Default window budget is `maxSyncedTokens` = 2_000_000 (31250 hashes at
`blockSizeTokens` = 64), matching `lruCapacityPerServer`. Steady-window
fixture is 4 prompts filling that budget (same shape as the earlier 256-hash
bench, scaled up).

| Mode | Median of 5 | Allocs/op | Relative to sync off |
| --- | ---: | ---: | ---: |
| `syncCrossReplicaState=false` | 37 µs | 424 | 1.0x |
| Syncer configured, empty peer window | 40 µs | 424 | 1.06x |
| Syncer configured, steady peer window (default budget, already merged) | 38 µs | 520 | 1.03x |

Empty-window overhead is a `syncerActive` check plus a miss on the optional
attribute. Steady-state merge skips work when the sync attribute digest is
unchanged (Peek + cached `ContentDigest`), so cost stays near sync-off.
Novel prompts (window slides) pay snapshot, prompt-key, and indexer `Add` work.
Reproduce with:

```bash
go test ./pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/ \
  -run='^$' -bench=BenchmarkProduceCrossSync -benchmem -count=5
```

**`redis-syncer` publish/read (miniredis, same host):**

| Operation | Payload | Median of 5 |
| --- | --- | ---: |
| `Set` (publish) | 4 prompts × 64 hashes (default window shape) | 47 µs |
| `Set` (publish) | 1 prompt × 256 hashes | 48 µs |
| `Set` (publish) | 8 prompts × 256 hashes | 88 µs |
| `Get` (cached peer aggregate) | after prior `Set` | 0.40 µs |

Reproduce with:

```bash
go test ./pkg/epp/framework/plugins/datalayer/cross_plugin/redis/ \
  -run='^$' -bench='BenchmarkSyncer(Set|Get)' -benchmem -count=5
```

Operational takeaways:

- Steady peer windows are cheap on the request path when the attribute is
  unchanged; cost spikes when the window slides (novel peer prompts). Prefer
  sizing `maxSyncedTokens` for publish payload and hit-rate, not for steady CPU.
- Publish latency is on the sync interval, not in `Produce`. Keep `stateTTL`
  above the refresh interval plus observed publish RTT (include Redis/Valkey
  network time in cluster measurements).
- Redis/Valkey must support hash-field TTL (`HPEXPIRE`: Redis 7.4+ or Valkey 9+).

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

Standalone mode supports two Envoy proxy topologies. In the default `sidecar` mode, each EPP pod includes one proxy container, so the EPP replica count also determines the proxy replica count. With `router.proxy.mode: service`, the proxy runs in a separate Deployment and Service. Set `router.proxy.replicas` to scale service-mode proxies independently from EPP:

```yaml
router:
  proxy:
    mode: service
    replicas: 3
```

Sizing each Envoy proxy container depends primarily on the request throughput handled by that replica and the request and response payload size. The `router.proxy.resources` setting applies to each proxy container in either topology.

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

1. **Fully Active-Active**: Multiple EPP replicas run concurrently and share load across all instances. Suitable when scheduling algorithms and plugins do not require unified state across pods or there is a synchronization mechanism in place; see [Active-Active Mode](#active-active-mode).
2. **Active-Passive**: Traffic routes to a single primary replica set while standby replicas remain available for failover.
   - **Priority Routing (Standalone Service Mode, Recommended)**: Uses Envoy Priority Routing and local outlier detection to route traffic to primary EPP pods and shift to warm standby pods on connection failure; see [Priority Routing (Standalone Service Mode)](#priority-routing-standalone-service-mode).
   - **GKE Preferred Backends (Gateway Mode, Recommended on GKE)**: Uses GKE `GCPBackendPolicy` (`PREFERRED` and `DEFAULT` tiers) and `HealthCheckPolicy` for warm-standby failover and rate/capacity spillover; see [GKE Preferred Backends (Gateway Mode)](#gke-preferred-backends-gateway-mode).
   - **Leader Election with Fail-Open**: Uses Kubernetes `coordination.k8s.io/Lease` coordination so only the elected leader serves inference extension requests while standby pods remain out of rotation until acquiring the lease; see [Leader Election and Fail-Open](#leader-election-and-fail-open).

### Active-Active Mode

To run multiple EPP replicas concurrently with shared in-flight load, configure
active-active mode and the Redis syncer:

```yaml
router:
  epp:
    replicas: 3
    haMode: all-active
    pluginsConfigFile: payload-agnostic.yaml
    crossReplica:
      enabled: true
      redis:
        address: redis.example:6379
```

For Redis provisioning and authentication requirements, see
[Sharing in-flight load with Redis](#sharing-in-flight-load-with-redis-alpha).
For active-active operation without synchronization, omit `crossReplica` and
choose plugins compatible with local state.

See [Scaling Modes (Active-Active vs. Active-Passive)](#scaling-modes-active-active-vs-active-passive) for throughput scaling factors, flow control scope, and plugin compatibility requirements.

### Priority Routing (Standalone Service Mode)

Priority Routing is the recommended Active-Passive configuration in standalone service mode (`router.proxy.mode: service`). When priority routing is enabled (`router.proxy.priorityRouting.enabled: true`), the router suppresses leader election so all EPP replicas stay warm and synced, and uses [Envoy Priority Routing](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/priority) to organize EPP endpoints into distinct priority tiers:
* **Priority 0 (Primary / Active)**: Handles 100% of steady-state scheduling traffic.
* **Priority 1 (Standby / Passive)**: Warm standby pods ready to accept failover traffic upon primary pod failure.

#### Architecture and Failover Mechanics

1. **Deterministic Endpoint Discovery**: EPP pods run as a StatefulSet with a headless Service (`publishNotReadyAddresses: true`). Envoy targets individual pod DNS entries (`<release>-epp-0`, `<release>-epp-1`, etc.) mapped to distinct priority levels.
2. **Active Health Probing**: Envoy actively probes EPP Port 9002 via gRPC health check (`grpc.health.v1.Health`).
3. **Outlier Detection Failover**: When priority routing is enabled, if a primary pod fails or crashes, Envoy's passive Outlier Detection (`connectTimeout: 0.250s`, `consecutive_local_origin_failure: 2`) detects TCP connection failure and ejects the primary host, shifting traffic to warm Priority 1 standbys in **sub-second time** (`< 1s`, `0.00s–0.08s` fail window in `1P1S` benchmarks) without lease expiration delays.
4. **Graceful Pod Termination**: EPP pods include a native `lifecycle.preStop.sleep` hook (5 seconds) during planned deletion or rollout on Kubernetes 1.30+ with `PodLifecycleSleepAction` enabled. This gives Envoy active health checks time to detect pod shutdown and redirect new traffic to standby endpoints before SIGTERM, allowing in-flight gRPC streams to drain.
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

| Parameter | Default | Description |
|---|---|---|
| `router.proxy.mode` | `sidecar` | Must be set to `service` to enable Priority Routing. |
| `router.proxy.priorityRouting.enabled` | `false` | Enable Envoy Priority Routing for the `ext_proc` cluster. |
| `router.proxy.priorityRouting.primaryReplicas` | `1` | Number of active EPP pod ordinals in Priority 0. |
| `router.proxy.priorityRouting.standbyReplicas` | `1` | Number of warm standby EPP pod ordinals in Priority 1. |
| `router.proxy.priorityRouting.healthyPanicThreshold` | `10.0` | Threshold percentage to prevent panic routing during primary ejection. |
| `router.proxy.priorityRouting.dnsRefreshRate` | `5s` | DNS resolution refresh rate for headless EPP endpoints. |
| `router.proxy.priorityRouting.connectTimeout` | `0.250s` | Connection timeout to detect unreachable primary pods. |
| `router.proxy.healthCheckInterval` | `10s` | Active gRPC health check probe interval. |
| `router.proxy.healthCheckTimeout` | `2s` | Health check probe timeout. |
| `router.proxy.healthCheckUnhealthyThreshold` | `3` | Number of failed probes before marking an endpoint unhealthy. |
| `router.proxy.healthCheckHealthyThreshold` | `2` | Number of passing probes required before admitting recreated pods. |
| `router.epp.terminationGracePeriodSeconds` | `130` | Grace period (seconds) before SIGKILL on pod teardown. |

### GKE Preferred Backends (Gateway Mode)

In GKE Gateway mode (`llm-d-router-gateway`), `provider.gke.preferredBackends.enabled: true` is the recommended Active-Passive configuration. It deploys EPP as a `StatefulSet` (with leader election disabled so all replicas stay warm) and creates per-ordinal `ClusterIP` Services with zonal NEGs (`cloud.google.com/neg`), `GCPBackendPolicy`, and `HealthCheckPolicy` resources.

#### Architecture and Failover Mechanics

1. **Priority Tiers and NEG Attachment**: Primary active pod ordinals (`preferredReplicas`) are pinned to `backendPreference: PREFERRED`, and warm standby pod ordinals (`defaultReplicas`) are pinned to `backendPreference: DEFAULT`. The `InferencePool` anchors `endpointPickerRef` to the standby Service (`<release>-epp-backup`) while primary zonal NEGs are attached to the GKE `BackendService`.
2. **Rate and Capacity Spillover**: When incoming traffic exceeds `capacityScalerPercent` of `maxRatePerEndpoint` (`balancingMode: RATE`), GKE's managed Envoy data plane spills excess burst traffic over to the `DEFAULT` standby tier immediately in memory without dropping connections.
3. **Health-Check Failover**: When a primary pod fails or drains, Cloud Load Balancing gRPC health checks on port 9003 mark the primary NEG unhealthy and shift traffic to warm `DEFAULT` standby pods (completing within a `~3s` health-prober window in single-primary `1P1S` drain benchmarks, or with zero dropped requests when `preferredReplicas >= 2`).

#### Helm Configuration

```yaml
provider:
  name: gke
  gke:
    preferredBackends:
      enabled: true
      preferredReplicas: 1
      defaultReplicas: 1
      balancingMode: RATE
      maxRatePerEndpoint: 100
      capacityScalerPercent: 100
```

| Parameter | Default | Description |
|---|---|---|
| `provider.name` | `none` | Must be set to `gke` to enable GKE Gateway resources. |
| `provider.gke.preferredBackends.enabled` | `false` | Enable Preferred Backends high availability routing for GKE. |
| `provider.gke.preferredBackends.preferredReplicas` | `1` | Replica count for primary active pod ordinals pinned to the `PREFERRED` tier. |
| `provider.gke.preferredBackends.defaultReplicas` | `1` | Replica count for warm standby pod ordinals pinned to the `DEFAULT` tier. |
| `provider.gke.preferredBackends.balancingMode` | `RATE` | Load balancing calculation mode (`RATE`, `UTILIZATION`, or `CONNECTION`). |
| `provider.gke.preferredBackends.maxRatePerEndpoint` | `100` | Maximum requests per second per primary pod instance before spilling over to standby tiers. |
| `provider.gke.preferredBackends.capacityScalerPercent` | `100` | Effective capacity ceiling percentage (`0` to `100`) before spillover. |

### Leader Election and Fail-Open

In multi-replica deployments without priority routing or GKE preferred backends, the router coordinates active-passive replicas using Kubernetes lease-based leader election.

#### Architecture and Failover Mechanics

- **Leader Coordination**: EPP replicas contend for a `coordination.k8s.io/Lease`. Helm enables leader election for fixed `router.epp.replicas > 1` unless explicitly disabled, `router.epp.haMode: all-active` or cross-replica synchronization is selected, or priority routing or GKE preferred backends is enabled. The elected leader responds to active gRPC extension requests on Port 9002, while standby replicas do not serve external processing traffic until acquiring the lease. See [Active-Active Mode](#active-active-mode) to disable leader election for concurrent active-active replicas.
- **Fail-Open Behavior (Gateway vs. Standalone)**:
  - **Gateway Mode (`llm-d-router-gateway`)**: With `router.inferencePool.failureMode: FailOpen` (the default), the Gateway bypasses an unreachable EPP during a leader transition and routes requests directly to healthy model-server backends in the `InferencePool` to preserve request availability.
  - **Standalone Mode (`llm-d-router-standalone`)**: With `router.proxy.failOpen: true` (the default) and `proxyType: envoy`, the chart sets `failure_mode_allow: true` on Envoy's `ext_proc` filter and adds a fallback route. Requests that EPP schedules carry the `x-gateway-destination-endpoint` header and go to `original_destination_cluster`; requests that pass the filter because EPP is unreachable carry no header and go to the `inference_pool_fallback` cluster, which load-balances (`LEAST_REQUEST`) across the ready pods of a headless Service (`<release>-epp-model-servers`) selecting `router.modelServers.matchLabels` on the first `router.modelServers.targetPorts` entry. Envoy removes any client-supplied `x-gateway-destination-endpoint` header before routing, so only EPP picks the endpoint. Agentgateway has no fail-open setting and rejects requests while EPP is unreachable.
- **Switchover Disadvantage (10-30s EPP Unavailability Window)**: When the active leader fails, Kubernetes lease expiration (`--ha-lease-duration`, default `15s`), standby readiness probe transition, and Service endpoint propagation take **10 to 30 seconds** before a standby replica begins serving traffic. During this window, EPP is unavailable: in Gateway mode with `FailOpen` and in standalone Envoy mode with `router.proxy.failOpen: true`, **routing is purely unoptimized** (requests bypass KV-cache affinity, prefix-cache scoring, load-aware scheduling, and flow control). Use [Priority Routing](#priority-routing-standalone-service-mode) or [GKE Preferred Backends](#gke-preferred-backends-gateway-mode) as the recommended Active-Passive setup.

#### Helm Configuration

```yaml
router:
  epp:
    replicas: 2
    flags:
      ha-enable-leader-election: true
  inferencePool:
    failureMode: FailOpen
```

| Parameter | Default | Description |
|---|---|---|
| `router.epp.replicas` | `1` | Number of EPP replicas (1 active leader and `replicas - 1` standbys). |
| `router.epp.flags.ha-enable-leader-election` | `true` (when `replicas > 1`) | Enable Kubernetes Lease leader election across EPP replicas. |
| `router.inferencePool.failureMode` | `FailOpen` | Gateway `InferencePool` failure policy when EPP is unreachable (`FailOpen` or `FailClose`). |
| `router.proxy.failOpen` | `true` | Sets `failure_mode_allow` on the standalone Envoy `ext_proc` filter and routes requests EPP did not schedule to the model servers through the `inference_pool_fallback` cluster. |

### Horizontal Pod Autoscaling (HPA)

EPP supports horizontal pod autoscaling through Kubernetes HorizontalPodAutoscaler (HPA v2). When autoscaling is enabled, Helm omits `spec.replicas` on the EPP Deployment, and the HPA controller manages replica counts.

#### Operational Prerequisites and Constraints

- **Active-Active Mode Required**: Autoscaling requires Active-Active EPP operation. Standby replicas in leader-elected setups remain `NotReady` by design, which blocks HPA stabilization. The chart enforces active-active mode when autoscaling is enabled and blocks explicit leader election (`--ha-enable-leader-election`).
- **Incompatible with StatefulSet Topologies**: Priority routing and GKE preferred backends render EPP as a StatefulSet with fixed ordinal hostnames (`<name>-0`, `<name>-1`) for static Envoy routing. Autoscaling requires a standard Deployment managing a dynamically changing replica set in active-active mode. Enabling autoscaling alongside `router.proxy.priorityRouting.enabled: true` or GKE preferred backends fails chart validation.
- **RollingUpdate Strategy**: The deployment defaults to `RollingUpdate` strategy (`maxUnavailable: 0`, `maxSurge: 1`) under autoscaling to keep serving capacity during scale events. Setting `router.epp.deploymentStrategy` overrides this default.
- **Fail-Open During Capacity Gaps**: The rollout strategy does not cover losing the only Ready EPP replica while new replicas are still starting. In `sidecar` mode the proxy shares the EPP pod, so fail-open cannot keep traffic flowing once that pod is gone. In Gateway mode, `router.inferencePool.failureMode: FailOpen` routes requests directly to healthy `InferencePool` backends, without EPP scoring, while no EPP replica is reachable. In standalone `service` mode with Envoy, `router.proxy.failOpen: true` routes requests directly to the model servers, without EPP scoring, through the `inference_pool_fallback` cluster; see [Leader Election and Fail-Open](#leader-election-and-fail-open) and [Standalone Proxy Autoscaling (Service Mode)](#standalone-proxy-autoscaling-service-mode). Fail-open applies only when EPP is unreachable, not when a draining EPP returns an error response.
- **Replica Count Configuration**: When autoscaling is enabled, `router.epp.replicas` is ignored. Replica counts are controlled by `autoscaling.minReplicas` and `autoscaling.maxReplicas`.
- **Plugin Compatibility**: Autoscaling requires active-active compatible plugins; see [Scaling Modes (Active-Active vs. Active-Passive)](#scaling-modes-active-active-vs-active-passive).

#### Target Utilization Guidance

- **Target CPU Utilization**: The recommended starting default is **80%**. This leaves headroom to absorb traffic spikes while new pods initialize and pass readiness probes. Higher utilization leaves less headroom for traffic bursts while new pods start up. Operators should tune this target based on their workload shape, token lengths, and latency SLAs.
- **Container Sizing**: Set container CPU requests based on expected steady-state per-pod load (refer to the sizing guidelines in Section 1 for CPU core-to-throughput estimates).

#### Helm Configuration

```yaml
router:
  epp:
    autoscaling:
      enabled: true
      minReplicas: 1
      maxReplicas: 5
      targetCPUUtilizationPercentage: 80
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
    resources:
      requests:
        cpu: "8"
        memory: 16Gi
      limits:
        cpu: "8"
        memory: 16Gi
```

See `router.epp.autoscaling` in `config/charts/routerlib/values.yaml` for all fields and defaults. Custom metrics can be supplied via `autoscaling.metrics` to replace auto-generated CPU and memory metrics (target percentage fields remain range-validated if defined).

### Standalone Proxy Sidecar Termination

In standalone sidecar mode, the proxy container includes a five-second native `lifecycle.preStop.sleep` hook on Kubernetes 1.30+ with `PodLifecycleSleepAction` enabled. The delay allows Service endpoint removal to propagate before the proxy receives SIGTERM.

### Standalone Proxy Autoscaling (Service Mode)

In standalone service mode (`router.proxy.mode: service`), the proxy runs as an independent Deployment and Service that can be autoscaled using Kubernetes HorizontalPodAutoscaler (HPA v2). When enabled via `router.proxy.autoscaling.enabled: true`, Helm omits `spec.replicas` on the proxy Deployment and generates an HPA targeting the proxy Deployment.

#### Operational Prerequisites and Constraints

- **Service Mode Required**: Proxy autoscaling is supported only when `router.proxy.mode: service` and `router.proxy.enabled: true`. In `sidecar` mode, the proxy lifecycle and replica count are tied to the EPP pod.
- **Replica Count Configuration**: When autoscaling is enabled, `router.proxy.replicas` is ignored. Replica counts are managed by `router.proxy.autoscaling.minReplicas` and `router.proxy.autoscaling.maxReplicas`.
- **Drain and Termination Grace Period**: Envoy drains in-flight connections over a 60-second window (`--drain-time-s 60`). The proxy pod defaults `terminationGracePeriodSeconds: 70` and includes a 5-second `preStop` delay on Kubernetes 1.30+ to allow Service endpoint deregistration before Envoy terminates listeners.

#### Helm Configuration

```yaml
router:
  proxy:
    mode: service
    autoscaling:
      enabled: true
      minReplicas: 2
      maxReplicas: 10
      targetCPUUtilizationPercentage: 80
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
    resources:
      requests:
        cpu: "4"
        memory: 8Gi
      limits:
        memory: 16Gi
```

See `router.proxy.autoscaling` in `config/charts/llm-d-router-standalone/values.yaml` for all fields and defaults.
