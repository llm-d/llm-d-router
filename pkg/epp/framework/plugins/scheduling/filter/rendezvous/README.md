# Multi-cluster Rendezvous Filter

**Type:** `multicluster-rendezvous-filter` (Alpha: needs `--allow-experimental-plugins=true`)

Keeps each conversation on one cluster of a multi-cluster hub. Clusters take
conversations in proportion to their ready pods; a conversation moves only when
its cluster is full or down. Only moved conversations are stored, in memory per
hub replica.

## Decision

| Step | Rule |
| --- | --- |
| 1. Key | The `agent-identity` attribute (set by the [agent-identity](../../../requesthandling/requestheader/agentidentity/README.md) plugin from agent session headers: Claude Code, OpenCode, Codex), then `x-session-id`, else the conversation up to its first user message ([rendezvous.Key](../../util/rendezvous/rendezvous.go)). No key: the pinned cluster if it has room, else all clusters with room (the scorers pick). |
| 2. Moved | A key that spilled earlier goes to its spill cluster while it has room. |
| 3. Pin | A tenant (`x-llm-d-inference-fairness-id`) in a cluster's `pins` label goes there. |
| 4. Hash | Weighted rendezvous hash; weight = ready pods (`llm_d_epp_ready_endpoints`; unreported: 1; 0: excluded). A cluster added, removed or reweighted moves only its own keys. |
| 5. Spill | Home has no room: the filter returns all clusters with room, the scorers pick, and `PreRequest` keeps the pick until unused for `stickTtlSeconds`. |
| 6. No room anywhere | A spilled conversation stays put; others stay home unless home is down or has no ready pods. |

## Room

A cluster has room when it has ready pods, its metrics are at most `maxMetricsAgeSeconds` (default 30 s) old, and:

- **With flow control** (`llm_d_epp_flow_control_*`): no band of the request's
  priority or higher has a mean wait (dispatched, last 30 s) above
  `maxWaitSeconds`, or held requests with no dispatch for that long.
- **Without flow control**: the average vLLM queue per pod is below `maxQueue`.

Also:

- Ready-pod changes are held before they apply, per hub replica; see the
  [extractor README](../../../datalayer/extractor/metrics/README.md#multi-cluster-support).
- The scrape interval (default 50 ms) must stay well below `maxWaitSeconds`.
- `multicluster-metrics-extractor` is required, or the config fails at startup.
- Give the hub the clusters' objectives with `--objectives-file`; otherwise every request is priority 0.
- The moved table is per hub replica (not shared through a `CrossReplicaSyncer`), so with active-active replicas a spilled conversation can return home on
  another replica (measured: 30-60% of spilled sessions). Run one replica, or active-passive (chart
  `router.proxy.mode: service` with `router.proxy.priorityRouting.enabled: true`; measured: 0%).

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `maxWaitSeconds` | `10` | Flow-control wait limit. Must be > 0. |
| `maxQueue` | `4` | vLLM queue-per-pod limit, for clusters without flow control. Must be > 0. |
| `stickTtlSeconds` | `1800` | Time a spilled conversation stays after its last turn; `0`: returns once home has room. At most one year. |
| `maxMetricsAgeSeconds` | `30` | A cluster whose metrics are older is down. Lower fails over faster when a cluster router stops; too low marks a slow scrape as down. Must be > 0 and at most 3600. |
| `guardPluginName` | none | A fail-closed filter replacing the built-in room check; excludes `maxQueue` and `maxWaitSeconds`. |

Cluster label `pins` (in `clusters.yaml`): comma-separated tenants for that cluster.

## Metrics

`llm_d_epp_multicluster_rendezvous_decisions_total{cluster, decision}` counts requests by decision (`home`,
`pin`, `kept`, `spill`, `failover`); `llm_d_epp_multicluster_rendezvous_moved_entries` is the size of the moved
table. The spill rate per cluster is `rate(...{decision="spill"}[5m])`.

## Example

Use TTFT and KV-cache scorers. TTFT is observed per cluster from streamed responses the hub routes and
includes the cluster's flow-control wait; until a cluster has observations, KV-cache use ranks it. Behind
flow control the vLLM queue stays near 0, so `multicluster-queue-scorer` only suits clusters without it.

```yaml
plugins:
- type: multicluster-file-discovery
  name: discovery
  parameters: {path: /etc/epp/clusters/clusters.yaml, watchFile: true}
- type: multicluster-metrics-data-source
  name: metrics-source
  parameters: {scheme: http}   # the cluster routers' metrics port
- type: multicluster-metrics-extractor
  name: metrics-extractor
- type: agent-identity               # keys agent sessions by their session header
- type: multicluster-rendezvous-filter
  name: rendezvous
- type: latency-observer-producer-hub
  name: ttft-observer
  parameters: {intervalDuration: 1s, minRequests: 10}
- type: latency-observation-scorer-hub
  name: ttft
  parameters: {ttftPercentilesProducerName: ttft-observer}
- type: multicluster-kv-cache-utilization-scorer
  name: kv-cache
- type: max-score-picker
  name: picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: rendezvous
  - {pluginRef: ttft, weight: 1}
  - {pluginRef: kv-cache, weight: 1}
  - pluginRef: picker
dataLayer:
  discovery:
    endpoints:
      pluginRef: discovery
  sources:
  - pluginRef: metrics-source
    extractors:
    - pluginRef: metrics-extractor
  - pluginRef: ttft-observer
```

The key, hash and guard lookup are shared with the
[DisaggregatedSet rollout screener](../../../requesthandling/screener/disaggregatedsetrollout/README.md).
