# Utilization Filter Plugin

**Type:** `utilization-filter`

This plugin filters candidate endpoints by utilization metrics.

## What it does

For each scheduling cycle, the plugin evaluates the configured `conditions` against each candidate endpoint. Each condition caps one metric; an endpoint is kept only when it satisfies every condition, and dropped as soon as any condition's metric exceeds its `maxValue`. Multiple conditions therefore express an OR of overload rules: `conditions: [{running-requests <= 40}, {waiting-queue <= 0}]` drops an endpoint when it has more than 40 running requests or a non-empty queue.

Supported metrics:

- `active-requests` — the in-flight request count tracked by the EPP, produced by the `inflight-load-producer` data producer. The count covers every request the EPP has routed to the endpoint and not yet seen complete, so it includes both running and queued requests; the EPP does not distinguish the two.
- `running-requests` — the running request count reported by the model server.
- `waiting-queue` — the waiting queue size reported by the model server.
- `kv-cache-utilization` — the KV cache usage fraction (0 to 1) reported by the model server.

Endpoints missing a metric are treated as having a zero value for it. When every endpoint is filtered out and `fallbackOnEmpty` is `true`, the original candidate list is returned unchanged, so the request can still be routed somewhere.

## Inputs consumed

When a condition uses `active-requests`, the plugin consumes `InFlightLoadDataKey` (`InFlightLoad`) — the per-endpoint in-flight load snapshot maintained by the `inflight-load-producer` data producer. The model-server metrics are read directly from the endpoint's scraped metrics and need no producer.

## Configuration

| Parameter                  | Required | Description                                                                                     |
|----------------------------|----------|--------------------------------------------------------------------------------------------------|
| `conditions`               | yes      | List of metric caps; at least one. An endpoint must satisfy all of them to be kept.               |
| `conditions[].metric`      | yes      | One of `active-requests`, `running-requests`, `waiting-queue`, `kv-cache-utilization`.            |
| `conditions[].maxValue`    | yes      | Maximum metric value an endpoint may have and still pass the condition. Must be non-negative; for `kv-cache-utilization` it is a fraction and must be at most 1. |
| `fallbackOnEmpty`          | no       | When `true`, return the unfiltered candidates if every endpoint was dropped. Default `false`.     |
| `inFlightLoadProducerName` | no       | `active-requests` only: name of the in-flight load producer instance to consume from. Defaults to the default producer. |

**Configuration Example:**
```yaml
plugins:
  - type: utilization-filter
    name: drop-overloaded-endpoints
    parameters:
      conditions:
        - metric: running-requests
          maxValue: 40
        - metric: waiting-queue
          maxValue: 0
        - metric: kv-cache-utilization
          maxValue: 0.8
      fallbackOnEmpty: true
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: drop-overloaded-endpoints
```

## See also

The `active-request-scorer`, `running-requests-scorer`, `queue-scorer`, and `kv-cache-utilization-scorer` plugins are the scoring counterparts of this filter: instead of dropping endpoints, they rank them by the same signals. The `endpoint-attribute-filter` plugin filters by arbitrary custom metrics configured through the metrics extraction layer.
