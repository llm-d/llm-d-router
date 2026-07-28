# Active Request Filter Plugin

**Type:** `active-request-filter`

This plugin filters candidate endpoints by the in-flight request count the EPP tracks for each endpoint.

## What it does

For each scheduling cycle, the plugin reads the in-flight load attribute produced by the `inflight-load-producer` data producer from each candidate endpoint and keeps only the endpoints whose in-flight request count is at most `maxRequests`. The count covers every request the EPP has routed to the endpoint and not yet seen complete, so it includes both running and queued requests; the EPP does not distinguish the two. Endpoints without the attribute are treated as having zero in-flight requests and kept.

When every endpoint is filtered out and `fallbackOnEmpty` is `true`, the original candidate list is returned unchanged, so the request can still be routed somewhere.

## Inputs consumed

The plugin consumes:

- `InFlightLoadDataKey` (`InFlightLoad`) — the per-endpoint in-flight load snapshot maintained by the `inflight-load-producer` data producer.

## Configuration

| Parameter                  | Required | Description                                                                                   |
|----------------------------|----------|-----------------------------------------------------------------------------------------------|
| `maxRequests`              | yes      | Maximum in-flight request count an endpoint may have and still be kept. Must be positive.      |
| `fallbackOnEmpty`          | no       | When `true`, return the unfiltered candidates if every endpoint was dropped. Default `false`.  |
| `inFlightLoadProducerName` | no       | Name of the in-flight load producer instance to consume from. Defaults to the default producer. |

**Configuration Example:**
```yaml
plugins:
  - type: active-request-filter
    name: drop-busy-endpoints
    parameters:
      maxRequests: 10
      fallbackOnEmpty: true
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: drop-busy-endpoints
```

## See also

The `active-request-scorer` plugin is the scoring counterpart of this filter: instead of dropping endpoints, it ranks them by the same in-flight request count.
