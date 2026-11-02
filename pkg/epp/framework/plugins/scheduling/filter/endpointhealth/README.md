# Endpoint Health Filter Plugin

**Type:** `endpoint-health-filter`

This plugin drops candidate endpoints that are failing, and keeps routing to them again once they recover. It works for every discovery type, including file discovery, where endpoints have no readiness signal of their own.

## What it does

The plugin tracks two independent health signals per endpoint. An endpoint is excluded while either signal marks it unhealthy.

- **Poll health.** The datalayer collector records every failed poll of a polling data source on the endpoint (see [Inputs consumed](#inputs-consumed)). An endpoint is marked unhealthy once the source configured by `pollSource` has failed at least `pollFailureThreshold` consecutive polls spanning at least `pollFailureDuration`. The first successful poll clears the mark.
- **Response health.** The plugin reads the status of every response an endpoint returns. After `responseFailureThreshold` consecutive responses with a status in `responseFailureCodes`, the endpoint is ejected for `ejectionDuration`. Each further ejection without a successful response in between lasts one `ejectionDuration` longer, up to `maxEjectionDuration`. Any other status resets the count and the backoff. Responses that arrive while the endpoint is ejected are ignored.

When every candidate is unhealthy, the plugin keeps all of them, so a pool is never left without endpoints.

The health state belongs to the endpoint's address. An endpoint recreated under the same name at another address, such as a restarted pod, starts healthy, and responses from its previous address are ignored.

### Choice of defaults

The defaults avoid excluding an endpoint that is slow but working:

- A single failed poll never marks an endpoint. Requiring both a count and a duration also keeps a burst of fast failures shorter than `pollFailureDuration`, or one failure of a source with a long interval, from marking it.
- vLLM serves `/metrics` from its API server process, so a long prefill in the engine does not fail polls. An endpoint is marked only after its polls fail without a single success for `pollFailureDuration`.
- `responseFailureCodes` defaults to 502 and 503, the statuses a proxy in front of the model server returns when it cannot reach it. A model server returns 500 for errors specific to one request, and requests of one conversation tend to reach the same endpoint, so counting 500 would let one client eject a working endpoint. A proxy returns 504 when a working but slow model server exceeds its timeout.
- Poll health clears on the first successful poll without a hysteresis count: marking already takes `pollFailureDuration` of uninterrupted failures, so an endpoint that answers some polls is never marked, and an endpoint that answers but fails requests is caught by response health.

Response health reacts to hard failures within a few requests. Poll health is the slower signal that also covers an endpoint that receives no traffic, and failures the EPP never sees as a response status.

### Evaluation

A response ejection starts when the response that completes the failure run arrives. Poll health and the end of an ejection are evaluated when the filter runs, so these transitions are logged and counted on the first scheduling cycle that observes them. Each transition logs one line at INFO level that names the endpoint. A scheduling cycle that keeps every candidate because all are unhealthy is counted in `llm_d_epp_endpoint_health_fail_open_total` and logged at DEBUG level. The metrics are listed in [docs/metrics.md](../../../../../../../docs/metrics.md#endpoint-health).

### Placement

Reference the filter first in every scheduling profile that must skip failing endpoints. The fail-open considers only the candidates the filter receives. After a filter that narrows the candidates to the endpoint holding a session or a cached prefix, a failing endpoint can be the only candidate left, and the filter then keeps it.

## Inputs consumed

When poll health is enabled, the plugin consumes `PollStatusDataKey(pollSource)` (`PollStatus`), which the datalayer collector writes for every polling data source. The EPP fails to start when `pollSource` names no configured polling data source.

Response health reads the `:status` response header in the response-received extension point, so the proxy must forward response headers to the EPP. A response is counted toward the endpoint the EPP picked first. When the proxy reports in the `x-gateway-destination-endpoint-served` metadata that another picked endpoint served the request, the response is ignored. A request that fails without a response, such as a refused connection that the proxy answers itself, is not seen by response health; poll health covers an endpoint that refuses connections.

## Configuration

| Parameter                  | Required | Description |
|----------------------------|----------|-------------|
| `pollSource`               | no       | Name of the polling data source whose polls decide poll health. Default `metrics-data-source`, the default metrics source. |
| `pollFailureThreshold`     | no       | Consecutive failed polls needed to mark an endpoint unhealthy. `0` disables poll health. Default `3`. |
| `pollFailureDuration`      | no       | How long polls must keep failing to mark an endpoint unhealthy. Must cover several poll intervals of `pollSource`. Default `10s`. |
| `responseFailureThreshold` | no       | Consecutive failed responses that eject an endpoint. `0` disables response health. Default `5`. |
| `responseFailureCodes`     | no       | Response statuses counted as failures. Each must be a 5xx status. Default `[502, 503]`. |
| `ejectionDuration`         | no       | Length of the first ejection. Default `30s`. |
| `maxEjectionDuration`      | no       | Upper bound on the length of an ejection. Must be at least `ejectionDuration`. Default `5m`. |

`pollFailureThreshold` and `responseFailureThreshold` cannot both be `0`.

**Configuration Example:**
```yaml
plugins:
  - type: endpoint-health-filter
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: endpoint-health-filter
```

## See also

The `utilization-filter` plugin and the saturation detector filter drop endpoints by load.
