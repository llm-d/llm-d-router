# Plugin Debug State

The router exposes a plugin state debug endpoint on the metrics/admin server:

```text
/debug/plugins/state
```

The endpoint is intended for on-demand operational debugging. It returns a snapshot of configured plugins and any sanitized internal state exposed through the optional `plugin.StateDumper` interface. Plugins that do not implement `StateDumper` are still listed with a message indicating that state collection is unsupported.

The endpoint is registered on the same metrics/admin server as the metrics endpoint and pprof handlers. Operators should use the existing metrics/admin server exposure and authentication controls to restrict access. When metrics endpoint authentication is enabled, the metrics/admin server authentication and authorization filters apply to this endpoint.

This endpoint is available in the EPP controller-manager server path. The standalone file-discovery mode uses a separate metrics mux and does not expose plugin debug state.

## Response format

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "plugins": {
    "inflight-load-producer": {
      "type": "inflight-load-producer",
      "state": {
        "endpoints": [],
        "totalEndpoints": 0,
        "maxEndpoints": 100,
        "truncated": false
      }
    },
    "max-score-picker": {
      "type": "max-score-picker",
      "message": "plugin does not support state collection"
    }
  }
}
```

Each `plugins` key is the configured plugin name. Entries include the plugin type and either:

- `state`: plugin-defined JSON state for plugins that implement `StateDumper`.
- `message`: an explanation for plugins that do not expose debug state.

## Implementing StateDumper

`StateDumper` implementations return `json.RawMessage` so each plugin owns serialization:

```go
type StateDumper interface {
    DumpState() (json.RawMessage, error)
}
```

Use state dumps for point-in-time debugging information that is hard to understand from metrics alone. Prefer metrics for numeric time series, alerting, dashboards, and aggregation over time.

State dumps must not include request payloads, credentials, tokens, or other sensitive values. Dumps should stay bounded; summarize, cap, or omit large and high-cardinality state.

The `inflight-load-producer` implementation reports the busiest endpoints up to a fixed cap, along with `totalEndpoints`, `maxEndpoints`, and `truncated` fields so operators can tell when the dump is partial.
