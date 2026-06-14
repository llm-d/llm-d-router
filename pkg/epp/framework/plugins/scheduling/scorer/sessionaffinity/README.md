# Session Affinity Scorer

**Type:** `session-affinity-scorer`

Scores candidate pods by giving a higher score to the pod that was previously used for the same session, and zero to the rest. Enables sticky routing for stateful workloads where reusing the same pod reduces latency or preserves context.

The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the scorer writes that same header on the response so the client can echo it back on the next request.

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `x-session-token` | Request and response header carrying the session token. When set, only this header is read; the default is ignored. |

```yaml
- type: session-affinity-scorer
  parameters:
    headerName: x-session-token
```

## Relationship to the session affinity filter

The [session affinity filter](../../filter/sessionaffinity/README.md) (`session-affinity-filter`) provides the same affinity behavior as a hard constraint and writes the same response header. Configuring both alongside the scorer is unnecessary and can be misleading; see [Relationship to the session affinity scorer](../../filter/sessionaffinity/README.md#relationship-to-the-session-affinity-scorer) for details. Use the scorer for a soft preference that can be outweighed by other scorers, or the filter for a hard pin.
