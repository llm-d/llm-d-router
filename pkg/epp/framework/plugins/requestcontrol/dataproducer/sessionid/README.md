# Session ID Producer Plugin

**Type:** `session-id-producer`

Extracts a session identifier from each inference request and tracks which endpoint each session was last routed to. The producer publishes two attributes on the `InferenceRequest` attribute store:

- `SessionID`, when the configured source carries a non-empty value.
- `BoundEndpoint`, when the session is currently bound to an endpoint by a previous request.

The post-schedule `PreRequest` hook writes the endpoint chosen by the primary profile into the binding cache, and the pre-schedule `Produce` step reads it back onto the request. Both operations refresh the binding's TTL, so an active session keeps its binding alive. Bindings live in an in-memory, size-bounded, time-expiring cache; nothing is written to the response. Affinity-aware scorers and filters consume the attributes via the framework's data dependency mechanism without needing to know how the session was carried on the wire.

The producer is a no-op when the configured source is absent or empty; consumers must treat the missing attributes as "no session preference".

## Parameters

### Source selection

When the producer is configured explicitly, exactly one of `headerName` or `cookieName` must be set. When it is auto-instantiated as the default for `SessionIDDataKey` or `BoundEndpointDataKey` (no `parameters` block), `headerName` defaults to `x-session-id`; configure the producer explicitly to use a different header or a cookie.

| Name | Type | Description |
|------|------|-------------|
| `headerName` | `string` | Name of the request header whose value is the session identifier. Comparison is case-insensitive (header names in the request are lowercased). |
| `cookieName` | `string` | Name of the cookie within the standard `Cookie` request header whose value is the session identifier. |

### Binding store (optional)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lruSize` | `int` | `1024` | Maximum number of concurrent session bindings retained. Must be `> 0` when set. |
| `ttl` | `string` (Go duration) | `30m` | Lifetime of a binding without activity ("30m", "1h"). Each `PreRequest` write and each `Produce` read refreshes the entry. Must be `> 0` when set. |

## Examples

```yaml
plugins:
  - type: session-id-producer
    parameters:
      headerName: x-my-session-id
```

```yaml
plugins:
  - type: session-id-producer
    parameters:
      cookieName: llm-d-session
      ttl: 1h
      lruSize: 4096
```

## Multiple instances

Each `session-id-producer` instance owns its own private binding cache. Configuring two instances and pointing the session-affinity filter and scorer at different ones causes them to pin different endpoints for the same session and to record bindings twice per request. If you configure both consumers, point them at the same producer instance. See [Session Affinity Filter](../../../scheduling/filter/sessionaffinity/README.md#difference-from-session-affinity-scorer) for the consumer-side framing.

## Related Documentation

- [Session Affinity Filter](../../../scheduling/filter/sessionaffinity/README.md)
- [Session Affinity Scorer](../../../scheduling/scorer/sessionaffinity/README.md)
- [Session Attributes](../../../datalayer/attribute/session/README.md)
