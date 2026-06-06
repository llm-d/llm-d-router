# Session Attributes

Per-request session identity used by affinity-aware scorers and filters.

## `SessionID`

Holds the session identifier extracted from a request. Stored on the
`InferenceRequest` attribute store (one entry per request, not per endpoint).

- **Key**: `SessionIDDataKey` (default producer: `session-id-producer`)
- **Type**: `SessionID` (string alias)
- **Reader helper**: `session.ReadSessionID(request)` returns the value and a
  presence boolean. Consumers should prefer this over reading the attribute
  directly so the storage choice stays encapsulated.

## `BoundEndpoint`

Identifies the network destination a session is pinned to. Stored in
canonical `host:port` form (the result of `net.JoinHostPort` over the
endpoint's `Address` and `Port`). The same `session-id-producer` that
publishes `SessionID` also maintains the binding cache and writes
`BoundEndpoint` on subsequent requests for a known session.

Tying the binding to `host:port` rather than the K8s namespaced name means
a restarted pod with a fresh IP no longer matches a stale binding, which is
the desired behavior since the KV cache the binding existed to preserve
died with the previous process.

- **Key**: `BoundEndpointDataKey` (default producer: `session-id-producer`)
- **Type**: `BoundEndpoint` (string alias)

## Producers

- **`session-id-producer`** (Request Control): extracts the session
  identifier from a configured request header or named cookie, tracks
  selected endpoints in a TTL-expiring LRU, and publishes both attributes.
