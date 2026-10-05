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

## Producers

- **`session-id-producer`** (Request Control): extracts the session
  identifier from a configured request header or named cookie.

## `SessionCacheRequest`

Holds a session producer's cache lookup for a request: the identity the engine
reports the request's blocks under and the candidate engine-block prefixes to
resolve. Stored on the `InferenceRequest` attribute store.

- **Key**: `SessionCacheRequestDataKey` (no default producer; the
  `session-prefix-cache-producer` names the producer it consumes)
- **Type**: `SessionCacheRequest`
- **Consumer**: [`session-prefix-cache-producer`](../../../requestcontrol/dataproducer/sessionprefixcache/README.md)
