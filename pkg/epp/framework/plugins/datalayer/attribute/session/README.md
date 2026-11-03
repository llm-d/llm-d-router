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

`SessionCacheRequest` carries an external session producer's request stamp and
ordered engine-block paths. Its contract and configuration are documented in the
[session prefix cache producer](../../../requesthandling/dataproducer/sessionprefixcache/README.md).
