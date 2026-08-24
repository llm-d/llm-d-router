# Session Attributes

Per-request session identity and history used by session-aware scheduling
plugins.

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
  identifier from a configured request header or named cookie for general
  session-affinity use.

## `SessionState`

Holds the session history observed before the current request is dispatched.

- **Key**: `SessionStateDataKey` (default producer: `session-state-producer`)
- **Type**: `SessionState`
- **Fields**: `TurnsTaken`, `Duration`, and `LastSeenAt`
- **Reader helper**: `session.ReadSessionState(request)` returns the value and
  a presence boolean.

## State producers

- **`session-state-producer`** (Request Control): tracks history by the
  identity published by `agent-identity` and writes per-request session state.
  It does not consume `SessionIDDataKey`.
