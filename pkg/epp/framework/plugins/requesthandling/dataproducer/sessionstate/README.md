# Session State Producer

The `session-state-producer` tracks history for identities published by the
`agent-identity` request-header plugin. It publishes per-request session state
through the request attribute store and per-endpoint session snapshots through
each endpoint's attribute map.

The producer uses only the `agent-identity` attribute. `SessionIDDataKey` is a
separate attribute produced by `session-id-producer` from one configured header
or cookie for general session-affinity use.

## Configuration

Both plugins must be enabled, and Alpha plugins must be allowed by the EPP
process. `agent-identity` is a required data dependency, so configuration
loading fails if no plugin declares that it produces the agent identity
attribute.

```yaml
plugins:
- type: agent-identity
- type: session-state-producer
  parameters:
    evictionTtlSeconds: 3600
    evictionSweepSeconds: 300
```

| Parameter | Default | Description |
|---|---:|---|
| `evictionTtlSeconds` | `3600` | Maximum session idle time before its state is removed. Set to `0` to disable eviction. |
| `evictionSweepSeconds` | `300` | Interval between idle-state scans. Must be greater than `0`. |

## Produced data

The producer writes `SessionStateDataKey` from `RequestHeader`, before flow
control, so flow-control plugins such as fairness policies can consume it. The
attribute resolves on every read: a request waiting in a flow-control queue
observes turns dispatched and responses completed for its session in the
meantime. Read it with `sessionstate.ReadSessionState`. `Produce` is a no-op.

The producer also subscribes to `endpoint-notification-source` and attaches a
`DynamicAttribute` for `EndpointSessionsDataKey` (`sessionstate.EndpointSessions`)
on each endpoint. Calling `sessionstate.ReadEndpointSessions` returns a fresh
snapshot of all sessions currently bound to that endpoint.

The `sessionstate.SessionState` value contains:

- `SessionID`: session identifier published by `agent-identity`.
- `TurnsTaken`: requests of the session dispatched so far.
- `Duration`: elapsed time from the first observation of the session to the
  observation of the current request (or the snapshot read time for endpoint
  snapshots).
- `LastSeenAt`: time the preceding request was observed; for a new session it
  is the current request time.
- `LastResponseAt`: time the most recent response lifecycle of the session
  ended, for any termination cause.
- `InFlightRequests`: requests dispatched by EPP whose response lifecycle has
  not ended.
- `CompletedRequests`: requests whose responses ended naturally. A complete
  model error response is a natural completion.
- `TotalInputTokens`: prompt tokens reported by naturally completed responses.
- `TotalOutputTokens`: completion tokens reported by naturally completed
  responses.
- `TotalCachedTokens`: cached prompt tokens (`prompt_tokens_details.cached_tokens`)
  reported by naturally completed responses.
- `Endpoint`: `NamespacedName` string of the primary profile's picked endpoint
  from the latest dispatched turn, or `""` before dispatch or after that endpoint
  leaves the pool.
- `ContextTokens`: context token footprint of the latest finished turn.
- `InFlightTokens`: sum of prompt token estimates for turns currently in flight.

`SessionState.ContextSize()` returns `max(ContextTokens, InFlightTokens)`.
`EstimatePromptTokens(requestSizeBytes)` converts a raw request size in bytes
to a token estimate using `request.BytesPerToken`, with a minimum of `1`.

A turn is counted in `PreRequest` once per `RequestID` after scheduling selects
a target, binding or moving `Endpoint` when the primary profile picked one and
charging `EstimatePromptTokens(request.RequestSizeBytes)` into `InFlightTokens`.
When a response lifecycle ends (`EndOfStream`), `ResponseBody` removes the turn
from `InFlightRequests` and deducts its charged estimate from `InFlightTokens`
for every termination cause. `ContextTokens` updates to `Usage.TotalTokens` (or
`PromptTokens + CompletionTokens` on natural completion when `TotalTokens` is
zero), falling back to the turn's prompt estimate when it exceeds the stored
`ContextTokens`. Client disconnects, evictions, and internal errors do not
contribute to `CompletedRequests` or the cumulative token totals because their
usage may be incomplete.

When `endpoint-notification-source` delivers `EventDelete` for the active
endpoint object (or `EventAddOrUpdate` replaces it with a new endpoint object),
the producer clears `Endpoint` on affected sessions while keeping their history
and `ContextTokens`. State is local to one EPP replica.
A session is removed from both the session map and the endpoint index when its
latest request or response is older than the eviction TTL, at the next sweep.
Sessions are retained while a request is pending (observed at `RequestHeader`
but not yet dispatched, for example waiting in a flow-control queue) or in
flight. A pending request that is rejected before dispatch is released through
`RequestAbandoned`.
