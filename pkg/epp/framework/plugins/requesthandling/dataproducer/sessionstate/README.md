# Session State Producer

The `session-state-producer` tracks history for identities published by the
`agent-identity` request-header plugin and publishes per-request session state
for flow-control and scheduling plugins through the request attribute store.

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

The `sessionstate.SessionState` value contains:

- `TurnsTaken`: requests of the session dispatched so far.
- `Duration`: elapsed time from the first observation of the session to the
  observation of the current request.
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

A turn is counted in `PreRequest` only after scheduling selects at least one
target. Multiple scheduling profiles still count as one turn. The request is
also added to `InFlightRequests` at that point and removed when its response
lifecycle ends. Client disconnects, evictions, and internal errors do not
contribute to `CompletedRequests` or the token totals because their usage may
be incomplete. A naturally completed response without usage data contributes
zero tokens.

State is local to one EPP replica. A session is removed when its latest
request or response is older than the eviction TTL, at the next sweep.
Sessions are retained while a request is pending (observed at `RequestHeader`
but not yet dispatched, for example waiting in a flow-control queue) or in
flight. A pending request that is rejected before dispatch is released through
`RequestAbandoned`.
