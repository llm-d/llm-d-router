# Session State Producer

The `session-state-producer` tracks history for identities published by the
`agent-identity` request-header plugin and publishes per-request session state
for scheduling plugins through the request attribute store.

The producer uses only the `agent-identity` attribute. `SessionIDDataKey` is a
separate attribute produced by `session-id-producer` from one configured header
or cookie for general session-affinity use.

## Configuration

Both plugins must be enabled, and Alpha plugins must be allowed by the EPP
process.

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

The producer writes `SessionStateDataKey` with a `session.SessionState` value:

- `TurnsTaken`: requests dispatched before the current request.
- `Duration`: elapsed time since the session was first observed.
- `LastSeenAt`: time the preceding request was observed; for a new session it
  is the current request time.

The current request is marked as seen during data production. Its turn is
counted in `PreRequest` only after scheduling selects at least one target, so
the next request observes the increment. Multiple scheduling profiles still
count as one turn.

State is local to one EPP replica. Idle session state is removed according to
the configured eviction TTL and sweep interval.
