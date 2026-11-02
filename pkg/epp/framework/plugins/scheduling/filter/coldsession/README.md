# Cold Session Filter

**Type:** `cold-session-filter`
**Interfaces:** `scheduling.Filter`, `requestcontrol.PreRequest`, `requestcontrol.ResponseBodyProcessor`

The cold-session filter keeps a new (cold) session from piling onto a prefill
endpoint that is already busy with another cold, full-context prefill. It uses
the `SessionState` produced by `session-state-producer`:

- a request is **cold** when the session has `TurnsTaken == 0` and no other
  request from that session is in flight;
- a request is **warm** when the session has completed at least one turn, or
  already has a request in flight.

Filtering only applies to cold requests. A cold request is routed to candidates
whose in-flight cold-prefill count is below `maxConcurrentColdPrefills`. If every
candidate is at the limit, the filter fails open and returns all candidates, so
a cold session is never starved by this filter alone.

## Why use it

Agentic sessions often start with a 50K–100K-token prompt that occupies a
prefill endpoint for several seconds. Warm follow-up turns have a much smaller
fresh prefill and can be delayed behind those cold turns. This filter avoids
adding another cold prefill to an endpoint that is already paying that cost.

It is not admission control: it does not queue or reject cold requests when the
whole pool is busy. That remains flow-control policy work. The filter is the
routing-side mitigation that composes with `turn-priority` or any other
fairness policy.

## Configuration

```yaml
plugins:
  - type: session-state-producer
  - type: agent-identity
  - type: cold-session-filter
    parameters:
      profileName: prefill
      maxConcurrentColdPrefills: 1
      requestTtlSeconds: 600

schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: cold-session-filter
```

The filter requires the `session-state-producer` and its `agent-identity`
dependency. Without `SessionState` on the request, the filter is a no-op.

| Field | Default | Description |
|---|---|---|
| `profileName` | `prefill` | Scheduling profile whose selected endpoint is tracked. Falls back to the primary profile when absent. |
| `maxConcurrentColdPrefills` | `1` | Cold prefills allowed on one endpoint before new cold sessions are routed elsewhere. Must be at least 1. |
| `requestTtlSeconds` | `600` | Removes a tracked request if its response never reaches end-of-stream. |

## Trade-offs

- **No global cap.** When every endpoint is at the limit, cold requests still
  proceed. Combine with flow control for pool-wide admission control.
- **Per-EPP state.** The counters are local to one EPP process, the same
  limitation as the in-flight load producer.
- **Requires session identity.** Requests without a session header have no
  `SessionState` and are not filtered.
