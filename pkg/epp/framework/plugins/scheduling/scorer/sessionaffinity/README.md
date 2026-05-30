# Session Affinity Scorer

**Type:** `session-affinity-scorer`
**Interfaces:** `scheduling.Scorer`, `requestcontrol.ResponseHeaderProcessor`
**Category:** Soft affinity

Scores candidate pods by giving a high score to the pod previously used for the same session, and sets an HTTP cookie on the response so the client carries the session binding on subsequent requests.

## What it does

For each request the scorer reads the `SessionID` attribute published by a `session-id-producer`:

- **Session present, endpoint in candidates** — scores that endpoint `1.0`, all others `0.0`.
- **Session present, endpoint not in candidates** — scores all endpoints `0.0`.
- **Session absent** — scores all endpoints `0.0`.

On each response the scorer writes an `HttpOnly; SameSite=Lax` `Set-Cookie` header encoding the selected endpoint's `<namespace>/<name>` as base64. The cookie is omitted when the client already carries the correct value.

## Inputs consumed

- `SessionID` attribute — published by `session-id-producer` onto the `InferenceRequest` attribute store.

## Configuration

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `maxAge` | `int` | No | `0` | Cookie `Max-Age` in seconds. `0` means a session cookie (expires when the browser closes). Must be `>= 0`. |
| `sessionIDProducerName` | `string` | No | `""` (uses default producer) | Name of the `session-id-producer` instance to consume. Must match the `name` field of a configured `session-id-producer` plugin. |

### Example

```yaml
plugins:
  - type: session-id-producer
    name: session-producer
    parameters:
      cookieName: llm-d-session

  - type: session-affinity-scorer
    name: session-scorer
    parameters:
      sessionIDProducerName: session-producer
      maxAge: 3600

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-scorer
        weight: 1
```

## Difference from session-affinity-filter

Both plugins provide session stickiness but with different guarantees:

| | `session-affinity-scorer` | `session-affinity-filter` |
|---|---|---|
| **Guarantee** | Soft — session endpoint gets a high score but others remain | Hard — only the session endpoint is returned |
| **On pod unavailability** | Other scorers determine the winner | Falls back to full candidate set |
| **Cookie management** | Sets `llm-d-session` cookie on the response | None — read-only |

Use the scorer alone when a best-effort preference is sufficient. Use the filter when strict stickiness is required for correctness, or combine both for hard routing with automatic cookie issuance.

## Related Documentation

- [Session Affinity Filter](../../filter/sessionaffinity/README.md)
- [Session ID Producer](../../../requestcontrol/dataproducer/sessionid/README.md)
- [Session Attributes](../../../datalayer/attribute/session/README.md)
