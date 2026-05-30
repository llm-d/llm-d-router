# Session Affinity Filter

**Type:** `session-affinity-filter`
**Interfaces:** `scheduling.Filter`
**Category:** Hard affinity

Enforces strict session stickiness by removing all candidates except the one that owns the current session. When no session is present, all candidates pass through unchanged.

## What it does

For each request the filter reads the `SessionID` attribute published by a `session-id-producer`:

- **Session present, endpoint in candidates** — returns only that endpoint; all others are removed.
- **Session present, endpoint not in candidates** (e.g. pod was scaled down) — returns all candidates unchanged so other plugins can select a replacement.
- **Session absent** — returns all candidates unchanged (no-op).
- **Single candidate in list** — always returns it unchanged, skipping the lookup.

## How It Works

The session ID is a base64-encoded `<namespace>/<name>` string. The filter decodes it, looks for a matching candidate by `NamespacedName`, and either narrows the list to that one endpoint or falls back to the full set.

## Inputs consumed

- `SessionID` attribute — published by `session-id-producer` onto the `InferenceRequest` attribute store.

## Configuration

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sessionIDProducerName` | `string` | No | `""` (uses default producer) | Name of the `session-id-producer` instance to consume. Must match the `name` field of a configured `session-id-producer` plugin. |

### Example

```yaml
plugins:
  - type: session-id-producer
    name: session-producer
    parameters:
      cookieName: llm-d-session

  - type: session-affinity-filter
    name: session-filter
    parameters:
      sessionIDProducerName: session-producer

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-filter
```

## Difference from session-affinity-scorer

Both plugins provide session stickiness but with different guarantees:

| | `session-affinity-filter` | `session-affinity-scorer` |
|---|---|---|
| **Guarantee** | Hard — only the session endpoint is returned | Soft — session endpoint gets a high score but others remain |
| **On pod unavailability** | Falls back to full candidate set | Other scorers determine the winner |
| **Cookie management** | None — read-only | Sets `llm-d-session` cookie on the response |

Use the filter when strict stickiness is required for correctness. Use the scorer alone when a best-effort preference is sufficient, or combine both for hard routing with automatic cookie issuance.

## Related Documentation

- [Session Affinity Scorer](../../scorer/sessionaffinity/README.md)
- [Session ID Producer](../../../requestcontrol/dataproducer/sessionid/README.md)
- [Session Attributes](../../../datalayer/attribute/session/README.md)
