# Session Affinity Filter

**Type:** `session-affinity-filter`
**Interfaces:** `scheduling.Filter`
**Category:** Hard affinity

Pins each session to the endpoint that served its previous request. When the session is bound to one of the current candidates, every other candidate is removed; otherwise all candidates pass through.

## What it does

For each request the filter reads the `BoundEndpoint` attribute published by a `session-id-producer`:

- **Binding present, endpoint in candidates**: returns only that endpoint.
- **Binding present, endpoint not in candidates** (e.g. the endpoint was scaled down): returns all candidates unchanged so other plugins can select a replacement.
- **Binding absent** (no session, or session not yet bound): returns all candidates unchanged.
- **Single candidate in list**: always returns it unchanged, skipping the lookup.

## How it works

The `session-id-producer` keeps an in-memory, time-expiring map of session identifier to endpoint. Pre-schedule it publishes the bound endpoint on the request as `BoundEndpoint`. Post-schedule it records the endpoint chosen by the primary profile so the next request in the same session can be steered back. The filter only reads.

## Inputs consumed

- `BoundEndpoint` attribute, published by `session-id-producer`.

## Configuration

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sessionIDProducerName` | `string` | No | `""` (uses default producer) | Name of the `session-id-producer` instance whose `BoundEndpoint` this filter consumes. |

### Examples

**Custom session header.** A `session-id-producer` is configured explicitly with a non-default header; the filter consumes from the default producer.

```yaml
plugins:
  - type: session-id-producer
    parameters:
      headerName: x-my-session-id

  - type: session-affinity-filter

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-affinity-filter
```

**Defaults only.** No `session-id-producer` is declared. The framework auto-instantiates one as the default producer for `BoundEndpointDataKey`, with `headerName` defaulting to `x-session-id`. Clients must send their session identifier in that header for affinity to take effect.

```yaml
plugins:
  - type: session-affinity-filter

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: session-affinity-filter
```

## Difference from session-affinity-scorer

Both plugins consume the same `BoundEndpoint` attribute but differ in how strictly they enforce stickiness:

| | `session-affinity-filter` | `session-affinity-scorer` |
|---|---|---|
| **Guarantee** | Hard, only the bound endpoint is returned when present | Soft, bound endpoint gets the highest score, others remain |
| **On endpoint unavailability** | Falls back to the full candidate set | Other scorers determine the winner |

Use the filter when strict stickiness is required for correctness. Use the scorer when a best-effort preference is sufficient and other scorers should still get a vote. Configuring both at once is redundant: the filter dominates the scorer in every case, so the extra scoring work changes no outcomes.

Pairing the filter and scorer with *different* `session-id-producer` instances (different `sessionIDProducerName` values) makes things worse: each producer maintains its own private binding cache, so the filter and scorer can pin different endpoints for the same session, and bindings get written twice per request. If both are configured, point them at the same producer.

## Related Documentation

- [Session Affinity Scorer](../../scorer/sessionaffinity/README.md)
- [Session ID Producer](../../../requestcontrol/dataproducer/sessionid/README.md)
- [Session Attributes](../../../datalayer/attribute/session/README.md)
