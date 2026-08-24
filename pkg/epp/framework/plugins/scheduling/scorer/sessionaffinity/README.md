# Session Affinity Scorer

**Type:** `session-affinity-scorer`

Scores candidate pods by giving a higher score to the pod that was previously used for the same session, and zero to the rest. Enables sticky routing for stateful workloads where reusing the same pod reduces latency or preserves context.

Supports two algorithms, selected by the `strategy` parameter:

- `encoded_endpoint_header` (default): stateless. The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the scorer writes that same header on the response so the client can echo it back on the next request.
- `session_id`: stateful. The client supplies an opaque session identifier, read from one or more configured sources (a request header, or a request attribute published by an upstream plugin) tried in priority order. The scorer maintains a server-side, TTL-evicted binding from that identifier to the pod that served it. Nothing is written back to the client. An unbound session expresses no preference: the scorer gives every candidate zero and lets the picker place the request. A bound session whose pod is absent from the candidate set also scores zero everywhere, then rebinds to the pod the request was routed to.

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `strategy` | string | `encoded_endpoint_header` | `encoded_endpoint_header` or `session_id`. Only the config block matching the strategy is used. |
| `profileName` | string | | The name of the profile this instance is associated with. When set (e.g. `prefill`), the plugin looks up the target pod from the results of that profile in `SchedulingResult`. When empty, it defaults to the primary (decode) pod. |
| `encodedEndpointHeaderConfig` | object | | Config for `encoded_endpoint_header`; see below. |
| `sessionIdConfig` | object | | Config for `session_id`; see below. |

`encodedEndpointHeaderConfig`:

| Name | Type | Default | Description |
|---|---|---|---|
| `header` | string | `x-session-token` | Request and response header carrying the base64-encoded pod token. |

`sessionIdConfig`:

| Name | Type | Default | Description |
|---|---|---|---|
| `sources` | list | one `header` source `x-session-id` | Where the session identifier is read from, in priority order; the first non-empty match wins. Each source sets exactly one of `header` (a request header name) or `attribute` (a request-attribute key published by an upstream plugin). |
| `evictionTtlSeconds` | float | `300` | How long a session binding survives unused. |
| `evictionSweepSeconds` | float | `10` | How often expired bindings are swept. |

### Default Configuration (without PD disaggregation)

```yaml
- type: session-affinity-scorer
```

### Session ID Configuration

```yaml
- type: session-affinity-scorer
  parameters:
    strategy: session_id
    sessionIdConfig:
      sources:
        - header: x-session-id
      evictionTtlSeconds: 300
      evictionSweepSeconds: 10
```

To pin agent traffic, list the agent-identity attribute as a fallback source after the header:

```yaml
- type: session-affinity-scorer
  parameters:
    strategy: session_id
    sessionIdConfig:
      sources:
        - header: x-session-id
        - attribute: agent-identity
```

### PD Disaggregation Configuration

To support session affinity with PD disaggregation, configure two separate instances of the scorer: one for decode and one for prefill.

```yaml
# Instance for the decode profile (pins decode requests)
- name: session-affinity-decode
  type: session-affinity-scorer
  parameters:
    encodedEndpointHeaderConfig:
      header: x-session-token

# Instance for the prefill profile (pins prefill requests)
- name: session-affinity-prefill
  type: session-affinity-scorer
  parameters:
    profileName: prefill
    encodedEndpointHeaderConfig:
      header: x-session-token-prefill
```

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod. `session_id` supports the same pattern: configure one instance per profile, each with its own `profileName`.

## Relationship to the session affinity filter

The [session affinity filter](../../filter/sessionaffinity/README.md) (`session-affinity-filter`) provides the same affinity behavior as a hard constraint. Configuring both alongside the scorer is unnecessary and can be misleading; see [Relationship to the session affinity scorer](../../filter/sessionaffinity/README.md#relationship-to-the-session-affinity-scorer) for details. Use the scorer for a soft preference that can be outweighed by other scorers, or the filter for a hard pin.
