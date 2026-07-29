# Session Affinity Scorer

**Type:** `session-affinity-scorer`

Scores candidate pods by giving a higher score to the pod that was previously used for the same session, and zero to the rest. Enables sticky routing for stateful workloads where reusing the same pod reduces latency or preserves context.

Supports two algorithms, selected by the `algorithm` parameter:

- `encoded_endpoint_header` (default): stateless. The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the scorer writes that same header on the response so the client can echo it back on the next request.
- `session_id_header`: stateful. The client supplies an opaque session identifier (in the same header, or via an agent-identity request attribute if the header is absent), and the scorer maintains a server-side, TTL-evicted binding from that identifier to the pod that served it. Nothing is written back to the client.

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `algorithm` | string | `encoded_endpoint_header` | `encoded_endpoint_header` or `session_id_header`. |
| `headerName` | string | `x-session-token` | Request and response header carrying the session token. When set, only this header is read; the default is ignored. |
| `profileName` | string | | The name of the profile this instance is associated with. When set (e.g. `prefill`), the plugin looks up the target pod from the results of that profile in `SchedulingResult`. When empty, it defaults to the primary (decode) pod. |
| `evictionTtlSeconds` | float | `600` | How long a session binding survives unused. `session_id_header` only. |
| `evictionSweepSeconds` | float | `30` | How often expired bindings are swept. `session_id_header` only. |

### Default Configuration (without PD disaggregation)

```yaml
- type: session-affinity-scorer
  parameters:
    headerName: x-session-token
```

### Session ID Header Configuration

```yaml
- type: session-affinity-scorer
  parameters:
    algorithm: session_id_header
    headerName: x-session-token
    evictionTtlSeconds: 600
    evictionSweepSeconds: 30
```

### PD Disaggregation Configuration

To support session affinity with PD disaggregation, configure two separate instances of the scorer: one for decode and one for prefill.

```yaml
# Instance for the decode profile (pins decode requests)
- name: session-affinity-decode
  type: session-affinity-scorer
  parameters:
    headerName: x-session-token

# Instance for the prefill profile (pins prefill requests)
- name: session-affinity-prefill
  type: session-affinity-scorer
  parameters:
    headerName: x-session-token-prefill
    profileName: prefill
```

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod. `session_id_header` supports the same pattern: configure one instance per profile, each with its own `profileName`.

## Relationship to the session affinity filter

The [session affinity filter](../../filter/sessionaffinity/README.md) (`session-affinity-filter`) provides the same affinity behavior as a hard constraint and writes the same response header. Configuring both alongside the scorer is unnecessary and can be misleading; see [Relationship to the session affinity scorer](../../filter/sessionaffinity/README.md#relationship-to-the-session-affinity-scorer) for details. Use the scorer for a soft preference that can be outweighed by other scorers, or the filter for a hard pin.
