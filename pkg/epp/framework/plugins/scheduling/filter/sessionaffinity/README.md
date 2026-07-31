# Session Affinity Filter

**Type:** `session-affinity-filter`

Pins subsequent requests in a session to the same pod the first request was sent to, as a hard constraint. When the session pod is among the candidates the filter returns it as the sole endpoint; when there is no session or the session pod is no longer a candidate, the filter returns all candidates unchanged so downstream filters and scorers decide. The filter never returns an empty set.

Supports two algorithms, selected by the `strategy` parameter:

- `encoded_endpoint_header` (default): stateless. The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the filter writes that same header on the response so the client can echo it back on the next request. When the token cannot be decoded the filter returns all candidates.
- `session_id_header`: stateful. The client supplies an opaque session identifier (in the same header, or via an agent-identity request attribute if the header is absent), and the filter maintains a server-side, TTL-evicted binding from that identifier to the pod that served it. Nothing is written back to the client. An unbound session is pinned to the pod currently bound by the fewest sessions. A bound session whose pod is absent from the candidate set migrates to the present pod bound by the fewest sessions immediately.

## Parameters
 
| Name | Type | Default | Description |
|---|---|---|---|
| `strategy` | string | `encoded_endpoint_header` | `encoded_endpoint_header` or `session_id_header`. |
| `headerName` | string | `x-session-token` | Request and response header carrying the session token. When set, only this header is read; the default is ignored. |
| `profileName` | string | | The name of the profile this instance is associated with. When set (e.g. `prefill`), the plugin looks up the target pod from the results of that profile in `SchedulingResult`. When empty, it defaults to the primary (decode) pod. |
| `evictionTtlSeconds` | float | `300` | How long a session binding survives unused. `session_id_header` only. |
| `evictionSweepSeconds` | float | `10` | How often expired bindings are swept. `session_id_header` only. |

### Default Configuration (without PD disaggregation)

```yaml
- type: session-affinity-filter
  parameters:
    headerName: x-session-token
```

### Session ID Header Configuration

```yaml
- type: session-affinity-filter
  parameters:
    strategy: session_id_header
    headerName: x-session-token
    evictionTtlSeconds: 300
    evictionSweepSeconds: 10
```

### PD Disaggregation Configuration

To support session affinity with PD disaggregation, configure two separate instances of the filter: one for decode and one for prefill.

```yaml
# Instance for the decode profile (pins decode requests)
- name: session-affinity-decode
  type: session-affinity-filter
  parameters:
    headerName: x-session-token

# Instance for the prefill profile (pins prefill requests)
- name: session-affinity-prefill
  type: session-affinity-filter
  parameters:
    headerName: x-session-token-prefill
    profileName: prefill
```

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod. `session_id_header` supports the same pattern: configure one instance per profile, each with its own `profileName`.

## Relationship to the session affinity scorer

The [session affinity scorer](../../scorer/sessionaffinity/README.md) (`session-affinity-scorer`) provides the same affinity behavior as a soft preference.

Configuring both the filter and the scorer is unnecessary:

- Under `encoded_endpoint_header`, if they use the **same** `headerName` the configuration is redundant: both read and write the identical header, and the filter already restricts candidates to the session pod, so the scorer's contribution is moot. If they use **different** `headerName` values it is misleading: the response carries the same token under two different headers, so the client cannot tell which to echo back.
- Under `session_id_header`, the filter and scorer keep independent binding stores, so their routing decisions for a session can diverge; run only one.

Choose one: the filter for a hard pin, or the scorer for a soft preference that can be outweighed by other scorers.
