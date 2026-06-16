# Session Affinity Scorer

**Type:** `session-affinity-scorer`

Scores candidate pods by giving a higher score to the pod that was previously used for the same session, and zero to the rest. Enables sticky routing for stateful workloads where reusing the same pod reduces latency or preserves context.

The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the scorer writes that same header on the response so the client can echo it back on the next request.

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `x-session-token` | Request and response header carrying the session token. When set, only this header is read; the default is ignored. |
| `profileName` | string | | The name of the profile this instance is associated with. When set (e.g. `prefill`), the plugin looks up the target pod from the results of that profile in `SchedulingResult` during the response received phase. When empty, it defaults to the primary (decode) pod. |

### Default Configuration (without PD disaggregation)

```yaml
- type: session-affinity-scorer
  parameters:
    headerName: x-session-token
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

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod.

## Relationship to the session affinity filter

The [session affinity filter](../../filter/sessionaffinity/README.md) (`session-affinity-filter`) provides the same affinity behavior as a hard constraint and writes the same response header. Configuring both alongside the scorer is unnecessary and can be misleading; see [Relationship to the session affinity scorer](../../filter/sessionaffinity/README.md#relationship-to-the-session-affinity-scorer) for details. Use the scorer for a soft preference that can be outweighed by other scorers, or the filter for a hard pin.
