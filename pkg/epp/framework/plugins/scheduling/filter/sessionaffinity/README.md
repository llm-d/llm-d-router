# Session Affinity Filter

**Type:** `session-affinity-filter`

Pins subsequent requests in a session to the same pod the first request was sent to, as a hard constraint. When the session pod is among the candidates the filter returns it as the sole endpoint; when there is no session, the token cannot be decoded, or the session pod is no longer a candidate, the filter returns all candidates unchanged so downstream filters and scorers decide.

The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the filter writes that same header on the response so the client can echo it back on the next request.

## Parameters
 
| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `x-session-token` | Request and response header carrying the session token. When set, only this header is read; the default is ignored. |
| `profileName` | string | | The name of the profile this instance is associated with. When set (e.g. `prefill`), the plugin looks up the target pod from the results of that profile in `SchedulingResult` during the response received phase. When empty, it defaults to the primary (decode) pod. |

### Default Configuration (without PD disaggregation)

```yaml
- type: session-affinity-filter
  parameters:
    headerName: x-session-token
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

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod.

## Relationship to the session affinity scorer

The [session affinity scorer](../../scorer/sessionaffinity/README.md) (`session-affinity-scorer`) provides the same affinity behavior as a soft preference and writes the same response header.

Configuring both the filter and the scorer is unnecessary:

- If they use the **same** `headerName`, the configuration is redundant: both read and write the identical header, and the filter already restricts candidates to the session pod, so the scorer's contribution is moot.
- If they use **different** `headerName` values, the configuration is misleading: the response carries the same token under two different headers (both encode the chosen pod), so the client cannot tell which header to echo back.

Choose one: the filter for a hard pin, or the scorer for a soft preference that can be outweighed by other scorers.
