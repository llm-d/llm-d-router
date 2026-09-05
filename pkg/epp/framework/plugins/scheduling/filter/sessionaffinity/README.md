# Session Affinity Filter

**Type:** `session-affinity-filter`

Pins subsequent requests in a session to the same pod the first request was sent to, as a hard constraint. When the session pod is among the candidates the filter returns it as the sole endpoint; when there is no session or the session pod is no longer a candidate, the filter returns all candidates unchanged so downstream filters and scorers decide. The filter never returns an empty set.

Supports two algorithms, selected by the `strategy` parameter:

- `encoded_endpoint_header` (default): stateless. The session is carried in a request header whose value is the base64-encoded `namespace/name` of the previously selected pod. As a [`ResponseHeaderProcessor`](../../../../interface/requestcontrol/plugins.go), the filter writes that same header on the response so the client can echo it back on the next request. When the token cannot be decoded the filter returns all candidates.
- `session_id`: stateful. The client supplies an opaque session identifier, read from one or more configured sources (a request header, or a request attribute published by an upstream plugin) tried in priority order. The filter maintains a server-side, TTL-evicted binding from that identifier to the pod that served it. Nothing is written back to the client. An unbound session expresses no preference: the filter passes all candidates through and lets downstream plugins place the request. A bound session whose pod is absent from the candidate set also passes through, then rebinds to the pod the request was routed to.

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
- type: session-affinity-filter
```

### Session ID Configuration

```yaml
- type: session-affinity-filter
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
- type: session-affinity-filter
  parameters:
    strategy: session_id
    sessionIdConfig:
      sources:
        - header: x-session-id
        - attribute: agent-identity
```

### PD Disaggregation Configuration

To support session affinity with PD disaggregation, configure two separate instances of the filter: one for decode and one for prefill.

```yaml
# Instance for the decode profile (pins decode requests)
- name: session-affinity-decode
  type: session-affinity-filter
  parameters:
    encodedEndpointHeaderConfig:
      header: x-session-token

# Instance for the prefill profile (pins prefill requests)
- name: session-affinity-prefill
  type: session-affinity-filter
  parameters:
    profileName: prefill
    encodedEndpointHeaderConfig:
      header: x-session-token-prefill
```

The decode instance uses the default behavior (writing the decode pod to `x-session-token`). The prefill instance uses `profileName: prefill` to look up the prefill pod from the scheduling results and write it to `x-session-token-prefill`. This ensures that subsequent requests in the same session target both the same prefill pod and the same decode pod. `session_id` supports the same pattern: configure one instance per profile, each with its own `profileName`.

## Relationship to the session affinity scorer

The [session affinity scorer](../../scorer/sessionaffinity/README.md) (`session-affinity-scorer`) provides the same affinity behavior as a soft preference.

Configuring both the filter and the scorer is unnecessary:

- Under `encoded_endpoint_header`, if they use the **same** `header` the configuration is redundant: both read and write the identical header, and the filter already restricts candidates to the session pod, so the scorer's contribution is moot. If they use **different** `header` values it is misleading: the response carries the same token under two different headers, so the client cannot tell which to echo back.
- Under `session_id`, the filter and scorer keep independent binding stores, so their routing decisions for a session can diverge; run only one.

Choose one: the filter for a hard pin, or the scorer for a soft preference that can be outweighed by other scorers.

## Multi-cluster support

`multicluster-session-affinity-filter` is the cluster-scoped variant. Cluster identity comes from discovery, so it delegates to this filter unchanged and pins a session to its cluster.
