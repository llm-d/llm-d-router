# Session Manager

The Alpha `session-manager` plugin derives a scoped identity from the
`agent-identity` request attribute and mints a unique per-request stamp for the
session prefix-cache integration. Version 1 does not subscribe to KV events,
track residency, infer continuations or forks, or publish cache prefixes.

## Configuration

```yaml
- type: agent-identity
- type: session-manager
  parameters:
    deploymentID: payments-prod-eu1
    hmacKeyFile: /var/run/secrets/llm-d/session-manager-key
```

`hmacKeyFile` must contain the unpadded base64url encoding of exactly 32 random
bytes. Mount it from an operator-managed Kubernetes Secret. The manager reads
it once at startup and never exposes it. The client-provided identity is a
grouping signal, not authentication or tenant isolation.

## Hook order and outputs

The plugin consumes `agent-identity`, so the request-plugin dependency graph
orders its `RequestHeader` hook after `agent-identity`. The hook runs before
flow-control admission and fails open: a missing identity publishes nothing
and the hook always returns nil.

For a non-empty `agent-identity`, the hook publishes:

- a name-bound `SessionIdentity` containing an opaque HMAC-derived
  `SessionTag`, `AgentIdentityAttribute` source, and deployment/key/version
  scope;
- the same `SessionTag` as a plain string request attribute named
  `session-tag`.

The string form lets generic consumers use configuration such as:

```yaml
sessionIdConfig:
  sources:
  - attribute: session-tag
    producer: session-manager
```

After admission, the DataProducer hook publishes:

```text
SessionCacheRequest
  SessionID: unique process-nonce/counter stamp
  FullReport: false
  TotalTokens: 0
  Prefixes: []
```

`TotalTokens` remains zero because token producers run after flow control. The
`session-prefix-cache-producer` from #2716 consumes the stamp and owns request
body mutation, KV-event subscription, and residency indexing.

The `SessionCacheRequest` data contract in
`plugins/datalayer/attribute/session/data_types.go` mirrors #2716 verbatim so
either change can merge first without conflicting.
