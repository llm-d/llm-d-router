# Session Manager

The Alpha `session-manager` DataProducer consumes a scoped `SessionIdentity`
minted before flow-control admission and publishes a unique per-request stamp
for the session prefix-cache integration.

```yaml
- type: agent-identity
- type: session-identity
  parameters:
    deploymentID: payments-prod-eu1
    hmacKeyFile: /var/run/secrets/llm-d/session-identity-key
- type: session-manager
  parameters:
    sessionIdentityProducer: session-identity
```

`sessionIdentityProducer` defaults to `session-identity`. It names the plugin
instance whose typed `SessionIdentity` this manager consumes.

After admission, the plugin publishes:

```text
SessionCacheRequest
  SessionID: unique process-nonce/counter stamp
  FullReport: false
  TotalTokens: 0
  Prefixes: []
```

The `session-prefix-cache-producer` from #2716 consumes this request and owns
request-body mutation, KV-event subscription, and residency indexing. Version
1 does not subscribe to KV events, retain engine-block paths, infer
continuations or forks, or publish cache prefixes.

The `SessionCacheRequest` data contract in
`plugins/datalayer/attribute/session/data_types.go` is copied byte-for-byte
from #2716 so either PR can merge first without conflicting.
