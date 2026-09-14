# Session Manager

The Alpha `session-manager` DataProducer establishes a scoped identity shared by
session consumers and, when explicitly enabled, assigns request stamps for the
session-aware KV-event integration.

Version 1 does not infer continuations or forks, retain engine-block paths, or
publish `SessionCachePrefix`. A repeated external identity is only a grouping
signal.

## Configuration

```yaml
- type: agent-identity
- type: token-producer
  name: tokens
  parameters:
    modelName: example/model
- type: session-manager
  name: sessions
  parameters:
    deploymentID: payments-prod-eu1
    hmacKeyFile: /var/run/secrets/llm-d/session-manager-key
    tokenProducer: tokens
    eventCorrelationEnabled: true
    bindingTTL: 5m
    maxBindings: 100000
```

`hmacKeyFile` must contain the unpadded base64url encoding of exactly 32 random
bytes. Mount it from an operator-managed Kubernetes Secret. The manager reads it
once at startup and never exposes it.

Version 1 publishes no engine-block prefixes, so it neither requires nor
asserts cache compatibility across workers. Pod endpoints and event-local
cache-group metadata come from the existing KV-event discovery path. The
client-provided identity is not authentication or tenant isolation.

## Outputs

For a non-empty `agent-identity`, the manager publishes a name-bound
`SessionIdentity`:

- `SessionTag`: an opaque HMAC-derived grouping identity;
- `IdentitySource`: `AgentIdentityAttribute`;
- `ScopeVersion`: an opaque deployment/key/version scope.

Raw aliases are not retained or copied into manager output, logs, metrics, or
debug state.

When event correlation is enabled and the request has one non-empty tokenized
prompt in a mutable JSON envelope, the manager additionally publishes:

```text
SessionCacheRequest
  Stamp: unique process-nonce/counter value
  FullReport: false
  TotalTokens: whole prompt length
  Prefixes: []
```

The precise-prefix integration writes the stamp to vLLM request-body
`session_id` and requests incremental reporting. A client body value is
overwritten in this opt-in mode. Before dispatch, the manager binds the stamp
to the single endpoint selected by the primary scheduling profile. Stamped
local-GPU store events are classified against that bounded replica-local
binding; another endpoint, a reset source, or an unbound request fails open.
Version 1 creates no relation or session-derived cache-prefix signal.

## Operational boundary

- Enable Alpha plugins explicitly.
- Run one request-serving EPP replica per manager state domain.
- Missing identity, unsupported request shapes, expired or unknown stamps,
  restart, reset, and event gaps fail open.
- Keep the ordinary token-derived precise producer for production cache-aware
  scoring. Manager-backed precise mode is a correlation/integration
  configuration until a later version publishes defensible prefixes.
