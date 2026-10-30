# Session Identity

The Alpha `session-identity` RequestHeader plugin derives a scoped identity
from the `agent-identity` request attribute before flow-control admission.

```yaml
- type: agent-identity
- type: session-identity
  parameters:
    deploymentID: payments-prod-eu1
    hmacKeyFile: /var/run/secrets/llm-d/session-identity-key
```

`hmacKeyFile` must contain the unpadded base64url encoding of exactly 32 random
bytes. The plugin reads it once at startup.

For a non-empty `agent-identity`, the plugin publishes:

- a producer-bound typed `SessionIdentity`;
- the same opaque HMAC-derived `SessionTag` as a producer-bound plain string
  attribute named `session-tag`.

The typed value is intended for structured consumers such as Session Manager
and fairness policies. The string value supports generic consumers:

```yaml
sessionIdConfig:
  sources:
  - attribute: session-tag
    producer: session-identity
```

Missing identity fails open. Raw aliases are not copied into plugin output,
logs, metrics, or debug state.
