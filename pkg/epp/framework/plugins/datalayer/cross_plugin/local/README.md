# Local Cross-Replica Syncer

**Type:** `local-syncer`
**Interface:** `CrossReplicaSyncer`

Stores cross-replica state in the EPP process. It is intended for tests and
single-replica deployments that do not need synchronization between EPP
instances.

The runtime binds one state handle per contributor. The handle retains the live
local value reader and aggregation function. Each operation receives only its
endpoint.

## What It Does

- Reads the contributor's live local endpoint value and applies its aggregation
  function.
- Provides atomic `GetOrSet` coordination within one EPP process.
- Uses the local hostname to isolate state associated with the EPP replica.

## Registration

The stock EPP runner does not register this test-oriented plugin. Embedders and
tests can register `LocalSyncerFactory`; the plugin has no parameters. Once
registered, it can be selected in the data layer configuration:

```yaml
plugins:
  - type: local-syncer
    name: local-syncer

dataLayer:
  crossReplica:
    syncerPluginRef: local-syncer
```

## Limitations

- State is not shared between EPP replicas.
- State is lost when the EPP process exits.

## Related Documentation

- [Plugins Index](../../../README.md)
