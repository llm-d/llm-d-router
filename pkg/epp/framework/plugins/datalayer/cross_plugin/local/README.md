# Local Cross-Replica Syncer

**Type:** `local-syncer`
**Interface:** `CrossReplicaSyncer`

Stores cross-replica state in the EPP process. It is intended for tests and
single-replica deployments that do not need synchronization between EPP
instances.

## What It Does

- Stores endpoint state in memory and applies the contributor's aggregation
  function when state is set.
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
  crossReplicaSyncerPluginRef: local-syncer
```

## Limitations

- State is not shared between EPP replicas.
- State is lost when the EPP process exits.

## Related Documentation

- [Plugins Index](../../../README.md)
