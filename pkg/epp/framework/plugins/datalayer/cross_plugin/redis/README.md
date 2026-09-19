# Redis Cross-Replica Syncer

**Type:** `redis-state-store`
**Interface:** `CrossReplicaSyncer`

Shares endpoint and request coordination state between EPP replicas through
Redis. Each EPP publishes its local endpoint value. The plugin computes the
cross-replica aggregate during `Set` and serves `Get` from an in-process cache.

## Configuration

```yaml
plugins:
  - type: redis-state-store
    name: redis
    parameters:
      address: my-router-redis:6379
      ttl: 180s

dataLayer:
  crossReplicaSyncerPluginRef: redis
```

Parameters:

- `address`: Redis address. Defaults to `localhost:6379`.
- `password`: Optional Redis password.
- `db`: Redis database number. Defaults to `0`.
- `ttl`: Expiration for replica and coordination state. Defaults to `180s`.

The configured Redis server must support field expiration and `SET NX GET`.
Redis 7.4 or newer is required.

## State Model

Endpoint state is stored in one Redis hash per state key and endpoint. Each EPP
owns one field in that hash. `Set` refreshes the field TTL, reads the hash in the
same transaction, computes the aggregate, and caches it locally. `Get` only
reads the local aggregate cache.

Request-level coordination uses separate string keys and `SET NX GET` so the
first value stored for a request is selected atomically across EPP replicas.

## Deployment

The plugin is a Redis client and does not deploy a Redis server. The server must
be provisioned separately and reachable at the configured address.

## Related Documentation

- [Plugins Index](../../../README.md)
