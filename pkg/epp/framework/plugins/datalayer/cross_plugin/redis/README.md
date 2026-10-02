# Redis Cross-Replica Syncer

**Type:** `redis-state-store`
**Interface:** `CrossReplicaSyncer`

Shares endpoint and request coordination state between EPP replicas through
Redis. Each EPP publishes its local endpoint value. During `Set`, the plugin
prepares a peer aggregate in its in-process cache. `Get` reads the contributor's
live local value and combines it with the cached peer aggregate.

The runtime binds one state handle per contributor and endpoint. The handle
retains the endpoint key, live local value reader, and aggregation function used
by `Set`, `Get`, and `Delete`.

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

Values stored under the same state key must use a compatible `encoding/gob`
schema. Each EPP decodes peer values into the type supplied by its local
contributor without requiring global `gob.Register` calls.

## State Model

Endpoint state is stored in one Redis hash per state key and endpoint. Each EPP
owns one field in that hash. `Set` refreshes the field TTL, reads the hash in the
same transaction, and caches the prepared peer aggregate locally. The aggregate
expires with its oldest included value. `Get` reads the cache without accessing
Redis and combines it with the contributor's live local value. If no peer value
is available, `Get` aggregates the local value by itself.

Request-level coordination uses separate string keys and `SET NX GET` so the
first value stored for a request is selected atomically across EPP replicas.

## Scheduling and Admission

Periodic endpoint synchronization provides peer-aware scheduling state. It is
eventually consistent: two EPP replicas can make scheduling or admission
decisions before either observes the other's latest work.

This synchronization does not implement an atomic cluster-wide admission
quota. A deployment that requires a fixed admission limit can use separate
local-only admission producers, with one admitting EPP per fixed share and the
shares summing to the intended limit. Unused capacity is not borrowed between
shares automatically.

`GetOrSet` coordinates one value for one request identifier. It does not reserve
capacity across different requests.

## Deployment

The plugin is a Redis client and does not deploy a Redis server. The server must
be provisioned separately. Plugin construction does not connect to Redis, so an
unavailable server does not prevent the EPP from starting. Failed publications
leave the current peer cache in place until its entries expire. Scheduling then
uses local state until a later publication succeeds and refreshes the cache.

`GetOrSet` does not use a local fallback. It returns Redis errors because its
callers require cross-replica atomicity.

## Related Documentation

- [Plugins Index](../../../README.md)
