# Redis Syncer

**Type:** `redis-syncer`
**Interface:** `CrossReplicaSyncer`

Shares endpoint and request coordination state between EPP replicas through
Redis. Each EPP publishes its local endpoint value. During `Set`, the plugin
prepares a peer aggregate in its in-process cache. `Get` reads the contributor's
live local value and combines it with the cached peer aggregate.

## Configuration

The plugin also accepts `redis-state-store` as a compatibility alias.

```yaml
plugins:
  - type: redis-syncer
    name: redis
    parameters:
      address: my-router-redis:6379
      passwordFile: /var/run/secrets/redis/password
      stateTTL: 2s
      coordinationTTL: 180s

dataLayer:
  crossReplica:
    syncerPluginRef: redis
```

Parameters:

- `address`: Redis address. Defaults to `localhost:6379`.
- `passwordEnv`: Optional name of an environment variable containing the Redis
  password.
- `passwordFile`: Optional path to a file containing the Redis password.
  Trailing carriage returns and newlines are removed.
  `passwordEnv` and `passwordFile` are mutually exclusive. Plaintext
  `password` configuration is rejected.
- `db`: Redis database number. Defaults to `0`.
- `stateTTL`: Expiration for each replica's endpoint state and the local peer
  aggregate. Defaults to `2s` and must be at least `1ms`.
- `coordinationTTL`: Expiration for request-level coordination values. Defaults
  to `180s` and must be at least `1ms`.

### Publisher timing

Publisher settings are configured under `dataLayer.crossReplica`:

| Setting | Default | Purpose |
|---|---|---|
| `syncInterval` | `200ms` | Interval between endpoint publication ticks. |
| `publishTimeout` | `1s` | Context timeout shared by all contributors for one endpoint's publication or deletion. Must be positive. |

Each publication cycle visits endpoints sequentially. Contributors for one
endpoint publish concurrently, sharing a deadline that starts before acquiring
the publisher's read locks. A slow cycle can exceed `syncInterval`. Choose
`stateTTL` longer than the observed interval between successful refreshes of an
endpoint, with margin for Redis latency and variation in cycle duration.

Deletion waits for the publisher's write lock, then starts one timeout shared by
all contributor deletions. An earlier caller deadline takes precedence for both
publication and deletion.

The Redis client honors context deadlines. Cancellation cannot interrupt mutex
waits or local computation, so `publishTimeout` does not guarantee that the
entire operation returns within that duration. Scheduling-path `Get` calls do
not acquire the publisher mutex.

`GetOrSet` uses its caller's context to bound Redis access; `coordinationTTL`
controls how long the stored value is retained. Redis connection, read, and
write timeouts use the client library's defaults and have no plugin parameters.

The `v1alpha1` field names are listed in the
[configuration API migration guide](../../../../../../../docs/architecture.md#api-versions).

### Redis requirements

TLS and Redis ACL usernames are not supported. Connections use unencrypted TCP.
Optional password authentication uses the default Redis user. Managed Redis
services that require TLS or an ACL username cannot use this plugin.

The configured Redis server must support field expiration and `SET NX GET`.
Redis 7.4 or newer is required.

Values stored under the same state key must use a compatible `encoding/gob`
schema. Each EPP decodes peer values into the type supplied by its local
contributor without requiring global `gob.Register` calls.

## State Model

Endpoint state is stored in one Redis hash per state key and endpoint. Each EPP
owns one field in that hash, identified by its hostname and a process-specific
UUID. `Set` refreshes the field TTL, reads the hash in the same transaction,
and caches the prepared peer aggregate locally. The aggregate expires
`stateTTL` after the local refresh. `Get` reads the cache without accessing
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
are logged at error level with rate limiting and leave the current peer cache in
place until its entries expire. Scheduling then uses local state until a later
publication succeeds and refreshes the cache.

`GetOrSet` does not use a local fallback. It returns Redis errors because its
callers require cross-replica atomicity.

## Related Documentation

- [Plugins Index](../../../README.md)
