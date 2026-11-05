# Redis Syncer

**Type:** `redis-syncer`
**Interface:** `CrossReplicaSyncer`

Shares endpoint and request coordination state between EPP replicas through
Redis. Each EPP publishes its local endpoint value. During `Set`, the plugin
prepares a peer aggregate in its in-process cache. `Get` reads the contributor's
live local value and combines it with the cached peer aggregate.

## Configuration

Enable this Alpha plugin with `--allow-experimental-plugins=true`.

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
- `stateTTL`: Expiration for each replica's endpoint state in Redis. Defaults to
  `2s` and must be at least `1ms`.
- `coordinationTTL`: Expiration for request-level coordination values. Defaults
  to `180s` and must be at least `1ms`.
- `windowReplication`: How sequenced values such as approximate-prefix windows
  are published. `full` (default) writes the whole value every tick.
  `incremental` appends only new generations to a per-replica Redis stream.
  See [Incremental replication](#incremental-replication).
- `maxBytesPerSecond`: Stream bytes each replica may write, and separately
  read, per second in `incremental` mode. Defaults to `8388608` (8 MiB/s) and
  must not be negative; `0` selects the default. Unused budget accumulates up to
  one second. Values that are not sequenced ignore this limit.

### Publisher timing

Publisher settings are configured under `dataLayer.crossReplica`:

| Setting | Default | Purpose |
|---|---|---|
| `syncInterval` | `200ms` | Interval between endpoint publication ticks. |
| `publishTimeout` | `1s` | Context timeout for one endpoint publish (shared by its concurrent contributor writes), or one batch chunk per contributor when the syncer supports batching. Also bounds one endpoint deletion. Must be positive. |

The Redis syncer publishes endpoints in batches: the publisher calls the
syncer once per contributor per cycle with the live endpoint set, without
holding its endpoint lock, and each chunk runs under one `publishTimeout`. Syncers without batch
support visit endpoints sequentially, with concurrent contributor writes for
each endpoint. Endpoints removed while a batch runs are deleted again after
it finishes. A slow cycle can exceed `syncInterval`. Choose `stateTTL` longer
than the observed interval between successful refreshes of an endpoint, with
margin for Redis latency and variation in cycle duration.

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
and caches the prepared peer aggregate locally until the next successful `Set`
replaces it. A replica that stops publishing loses its field `stateTTL` after
its last write, and the next refresh drops it from the aggregate. `Get` reads
the cache without accessing Redis and combines it with the contributor's live
local value. If no peer value is available, `Get` returns the local value.

### Incremental replication

With `windowReplication: incremental`, a replica publishes a sequenced value
as a stream next to the hash, `{<hash key>}:stream:<replica field>`. Entry
`0-(g+1)` holds generation `g`; for approximate-prefix windows a generation is
one prompt, packed as little-endian 8-byte hashes. The replica's hash field
`stream:<replica field>` holds the marker `epoch base tip size`, naming the
generations `[base, tip)` the stream holds and their total bytes. Each tick
appends generations the replica has not written, trims entries below `base`
when the window evicts prompts, and refreshes the TTLs of the hash, marker
field, and stream. An unchanged window costs a few commands and no payload.

Readers fetch only the generations past the last tip they hold. When a write
is lost or a stream is missing or truncated, the writer starts a new epoch and
republishes from `base`; readers that see a new epoch or a gap reread from
`base`. When the byte budget is spent the writer stops appending and readers
stop fetching for that tick, and peers catch up on later ticks. A cold
window, after a restart or reset, reaches peers over several ticks.

Readers in either mode merge stream peers and whole-value peers, so a fleet
can switch modes one replica at a time: roll out the new version in `full`
mode, then switch replicas to `incremental`.

| Metric | Labels | Meaning |
|---|---|---|
| `redis_syncer_stream_bytes_total` | `state_key`, `direction` | Stream payload bytes written or read. |
| `redis_syncer_stream_budget_exhausted_total` | `state_key`, `direction` | Ticks that deferred stream work because the byte budget was spent. |
| `redis_syncer_stream_resets_total` | `state_key` | New epochs started by the writer. |
| `redis_syncer_stream_peer_lag_generations` | `state_key` | Generations a peer read was behind the peer's tip. |
| `redis_syncer_set_batch_duration_seconds` | `state_key` | Duration of one batched publication. |

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
are logged at error level with rate limiting and leave the peer cache unchanged,
so scheduling keeps counting the last known peer state until a publication
succeeds. A peer that stops while Redis is unreachable keeps counting until
then. Before its first successful publication an EPP uses local state only.

`GetOrSet` does not use a local fallback. It returns Redis errors because its
callers require cross-replica atomicity.

## Related Documentation

- [Plugins Index](../../../README.md)
