# KV-Events

The write side of the KV-cache subsystem: it subscribes to the KV-cache events
that model-serving engines emit as they admit and evict blocks, and applies them
to the [`kvblock.Index`](../kvcache/kvblock/README.md) so the
[`kvcache.Indexer`](../kvcache/README.md) scores against an up-to-date view of
what each pod holds.

## What It Does

Engines (vLLM, SGLang) publish block-stored / block-removed / all-cleared events
over ZMQ. The subsystem subscribes to those streams, decodes each engine's
message format via an [engine adapter](engineadapter/README.md), and translates
them into `Add` / `Evict` calls on the block index -- a `SubscriberManager`
owns the per-engine subscriptions and a sharded worker `Pool` does the
processing. This keeps cache-locality scoring real-time without polling the
engines.

## How It Works

- **Subscription.** A ZMQ subscriber receives messages from the engine pods.
  `SubscriberManager` maintains one subscription per engine endpoint and tears
  them down as pods come and go.
- **Decoding.** Each message is parsed by the engine adapter for its topic,
  producing canonical block-stored / block-removed / all-blocks-cleared events.
- **Worker pool.** `Pool` fans messages out to workers keyed by a sharding key
  so events for the same block are processed in order, then applies them to the
  index.
- **Reference-count dedup.** A dedup filter reference-counts block hashes so a
  removal is forwarded to the index only once no remaining announcement
  references the block, preventing premature eviction of blocks that multiple
  engines (or HMA groups) still hold. Suppressed and forwarded removals are
  counted in the [metrics](../kvcache/metrics/README.md).

## Key Types

| Symbol | Role |
|--------|------|
| `Pool` | Entry point; constructed with `NewPool(config)`. Subscribes, decodes, and applies events. |
| `SubscriberManager` | Manages per-endpoint ZMQ subscriptions across pod churn. |
| `Config` / `DefaultConfig` | Worker count, subscription, and dedup configuration. |

## Related Documentation

- [Engine Adapters](engineadapter/README.md) -- per-engine event decoding
- [KV-Block Index](../kvcache/kvblock/README.md) -- the index these events update
- [KV-Cache Indexer](../kvcache/README.md) -- reads the resulting index state
