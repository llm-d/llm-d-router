# KV events source

`kv-events-source` opens one subscription per discovered engine endpoint and hands
each decoded batch to every consumer bound to it. Consumers receive the same batch,
in order per endpoint, and must not mutate it. Different endpoints can run
concurrently. The source owns subscriptions, decoding, replay, and snapshot recovery.

```yaml
plugins:
- type: kv-events-source
  name: engine-events
  parameters:
    podDiscoveryConfig:
      podLabelSelector: app=vllm
      socketPort: 5557
      replaySocketPort: 5558
- type: precise-prefix-cache-producer
  name: cache
  parameters:
    kvEventsSourceRef: engine-events
```

Parameters use the existing `kvevents.Config`. Per-endpoint discovery is required;
global socket mode is unsupported. Live, replay, and snapshot ports are offset by
the endpoint's rank.

The source follows endpoint lifecycle events as an extractor of the
`endpoint-notification-source`, which it registers itself. It takes no
`dataLayer.sources` entry and no extractors. A plugin binds a consumer by naming the
source in its own configuration (`kvEventsSourceRef` on the precise producer) and
calling `AddConsumer` with a `kvevents.EventConsumer` when it is constructed; the
plugin loader instantiates the source first. Consumers bind before endpoint
discovery starts. The precise producer bound this way opens no subscribers of its
own.

`ProcessEvents` receives the serving endpoint, model, wire sequence, and decoded
batch. Wire `AllBlocksCleared` events remain in the batch. `Reset` invalidates
observations after an engine restart, a failed replay, a rebuild from a snapshot
or a full replay, subscriber attachment, or endpoint removal. A socket reconnect
keeps consumer state, and the first frame after it recovers missed events through
replay or a snapshot. With neither recovery port configured, a dropped frame or an
undecodable message keeps consumer state and the missed events stay missed.
`EventSource.Retired` identifies endpoint removal. Consumers that retain logical
lineage should mark it incomplete on reset and resolve current residency
separately.

A consumer that returns an error or panics is logged and reset for that endpoint,
and the other consumers keep their state for that batch. The source then rebuilds
the endpoint from its snapshot or replay buffer, which resets every consumer of that
endpoint and delivers the rebuilt state to all of them. Rebuilds for one endpoint
are at least 30 seconds apart. With neither recovery port configured, the reset
consumer rebuilds from live events alone.

`socketPort` must be positive. The source's state dump lists the endpoints with
an active subscriber and the plugins with a bound consumer.
