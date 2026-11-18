# KV events source

`kv-events-source` shares one subscription per discovered engine endpoint among
plugins implementing `kvevents.EventConsumer`. Consumers receive the same decoded
batch, in order per endpoint, and must not mutate it. Different endpoints can run
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
    externalEvents: true
dataLayer:
  sources:
  - pluginRef: engine-events
    extractors:
    - pluginRef: cache
```

Parameters use the existing `kvevents.Config`. Per-endpoint discovery is required;
global socket mode is unsupported. Live, replay, and snapshot ports are offset by
the endpoint's rank. Bind every consumer under the same source to share its stream.
Bindings close when endpoint discovery starts.

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

A consumer that returns an error or panics is reset for that endpoint and the
other consumers keep their state for that batch. The source then rebuilds the
endpoint from its snapshot or replay buffer, which resets every consumer of that
endpoint and delivers the rebuilt state to all of them. Rebuilds for one endpoint
are at least 30 seconds apart. With neither recovery port configured, the reset
consumer rebuilds from live events alone.

`socketPort` must be positive. The source's state dump lists the endpoints with
an active subscriber and the bound consumers.

The precise producer requires `externalEvents: true` to bind to this source. Its
own endpoint subscription hook is then inactive. Other consumers implement the
same interface and can be bound alongside it.
