# Precise Prefix Cache Scorer

**Type:** `precise-prefix-cache-scorer`

> **Deprecated.** Configure
> [`precise-prefix-cache-producer`](../../../requestcontrol/dataproducer/preciseprefixcache/)
> and the generic [`prefix-cache-scorer`](../prefix/) directly, with
> `prefixMatchInfoProducerName: precise-prefix-cache-producer`.

Composes a `precise-prefix-cache-producer` and a `prefix-cache-scorer`
behind a single plugin type. Two modes:

- If no `precise-prefix-cache-producer` is configured elsewhere, the
  plugin instantiates an internal one from its parameters and presents
  Scorer + DataProducer + PreRequest + EndpointExtractor by delegation.
  When `indexerConfig.tokenizersPoolConfig` is set, the plugin also
  owns a UDS tokenization pool and pre-tokenizes incoming prompts so
  no upstream `token-producer` is required.
- If a `precise-prefix-cache-producer` is already configured, the plugin
  returns a `prefix-cache-scorer` bound to it and ignores its own
  parameters with a warning.

Peer detection runs at plugin construction time. Declare the
`precise-prefix-cache-producer` ahead of this plugin in YAML so it is
already registered when this factory runs; otherwise both producers
end up active.

## Migration

```yaml
- type: precise-prefix-cache-scorer
  parameters:
    tokenProcessorConfig: { blockSize: 64 }
    kvEventsConfig: { discoverPods: true, podDiscoveryConfig: { socketPort: 5557 } }
```

becomes

```yaml
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig: { blockSize: 64 }
    kvEventsConfig: { discoverPods: true, podDiscoveryConfig: { socketPort: 5557 } }
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
```

Wire the producer to `endpoint-notification-source` under `dataLayer` as
in the producer's own README.

## Parameters

Parameters forward to the internal producer. See
[`precise-prefix-cache-producer`](../../../requestcontrol/dataproducer/preciseprefixcache/README.md)
for the full reference.
