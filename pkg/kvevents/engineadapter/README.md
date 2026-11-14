# Engine Adapters

Decodes the KV-cache event messages that model-serving engines publish into the
canonical events the [`kvevents.Pool`](../README.md) applies to the block index.
Each engine has its own topic scheme and wire format; an adapter hides those
differences behind one interface.

## What It Does

Given a raw ZMQ message (topic plus payload), an adapter parses the topic to
recover the model and sharding key, decodes the payload, and returns canonical
block-stored / block-removed / all-blocks-cleared events. It also exposes the
sharding key so the pool can order events for the same block.

## Adapters

| Engine | Constructor | Notes |
|--------|-------------|-------|
| vLLM | `NewVLLMAdapter` | msgpack payloads; supports HMA group identity and map-encoded events. |
| SGLang | `NewSGLangAdapter` | SGLang's topic scheme and payload format. |

`NewAdapter` selects the adapter registered for a given engine type
(`kvEventsConfig.engineType`).

## Adding an engine

vLLM and SGLang register their adapters in this package. An engine with its own
event format registers its adapter from an `init` function in its own package:

```go
func init() {
	engineadapter.Register("my-engine", func() kvevents.EngineAdapter { return NewMyEngineAdapter() })
}
```

An EPP binary that imports that package accepts `engineType: my-engine`. A
later registration under the same name replaces the earlier one. An adapter
decodes messages only: events arrive over the ZMQ subscriber, and the pool
recomputes block keys from each event's token IDs, so the engine must publish
the token IDs the router's tokenizer produces for the same prompt.

## Related Documentation

- [KV-Events](../README.md) -- consumes the decoded events
- [KV-Block Index](../../kvcache/kvblock/README.md) -- where the events land
