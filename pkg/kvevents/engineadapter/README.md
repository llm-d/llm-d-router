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

Topics take the form `kv@<pod-id>@<model-name>`. With per-pod subscriptions
the subscriber already knows the pod, so `kv@<model-name>` works too, and every
replica of a model server can run with the same arguments.

## Adapters

| Engine | Constructor | Notes |
|--------|-------------|-------|
| vLLM | `NewVLLMAdapter` | msgpack payloads; supports HMA group identity and map-encoded events. Skips an event with an unknown tag and lists the tag in `EventBatch.SkippedTags`. |
| SGLang | `NewSGLangAdapter` | SGLang's topic scheme and payload format. Rejects a message that holds an event with an unknown tag. |

`NewAdapter` selects the adapter registered for a given engine type
(`kvEventsConfig.engineType`).

### vLLM recovery checkpoints

The recovery wire format comes from an unmerged vLLM draft,
[PR #67160](https://github.com/vllm-project/vllm/pull/67160), stacked on
[#66881](https://github.com/vllm-project/vllm/pull/66881). It can change before
merge. The draft producer publishes an array-shaped batch containing tagged
MessagePack **maps**.
The adapter follows the [event definitions](https://github.com/ischencheng/vllm/blob/4ffb2a4b3a5c9a1cdae88262ed1ec35e6f88665d/vllm/distributed/kv_events.py#L148-L207)
and the [checkpoint publisher](https://github.com/ischencheng/vllm/blob/4ffb2a4b3a5c9a1cdae88262ed1ec35e6f88665d/vllm/v1/core/hybrid_checkpoint/manager.py#L269-L320).

| Event tag (`type`) | Required fields (any key order) | Optional fields |
| --- | --- | --- |
| `RecoveryUnitStored` | `unit_hash`, `num_tokens`, `frontier` | `session_id` |
| `RecoveryUnitRemoved` | `unit_hash`, `num_tokens`, `frontier`, `cause` | `tier`, `num_blocks` |

`unit_hash` identifies recurrent state on the publishing engine. The producer
uses a binary or integer hash. The adapter retains every byte of binary keys
up to 64 bytes. `num_tokens` gives the resumable prefix length.
`frontier: true` identifies a request's exact endpoint inside a block. A false
value identifies a block boundary.

On removal, `cause` describes why the state disappeared. The producer defines
`evicted-dead`, `evicted-interior`, `evicted-leaf`, `evicted-unranked`,
`orphan`, `superseded`, `frontier`, `replaced` and `allocation`. The optional
`tier` gives the last eviction ranking (`dead`, `interior` or `leaf`).
`num_blocks` counts blocks freed when the eviction selected that unit.
Consumers receive cause and tier strings verbatim.

The adapter delivers valid events in wire order and records unsupported
recovery events in `EventBatch.SkippedTags`. This includes positional recovery
arrays and malformed maps with an identifiable recovery tag. The pool counts
these tags in `llm_d_epp_kv_cache_events_unknown_tags_skipped_total`. An
unreadable MessagePack boundary still fails the batch. A consumer that requires
complete checkpoint history can return an error on a skipped recovery tag
to reset its source. The block index
logs recovery events at trace verbosity and indexes the surrounding block
events. Consumers must also handle `AllBlocksCleared` and endpoint resets.

#### Replay the vLLM producer fixture

`go test ./pkg/kvevents/engineadapter -run '^TestVLLMRecoveryProducerReplay$'`
replays 74 captured batches from two CPU schedulers. The producer's test helpers
generate synthetic token IDs and model outputs. The fixture wraps the original
MessagePack payloads in a gzipped MessagePack array of byte strings.
The test decodes each frame and repeats it with a malformed recovery map inserted
between events. It checks the skipped tag, event order and contents, timestamp,
and data-parallel rank.

To regenerate, use a vLLM checkout at the
[pinned producer revision](https://github.com/ischencheng/vllm/commit/4ffb2a4b3a5c9a1cdae88262ed1ec35e6f88665d)
with its test dependencies and the `facebook/opt-125m` test configuration
available. From the router repository root, set `VLLM_CHECKOUT` to that checkout
and run in its Python environment:

```sh
fixture_dir="$PWD/pkg/kvevents/engineadapter/testdata"
cd "$VLLM_CHECKOUT"
PYTHONHASHSEED=0 PYTHONPATH="$PWD" python "$fixture_dir/generate_vllm_recovery.py" \
  "$fixture_dir/vllm_recovery.msgpack.gz"
```

When the producer schema changes, regenerate from that revision and run the
replay before updating the pin. In a running router, rising skipped-tag counts
for `tag="RecoveryUnitStored"` or `tag="RecoveryUnitRemoved"` indicate recovery
payloads the adapter could not decode.

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
