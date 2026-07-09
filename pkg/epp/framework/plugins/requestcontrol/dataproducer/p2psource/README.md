# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Sets the `x-kv-cache-source-host-port` header to the endpoint holding the most cached prompt prefix, so the routing sidecar can pull those blocks over the P2P connector instead of recomputing them. Runs in the request handling's `DataProducer` phase before scheduling, then emits the header in `PreRequest` after the scheduling decision.

For each request the plugin consumes the per-endpoint `PrefixCacheMatchInfo` produced by a prefix-cache producer (`approx-prefix-cache-producer` or `precise-prefix-cache-producer`), finds the candidate endpoint caching the most prompt tokens, and stashes it on the request. After scheduling it compares that best-match peer against the pod that will compute the prefix — the `prefill` profile target under P/D disaggregation, otherwise the primary target — and sets the header only when the peer out-caches the computing pod by at least `minCachedTokenDelta` tokens. Any inbound value of the header is removed. When no peer out-caches the computing pod, the request proceeds unchanged.

**Parameters:**

- `prefixMatchInfoProducerName` (string, optional): Name of the prefix-cache producer instance to consume `PrefixCacheMatchInfo` from, e.g. `precise-prefix-cache-producer`. Empty selects the default (unnamed) producer.
- `minCachedTokenDelta` (int, optional, default: `1`): Minimum number of cached prompt tokens the best peer must hold beyond the computing pod for the header to be set. Must be `>= 1`. Higher values suppress pulls of short prefixes that are cheap to recompute.
- `prefillProfileName` (string, optional, default: `prefill`): Name of the P/D disaggregation prefill scheduling profile. The computing pod is read from this profile's target when present; otherwise the primary profile's target is used.

**Configuration Example:**

```yaml
plugins:
  - type: precise-prefix-cache-producer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
      kvEventsConfig:
        topicFilter: "kv@"
  - type: p2p-source-producer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
      prefillProfileName: prefill
      minCachedTokenDelta: 1
```

## Deployment Requirements

The emitted header only results in a KV transfer when the serving pods are
configured to serve and pull blocks over the P2P tier:

- vLLM runs the `OffloadingConnector` with a `p2p` secondary tier, and the routing sidecar consumes the header to inject the pull.
- `offload_prompt_only: false` in `kv_connector_extra_config` on any pod whose cache may be pulled — with the default (`true`), decode-phase (generated) blocks are never offloaded, so a pull of that content misses.
- Identical `--block-size` across peers; a mismatch makes vLLM reject the transfer (`block_len mismatch`).
- Identical `PYTHONHASHSEED` across peers, so block hashes match across processes.

---

## Related Documentation
- [Approximate Prefix Cache Producer](../approximateprefix/README.md)
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
