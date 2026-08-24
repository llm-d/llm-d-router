# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Sets the `x-kv-cache-source-host-port` header to an endpoint within one block of the most cached prompt prefix, so the routing sidecar can pull those blocks over the P2P connector instead of recomputing them. Runs in the request handling's `DataProducer` phase before scheduling, then emits the header in `PreRequest` after the scheduling decision.

For each request the plugin consumes the per-endpoint `PrefixCacheMatchInfo` of a prefix-cache producer (`approx-prefix-cache-producer` or `precise-prefix-cache-producer`) and picks a source among the endpoints caching within one block of the most prompt tokens, weighted by `1/(1+waiting queue)` with a request-ID hash as the sampling coordinate; when the producer supplies per-tier data, only CPU-tier blocks count, since pulls are served from the source's CPU tier. Sampling within the one-block band, rather than argmax, keeps pull traffic from concentrating on a single replica of a widely-cached prefix. After scheduling, the header is set only when the chosen peer out-caches the computing pod by at least `minCachedTokenDelta` tokens; any inbound header value is removed.

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
