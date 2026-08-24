# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Sets the `x-kv-cache-source-host-port` header to an endpoint within one block of the most cached prompt prefix, so the routing sidecar can pull those blocks over the P2P connector instead of recomputing them. Runs in the request handling's `DataProducer` phase before scheduling, then emits the header in `PreRequest` after the scheduling decision.

For each request the plugin consumes the per-endpoint `PrefixCacheMatchInfo` of a prefix-cache producer (`approx-prefix-cache-producer` or `precise-prefix-cache-producer`) and picks a source among the endpoints caching within one block of the most prompt tokens, weighted by `1/(1+waiting queue)` with a request-ID hash as the sampling coordinate; when the producer supplies per-tier data, only CPU-tier blocks count, since pulls are served from the source's CPU tier. Sampling within the one-block band, rather than argmax, keeps pull traffic from concentrating on a single replica of a widely-cached prefix. After scheduling, the header is set only when the chosen peer out-caches the computing pod by at least `minCachedTokenDelta` tokens; any inbound header value is removed.

The plugin also publishes a name-bound `ReusablePrefixTokens` request
attribute for scheduling plugins. Let `S` be the pull-servable prefix token
count on the source actually sampled by this plugin (the CPU-tier count when
per-tier data is available), and let `D` be `minCachedTokenDelta`. The
published floor is:

```text
G = max(0, S - D + 1)
```

Any computing pod either has at least `G` tokens locally or qualifies for a
pull from the sampled source under the same routing snapshot. A
[`context-length-aware`](../../../scheduling/scorer/contextlengthaware/README.md)
instance can consume this floor through `reusableTokensProducerName` and route
on the remaining prefill work. A request with no CPU-pullable source has no
floor attribute.

**Parameters:**

- `prefixMatchInfoProducerName` (string, optional): Name of the prefix-cache producer instance to consume `PrefixCacheMatchInfo` from, e.g. `precise-prefix-cache-producer`. Empty selects the default (unnamed) producer.
- `minCachedTokenDelta` (int, optional, default: `1`): Minimum number of cached prompt tokens the best peer must hold beyond the computing pod for the header to be set. Must be `>= 1`. Higher values suppress pulls of short prefixes that are cheap to recompute.
- `prefillProfileName` (string, optional, default: `prefill`): Name of the P/D disaggregation prefill scheduling profile. The computing pod is read from this profile's target when present; otherwise the primary profile's target is used.

**Configuration Example:**

```yaml
plugins:
  - type: precise-prefix-cache-producer
    name: precise-cache
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 64
      kvEventsConfig:
        topicFilter: "kv@"
      speculativeIndexing: false
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      prefillProfileName: prefill
      minCachedTokenDelta: 1
```

## Deployment Requirements

The emitted header only results in a KV transfer when the serving pods are
configured to serve and pull blocks over the P2P tier:

- vLLM runs the `OffloadingConnector` with a `p2p` secondary tier, and the routing sidecar consumes the header to inject the pull.
- `offload_prompt_only: false` in `kv_connector_extra_config` on any pod whose cache may be pulled. With the default (`true`), decode-phase (generated) blocks are never offloaded, so a pull of that content misses.
- Identical `--block-size` across peers; a mismatch makes vLLM reject the transfer (`block_len mismatch`).
- Identical `PYTHONHASHSEED` across peers, so block hashes match across processes.

Cache-aware prefill-work routing should use a `precise-prefix-cache-producer`
with per-tier cache data and `speculativeIndexing: false`. The reusable floor
uses confirmed CPU-tier blocks. Speculative local entries can otherwise
suppress the source header without providing a pullable prefix. Set
`prefillProfileName` to the profile selected by `disagg-profile-handler`, and
reference the consuming `context-length-aware` instance only from that prefill
profile.

---

## Related Documentation
- [Approximate Prefix Cache Producer](../approximateprefix/README.md)
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
