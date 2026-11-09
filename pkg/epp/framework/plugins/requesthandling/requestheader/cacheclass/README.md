# Cache Class Router

**Type:** `cache-class-router`  
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.PreRequest`, `flowcontrol.DispatchGate`  
**Stability:** Alpha

Classifies a request as a prefix-cache hit or miss before admission, and optionally
holds ready misses in the flow-control queue when the pool is saturated so hits can
still dispatch.

The plugin does not change customer priority. Hits and misses stay in the same
priority band as sibling fairness flows when this plugin is also configured as
`flowControl.dispatchGatePluginRef`.

## What It Does

1. **`RequestHeader`** — tokenizer-free prefix hash of the prompt, lookup in an
   EPP-local pool-wide LRU, attach a `CacheClass` attribute.
2. **`PreRequest`** — record the request's prefix blocks in the index only after
   admit and route, so shed traffic cannot warm the index.
3. **`Allow` (DispatchGate)** — when saturation is at or above `shedThreshold`,
   keep a ready miss queued; hits and not-ready classes dispatch.

Until the index holds `warmupBlocks` entries, classes are not `Ready`. The gate
ignores them so a cold router does not hold everything as a miss.

## Configuration

Register the plugin, list it under request-header processors, and optionally set
it as the dispatch gate:

```yaml
plugins:
  - name: cache-class
    type: cache-class-router
    parameters:
      shedThreshold: 0.85
      hitMatchRatio: 0.3
      minMatchedTokens: 1024
      warmupBlocks: 5000
      indexCapacity: 100000

schedulingProfiles: ... # unchanged

# request header processors include cache-class (name must match)
# flowControl:
#   dispatchGatePluginRef: cache-class
```

### Parameters

| Field | Default | Meaning |
|---|---|---|
| `blockSizeTokens` | 64 | Pseudo-token block size for hashing |
| `maxPrefixBlocks` | 256 | Cap on blocks hashed per request |
| `indexCapacity` | 100000 | Pool-wide LRU capacity (blocks) |
| `hitMatchRatio` | 0.3 | Match fraction for a hit |
| `minMatchedTokens` | 1024 | Floor on matched tokens for a hit; `0` disables |
| `shedThreshold` | 0.85 | Saturation at or above which ready misses stay queued |
| `warmupBlocks` | 5000 | Indexed blocks required before a class is Ready; `0` disables |

## Hit Criteria

A request is a hit only when both are true:

- matched prefix blocks / total blocks ≥ `hitMatchRatio`
- matched tokens ≥ `minMatchedTokens`

Matched tokens are `matchedBlocks * blockSizeTokens` (pseudo-tokens).

## Index Lifetime

The index is an LRU of prefix block hashes with capacity `indexCapacity`. There
is no time-based TTL on index entries. Pending classifications that never reach
`PreRequest` (shed or cancelled) are swept after 2 minutes for turned-away
metrics.
