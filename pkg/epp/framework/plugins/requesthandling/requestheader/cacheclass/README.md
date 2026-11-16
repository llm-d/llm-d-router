# Cache Class Router

**Type:** `cache-class-router`  
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.PreRequest`, `flowcontrol.DispatchGate`, `flowcontrol.DispatchPreference`, `flowcontrol.BandEnqueuePolicy`  
**Stability:** Alpha

Classifies a request as a prefix-cache hit or miss before admission, and optionally
holds ready misses in the flow-control queue when the pool is near saturation so
hits can still dispatch.

The plugin does not change customer priority. Hits and misses stay in the same
priority band as sibling fairness flows when this plugin is also configured as
`flowControl.dispatchGatePluginRef`.

## What It Does

1. **`RequestHeader`** — tokenizer-free prefix hash of the prompt, lookup in an
   EPP-local pool-wide LRU, attach a `CacheClass` attribute.
2. **`PreRequest`** — record the request's prefix blocks in the index only after
   admit and route, so shed traffic cannot warm the index.
3. **`missHold` (DispatchGate / DispatchPreference)** — `off` always allows.
   `prefer` (default) and `strict` keep a ready miss queued once saturation
   reaches `shedThreshold` until it falls below `releaseThreshold`. While
   latched, fairness picks hit-eligible queues first. `prefer` may fall back to
   the full band when no hit is waiting; `strict` does not.
4. **`EffectiveBandRequestCapacity` (BandEnqueuePolicy)** — under `strict`
   (or an explicit `hitQueueShare`), reserve band `maxRequests` slots for hits
   so held misses cannot fill the shared queue budget and starve hit enqueue.

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
      missHold: prefer
      hitMatchRatio: 0.3
      minMatchedTokens: 1024
      warmupBlocks: 5000
      indexCapacity: 100000

schedulingProfiles: ... # unchanged

# request header processors include cache-class (name must match)
# flowControl:
#   dispatchGatePluginRef: cache-class
#   defaultRequestTTL: 3s
```

`missHold: strict` holds ready misses until release with no empty-hit fallback,
and defaults `hitQueueShare` to `0.25` unless overridden.

### Parameters

| Field | Default | Meaning |
|---|---|---|
| `missHold` | `prefer` | `off`, `prefer`, or `strict` |
| `blockSizeTokens` | 64 | Pseudo-token block size for hashing |
| `maxPrefixBlocks` | 256 | Cap on blocks hashed per request |
| `indexCapacity` | 100000 | Pool-wide LRU capacity (blocks) |
| `hitMatchRatio` | 0.3 | Match fraction for a hit |
| `minMatchedTokens` | 1024 | Floor on matched tokens for a hit; `0` disables |
| `shedThreshold` | 0.80 | Saturation at or above which ready misses stay queued |
| `releaseThreshold` | min(0.70, shedThreshold) | Saturation below which a latched miss hold releases |
| `hitQueueShare` | `0` (`prefer`/`off`); `0.25` (`strict`) | Fraction of band `maxRequests` reserved for hits; `0` disables |
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

## Queue IDs

Ready hit/miss classes use sibling queue IDs (`parent + cacheclass suffix`) at
the same priority so a held miss does not block a hit. `ParentFlowID` strips the
suffix to recover the parent fairness ID for accounting.
