# cache-class-router

**Stability:** Alpha  
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.PreRequest`  
**Type:** `cache-class-router`

Tokenizer-free prefix classification before admission. Attaches a `CacheClass` request
attribute (hit/miss, match ratio, matched tokens, ready) for metrics and downstream
consumers. Does not change customer priority, fairness identity, or flow-control dispatch.

## Behavior

1. **`RequestHeader`** — byte-packs the prompt into pseudo-tokens, hashes leading blocks, and
   looks them up in a pool-wide LRU index. Writes `CacheClass` and stages hashes in a pending
   map keyed by request ID.
2. **`PreRequest`** — on admit, claims the pending entry and indexes the hashes so later
   requests with the same prefix can classify as hits.
3. **Warmup** — until the index holds `warmupBlocks`, `Ready` is false so consumers can treat
   the class as unclassified on a cold router.
4. **Metrics** — classification counts, match-ratio histogram, index size, and turned-away
   tokens for requests that never reach `PreRequest` (pending TTL sweep).

## Config

```yaml
plugins:
  - name: cache-class
    type: cache-class-router
    parameters:
      hitMatchRatio: 0.3
      minMatchedTokens: 1024
      warmupBlocks: 64
      indexCapacity: 100000
```

| Parameter | Default | Notes |
|---|---|---|
| `blockSizeTokens` | `64` | Pseudo-token block size |
| `maxPrefixBlocks` | `256` | Cap blocks hashed per request |
| `indexCapacity` | `100000` | LRU capacity |
| `hitMatchRatio` | `0.3` | Match fraction for hit |
| `minMatchedTokens` | `1024` | Floor below which ratio alone is not a hit |
| `warmupBlocks` | `5000` | Index size before `Ready`; `0` disables |

## Limits

- A hit means this EPP replica recently admitted a matching prefix. Entries do not expire with
  endpoint lifetime, and another EPP replica has a different view.
- The class is fixed at arrival for the life of the queued request.
- Pair class labels with response `cached_tokens` to measure accuracy before using the signal
  for capacity or sharing decisions.
