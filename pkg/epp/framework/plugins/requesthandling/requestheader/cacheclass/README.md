# cache-classifier

**Stability:** Alpha  
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `requesthandling.PreRequest`,
`requesthandling.RequestAbandoned`, `requesthandling.ResponseBodyProcessor`  
**Type:** `cache-classifier`

Tokenizer-free prefix classification before admission. Attaches a `CacheClass` request
attribute (hit/miss, match ratio, matched tokens, ready) for metrics and downstream
consumers. Does not change customer priority, fairness identity, or flow-control dispatch.

The admission-time index is **pool-wide on this EPP replica** and **separate from** the
approximate/precise prefix indexes used for placement scoring.

## Behavior

1. **`RequestHeader`** — byte-packs the prompt into pseudo-tokens, hashes leading blocks, and
   publishes `CacheClass` as a `DynamicRequestAttribute` so a request still in the FC queue
   observes the current index on every read. Stages hashes until PreRequest or abandon.
2. **`PreRequest`** — indexes hashes for admitted requests and records `MatchedTokens` at
   dispatch for later accuracy comparison.
3. **`RequestAbandoned`** — drops pending state when the request never reaches PreRequest
   (flow-control rejection, admission deny, etc.) and counts turned-away metrics.
4. **`ResponseBody`** — compares dispatch `MatchedTokens` with response `cached_tokens`.
5. **Warmup** — until the index holds `warmupBlocks`, `Ready` is false.

## Config

```yaml
plugins:
  - name: cache-classifier
    type: cache-classifier
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
| `indexCapacity` | `100000` | LRU capacity on this replica |
| `hitMatchRatio` | `0.3` | Match fraction for hit |
| `minMatchedTokens` | `1024` | Floor below which ratio alone is not a hit |
| `warmupBlocks` | `5000` | Index size before `Ready`; `0` disables |

## Limits

- A hit means this EPP replica recently admitted a matching prefix. Entries are not removed when
  an endpoint leaves, and another EPP replica has a different view.
- Reads use LRU `Peek` so classification does not refresh recent-ness; only admit (`Add`) does.
- Pair `cache_class_matched_vs_cached_tokens` with production traffic before using the signal for
  capacity or sharing decisions.
