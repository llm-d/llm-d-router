# Prefix Cache Attributes

This package defines the data structures for tracking prefix cache hits and status on endpoints.

## `PrefixCacheMatchInfo`

Contains information about how much of a request's prefix matched the cache on a specific endpoint.

- **Key**: `PrefixCacheMatchInfoDataKey`
- **Fields**:
  - `MatchBlocks`: Number of blocks that matched the cache.
  - `TotalBlocks`: Total number of blocks in the request prefix.
  - `BlockSizeTokens`: Number of tokens per block. This is fixed across endpoints.

This information is used by affinity-based scheduling scorers to prefer endpoints with high cache hits.

## Producers

The following plugins produce this attribute:

- **`approx-prefix-cache-producer`** (Request Control): Estimates cache hit rates by comparing the request prefix against a local index of recently seen prefixes for each endpoint.

