# Concurrency Attributes

This package defines the data structures for tracking real-time concurrency and load on endpoints.

## `InFlightLoad`

Captures the current real-time load of an endpoint as tracked by the EPP.

- **Key**: `InFlightLoadDataKey`
- **Fields**:
  - `Tokens`: Number of tokens currently in-flight.
  - `NonTextTokens`: The multimodal placeholder (image, audio, video) part of `Tokens`.
  - `Requests`: Number of requests currently in-flight.
- **Methods**:
  - `TokenLoad()`: `Tokens` split into a `TokenLoad`.

`Tokens` and `NonTextTokens` are separate `int64` fields because the Redis cross-replica syncer gob-encodes this struct, and replicas running different EPP versions must decode each other's values during a rolling upgrade.

### `TokenLoad`

A token count split into its two disjoint halves, so a consumer can price modalities differently.

- `Text`: The text-token part of the count.
- `NonText`: The multimodal placeholder (image, audio, video) part of the count.
- `Total()`: Both halves together.

## Producers

The following plugins produce this attribute:

- **`inflight-load-producer`** (Request Control): Tracks real-time token and request counters as they are dispatched and completed by the EPP.
