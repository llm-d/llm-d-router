# Concurrency Attributes

This package defines the data structures for tracking real-time concurrency and load on endpoints.

## `InFlightLoad`

Captures the current real-time load of an endpoint as tracked by the EPP.

- **Key**: `InFlightLoadDataKey`
- **Fields**:
  - `Tokens`: Number of tokens currently in-flight.
  - `Requests`: Number of requests currently in-flight.
  - `KVGrowthTokens`: Projected peak of the KV cache tokens the endpoint will need beyond its scraped usage. Zero unless the producer's `kvGrowthProjection` is configured.

## Producers

The following plugins produce this attribute:

- **`inflight-load-producer`** (Request Control): Tracks real-time token and request counters as they are dispatched and completed by the EPP.
