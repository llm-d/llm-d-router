# Concurrency Attributes

This package defines the data structures for tracking real-time concurrency and load on endpoints.

## `InFlightLoad`

Captures the current real-time load of an endpoint as tracked by the EPP.

- **Key**: `InFlightLoadDataKey`
- **Fields**:
  - `Tokens`: Number of tokens currently in-flight.
  - `Requests`: Number of requests currently in-flight.
  - `CompletionsPerSecond`: Requests per second that finished on this endpoint, over the last 30 s.

## Producers

The following plugins produce this attribute:

- **`inflight-load-producer`** (Request Control): Tracks real-time token and request counters as they are dispatched and completed by the EPP.
