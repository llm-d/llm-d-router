# Performance on the request path

## Avoid heap allocations and unhoisted work on the request path
Avoid unnecessary byte-to-string conversions, slice or map clones, untyped map unmarshaling, and unmutated payload re-serialization on per-request routing, scoring, and event-decoding paths. Hoist header normalization, lookup maps, and type checks to startup, pre-allocate known-length slices, and combine consecutive read-locked lookups into one pass.

## Reuse pooled HTTP transports and multiplex per-endpoint event subscriptions
Reuse HTTP clients with keep-alive connection pooling across backend calls and metric scrapes, rebuilding transports only on TLS certificate rotation. Multiplex per-endpoint ZMQ or event streams across in-process consumers instead of opening duplicate sockets per plugin, and batch background sweeps off the request path.

## Evaluate orthogonal load signals over filtered candidate subsets
Evaluate queue depth, in-flight requests, and KV-cache utilization as distinct signals over the filtered candidate subset, and decrement every in-flight counter on cancellation or error.
- Accept: Constant-time byte-length token estimation and single-endpoint soft-pinning session affinity when backed by engine contracts or multi-pod benchmarks.

## Score prefix caches contiguously and break near-ties by load
Score prefix caches contiguously from the start of each sequence by stopping block credit at the first missing block, salt cache keys by tenant, and break near-ties with load-weighted sampling.

## Gate routing policies and hot-path changes on reproducible before-and-after benchmarks
Back new scoring formulas, cache indexes, hot-path locks, and default threshold changes with reproducible hit-and-miss microbenchmarks and cluster comparisons against active routing baselines. Commit benchmark scripts and summary reports while excluding raw execution logs from git.
