# KV-Cache Metrics

Prometheus collectors for the KV-cache subsystem: block-index admissions,
evictions, and lookups, and the KV-event dedup filter's suppressed/forwarded
removals.

## What It Does

Defines the metric collectors and registers them with the router's metrics
registry. The [`kvblock`](../kvblock/README.md) instrumented index and the
[`kvevents`](../../kvevents/README.md) dedup filter record into them.

## Naming

Metrics are emitted under the router's standard `llm_d_epp` subsystem
(for example `llm_d_epp_kv_cache_index_lookup_hits_total`).

## Usage

| Symbol | Role |
|--------|------|
| `Register` | Registers all collectors with the router metrics registry (once). |
| `Collectors` | Returns the collectors, e.g. for a custom registry. |
| `StartMetricsLogging` | Optionally logs current metric values on an interval. |

## Related Documentation

- [KV-Block Index](../kvblock/README.md)
- [KV-Events](../../kvevents/README.md)
