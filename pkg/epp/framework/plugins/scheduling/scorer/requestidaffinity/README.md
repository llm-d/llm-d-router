# Request-ID Affinity Scorer

**Type:** `request-id-affinity-scorer`

Routes requests to a deterministic endpoint bucket derived from the numeric suffix of a request header (default `x-request-id`). Endpoints are sorted alphabetically by name for stable bucket assignment, then:

```
bucket = numericSuffix % modulus
targetIndex = bucket % len(endpoints)
```

The endpoint at `targetIndex` receives score `1.0`; all others receive `0.0`.

This is useful for prefill-fanout workloads where you want requests whose IDs end in `0, 8, 16, ...` to always reach instance 0, IDs ending in `1, 9, 17, ...` to reach instance 1, and so on.

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headerName` | string | `x-request-id` | Request header whose numeric suffix drives bucket assignment. |
| `modulus` | int | `0` | Divisor for `bucket = numericSuffix % modulus`. When `0`, `len(endpoints)` is used so every candidate endpoint gets exactly one bucket. Set to a fixed value (e.g. `8`) when the candidate-endpoint count varies but the bucket mapping must stay stable. |

## Bucket assignment

Endpoints are sorted by `namespace/name` before indexing, giving a stable and deterministic assignment regardless of the order in which endpoints arrive from the data layer. When `modulus > len(endpoints)`, `targetIndex = bucket % len(endpoints)` wraps the assignment so no endpoint is ever skipped.

When the header is absent or its value carries no numeric suffix, all endpoints score `0.0` (no preference).

## Example configuration

```yaml
# 8 prefill instances, fixed modulus keeps mapping stable even if one pod
# temporarily disappears from the candidate list.
- name: request-id-affinity
  type: request-id-affinity-scorer
  parameters:
    headerName: x-request-id
    modulus: 8
```

With this configuration a request carrying `x-request-id: req-17` computes
`17 % 8 = 1` and is directed to the alphabetically second prefill endpoint.
