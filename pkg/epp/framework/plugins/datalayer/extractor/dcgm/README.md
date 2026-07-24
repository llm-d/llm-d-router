# DCGM Extractor

**Type:** `dcgm-extractor`

The DCGM Extractor converts the Prometheus metrics response from a
`dcgm-data-source` into a per-endpoint GPU utilization attribute consumed by
GPU-aware filters and scorers.

## What it does

1. Receives the parsed Prometheus metric families forwarded by `dcgm-data-source`.
2. Looks up the `DCGM_FI_DEV_GPU_UTIL` metric family.
3. Keeps samples that belong to this endpoint:
   - If a sample has a `pod` label and the endpoint has a `PodName`, only
     matching samples are kept (DaemonSet multi-pod payloads).
   - If the sample has no `pod` label, it is kept (sidecar / unlabeled payloads).
4. Aggregates matching samples using `max` (highest-utilized GPU for the pod).
5. Normalizes the value from 0-100 to [0.0, 1.0].
6. Stores the result as a `GPUUtilization` attribute on the corresponding endpoint.

## Attributes produced

- `GPUUtilization` stored at attribute key `GPUUtilizationDataKey`
  (`GPUUtilization/dcgm-extractor`) on each endpoint.

```go
key := attrgpu.GPUUtilizationDataKey.String()
util, ok := attrgpu.ReadGPUUtilization(endpoint.GetAttributes(), key)
```

## Configuration

No configuration parameters.
