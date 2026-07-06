# GPU Utilization Filter

**Type:** `gpu-utilization-filter`

Filters out endpoints whose GPU compute utilization exceeds a configurable threshold. Endpoints without GPU data pass through unconditionally.

## Behavior

- Endpoints with GPU utilization **at or below** the threshold pass.
- Endpoints **without GPU data** pass (conservative: DCGM may not have scraped yet).
- If **all** endpoints with data exceed the threshold, all original endpoints are returned (graceful fallback).

## Configuration

- `threshold` (float64, optional, default: `0.90`): Maximum GPU utilization in [0.0, 1.0].

```yaml
- type: gpu-utilization-filter
  name: gpu-filter
  parameters:
    threshold: 0.85
```

## Data dependency

Declares a **Required** `Consumes` dependency on `GPUUtilization` (`GPUUtilization/dcgm-extractor`). The framework validates at init that a matching producer is configured.
