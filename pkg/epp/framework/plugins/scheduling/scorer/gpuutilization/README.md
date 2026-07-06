# GPU Utilization Scorer

**Type:** `gpu-utilization-scorer`

Scores endpoints inversely to their GPU compute utilization: `score = 1.0 - utilization`. Endpoints with lower GPU utilization receive higher scores, spreading load toward less-busy GPUs.

## Behavior

- **Category:** `Distribution`
- Endpoints **without GPU data** receive score `0` (prefer endpoints with known-low utilization).

## Configuration

No configuration parameters.

```yaml
- type: gpu-utilization-scorer
  name: gpu-scorer
```

## Data dependency

Declares a **Required** `Consumes` dependency on `GPUUtilization` (`GPUUtilization/dcgm-extractor`). The framework validates at init that a matching producer is configured.
