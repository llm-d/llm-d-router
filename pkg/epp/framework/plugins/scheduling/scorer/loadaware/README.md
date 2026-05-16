# Load Aware Scorer

**Type:** `load-aware-scorer`
**Interfaces**: `scheduling.Scorer`
**Category**: Distribution

Scores pods based on their current load, measured by the number of requests waiting in each pod's queue, so the scheduler can route new traffic away from busy endpoints.

## What it does

For each candidate endpoint, the scorer reads the pod's current waiting-queue size and assigns a score in `[0, 0.5]`:

- Pods with an empty waiting queue score `0.5`.
- Pods with requests queued score between `0.5` and `0.0` — the more queued, the lower the score.
- Pods with queue depth at or beyond `threshold` score `0.0`.

## How It Works

The score is computed from the live `WaitingQueueSize` metric using a piecewise linear function:

```
waitingRequests == 0                     → score = 0.5
waitingRequests >= threshold             → score = 0.0
0 < waitingRequests < threshold          → score = 0.5 * (1 - waitingRequests / threshold)
```

Note that the maximum score is capped at `0.5`, not `1.0`. This reflects that an empty queue is the best *observable* signal of availability given current metrics, but does not necessarily indicate spare capacity. A future extension could raise idle pods above `0.5` once capacity-headroom information becomes available.

## Inputs consumed

- `WaitingQueueSize` — live pod metric read from `endpoint.GetMetrics()`, populated by the metrics data pipeline (`metrics-data-source` + `core-metrics-extractor`).

## Configuration

**Location:** `schedulingProfiles[N].plugins`

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `threshold` | `int` | No | `128` | Queue depth at which the score reaches `0.0`. Non-positive values are rejected and replaced by the default. |

### Example

```yaml
plugins:
  - type: load-aware-scorer
    name: load-balancer
    config:
      threshold: 100

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: load-balancer
        weight: 5
```

## Limitations

- Its maximum of 0.5 (vs. the usual 1.0 ceiling of other scorers) means its effective pull under weighted aggregation is roughly half its configured weight — raise weight to compensate.
- Treats all queued requests equally regardless of size or remaining work — a queue of one long request scores the same as a queue of one short request.
- Depends on accurate, fresh `WaitingQueueSize` metrics; stale or missing metrics make the scorer degenerate (all pods tied at `0.5`).

## Related Documentation

- [Active Request Scorer](../activerequest/README.md) — alternative distribution scorer based on in-flight request counts instead of queue depth