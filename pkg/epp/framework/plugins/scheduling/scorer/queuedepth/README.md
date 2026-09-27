# Queue Depth Scorer Plugin

**Type:** `queue-scorer`

This plugin scores candidate endpoints by current waiting-queue depth.


## What it does

For each scheduling cycle, the plugin reads `WaitingQueueSize` from endpoint metrics and computes a normalized score:

\[
\text{score(endpoint)} = \frac{\maxQueue - \text{queue(endpoint)}}{\maxQueue - \minQueue}
\]

So:

- shortest queue gets score `1.0`
- longest queue gets score `0.0`
- others are linearly scaled between them

If all endpoints with written metrics have the same queue size (`maxQueue = minQueue`), those endpoints receive a neutral score of `1.0`.

Endpoints with no written metrics (nil metrics, or a zero `UpdateTime`) are left unscored and do not participate in the min-max range.
The scheduler treats an omitted score as a zero contribution from this plugin.
An endpoint that has reported a waiting-queue size of 0 still scores as the shortest queue.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests away from endpoints with deeper backlogs.

## Inputs consumed

The plugin consumes:

- `metrics.WaitingQueueSizeKey` (`int`)

## Configuration

This scorer currently has no runtime parameters.

**Configuration Example:**
```yaml
plugins:
  - type: queue-scorer
    name: queue-depth
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-depth
        weight: 1
```

## Multi-cluster support

`multicluster-queue-scorer` is the cluster-scoped variant. The stock per-pod queue field is empty on a cluster-endpoint, so it scores on the `llm-d.ai/multicluster-queue-size` attribute. The `multicluster-metrics-data-source` scrapes the peer cluster and the `multicluster-metrics-extractor` writes that attribute from the pool's aggregate queue metric. Configure both in `dataLayer`, otherwise this scorer sees no pool metric and returns no score. See the [wiring example](../../../README.md#example).
