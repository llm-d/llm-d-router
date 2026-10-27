# KV Cache Utilization Scorer Plugin

**Type:** `kv-cache-utilization-scorer`

This plugin scores candidate endpoints using each endpoint's current KV-cache utilization.


## What it does

For each candidate endpoint, the plugin computes:

```
  {score(endpoint)} = 1 - {kvCacheUsagePercent}
```

Where `kvCacheUsagePercent` is read from endpoint metrics.

This means:

- lower KV-cache usage -> higher score
- higher KV-cache usage -> lower score

Endpoints with no written metrics (nil metrics, or a zero `UpdateTime`) are left unscored.
The scheduler treats an omitted score as a zero contribution from this plugin.
An endpoint that has reported KV-cache usage of 0 still scores 1.0.

## Scheduling intent

The scorer returns category `Distribution`, so it helps spread traffic away from endpoints with high KV-cache pressure.

## Inputs consumed

The plugin consumes:

- `metrics.KVCacheUsagePercentKey` (`float64`)

## Configuration

This scorer currently has no runtime parameters.

**Configuration Example:**
```yaml
plugins:
  - type: kv-cache-utilization-scorer
    name: kv-cache-util
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: kv-cache-util
        weight: 1
```

## Multi-cluster support

`multicluster-kv-cache-utilization-scorer` is the cluster-scoped variant. The stock per-pod KV field is empty on a cluster-endpoint, so it scores on the `llm-d.ai/multicluster-kv-cache-utilization` attribute. The `multicluster-metrics-data-source` scrapes the peer cluster and the `multicluster-metrics-extractor` writes that attribute from the pool's aggregate KV metric. Configure both in `dataLayer`, otherwise this scorer sees no pool metric and returns no score. See the [wiring example](../../../README.md#example).
