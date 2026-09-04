# LMetric Scorer

**Type:** `lmetric-scorer`

This scorer ranks endpoints by the LMetric cost for precise prefix-cache routing:

```text
cost(endpoint) = PToken(endpoint) * BS(endpoint)
```

Where:

```text
PToken = (totalBlocks - matchBlocks) * blockSizeTokens
BS = inFlightRequests + 1
```

The endpoint with the lowest cost receives the highest score. Scores are normalized across candidates:

```text
score(endpoint) = (maxCost - cost(endpoint)) / (maxCost - minCost)
```

If all candidate costs are equal, every endpoint receives score `1.0`.

## Scheduling intent

LMetric combines prefix locality and current request load into one affinity score. It prefers endpoints that can reuse more cached prefix blocks and have fewer in-flight requests.

## Inputs consumed

The plugin consumes:

- `prefix.PrefixCacheMatchInfo`, produced by `precise-prefix-cache-producer`
- `concurrency.InFlightLoad`, produced by `inflight-load-producer`

## Configuration

```yaml
plugins:
  - type: inflight-load-producer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
  - type: precise-prefix-cache-producer
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 16
  - type: lmetric-scorer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
      inFlightLoadProducerName: inflight-load-producer

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: lmetric-scorer
        weight: 1
```
