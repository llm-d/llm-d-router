# Prefix Cache Scorer Plugin

**Type:** `prefix-cache-scorer`

Scores candidate endpoints using `PrefixCacheMatchInfo` prepared earlier in the request pipeline.

## What it does

For each candidate endpoint, the scorer reads the `PrefixCacheMatchInfo` attribute and computes a score. By default, this is the ratio of matched blocks to total blocks. Optionally, it can also factor in the absolute length of the prefix.

The score is computed as:

```text
score = prefixLengthWeight * matchLengthRatio + (1.0 - prefixLengthWeight) * matchRatio
```

Where:
- `matchRatio = matchBlocks / totalBlocks` (the fraction of the request prefix matched)
- `matchLengthRatio = min(1.0, (totalBlocks * blockSize / maxModelLen) ^ 2)` (normalized square of the total request prefix length relative to `maxModelLen`)

If `prefixLengthWeight` is `0.0` (the default), the score simplifies to just the `matchRatio`.

If the attribute is missing, has the wrong type, or `totalBlocks` is zero, the endpoint receives score `0`.

## Inputs consumed

This scorer consumes:

- `PrefixCacheMatchInfo`

The attribute is typically produced by the approximate prefix cache data producer before scheduling.

## Configuration

This plugin supports the following configuration parameters:

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `prefixLengthWeight` | float | Weight of the absolute prefix length in the score. Must be between `0.0` and `1.0`. | `0.0` |
| `maxModelLen` | integer | The maximum context length (in tokens) supported by the model. Required if `prefixLengthWeight` > `0.0`. | `8192` |

Example configuration:

```yaml
plugins:
- type: prefix-cache-scorer
  parameters:
    prefixLengthWeight: 0.5
    maxModelLen: 16384
```

## Operational notes

- The scorer itself does not hash prompts or maintain cache state.
- It only converts previously prepared prefix match data into endpoint scores.
- To be useful, it should be used together with a data producer that populates `PrefixCacheMatchInfo`.
