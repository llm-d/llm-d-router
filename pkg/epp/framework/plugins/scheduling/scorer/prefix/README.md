# Prefix Cache Scorer Plugin

**Type:** `prefix-cache-scorer`

Scores candidate endpoints using `PrefixCacheMatchInfo` prepared earlier in the request pipeline. By default, the score is the ratio of matched blocks to total blocks. Optionally, it can also factor in the absolute length of the prefix.

## What it does

For each candidate endpoint, the scorer reads the `PrefixCacheMatchInfo` attribute and computes a score.

The score is computed as:

```text
score = 0
score += prefixLengthWeight * matchLengthRatio
score += (1.0 - prefixLengthWeight) * matchRatio
```

Where:
- `matchRatio = matchBlocks / totalBlocks` (the fraction of the request prefix matched)
- `matchLengthRatio = min(1.0, matchedBlocks * blockSize / prefillSaturationTokens) ^ 2` (square of normalized matched tokens relative to `prefillSaturationTokens`)

If `prefixLengthWeight` is `0.0` (the default), the score simplifies to just the `matchRatio`.

If the attribute is missing, has the wrong type, or `totalBlocks` is zero, the endpoint receives score `0`.

The square introduces a non-linearity motivated by the fact that attention computation grows quadratically as a function of prompt length. This ensures that the scoring function assigns a higher priority to longer requests, where the computational and latency savings of KV-cache reuse are significantly more critical.

## Inputs consumed

This scorer consumes:

- `PrefixCacheMatchInfo`

The attribute is typically produced by the approximate prefix cache data producer before scheduling.

## Configuration

This plugin supports the following optional configuration parameters:

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `prefixLengthWeight` | float | Weight of the absolute prefix length in the score. Must be between `0.0` and `1.0`. | `0.0` |
| `prefillSaturationTokens` | integer | The number of tokens at which the value of KV cache saturates. Required if `prefixLengthWeight` > `0.0`. | `8192` |

Example configuration:

```yaml
plugins:
- type: prefix-cache-scorer
  parameters:
    prefixLengthWeight: 0.5
    prefillSaturationTokens: 16384
```

## Operational notes

- The scorer itself does not hash prompts or maintain cache state.
- It only converts previously prepared prefix match data into endpoint scores.
- To be useful, it should be used together with a data producer that populates `PrefixCacheMatchInfo`.
- `prefillSaturationTokens` should be tuned to match a specific workload. The general recommendation is to set this value to the P95 prompt length of the workload.
- Do not set `prefillSaturationTokens` to the model's maximum context limit if the actual workload lengths are much lower.
