# Prefix Cache Scorer Plugin

**Type:** `prefix-cache-scorer`

Scores candidate endpoints based on how much of the request's prefix is already cached. It can consider both the *ratio* of the prefix matched and the *absolute length* of the match.

## What it does

The scorer computes a score between `0.0` and `1.0` for each endpoint using a weighted combination of two metrics:

1.  **Match Ratio**: The proportion of the request prefix that is cached.
2.  **Match Length**: The absolute number of cached tokens, normalized against a scale factor.

### Scoring Formula

```text
score = 0
score += matchLengthWeight * matchLengthRatio
score += (1.0 - matchLengthWeight) * matchRatio
```

Where:
*   **`matchRatio`** = `matchBlocks` / `totalBlocks`
*   **`matchLengthRatio`** = `min(1.0, matchedTokens / matchLengthScaleTokens) ^ 2`
    *   `matchedTokens` = `matchBlocks` * `blockSize`
    *   **`matchLengthScaleTokens`** is the scaling factor.

If `matchLengthWeight` is `0.0` (default), the score simplifies to just the `matchRatio`.

If the attribute is missing, has the wrong type, or `totalBlocks` is zero, the endpoint receives score `0`.

### Why the Quadratic Term?

The squaring of `matchLengthRatio` introduces a non-linearity motivated by the fact that attention computation grows quadratically as a function of prompt length. This ensures that the scoring function assigns a higher priority to longer requests, where the computational and latency savings of KV-cache reuse are significantly more critical.
## Inputs Consumed

- `PrefixCacheMatchInfo`

The attribute is typically produced by the approximate prefix cache data producer before scheduling.

## Configuration

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `matchLengthWeight` | float | Weight of the absolute prefix length in the score. Must be between `0.0` and `1.0`. | `0.0` |
| `matchLengthScaleTokens` | integer | The number of tokens used to normalize `matchLengthRatio`. | `8192` |

### Example

```yaml
plugins:
- type: prefix-cache-scorer
  parameters:
    matchLengthWeight: 0.5
    matchLengthScaleTokens: 16384
```

## Operational Notes

*   **Tuning `matchLengthScaleTokens`**: This should be set to reflect your workload. A good rule of thumb is to set it to the **P95 prompt length** of your typical requests. Setting it too high (e.g., to the model's maximum limit when actual requests are short) will compress the score range and make the scorer less effective.
*   The scorer is stateless; it does not manage cache state or hash prompts itself. It relies entirely on the data producer.
