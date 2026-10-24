# Weighted Random Picker

**Type:** `weighted-random-picker`

Selects endpoint(s) randomly, where the probability of an endpoint being selected is proportional to its score.

## What it does

1.  Receives a list of `ScoredEndpoint` candidates.
2.  If all candidates have a score of zero or less, it delegates to the `random-picker` for uniform selection.
3.  Uses the **A-Res (Algorithm for Reservoir Sampling)** algorithm to perform mathematically correct weighted random sampling.
    - Generates a random key for each endpoint based on its score: $key_i = U_i^{(1/w_i)}$ where $U_i$ is a random number in $(0,1)$ and $w_i$ is the endpoint's effective weight, $w_i = Score_i^{1/T}$ ($T$ is the optional temperature setting; at its default of $T=1$, $w_i$ is the score itself).
    - Selects the candidates with the largest keys.
4.  Returns the top `maxNumOfEndpoints` candidates.

## Behavioral Intent

This picker resolves the trade-off between `max-score-picker` and `random-picker`. It prefers higher-scoring endpoints while maintaining exploration and avoiding extreme hot-spotting. It is ideal when paired with scorers that produce continuous scores (e.g., Token Load or Latency).

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results and utilizes the `Score` value.

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.
- `temperature` (default 1.0)
  - Tunes how sharply selection favors higher scores. Selection probability is proportional to $Score_i^{1/T}$, equivalent to applying softmax temperature to the log-scores. Must be > 0; non-positive values fall back to the default.
    - `T = 1`: probability proportional to score (default).
    - `T < 1`: sharpens toward the highest-scoring endpoint; as $T \to 0$ it approaches `max-score-picker`.
    - `T > 1`: flattens toward uniform selection; as $T \to \infty$ it approaches `random-picker`.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
