# Weighted Random Picker

**Type:** `weighted-random-picker`

Selects positive-score endpoints with weights proportional to `score^(1/temperature)` and optional exploration of zero-score candidates. The default temperature of 1.0 gives weights proportional to score.

## What it does

1.  Receives a list of `ScoredEndpoint` candidates.
2.  If all candidates have a score of zero or less, it delegates to the `random-picker` for uniform selection.
3.  Uses the **A-Res (Algorithm for Reservoir Sampling)** algorithm to perform mathematically correct weighted random sampling.
    - Generates a random key for each endpoint based on its score: $key_i = U_i^{(1/w_i)}$ where $U_i$ is a random number in $(0,1)$ and $w_i$ is the endpoint's effective weight, $w_i = Score_i^{1/T}$ ($T$ is the optional temperature setting; at its default of $T=1$, $w_i$ is the score itself).
    - Selects the candidates with the largest keys.
4.  With `explorationBudget: 0`, returns the top `maxNumOfEndpoints` candidates. Only candidates with a
    positive score are eligible: zero and negative-score endpoints are never
    returned, including as lower-priority fallback entries when
    `maxNumOfEndpoints` exceeds the number of positive-score candidates.

## Behavioral Intent

This picker resolves the trade-off between `max-score-picker` and `random-picker`. It prefers higher-scoring endpoints while maintaining exploration and avoiding extreme hot-spotting. It is ideal when paired with scorers that produce continuous scores (e.g., Token Load or Latency).

With `explorationBudget > 0`, zero-score endpoints can be selected.
Endpoints that must never be selected must be removed by an appropriate filter.
For example, when `latency-scorer` applies its idle preference, busy endpoints receive zero
scores while idle endpoints are available. Exploration can select those busy endpoints.

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results and utilizes the `Score` value.

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.
- `temperature` (default 1.0)
  - Tunes how sharply selection favors higher scores. Selection probability is proportional to $Score_i^{1/T}$, equivalent to applying softmax temperature to the log-scores. Must be > 0; a non-positive value is rejected and the picker fails to load. The implementation normalizes scores by the maximum and computes the sampling keys in log space, which preserves these probabilities and keeps the keys finite and distinct at low temperatures.
    - `T = 1`: probability proportional to score (default).
    - `T < 1`: sharpens toward the highest-scoring endpoint; as $T \to 0$ it approaches `max-score-picker`.
    - `T > 1`: flattens toward uniform selection; as $T \to \infty$ it approaches `random-picker`.

- `explorationBudget` (default 0)
  - A finite probability in `[0,1]`. Invalid values prevent the plugin from loading.
  - When positive-score and zero-score candidates both exist, one exploration decision is made per call. With this probability, one uniformly chosen zero-score endpoint is placed first and the remaining slots use weighted positive-score sampling without replacement. Otherwise, all returned endpoints use weighted positive-score sampling. `temperature` controls the positive-score weighting in both cases.
  - For any `maxNumOfEndpoints`, the probability of including a zero-score endpoint and the probability of placing one first both equal the budget. Exactly one zero-score endpoint is returned on exploration; the probability does not grow with the zero-score group or list size. Negative scores remain excluded when positive scores exist. The all-non-positive random fallback is unaffected.
  - The first endpoint is the primary target and later entries are fallback candidates. Exploration changes the primary target intentionally. Consumers may sample the list or retry the selected host, so the fraction of inference attempts sent to zero-score endpoints can differ from the budget. Returned lists can be shorter than `maxNumOfEndpoints`.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
