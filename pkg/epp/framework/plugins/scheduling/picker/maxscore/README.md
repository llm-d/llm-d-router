# Max Score Picker

**Type:** `max-score-picker`

Selects the endpoint(s) with the highest score calculated during the scoring phase.

> [!NOTE]
> This plugin is enabled by default if no other picker is specified. You do not need to explicitly declare it in your configuration.

## What it does

1. Receives a list of `ScoredEndpoint` candidates.
2. Sorts candidates by score in descending order using stable sort.
3. Rotates candidates within each equal-score tier up to `maxNumOfEndpoints` using round-robin rotation for deterministic tie-breaking.
4. Returns the top `maxNumOfEndpoints` candidates.

## Behavioral Intent

This picker maximizes adherence to scoring objectives (such as cache affinity or lowest load). For candidates with equal scores, deterministic round-robin rotation distributes requests across tied endpoints to prevent traffic concentration on a single winner.

## Inputs consumed

- Consumes the list of `ScoredEndpoint` results from the scoring phase.

## Configuration

The plugin config supports:

- `maxNumOfEndpoints` (default 1)
  - The maximum number of endpoints to pick and return. Must be > 0. If more candidates are available than this limit, only the top subset is returned.

> [!TIP]
> In most production scenarios, `maxNumOfEndpoints` is left at its default value of `1` to select a single target endpoint for the request.
