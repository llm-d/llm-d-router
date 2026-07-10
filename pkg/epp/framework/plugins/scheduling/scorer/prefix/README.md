# Prefix Cache Scorer Plugin

**Type:** `prefix-cache-scorer`

Scores candidate endpoints using `PrefixCacheMatchInfo` prepared earlier in the request pipeline.

## What it does

For each candidate endpoint, the scorer reads the `PrefixCacheMatchInfo` attribute and computes:

```text
score = matchBlocks / totalBlocks
```

This produces a normalized score in the range `[0, 1]`:

- higher score: more of the request prefix is expected to be reusable from cache
- lower score: less prefix cache reuse is expected

If the attribute is missing, has the wrong type, or `totalBlocks` is zero, the endpoint receives score `0`.

## Inputs consumed

This scorer consumes:

- `PrefixCacheMatchInfo`

The attribute is typically produced by the approximate prefix cache data producer before scheduling.

## Configuration

This plugin does not define any plugin-specific parameters.

### Agentic Workload Example

For agentic workloads, sessions typically consist of long-running, multi-turn interactions where the system prompt is shared but session-specific context diverges rapidly.

To optimize routing for these workloads, configure both the `prefix-cache-scorer` (with length awareness enabled) and the `session-affinity-scorer`. Length awareness allows shorter sessions to migrate freely when pods are overloaded, while session affinity keeps the rapidly diverging chat history sticky to the original pod:

```yaml
- type: prefix-cache-scorer
  parameters:
    matchLengthWeight: 0.8
    matchLengthScaleTokens: 16000

- type: session-affinity-scorer
  parameters:
    headerName: x-session-token
```

## Operational notes

- The scorer itself does not hash prompts or maintain cache state.
- It only converts previously prepared prefix match data into endpoint scores.
- To be useful, it should be used together with a data producer that populates `PrefixCacheMatchInfo`.
