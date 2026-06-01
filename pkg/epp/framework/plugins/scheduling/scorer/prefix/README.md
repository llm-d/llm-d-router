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

## Operational notes

- The scorer itself does not hash prompts or maintain cache state.
- It only converts previously prepared prefix match data into endpoint scores.
- To be useful, it should be used together with a data producer that populates `PrefixCacheMatchInfo`.
