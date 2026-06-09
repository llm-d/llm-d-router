# Multimodal Embeddings Cache Scorer Plugin

**Type:** `mm-embeddings-cache-scorer`

Scores candidate endpoints using multimodal embeddings cache match data produced
by `mm-embeddings-cache-producer`.

## What It Does

For each candidate endpoint, the scorer reads `EncoderCacheMatchInfo` and computes:

```text
score = matchedItemSize / totalRequestItemSize
```

When tokenized multimodal metadata is available, item weights come from
multimodal placeholder lengths, so larger multimodal inputs contribute more to
the score. Otherwise, every unique multimodal item has item weight `1`, so the
score is the fraction of unique request multimodal hashes that are likely cached
on the endpoint.

This produces a normalized score in the range `[0, 1]`:

- higher score: more request multimodal content is expected to reuse endpoint-local
  embeddings cache
- lower score: less multimodal cache reuse is expected

If the attribute is missing, has the wrong type, or total request item size is zero,
the endpoint receives score `0`.

## Inputs Consumed

This scorer consumes:

- `MultiModalEncoderCacheMatchInfoKey` (`EncoderCacheMatchInfo`)

The attribute is produced by `mm-embeddings-cache-producer` before scheduling.

## Configuration

- `producerName` (string, optional): scopes the consumed data key to a named
  producer instance. Leave empty to consume from the default producer.

Weighting is controlled by the match data emitted by the producer; the scorer
remains size-agnostic.

**Unweighted configuration example:**

```yaml
plugins:
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSizeInMBPerServer: 2048
  - type: mm-embeddings-cache-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: mm-embeddings-cache-scorer
        weight: 1
      - pluginRef: max-score-picker
```

**Tokenized multimodal weight example:**

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: hf-repo/model-name
      vllm:
        url: http://localhost:8000
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSizeInMBPerServer: 2048
  - type: mm-embeddings-cache-scorer
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: mm-embeddings-cache-scorer
        weight: 4
```

## Operational Notes

- The scorer does not hash request media and does not maintain cache state.
- It only converts producer-generated match data into endpoint scores.
- KV-prefix cache affinity remains owned by `precise-prefix-cache-producer`.
