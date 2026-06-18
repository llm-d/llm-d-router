# Multimodal Embeddings Cache Producer Plugin

**Type:** `mm-embeddings-cache-producer`

Produces multimodal embeddings cache match data for downstream scheduling plugins.

## What It Does

For each request, the producer extracts stable multimodal item hashes from:

- `TokenizedPrompt.MultiModalFeatures`, when tokenized multimodal metadata is
  available on the request
- typed OpenAI chat-completions structured media blocks, as a lightweight fallback
- `Generate.Features.MMHashes`, when present

It keeps an in-memory LRU map from multimodal hash to the set of pods that recently
handled that item. During scheduling, it attaches `EncoderCacheMatchInfo` to each
endpoint so scorers can prefer pods that are likely to have already processed the
same image, video, or audio input.

Repeated references to the same multimodal hash within one request count once.

## Item Weights

Each matched multimodal item contributes to encoder-cache affinity. The scorer
computes `matchedWeight / totalWeight`; this producer defines the per-item
weight in that ratio.

- When tokenized multimodal metadata is available, each item weight is
  `MultiModalFeature.Length` (falling back to `1` when length is zero).
- Without tokenized multimodal metadata, each unique multimodal hash has item
  weight `1`.

## Inputs Consumed

This producer declares `TokenizedPrompt` from `token-producer` as an optional
dependency. If `token-producer` is configured, this producer runs after it and
uses `TokenizedPrompt.MultiModalFeatures` placeholder lengths. If tokenized
prompt data is absent at runtime, the producer falls back to typed structured
chat-completions media blocks with item weight `1` per hash.

## Data Produced

This plugin produces:

- `MultiModalEncoderCacheMatchInfoKey` (`EncoderCacheMatchInfo`)

## Configuration

The producer supports the following runtime parameters:

- `cacheSizeInMBPerServer` (integer, default: `2048`, 2 GiB): per-endpoint memory budget in
  mebibytes (MiB) for the best-effort pod-affinity LRU.

**Lightweight configuration example (tokenizer-free chat media):**

```yaml
plugins:
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSizeInMBPerServer: 2048
  - type: mm-embeddings-cache-scorer
schedulingProfiles:
  - name: encoder-cache-aware
    plugins:
      - pluginRef: mm-embeddings-cache-scorer
        weight: 4
      - pluginRef: kv-cache-utilization-scorer
        weight: 2
      - pluginRef: queue-scorer
        weight: 2
```

**Tokenized multimodal weight example:**

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: Qwen/Qwen2.5-1.5B-Instruct
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

- The cache is a best-effort routing signal, not a correctness dependency.
- `cacheSizeInMBPerServer` bounds the EPP routing LRU only; it is not a model-server
  encoder cache capacity knob.
- Structured chat media blocks are enough for lightweight cache affinity;
  `token-producer` is not required.
- Configure `token-producer` when multimodal placeholder lengths should
  influence affinity scores.
