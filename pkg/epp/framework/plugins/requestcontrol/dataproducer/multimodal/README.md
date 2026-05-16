# Multimodal Embeddings Cache Producer Plugin

**Type:** `mm-embeddings-cache-producer`

Produces multimodal embeddings cache match data for downstream scheduling plugins.

## What It Does

For each request, the producer extracts stable multimodal item hashes from:

- `TokenizedPrompt.MultiModalFeatures`, when a `token-producer` is configured
- typed OpenAI chat-completions structured media blocks, as a lightweight fallback

It keeps an in-memory LRU map from multimodal hash to the set of pods that recently
handled that item. During scheduling, it attaches `EncoderCacheMatchInfo` to each
endpoint so scorers can prefer pods that are likely to have already processed the
same image, video, or audio input.

Repeated references to the same multimodal hash within one request count once.

## Inputs Consumed

This plugin declares:

- `TokenizedPrompt`

When `token-producer` is present, this orders tokenization before multimodal match
data production. If tokenized prompt data is absent at runtime, the producer falls
back to typed structured chat-completions media blocks.

## Data Produced

This plugin produces:

- `MultiModalEncoderCacheMatchInfoKey` (`EncoderCacheMatchInfo`)

## Configuration

The producer supports the following runtime parameters:

- `cacheSize` (integer, default: `10000`): maximum number of multimodal hash entries
  retained in the best-effort pod-affinity cache.

**Configuration Example:**

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: Qwen/Qwen2.5-1.5B-Instruct
      vllm:
        http: http://localhost:8000
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSize: 10000
  - type: mm-embeddings-cache-scorer
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: mm-embeddings-cache-scorer
        weight: 4
```

## Operational Notes

- The cache is a best-effort routing signal, not a correctness dependency.
- Endpoint delete events can remove stale pod entries when `endpoint-notification-source`
  is wired through `dataLayer`.
- The producer remains tokenizer-free for request shapes where typed media blocks are
  sufficient; `token-producer` is only required when relying on upstream multimodal
  metadata.
