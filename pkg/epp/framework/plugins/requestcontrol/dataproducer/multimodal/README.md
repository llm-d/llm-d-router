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

- `cacheSizeInMBPerServer` (integer, default: `4096`, 4 GiB): per-endpoint budget in
  mebibytes (MiB) for the best-effort pod-affinity LRU. Values `<= 0` or unspecified fall back
  to the default.

### Sizing `cacheSizeInMBPerServer`

This budget describes the **model server's** encoder cache, not the EPP's memory. The LRU
holds content encode hashes (a few tens of bytes each), so the MiB figure is used only to
derive how many entries to remember per endpoint:

```
entries per endpoint = cacheSizeInMBPerServer MiB / 2 MiB    (assumed size per tracked item)
```

With the default that is `4096 / 2 = 2048` entries per endpoint. The 2 MiB divisor is a fixed
assumption in the plugin, not a measurement of your actual payloads.

Set the value to approximate the encoder cache capacity configured on the model servers this
pool routes to:

- **Too high** — the producer keeps claiming a pod holds an item the server has already
  evicted, so the scorer sends work to a pod that has to re-encode it. The routing signal
  degrades quietly; watch `encoder_cache_hit_ratio` rather than expecting an error.
- **Too low** — real cache hits are forgotten early and affinity opportunities are missed.

What the plugin ultimately needs is the correct target entry count. The configuration is
expressed in MiB, and the entry count is derived from it using a fixed 2 MiB per entry
baseline. If your workload deviates from that baseline (long video versus small images),
scale the configured MiB so the plugin allocates the right number of slots.

**Configuration Examples:**

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

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: Qwen/Qwen2.5-1.5B-Instruct
      vllm:
        http: http://localhost:8000
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
- Per-endpoint state is managed and cleaned up automatically, with no manual intervention
  required. State removal happens through two distinct mechanisms:
  - **Event-driven:** an endpoint delete event drops that endpoint's state immediately.
  - **Periodic sweep:** every 2 minutes, entries for pods no longer in the pod list are
    discarded, covering any delete event that was missed.
- The producer emits three metrics on the EPP's own `/metrics` endpoint, under the
  `llm_d_epp_` subsystem, to show how often the affinity signal finds a match:
  - `encoder_cache_queries_total` — every item-hash lookup against the LRU;
    labels `{plugin_type, plugin_name, modality}`.
  - `encoder_cache_hits_total` — the subset of those lookups that matched, per endpoint;
    labels `{plugin_type, plugin_name, pod, modality}`.
  - `encoder_cache_hit_ratio` — histogram of matched items over total items per endpoint
    for a single lookup; labels `{plugin_type, plugin_name}`.
- The producer remains tokenizer-free for request shapes where typed media blocks are
  sufficient; `token-producer` is only required when relying on upstream multimodal
  metadata.
