# Multimodal Embeddings Cache Producer Plugin

**Type:** `mm-embeddings-cache-producer`

Produces multimodal embeddings cache match data for downstream scheduling plugins.

## What It Does

For each request, the producer extracts stable multimodal item hashes from:

- `TokenizedRequest.Prompts[].MultiModalFeatures`, when tokenized multimodal
  metadata is available on the request
- typed OpenAI chat-completions structured media blocks, as a lightweight fallback
- `Generate.Features.MMHashes`, when present

It keeps an in-memory model of each pod's encoder cache. In embedding-capacity
mode, entries are sized in encoder embeddings and use the same oldest-freeable
eviction rule as vLLM: entries referenced by active requests are pinned, while
zero-reference entries are reclaimed from oldest to newest when capacity is
needed. During scheduling, it attaches `EncoderCacheMatchInfo` to each endpoint
so scorers can prefer pods that are likely to have already processed the same
image, video, or audio input.

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

This producer declares `TokenizedRequest` from `token-producer` as an optional
dependency. If `token-producer` is configured, this producer runs after it and
uses `TokenizedRequest` multimodal placeholder lengths. If tokenized request
data is absent at runtime, the producer falls back to typed structured
chat-completions media blocks (or generate feature hashes) with item weight `1`
per hash.

## Data Produced

This plugin produces:

- `MultiModalEncoderCacheMatchInfoKey` (`EncoderCacheMatchInfo`)

## Configuration

The producer supports two mutually exclusive capacity parameters:

- `cacheSizeInEmbeddingsPerServer` (integer): per-endpoint capacity in encoder
  embeddings. When set, item sizes from tokenized multimodal features control
  both affinity weight and cache eviction. Configure `token-producer` so these
  lengths are present; fallback request shapes have unit size.
- `cacheSizeInMBPerServer` (integer, default: `4096`, 4 GiB): compatibility
  mode for existing configurations. It converts the MiB value to a fixed item
  count using the historical 2 MiB-per-item assumption, and every item consumes
  one slot.

Configure only one parameter. Setting both returns a configuration error.

### Sizing `cacheSizeInEmbeddingsPerServer`

Use the model server's encoder cache capacity, measured in encoder embeddings.
For vLLM versions where `encoder_cache_size` is derived from scheduler settings,
the effective value is the larger of `max_num_batched_tokens` and the largest
supported multimodal item. Check the deployed vLLM version and model limits,
then give every EPP instance the same effective capacity as the servers it
routes to.

For example, a vLLM server whose effective encoder cache size is 8192 should use:

```yaml
- type: mm-embeddings-cache-producer
  parameters:
    cacheSizeInEmbeddingsPerServer: 8192
```

This value describes the model server's GPU-side encoder cache. The producer
stores only hashes and bookkeeping in EPP memory.

### Compatibility sizing with `cacheSizeInMBPerServer`

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

Compatibility mode cannot represent mixed-size media accurately. Prefer
`cacheSizeInEmbeddingsPerServer` when tokenized multimodal feature lengths are
available.

**Configuration Examples:**

```yaml
plugins:
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSizeInEmbeddingsPerServer: 8192
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
        url: http://localhost:8000
  - type: mm-embeddings-cache-producer
    parameters:
      cacheSizeInEmbeddingsPerServer: 8192
  - type: mm-embeddings-cache-scorer
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: mm-embeddings-cache-scorer
        weight: 4
```

## Operational Notes

- The cache is a best-effort routing signal, not a correctness dependency.
- Request references are acquired before forwarding and released when the first
  response body chunk arrives. Cancellation and state cleanup also release them,
  so active encoder outputs are not selected for eviction.
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
