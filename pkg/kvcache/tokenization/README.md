# Tokenization Types

Shared types for tokenized-prompt handling used by the tokenizer plugin and the
prefix-cache scorers: the multimodal metadata a tokenizer produces, and the chat
request/response data model it consumes.

## What It Does

- **`MultiModalFeatures`.** Per-modality multimodal metadata produced alongside
  token IDs: content hashes (`MMHashes`) and placeholder token ranges
  (`MMPlaceholders`, expressed as [`kvblock.PlaceholderRange`](../kvblock/README.md)).
  The [`kvcache.Indexer`](../README.md) folds these into per-block extra
  features so multimodal content taints the block hash.
- **`types`** (subpackage). The chat data model -- conversations, content
  blocks (text and image), render requests/responses, and token offsets -- with
  the custom JSON handling the OpenAI-compatible request shapes require.

## Scope

This package holds only tokenization types. The tokenizer backends that produce
them live with the `token-producer` plugin.

## Related Documentation

- [KV-Cache Indexer](../README.md) -- consumes `MultiModalFeatures`
- [KV-Block Index](../kvblock/README.md) -- defines `PlaceholderRange`
