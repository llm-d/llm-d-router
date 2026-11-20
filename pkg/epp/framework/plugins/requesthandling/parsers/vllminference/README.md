# vLLM Inference gRPC Parser

`vllminference-parser` parses requests and responses for the upstream vLLM gRPC service (`vllm.Inference`).

## Claimed Paths and Protocols

* Path suffixes: `vllm.Inference/Generate`, `vllm.Inference/GenerateStream`
* Protocol: `kubernetes.io/h2c`

`vllminference-parser` does not claim `vllm.Control` RPCs (`GetModelInfo`, `GetServerInfo`, `ListModels`) or `grpc.health.v1.Health` checks. To forward unmatched gRPC methods without payload inspection, configure `passthrough-parser` after `vllminference-parser`.

## Request and Response Behavior

* The parser accepts single-sequence `GenerateRequest` messages that set `text` or `token_ids`. Requests with `sampling.num_sequences > 1` return `400`.
* On text prompts, the parser operates in `EPP-Text-Text` mode ([#3993](https://github.com/llm-d/llm-d-router/issues/3993)). The tokenizer tokenizes `text` for prefix-cache scoring only. The parser forwards `text` unchanged, does not rewrite `text` to `token_ids`, and does not inject a render fingerprint.
* The parser forwards `kv.kv_transfer_params` and `kv.ec_transfer_params` from the client unchanged ([#3979](https://github.com/llm-d/llm-d-router/issues/3979)).
* The parser extracts `kv.cache_salt`, `stopping.max_new_tokens`, `PreprocessedMediaFeatures` (`identifier`, `mm_hash`, `offset`, `length`, `modality`), and raw `MediaItem.uuid` values for cache scoring.
* A request with `lora_name` set is routed and scored by the adapter name. A model rewrite targets `lora_name` and leaves `GenerateRequest.model` as the base model. A rewrite to the base model clears `lora_name`.
* A body with bytes after the single gRPC frame returns `400`.
* The parser drops a client-supplied `kv_hints` (`GenerateRequest` field 17). A hint names a peer the engine connects to, so only the EPP may set it after scheduling ([#4194](https://github.com/llm-d/llm-d-router/issues/4194)).
