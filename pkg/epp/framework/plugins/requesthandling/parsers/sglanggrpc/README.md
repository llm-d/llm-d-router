# SGLang gRPC Parser

`sglanggrpc-parser` parses requests and responses for SGLang's native gRPC service (`sglang.api.v1.SglangService`).

## Claimed Paths and Protocols

* Path suffix: `sglang.api.v1.SglangService/Generate`
* Protocol: `kubernetes.io/h2c`

Other `sglang.api.v1.SglangService` RPCs (`HealthCheck`, `GetModelInfo`, `GetServerInfo`, `ListModels`, `Detokenize`, `Abort`, and HiCache control RPCs) are not claimed by `sglanggrpc-parser`; configure `passthrough-parser` after `sglanggrpc-parser` so unmatched gRPC methods pass through without interpretation.

## Request and Response Behavior

* Accepts single-prompt `GenerateRequest` messages carrying `input_ids.one` or `text.one`. Batched `many` variants return `400`.
* Composes `(extra_key, cache_salt)` identically to `sglanghttp-parser` for prefix-cache isolation.
* Extracts `mm_hashes.one` for multimodal encoder-cache affinity and `sampling_params.one.max_new_tokens` for output-token estimation.
* Implements `PriorityRewriter` to stamp `GenerateRequest.priority` when `propagatePriority` is enabled.
* Extracts `prompt_tokens`, `completion_tokens`, and `cached_tokens` from terminal `GenerateStreamItem` frames (`meta_info.finish_reason` set).
