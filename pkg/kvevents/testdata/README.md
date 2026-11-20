# KV-event captures

## lmcache_mp_kv_events.json

The KV-event messages one vLLM engine published while it served one prompt
through `LMCacheMPConnector` and an LMCache multiprocess (MP) server: 12
messages with the sequence numbers 0 to 11. `lmcache_capture_test.go` and
`test/integration/epp/precise_lmcache_events_test.go` replay it.

### Format

| Field | Content |
| --- | --- |
| `model` | Model name in the topic of every message. |
| `lmcache_chunk_size` | LMCache chunk size in tokens. |
| `engine_block_size` | Size in tokens of the engine's GPU blocks. |
| `stack` | Versions of vLLM, Python and the libraries, the commit of the LMCache source tree, the model snapshot and the GPU model. |
| `commands` | Command lines of the LMCache server and the engine, without the interpreter path. |
| `environment` | Environment variables set for both processes, apart from a `PYTHONPATH` that names the LMCache source tree. |
| `prompt_token_ids` | The prompt as the engine's `/v1/chat/completions/render` route tokenizes it. |
| `stages[]` | The number of `messages` of the stage, the warm-up `prompts` and the `fillers` sent. For a stage that sent the prompt: `prompt_tokens`, `lmcache_cached_tokens` (the `num_lmcache_cached_tokens` the engine reports for the request) and the `completion`. |
| `messages[].stage` | The stage the message belongs to. |
| `messages[].frames` | The frames of one multipart message as received, base64 encoded: topic, 8-byte big-endian sequence number, msgpack event batch. |

### Stages

The prompt has 548 tokens: two LMCache chunks of 256 tokens, eight engine
blocks of 64 tokens, and 36 more tokens that neither tier caches. Every request
is a chat completion with `temperature` 0.

| Stage | Requests | Events published |
| --- | --- | --- |
| `warm-up` | One prompt of 72 to 120 tokens. | One GPU `BlockStored` of one block. |
| `cold` | The prompt. The engine loaded 0 tokens from LMCache. | GPU `BlockStored`: 8 hashes, 512 tokens, `block_size` 64, no parent. |
| `cpu-stored` | Prompts shorter than one engine block, until the engine published the stores. | Two CPU `BlockStored`: one hash and 256 tokens each, `block_size` 256. The second names the first as parent. |
| `gpu-evicted` | Four filler prompts of 196 to 244 tokens, which is less than one chunk. | Per filler, GPU `BlockRemoved` events of one hash each and the filler's GPU `BlockStored`. The 8 blocks of the prompt are among the removed. |
| `cpu-load` | The prompt. The engine loaded 512 tokens from LMCache and returned the completion of `cold`. | GPU `BlockRemoved` for filler blocks and the prompt's GPU `BlockStored` under the hashes of `cold`. |
| `gpu-reset` | `POST /reset_prefix_cache` on the engine, then prompts shorter than one engine block. LMCache was not cleared. | `AllBlocksCleared`. |
| `cpu-load-2` | The prompt. The engine loaded 512 tokens from LMCache and returned the completion of `cold`. | The prompt's GPU `BlockStored`. No CPU `BlockStored`: LMCache announces a chunk once. |
| `cpu-removed` | `POST /cache/clear` on the LMCache server, then prompts shorter than one engine block. | Two CPU `BlockRemoved`, one chunk hash each. |

### Provenance

Captured on 2026-11-19 on one GPU from the processes in `commands`, driven by
the requests of the stage table. These settings differ from a deployment:

- `--num-gpu-blocks-override 12` gives the engine a GPU cache that four filler
  prompts fill, and requires `lmcache.mp.use_vmm_api: false`.
- `--enable-scale-out` serves the render route and `VLLM_SERVER_DEV_MODE=1` the
  reset route.
- The publisher's `endpoint` has no wildcard, so the engine connects to a SUB
  socket that the capture bound on the loopback interface.
- `PYTHONHASHSEED=0` seeds the engine's block hashes.

### Limits

- One engine, one prompt, no LoRA adapter, no cache salt, no multimodal input.
- vLLM 0.34.0 publishes connector events only while its scheduler steps. The
  short prompts of `cpu-stored`, `gpu-reset` and `cpu-removed` make it step, so
  the capture does not show publication by an idle engine.
- The capture records what this stack published. It does not show that another
  version publishes the same.
