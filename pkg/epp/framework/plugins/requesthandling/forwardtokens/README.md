# Forward Prompt Token IDs Plugin

**Type:** `forward-prompt-token-ids`

`PreRequest` plugin that forwards the token IDs the `token-producer` already rendered to the model server, so
the prompt is tokenized once per request instead of once in the router and again on every model server:

- `/v1/chat/completions`, `/v1/messages`, `/v1/responses`: written as `kv_transfer_params.prompt_token_ids`.
  vLLM's chat path, which serves all three, reads that field and skips its own chat templating and
  tokenization ([vLLM #48145](https://github.com/vllm-project/vllm/pull/48145), first shipped in v0.27.0; see the version section below).
- `/v1/completions`: the text `prompt` is replaced with its token IDs, which vLLM uses without tokenizing.

At long context the second tokenization is a large share of first-token latency: about 1.8 ms per 1k
prompt tokens for a byte-level BPE tokenizer, single-threaded.

## Configuration

Because the plugin is Alpha, the EPP process must include `--allow-experimental-plugins`.

```yaml
- type: forward-prompt-token-ids
  parameters:
    minTokens: 1024
```

`minTokens` skips prompts shorter than the given count, where re-tokenizing costs the server little and the
ID list only grows the body. `0`, the default, forwards every prompt.

The plugin requires a `token-producer` with a `vllm` backend. The EPP refuses to start with the `estimate`
backend, including the one it auto-creates when no `token-producer` is configured: its IDs are packed bytes,
not engine tokens.

### Token parity

The IDs must come from the same tokenizer and chat template the model server uses, or the server serves a
different prompt from the one the client sent. The plugin cannot check this, so the `token-producer`
configuration is part of the contract. Setups that break parity:

- a renderer reached through a separate `vllm.url` that runs a different vLLM version, a different
  chat-template flag set (`--chat-template`, `--reasoning-effort-map`, `--enable-auto-tool-choice`), or a
  different model revision than the serving pods;
- a renderer pool (`endpointDiscovery`) whose members are configured differently from the serving pods;
- any renderer whose tokenizer mode differs from the serving pods (`--tokenizer-mode`);
- **model-server rollouts** that change the chat template, tokenizer or vLLM version: during the roll,
  the token-producer can render on an old pod and the scheduler can pick a new pod (or the reverse),
  with no error. Before this plugin that skew only blurred prefix-cache scores; with it, the server
  silently runs the other revision's prompt. Disable the plugin for the duration of such rollouts.

### Request size

Each forwarded token adds about 6 bytes to the request body (measured 6.14 bytes per token with a
129k-token vocabulary), so a 1M-token prompt grows by about 6.4 MB, roughly 2.5x the text it replaces
(text is about 4 bytes per token). A gateway or ext_proc request-body limit that fits today's prompts
can start rejecting long requests once the plugin is on; raise the limit accordingly.

### vLLM version

The serving pods must run vLLM **v0.27.0 or newer**, which first ships
[vLLM #48145](https://github.com/vllm-project/vllm/pull/48145) (the chat path reading
`kv_transfer_params.prompt_token_ids`). On older versions the forwarded IDs are inert bytes in every
request: the engine re-renders the prompt and the body grows by the ID list for nothing. The
`/v1/completions` path (token array as prompt) works on older versions, but this plugin pairs it with
the chat path, so the floor applies.

## Behavior

The plugin forwards IDs only for single-prompt, text-only requests. It skips:

- requests with no tokenization result;
- requests whose IDs came from the token-producer's legacy `/v1/messages` or `/v1/responses`
  translation (marked `LegacyTranslated`): those translations do not guarantee token parity
  with what the model server renders from the client's request;
- native generate requests and `/v1/completions` prompts that already carry token IDs;
- `/v1/completions` requests with `echo` or `suffix`, whose response or rendering depends on the prompt text;
- `/v1/responses` requests with `previous_response_id`, `conversation` or other stateful fields: the worker
  renders them from history it stores, which the router's render does not include;
- requests with more than one prompt;
- prompts with multimodal features, which vLLM renders from `messages` regardless
  (forwarding the processed multimodal features alongside the IDs is deferred to a
  follow-up PR, pending vllm-project/vllm#64721);
- chat requests with `echo` or `return_prompt_text` set: the former makes vLLM discard the
  forwarded IDs anyway, and the latter would return `null` prompt text on the reuse branch;
- payloads that are not mutable JSON, or whose `kv_transfer_params` is not an object;
- requests whose `kv_transfer_params` already carries `prompt_token_ids`, which are kept as
  sent (an explicit `null` counts as absent and is replaced).

Other `kv_transfer_params` fields are preserved.

## Deployment models

| Deployment | Where the IDs go |
| --- | --- |
| Aggregated | The EPP forwards the request unchanged, so the serving pod gets the IDs. |
| P/D sidecar, `nixlv2` (serial and MoRI-IO parallel WRITE) | Carried into the prefill and the decode request. |
| P/D sidecar, `offloading` (P/D and decode-only cache-source pull) | Carried into every request it builds. |
| P/D sidecar, `mooncake` | Carried into the prefill and the decode request. |
| P/D sidecar, `shared-storage`, `sglang` | The client's `kv_transfer_params` is forwarded as sent. |

Each P/D connector replaces `kv_transfer_params` with its own fields on at least one hop; without the
carry-over the IDs the plugin adds would be dropped before they reach a model server. Chunked decode drops
`kv_transfer_params` from the second chunk on, which is correct: later chunks continue a changed prompt.

SGLang has no equivalent field, so an SGLang engine ignores the IDs and tokenizes as before.
