# Forward Prompt Token IDs Plugin

**Type:** `forward-prompt-token-ids`

`PreRequest` plugin that hands the token IDs the `token-producer` already rendered to the model server, so
the prompt is tokenized once per request instead of once in the router and again on every model server:

- `/v1/chat/completions`, `/v1/messages`, `/v1/responses`: written as `kv_transfer_params.prompt_token_ids`.
  vLLM's chat path, which serves all three, reads that field and skips its own chat templating and
  tokenization ([vLLM #48145](https://github.com/vllm-project/vllm/pull/48145)).
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
not engine tokens. The IDs must come from the same tokenizer and chat template the model server uses, or the
server serves a different prompt from the one the client sent.

## Behaviour

The plugin forwards IDs only for single-prompt, text-only requests. It skips:

- requests with no tokenization result;
- native generate requests and `/v1/completions` prompts that already carry token IDs;
- `/v1/completions` requests with `echo` or `suffix`, whose response or rendering depends on the prompt text;
- `/v1/responses` requests with `previous_response_id`, `conversation` or other stateful fields: the worker
  renders them from history it stores, which the router's render does not include;
- requests with more than one prompt;
- prompts with multimodal features, which vLLM renders from `messages` regardless;
- payloads that are not mutable JSON, or whose `kv_transfer_params` is not an object;
- requests whose `kv_transfer_params` already carries `prompt_token_ids`, which are kept as sent.

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
