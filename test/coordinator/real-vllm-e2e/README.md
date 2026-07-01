# Real vLLM E2E Tests

End-to-end tests that exercise `RenderStep`, `EncodeStep`, and `PrefillStep`
against a real vLLM instance.

These tests require the `e2e` build tag and are **not** compiled or run during
`go test ./...`. They must be invoked explicitly with `-tags=e2e`.

## Prerequisites

Two vLLM endpoints are required:

- **Rendering service** at `RENDER_E2E_URL` (default `http://localhost:8000`),
  exposing `POST /v1/chat/completions/render` and `POST /v1/completions/render`.
- **Encode/prefill endpoint** at `ENCODE_E2E_GATEWAY` (default
  `http://localhost:8080`), receiving requests with the `EPP-Phase` header set
  to `encode` or `prefill`. This is typically vLLM fronted by the EPP.

The model named by `ENCODE_E2E_MODEL` / `RENDER_E2E_MODEL` (default
`Qwen/Qwen3-VL-2B-Instruct`) must be loaded. Image tests read
`test-data/200.jpg` and send it inline as a base64 JPEG, so the model must
be vision-capable.

## Environment variables

| Variable             | Default                      | Purpose                                                   |
| -------------------- | ---------------------------- | --------------------------------------------------------- |
| `RENDER_E2E_URL`     | `http://localhost:8000`      | Base URL of the rendering service.                        |
| `RENDER_E2E_MODEL`   | `Qwen/Qwen3-VL-2B-Instruct` | Model name used in render tests.                          |
| `ENCODE_E2E_GATEWAY` | `http://localhost:8080`      | Base URL of the encode/prefill endpoint (vLLM + EPP).     |
| `ENCODE_E2E_MODEL`   | `Qwen/Qwen3-VL-2B-Instruct` | Model name used in encode and prefill tests.              |
| `ENCODE_E2E_EC`      | `ec-nixl`                    | EC connector name passed to `EncodeStep` / `PrefillStep`. |
| `ENCODE_E2E_KV`      | `kv-nixl`                    | KV connector name passed to `PrefillStep`.                |

`RENDER_E2E_MODEL` and `ENCODE_E2E_MODEL` both default to the same value;
set them independently when the two endpoints serve different deployments.

## Running

From the repo root:

```sh
# All e2e tests, default config
go test -tags=e2e ./test/real-vllm-e2e/...

# Verbose output
go test -tags=e2e -v ./test/real-vllm-e2e/...

# Only render tests
go test -tags=e2e -v -run TestE2E_ChatCompletions ./test/real-vllm-e2e/...
go test -tags=e2e -v -run TestE2E_Completions ./test/real-vllm-e2e/...

# Only encode tests
go test -tags=e2e -v -run TestE2E_Encode ./test/real-vllm-e2e/...

# Only prefill tests
go test -tags=e2e -v -run TestE2E_Prefill ./test/real-vllm-e2e/...

# Single test
go test -tags=e2e -v -run TestE2E_Encode_RenderThenEncode ./test/real-vllm-e2e/...

# Override endpoints
RENDER_E2E_URL=http://vllm-render:8000 \
ENCODE_E2E_GATEWAY=http://vllm-epp:8080 \
ENCODE_E2E_MODEL=Qwen/Qwen3-VL-2B-Instruct \
  go test -tags=e2e -v ./test/real-vllm-e2e/...
```

## What each test covers

### Render tests (`render_test.go`)

| Test                                    | Endpoint                      | Asserts                                                                                  |
| --------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- |
| `TestE2E_ChatCompletions_SimpleMessage` | `/v1/chat/completions/render` | Non-empty `TokenIDs`.                                                                    |
| `TestE2E_ChatCompletions_TwoImages`     | `/v1/chat/completions/render` | Non-empty `TokenIDs`; both `MultimodalEntries` get `Hash`, `Placeholder`, `KwargsData`. |
| `TestE2E_Completions_TextPrompt`        | `/v1/completions/render`      | Non-empty `TokenIDs`; `Body["prompt"]` rewritten to `[]int`.                            |
| `TestE2E_Completions_TokenArray`        | (short-circuits, no network)  | Token array preserved as-is; verifies skip path with an unreachable render host.        |
| `TestE2E_DumpRenderChatCompletions`     | `/v1/chat/completions/render` | Diagnostic: logs request/response sizes and per-modality feature counts.                |

### Encode tests (`encode_test.go`)

Each table-driven test runs two sub-tests: `ChatCompletions`
(`use_openai_format=true` → `/v1/chat/completions`) and `Generate`
(`use_openai_format=false` → `/inference/v1/generate`).

| Test                                        | Target endpoint(s)                               | Asserts                                                                                                                            |
| ------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `TestE2E_Encode_Synthetic`                  | `/v1/chat/completions`                           | Hand-crafted multimodal entries accepted; `ECTransferParams` populated. Skips `Generate` (synthetic `kwargs_data` fails deserialization). |
| `TestE2E_Encode_RenderThenEncode`           | `/v1/chat/completions`, `/inference/v1/generate` | Render populates `Hash`/`KwargsData`/`Placeholder`; encode succeeds and populates `ECTransferParams` for each entry.               |
| `TestE2E_Encode_DumpChatCompletionsRequest` | `/v1/chat/completions`                           | Diagnostic: logs full request wire shape and response.                                                                             |
| `TestE2E_Encode_DumpGenerateRequest`        | `/inference/v1/generate`                         | Diagnostic: renders to obtain real `KwargsData`, then logs generate request/response.                                              |
| `TestE2E_Encode_DiagnoseGenerateFailure`    | `/inference/v1/generate`                         | Diagnostic: runs real `EncodeStep` through a capturing transport and logs exact wire bytes + response.                             |

### Prefill tests (`prefill_test.go`)

Each table-driven test runs `ChatCompletions` and `Generate` sub-tests where
applicable (same two variants as the encode tests).

| Test                                         | Pipeline                              | Asserts                                                                          |
| -------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| `TestE2E_Prefill_ChatCompletions_Multimodal` | render -> encode -> prefill           | Full pipeline succeeds; `ECTransferParams` populated; `KVTransferParams` logged. |
| `TestE2E_Prefill_ChatCompletions_TextOnly`   | render -> prefill (no encode)         | No `MultimodalEntries` after render; prefill succeeds.                           |
| `TestE2E_Prefill_Completions_TextPrompt`     | render -> prefill on `/v1/completions`| Non-empty `TokenIDs` after render; prefill routes to `/v1/completions`.          |
| `TestE2E_Prefill_Completions_TokenArray`     | render (skipped) -> prefill           | Pre-tokenized prompt bypasses render; prefill succeeds with token array.         |

`KVTransferParams` is logged but not asserted non-nil: vLLM returns
`kv_transfer_params: null` when the cluster is not in disaggregated P/D mode.

## Troubleshooting

- **`connection refused` on render** — the rendering service is not running, or
  `RENDER_E2E_URL` is wrong.
- **`connection refused` on encode/prefill** — the vLLM+EPP endpoint is not
  running, or `ENCODE_E2E_GATEWAY` is wrong.
- **HTTP 404 from render** — the service is reachable but does not expose
  `/render`-suffixed routes. Confirm `RENDER_E2E_URL` targets a vLLM instance
  with the render extension loaded, not a plain vLLM server.
- **HTTP 400 on image tests** — the model is not vision-capable. Set
  `ENCODE_E2E_MODEL` / `RENDER_E2E_MODEL` to a vision model.
- **No tests run with `go test ./test/real-vllm-e2e/...`** — you forgot
  `-tags=e2e`. Without the tag the package contains no buildable Go source.
