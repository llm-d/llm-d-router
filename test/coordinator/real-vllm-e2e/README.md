# Render Step E2E Tests

End-to-end tests that exercise `RenderStep` against a real rendering service.

These tests are gated by the `e2e` build tag, so they are **not** compiled or
run during `go test ./...`. They must be invoked explicitly with
`-tags=e2e`.

## Prerequisites

A rendering service reachable at the URL you point the tests at. By default the
tests target `http://localhost:8000`; the service must expose:

- `POST /v1/chat/completions/render`
- `POST /v1/completions/render`

The model named by `RENDER_E2E_MODEL` (default `Qwen/Qwen3-VL-2B-Instruct`)
must be loaded by the rendering service. The image tests send inline
base64 PNG images, so the model must be vision-capable.

## Environment variables

| Variable           | Default                          | Purpose                                        |
| ------------------ | -------------------------------- | ---------------------------------------------- |
| `RENDER_E2E_URL`   | `http://localhost:8000`          | Base URL of the rendering service.             |
| `RENDER_E2E_MODEL` | `Qwen/Qwen3-VL-2B-Instruct`      | Model name sent in the request body.           |

## Running

From the repo root:

```sh
# All e2e tests, default config
go test -tags=e2e ./test/real-vllm-e2e/...

# Verbose output
go test -tags=e2e -v ./test/real-vllm-e2e/...

# Single test
go test -tags=e2e -v -run TestE2E_ChatCompletions_TwoImages ./test/real-vllm-e2e/...

# Override host/model
RENDER_E2E_URL=http://my-render-service:8000 \
RENDER_E2E_MODEL=Qwen/Qwen3-VL-2B-Instruct \
  go test -tags=e2e -v ./test/real-vllm-e2e/...
```

## What each test covers

| Test                                    | Endpoint                          | Asserts                                                                              |
| --------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------ |
| `TestE2E_ChatCompletions_SimpleMessage` | `/v1/chat/completions/render`     | Non-empty `TokenIDs`.                                                                |
| `TestE2E_ChatCompletions_TwoImages`     | `/v1/chat/completions/render`     | Non-empty `TokenIDs`; both `MultimodalEntries` get `Hash`, `Placeholder`, `KwargsData` populated. |
| `TestE2E_Completions_TextPrompt`        | `/v1/completions/render`          | Non-empty `TokenIDs`; `Body["prompt"]` rewritten to `[]int`.                         |
| `TestE2E_Completions_TokenArray`        | (none, short-circuits)            | Token array preserved as-is; verifies the skip path by pointing at an unreachable host. |

## Troubleshooting

- **`render request failed: ... connection refused`** — the rendering service
  is not running, or `RENDER_E2E_URL` points at the wrong host/port.
- **`render service returned HTTP 404`** — the service is reachable but does
  not expose `/render` suffixed paths under the OpenAI routes. Check that you
  are pointing at the rendering service, not a vanilla vLLM server.
- **`render service returned HTTP 400` on the two-images test** — the model
  loaded by the rendering service is not vision-capable. Set
  `RENDER_E2E_MODEL` to a vision model.
- **No tests run with `go test ./test/real-vllm-e2e/...`** — you forgot `-tags=e2e`.
  Without the tag the package contains no buildable Go source.
