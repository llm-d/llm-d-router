# Token Producer Plugin

**Type:** `token-producer`

`DataProducer` plugin that tokenizes the request prompt and publishes
`TokenIDs` (and a flat sorted `MultiModalFeatures` list) on
`InferenceRequestBody.TokenizedRequest` for downstream consumers (scorers,
filters, other data producers).

Implements `requestcontrol.DataProducer` and runs in the `PrepareRequestData`
phase, before filters and scorers. The plugin is idempotent: if
`InferenceRequestBody.TokenizedRequest` is already populated by an earlier
producer, tokenization is skipped. Multi-modal features are flattened into the
upstream list shape, sorted by placeholder offset.

> [!NOTE]
> Legacy alias `tokenizer` is still accepted but logs a deprecation warning at
> instantiation. Prefer `token-producer` in new configs.

## Backend

Backend selection:

- **`estimate`** (default): tokenizer-free byte-packing — no model, no service.
  Selected when no backend is set, and auto-created by the framework for any
  config whose plugins consume `TokenizedPrompt` (prefix cache, context-length,
  P/D routing) without declaring a `token-producer`.
- **`vllm`** (or `modelName`): forwards requests to vLLM's native
  `/v1/completions/render`, `/v1/chat/completions/render`, and
  `/v1/messages/render` endpoints over HTTP or HTTPS. TLS is driven by the URL
  scheme (`https://`). For in-cluster endpoints using self-signed or private CA
  certificates, configure `vllm.caCertPath` to trust the CA, and optionally
  `vllm.clientCertPath`/`vllm.clientKeyPath` for mTLS.

## Native render contract

The renderer sends the original HTTP JSON body when EPP has not mutated it.
It does not substitute the model, translate protocols, rewrite messages or
tools, or reconstruct content from routing projections. Model rewrites happen
before token production and apply to both rendering and forwarding.

The parsed payload keeps nested objects, arrays, Completions `prompt`, and
Messages `system` as `json.RawMessage`.
Envelope mutations can therefore change routing fields without reordering
tool schemas or other nested content. Plugins use the typed protocol
projections to read content. Prompt-affecting mutations must finish before
token production; metadata added afterward must not affect tokenization.

Completions requests with token arrays still go through vLLM, which owns
truncation and other input preprocessing. Native Generate requests already
carry final tokens and bypass rendering. Direct requests to the three
`/render` endpoints retain model routing and pass through without local token
production or response parsing.

Native gRPC text has no HTTP JSON envelope. Its compatibility path submits
the text to Completions rendering with the configured `modelName`; this does
not establish native gRPC token parity. Pretokenized gRPC requests use their
parser-provided tokens without rendering. This exception does not apply to
HTTP requests or the HTTP JSON embedded in Vertex AI gRPC requests.

Chat and Messages requests use the larger of `vllm.timeout` and
`vllm.mmTimeout`. The renderer does not inspect content to select a timeout.

Token parity requires matching render/serve model, tokenizer, template,
processor, and parser configuration, plus deterministic upstream rendering.
The serving path must preserve the effective request after EPP. Sidecar
prefill/decode mutations and coordinator rendering are separate paths; this
EPP contract does not establish parity for them.

> [!WARNING]
> The `estimate` backend approximates token boundaries (≈4 bytes/token); its
> token IDs do not correspond to engine tokens. The precise prefix-cache scorer
> requires real tokens — configure a `vllm` `token-producer` explicitly for it.
> If omitted, the auto-created `estimate` producer satisfies the dependency but
> silently degrades precise cache correlation.

## Config

| Parameter                  | Default                 | Description                                                                  |
| -------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `modelName`                | – (required for `vllm`) | Model for startup probes and native gRPC text compatibility; HTTP requests retain their effective model. |
| `vllm.url`                 | `http://localhost:8000` | Base URL of the vLLM render endpoint (no trailing slash).                    |
| `vllm.timeout`             | `5s`                    | Completions timeout and minimum Chat/Messages timeout.                      |
| `vllm.mmTimeout`           | `30s`                   | Chat/Messages timeout budget, including multimodal processing.               |
| `vllm.caCertPath`          | system CA pool          | PEM CA bundle for verifying the render endpoint when using `https://`.       |
| `vllm.clientCertPath`      | –                       | Client certificate for mTLS with the render endpoint; requires `clientKeyPath`. |
| `vllm.clientKeyPath`       | –                       | Client private key for mTLS; requires `clientCertPath`.                      |
| `vllm.insecureSkipVerify`  | `false`                 | Skip server certificate verification when using `https://`; `caCertPath` is ignored when set. |

The `estimate` backend tunes multimodal image placeholder estimation (empty uses
the defaults below):

| Parameter                          | Default   | Description                                                                |
| ---------------------------------- | --------- | -------------------------------------------------------------------------- |
| `estimate.image.mode`              | `dynamic` | `dynamic` (width×height/factor) or `static` (a constant per-image count).  |
| `estimate.image.defaultResolution` | 640×360   | Dynamic-mode fallback when an image's dimensions can't be decoded.         |
| `estimate.image.dynamic.factor`    | `1024`    | Dynamic-mode pixels-per-placeholder-token divisor.                         |
| `estimate.image.static.staticToken`| –         | Static-mode per-image placeholder count.                                   |

Video estimation is `min(frames × tokensPerFrame, maxVideoTokens)`. The per-frame
token count and the frame count are configured independently, so the two common
model shapes are mode combinations: qwen3 is `tokensPerFrame.mode=dynamic` +
`frames.mode=sampled`; gemma4 is `tokensPerFrame.mode=static` +
`frames.mode=strided`. Video duration, resolution, and source FPS come from the
`x-llm-d-video-*` request headers below when present; otherwise each falls back to
its config value and then the built-in default. Headers are request-level, so they
apply to every video in the request.

| Request header                  | Format          | Description                                     |
| ------------------------------- | --------------- | ----------------------------------------------- |
| `x-llm-d-video-duration-seconds`| float seconds   | Video length; overrides `defaultDuration`.      |
| `x-llm-d-video-resolution`      | `WIDTHxHEIGHT`  | Frame resolution; overrides `defaultResolution`.|
| `x-llm-d-video-fps`             | float           | Source frame rate; overrides `frames.strided.defaultSourceFPS` (strided mode). |

| Parameter                             | Default   | Description                                                                                                                                                 |
| ------------------------------------- | --------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `estimate.video.tokensPerFrame.mode`  | `dynamic` | `dynamic` (width×height/factor) or `static` (a constant per-frame count).                                                                                   |
| `estimate.video.tokensPerFrame.dynamic.factor`| `1024` | Dynamic-mode pixels-per-placeholder-token divisor.                                                                                                          |
| `estimate.video.tokensPerFrame.static.numTokensPerFrame` | – | Static-mode per-frame placeholder count.                                                                                                                    |
| `estimate.video.frames.mode`          | `sampled` | `sampled` (clamp(duration×sampleFPS, minFrames, maxFrames) / temporalPatchSize) or `strided` (clamp(duration×sourceFPS/frameStride, minFrames, maxFrames)). |
| `estimate.video.frames.minFrames`     | –         | Sampled/strided frame floor (0 = none). Models a processor's minimum frames.                                                                                |
| `estimate.video.frames.maxFrames`     | –         | Sampled/strided frame cap (0 = uncapped).                                                                                                                   |
| `estimate.video.frames.sampled.sampleFPS`     | `1`       | Sampled-mode sampling rate.                                                                                                                                 |
| `estimate.video.frames.sampled.temporalPatchSize` | –     | Sampled-mode: merge every N sampled frames into one token group (qwen3-vl = 2; <2 = no merge).                                                              |
| `estimate.video.frames.strided.defaultSourceFPS` | `24`   | Strided-mode source frame rate; fallback for the `x-llm-d-video-fps` header.                                                                                |
| `estimate.video.frames.strided.frameStride`   | `1`       | Strided-mode divisor: keep every Nth source frame.                                                                                                          |
| `estimate.video.defaultResolution`    | 640×360   | Per-frame resolution for dynamic tokens-per-frame; fallback for the `x-llm-d-video-resolution` header.                                                      |
| `estimate.video.defaultDuration`      | `10`      | Video length in seconds for frame counting; fallback for the `x-llm-d-video-duration-seconds` header.                                                       |
| `estimate.video.maxVideoTokens`       | –         | Overall placeholder cap for a video (0 = uncapped).                                                                                                         |

## Failure mode

Per-request errors are returned to the Director, which currently logs and
continues; downstream scorers fall back to their own paths. A missing native
endpoint, render error, or empty token result does not publish token IDs.
The vLLM backend does not fall back to protocol conversion or estimation.

## Deployment

Use a vLLM build exposing the native render endpoint for each protocol in use.
Messages rendering requires [native `/v1/messages/render` support](https://github.com/vllm-project/vllm/pull/45803);
older builds are not supported by a local conversion fallback.
`vllm launch render <model>` exposes render endpoints without a GPU.
For `vllm serve <model>`, set `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`.

The renderer can run beside EPP or behind a dedicated Service. It must accept
the effective model name used for inference, including configured served-model
aliases and adapters. `modelName` does not replace those names in render
requests. When the inbound request carries an
`Authorization` header, it is forwarded verbatim on render requests, so an
endpoint started with `--api-key` accepts them; the startup warmup probe
sends no `Authorization` header, so against such an endpoint it is skipped
and the first request pays the cold-start cost.

```yaml
# EPP pod spec
containers:
- name: vllm-render
  image: vllm/vllm-openai:latest          # any image shipping `vllm launch render`
  command: ["vllm", "launch", "render"]
  args: ["${MODEL_NAME}", "--port=8000"]
  ports: [{name: render-http, containerPort: 8000}]
  readinessProbe: {httpGet: {path: /health, port: 8000}, periodSeconds: 5}
```

Plugin config — sidecar (loopback):

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "http://localhost:8000"       # optional; this is the default
```

Plugin config — dedicated render Service:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "http://vllm-render.default.svc.cluster.local:8000"
```

Plugin config — dedicated render Service with TLS:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "https://vllm-render.default.svc.cluster.local:8000"
      caCertPath: "/path/to/ca.crt"
```

The render endpoint must also be serving TLS. When using `vllm launch render`,
pass `--ssl-certfile` and `--ssl-keyfile` so the process listens over HTTPS:

```yaml
containers:
- name: vllm-render
  command: ["vllm", "launch", "render"]
  args:
    - "${MODEL_NAME}"
    - "--port=8000"
    - "--ssl-certfile=/path/to/tls.crt"
    - "--ssl-keyfile=/path/to/tls.key"
```

A complete sample config that pairs this with `precise-prefix-cache-producer` and `prefix-cache-scorer` is at [`deploy/config/sim-epp-tokenizer-vllm-http-config.yaml`](../../../../../../../deploy/config/sim-epp-tokenizer-vllm-http-config.yaml).

---

## Related Documentation
- [Precise Prefix Cache Scorer](../../../scheduling/scorer/preciseprefixcache/README.md)
- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
