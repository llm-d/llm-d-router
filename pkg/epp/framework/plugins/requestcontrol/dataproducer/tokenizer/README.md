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

## Backend

Backend selection:

- **`estimate`** (default): tokenizer-free byte-packing — no model, no service.
  Selected when no backend is set, and auto-created by the framework for any
  config whose plugins consume `TokenizedPrompt` (prefix cache, context-length,
  P/D routing) without declaring a `token-producer`.
- **`vllm`** (or `modelName`): forwards requests to vLLM's native
  `/v1/completions/render`, `/v1/chat/completions/render`, and
  `/v1/messages/render` endpoints over HTTP or HTTPS. Messages endpoint selection
  is controlled by `vllm.messagesRenderMode`. TLS is driven by the URL
  scheme (`https://`). For in-cluster endpoints using self-signed or private CA
  certificates, configure `vllm.caCertPath` to trust the CA, and optionally
  `vllm.clientCertPath`/`vllm.clientKeyPath` for mTLS. The HTTP renderer uses
  either one configured URL or endpoints supplied by data-layer discovery.

## Messages rendering

`vllm.messagesRenderMode: auto` probes `/v1/messages/render` with a small text
request using `modelName`. A successful probe selects native pass-through. A
404 or 405 selects legacy conversion only if the same probe succeeds at
`/v1/chat/completions/render`. Other discovery errors leave the mode unresolved.
Discovery runs during warmup and, if unresolved, on the next Messages request
using its Authorization header. User-request errors do not change the mode.

The selection is cached for the plugin lifetime. The configured renderer URL
must serve a consistent vLLM version and accept `modelName`. Restart EPP to
rediscover capabilities after a renderer upgrade. Explicit `native` and
`legacy` modes bypass discovery. Neither mode uses `/tokenize` or falls back
to estimation when rendering fails.

Legacy conversion uses the configured `modelName` and logs a deprecation
warning once per plugin instance. Conversion does not guarantee token parity
with inference. The forwarded request is unchanged.

The compatibility implementation and tests are contained in
`legacy_messages.go` and `legacy_messages_test.go`. Its integration points are
the `MessagesRenderMode` configuration field, `configureLegacyMessages` in the
plugin constructor, and `legacyMessages` in warmup and Messages dispatch.
The native rendering and token production implementations do not use
legacy conversion helpers or wire types. Helpers required by estimation are
owned by `estimate.go`.

## Native render contract

Completions and Chat Completions use native rendering. Messages uses this
contract when native rendering is selected.

The renderer sends the original HTTP JSON body when EPP has not mutated it.
It does not substitute the model, translate protocols, rewrite messages or
tools, or reconstruct content from routing projections. Model rewrites happen
before token production and apply to both rendering and forwarding.
`vllm.prefillOnly` rewrites only the output budget fields of the render copy;
see [Render-only output budget](#render-only-output-budget).

The parsed payload keeps nested objects, arrays, Completions `prompt`, and
Messages `system` as `json.RawMessage`.
Envelope mutations can therefore change routing fields without reordering
tool schemas or other nested content. Plugins use the typed protocol
projections to read content. Prompt-affecting mutations must finish before
token production; metadata added afterward must not affect tokenization.

Completions requests with token arrays go through vLLM, which owns truncation
and other input preprocessing. These requests require a render round trip and
an available renderer. Native Generate requests carry final tokens and bypass
rendering. Direct requests to the three `/render` endpoints retain model
routing and pass through without local token
production or response parsing.

Native gRPC text has no HTTP JSON envelope. Its compatibility path submits
the text to Completions rendering with the configured `modelName`; this does
not establish native gRPC token parity. Pretokenized gRPC requests use their
parser-provided tokens without rendering. This exception does not apply to
HTTP requests or the HTTP JSON embedded in Vertex AI gRPC requests.

Chat and Messages requests use the larger of `vllm.timeout` and
`vllm.mmTimeout`, including text-only requests. The default is 30 seconds.
The renderer does not inspect content to select a timeout. Set both values
to `5s` for a five-second render budget on all endpoints; multimodal requests
share that limit. An earlier caller deadline takes precedence.

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
| `modelName`                | - (required for `vllm`) | Model for startup probes, model-limit discovery, native gRPC text, and legacy Messages conversion. Native HTTP rendering retains the effective request model. |
| `vllm.messagesRenderMode`  | `auto`                 | Discover Messages rendering, or force `native` (pass-through) or `legacy` (deprecated conversion). |
| `vllm.url`                 | `http://localhost:8000` | Base URL of one vLLM render endpoint. Mutually exclusive with `endpointDiscovery`. |
| `vllm.endpointDiscovery`   | unset                   | Use endpoints published by data-layer discovery.                              |
| `vllm.endpointDiscovery.portRules` | empty             | Optional render port mappings; see [Endpoint discovery](#endpoint-discovery). |
| `vllm.endpointDiscovery.loadBalancer.type` | `round-robin` | Selection algorithm; `round-robin` is the only built-in algorithm. |
| `vllm.endpointDiscovery.attemptTimeout` | unset         | Optional positive duration limiting each render attempt, e.g. `1s`. |
| `vllm.endpointDiscovery.discoverModelLimits` | `false` | Probe model context capacity; see [Context limits](#context-limits). |
| `vllm.endpointDiscovery.minModelLen` | `0` | Minimum eligible renderer context capacity. |
| `vllm.endpointDiscovery.contextLimitLabel` | unset | Label supplying a positive context capacity, optionally bounded by probes. |
| `vllm.prefillOnly` | `false` | Use a one-token output budget only for rendering; see [Render-only output budget](#render-only-output-budget). |
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

Per-request errors are returned to the Director, which logs and continues;
downstream scorers fall back to their own paths. A missing selected render
endpoint, render error, or empty token result does not publish token IDs.
Neither Messages mode falls back to the other mode or to estimation.

## Deployment

Use a vLLM build exposing the render endpoints selected by the configuration.
Native Messages rendering requires [native `/v1/messages/render` support](https://github.com/vllm-project/vllm/pull/45803).
The deprecated `legacy` mode uses `/v1/chat/completions/render` for Messages.
`vllm launch render <model>` exposes render endpoints without a GPU.
The standalone Rust renderer `vllm-rs render <model>` exposes Completions
and Chat Completions rendering.
For `vllm serve <model>`, set `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1`.

The renderer can run beside EPP or behind a dedicated Service. Native HTTP
rendering requires it to accept the effective inference model, including
configured served-model aliases and adapters. `modelName` does not replace
those names in native render requests. When the inbound request carries an
`Authorization` header, it is forwarded verbatim on render requests, so an
endpoint started with `--api-key` accepts them; a bad token then fails at
render, the same way it fails at inference.

The startup warmup probe has no inbound request to borrow a credential
from. When `VLLM_API_KEY` is set on the EPP container, the probe sends
`Authorization: Bearer $VLLM_API_KEY`; the variable is warmup-only, and
request paths keep forwarding the client's `Authorization` header.

```yaml
# EPP pod spec (Python renderer)
containers:
- name: vllm-render
  image: vllm/vllm-openai:latest          # any image shipping `vllm launch render`
  command: ["vllm", "launch", "render"]
  args: ["${MODEL_NAME}", "--port=8000"]
  ports: [{name: render-http, containerPort: 8000}]
  readinessProbe: {httpGet: {path: /health, port: 8000}, periodSeconds: 5}

# EPP pod spec (Rust renderer, ~40x lighter image)
- name: vllm-render
  image: vllm/vllm-rs:latest              # image shipping `vllm-rs render`
  command: ["vllm-rs", "render"]
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
      messagesRenderMode: native
```

Plugin config — dedicated render Service:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "http://vllm-render.default.svc.cluster.local:8000"
      messagesRenderMode: native
```

Plugin config — dedicated render Service with TLS:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "https://vllm-render.default.svc.cluster.local:8000"
      messagesRenderMode: native
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

When the render endpoint requires a key — a `vllm launch render` endpoint
or a `vllm serve` instance started with `--api-key` — start the render
container with the key and expose it to the EPP container via
`VLLM_API_KEY` so the warmup probe authenticates:

```yaml
# vllm-api-key is the same Secret the endpoint reads its --api-key from
containers:
- name: epp
  env:
  - name: VLLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: vllm-api-key
        key: api-key
- name: vllm-render
  image: vllm/vllm-openai:latest
  command: ["vllm", "launch", "render"]
  args:
    - "${MODEL_NAME}"
    - "--port=8000"
    - "--api-key=$(VLLM_API_KEY)"
  env:
  - name: VLLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: vllm-api-key
        key: api-key
```

### Endpoint discovery

Set `vllm.endpointDiscovery: {}` to use discovered inference addresses and
ports with round-robin balancing. Omitted or empty `portRules` keeps each
endpoint's inference port. Use this when `/v1/*/render` is served on the same
listener as inference.

When render uses a different listener, configure ordered `portRules`. The
first matching Kubernetes label selector resolves the port as
`basePort + RankIndex`. `RankIndex` is the endpoint's zero-based position in
the `InferencePool`'s `targetPorts`, not a port-number difference. For example,
a decode endpoint at index 3 uses render port `8203` with `basePort: 8200`,
even when its inference port is `8003`.

This configuration maps prefill and decode endpoints to separate render port
ranges:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      endpointDiscovery:
        portRules:
          - selector:
              matchLabels:
                llm-d.ai/role: prefill
            basePort: 8000
          - selector:
              matchLabels:
                llm-d.ai/role: decode
            basePort: 8200
        loadBalancer:
          type: round-robin
```

The Kubernetes discovery path supplies Ready `InferencePool` endpoints and
removes endpoints when their pods become unready or leave the pool. Port rules
map these endpoints; they do not discover extra pods or ranks. Other discovery
plugins feed the same renderer path; `file-discovery`, for example, can supply
explicit render addresses and ports. All selected endpoints must serve the
configured model and expose the render routes. Discovered URLs use HTTP;
nonempty `caCertPath`, `clientCertPath`, or `clientKeyPath`, or
`insecureSkipVerify: true`, are rejected with `endpointDiscovery` at startup.
Use `vllm.url` for an HTTPS endpoint.

Each rule's `basePort` is required and must be between 1 and 65535. An empty or
omitted `selector` matches all endpoints, so a final catch-all rule can provide
a default base port. Nonempty rule lists have no inference-port fallback:
unmatched endpoints and endpoints whose resolved ports are out of range are
excluded, and the data layer logs the error. Invalid selectors and base ports
reject plugin configuration at startup.

Transport failures, attempt timeouts, HTTP 408, HTTP 429, and HTTP 5xx permit
one retry on a different discovered URL. Other HTTP errors return immediately.
Both attempts share the request's render timeout, capped by the caller's
deadline. By default, each attempt can use the full remaining request budget.
Set `vllm.endpointDiscovery.attemptTimeout` to opt into a shorter deadline on
each attempt, including retries. For example, `attemptTimeout: 1s` with
`timeout: 5s` permits retrying a slow endpoint after one second; the alternate
also has at most one second. Choose an attempt timeout that accommodates normal
render latency, including multimodal processing. Without an attempt timeout,
a slow endpoint can consume the request budget and leave no time for a retry.

With no eligible discovered endpoints, rendering returns an error. Render
failures do not remove URLs from subsequent requests; there is no render
circuit breaker. Retry exclusions preserve round-robin cursor progression
across the eligible endpoint list.

Each named token producer maintains its own endpoint set and balancing state.
Its HTTP/1.1 transport retains up to 16 idle connections per endpoint, with no
global idle-connection cap; idle connections expire after 90 seconds.
Alternative algorithms can be implemented in the tokenizer package through
`endpointLoadBalancer` and registered in `endpointLoadBalancerFactories`.
The picker supplies an independent snapshot without holding its endpoint lock;
algorithms must support concurrent calls to `Pick` and skip excluded URLs.

#### Context limits

Set `vllm.endpointDiscovery.discoverModelLimits: true` to verify each target's
configured model and `max_model_len` through `GET /v1/models`. Unknown targets,
failed probes, and observations older than 90 seconds are excluded. Probes run
every 30 seconds with at most four concurrent requests and a two-second timeout
per target. New endpoints can wait until the next probe cycle. Probes use
`VLLM_API_KEY` when set; they do not use an incoming request's credentials.
The models endpoint verifies model metadata, not render-route availability.

`vllm.endpointDiscovery.minModelLen` sets a nonnegative minimum context capacity
for the renderer pool. It requires model-limit discovery or
`vllm.endpointDiscovery.contextLimitLabel`, which names a metadata label holding
a positive integer capacity. With both sources configured, the smaller limit
applies. Missing or invalid labels exclude the endpoint.

```yaml
vllm:
  endpointDiscovery:
    discoverModelLimits: true
    minModelLen: 265088
    contextLimitLabel: llm-d.ai/render-context-limit
```

These options are unset by default. They select a renderer pool with a known
minimum capacity; they do not route by each request's token count, truncate
prompts, or validate the inference endpoint's capacity. Rendering errors retain
the existing token-producer error behavior.

#### Render-only output budget

Set `vllm.prefillOnly: true` when the renderer should validate the full prompt
without reserving the client's generation budget. The render copy uses
`max_tokens: 1`, caps `max_completion_tokens` at one when present, and sets
`min_tokens` to zero when present. The inference payload, prompt, and requested
generation budget are unchanged. The render copy re-serializes the JSON
envelope; nested content such as messages and tool schemas keeps its key order.
This option is false by default and also works with `vllm.url`.

This avoids rejecting a prompt solely because its requested output would
exceed the renderer's context capacity. The prompt itself must still fit;
this option does not enable partial matching or extend inference context limits.

Requests with a non-null `truncate_prompt_tokens` retain their original output
budget because automatic truncation depends on it.

A complete sample config that pairs this with `precise-prefix-cache-producer` and `prefix-cache-scorer` is at [`deploy/config/sim-epp-tokenizer-vllm-http-config.yaml`](../../../../../../../deploy/config/sim-epp-tokenizer-vllm-http-config.yaml).

## Live verification

The live tests use the actual parsers and token producer with a running vLLM
endpoint. They are opt-in and do not call `/tokenize`. Configure
`VLLM_RENDER_TEST_URL`, `VLLM_RENDER_TEST_MODEL`, and `VLLM_RENDER_TEST_ALIAS`.
Use two accepted names for the same model to exercise alias rewrites. Set
`VLLM_RENDER_TEST_AUTHORIZATION` to the full Authorization header when required.
Tool cases require the renderer's tool-choice and tool-parser configuration.

Run the appropriate tests in the builder with these variables forwarded:

- `TestNativeRenderLive`: all three native endpoints, content cases, model
  rewrites, direct render bypass, and exact token comparison after repackaging.
- `TestRenderLiveProtocolHandoff`: real renderer tokens through SGLang, vLLM
  HTTP and gRPC parsers; vLLM gRPC text and Vertex AI Chat rendering. This does
  not contact SGLang or gRPC serving endpoints.
- `TestMessagesDiscoveryLive`: native selection, simulated Chat-only support,
  authentication retry, and model errors. Requires native Messages rendering.
- `TestLegacyMessagesRenderLive`: automatic selection against a Chat-only
  renderer, with text, system, structured text, and tool-schema fixtures.
- `TestRenderServingLive`: set `VLLM_RENDER_TEST_SERVE=1` and point the URL at a
  server supporting both rendering and inference. Five Chat/Completions cases
  compare exact serving prompt IDs. Generate checks supplied IDs and serving
  token counts. Inference is serial and limited to one output token per prompt.

Render-only comparisons do not establish serving parity. These tests do not
cover loaded LoRA adapters, multimodal serving, or a deployed EPP/sidecar path.

---

## Related Documentation
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
- [Prefix Cache Scorer](../../../scheduling/scorer/prefix/README.md)
- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
