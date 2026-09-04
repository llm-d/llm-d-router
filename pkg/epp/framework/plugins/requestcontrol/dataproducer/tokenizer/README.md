# Token Producer Plugin

**Type:** `token-producer`

`DataProducer` plugin that tokenizes the request prompt and publishes
`TokenIDs` (and a flat sorted `MultiModalFeatures` list) on
`InferenceRequestBody.TokenizedPrompt` for downstream consumers (scorers,
filters, other data producers).

Implements `requestcontrol.DataProducer` and runs in the `PrepareRequestData`
phase, before filters and scorers. The plugin is idempotent: if
`InferenceRequestBody.TokenizedPrompt` is already populated by an earlier
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
- **`vllm`** (or `modelName`): calls vLLM's `/v1/completions/render` and
  `/v1/chat/completions/render` over HTTP or HTTPS. TLS is driven by the URL
  scheme (`https://`). For in-cluster endpoints using self-signed or private CA
  certificates, configure `vllm.caCertPath` to trust the CA, and optionally
  `vllm.clientCertPath`/`vllm.clientKeyPath` for mTLS. Future protocol fields
  (e.g. `grpc`) can be added under the same `vllm` block. The HTTP renderer uses
  either one configured URL or endpoints supplied by data-layer discovery.

> [!WARNING]
> The `estimate` backend approximates token boundaries (≈4 bytes/token); its
> token IDs do not correspond to engine tokens. The precise prefix-cache scorer
> requires real tokens — configure a `vllm` `token-producer` explicitly for it.
> If omitted, the auto-created `estimate` producer satisfies the dependency but
> silently degrades precise cache correlation.

## Config

| Parameter                  | Default                 | Description                                                                  |
| -------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `modelName`                | – (required for `vllm`) | Model whose tokenizer should be loaded / sent in render requests.            |
| `vllm.url`                 | `http://localhost:8000` | Base URL of one vLLM render endpoint. Mutually exclusive with `endpointDiscovery`. |
| `vllm.endpointDiscovery`   | unset                   | Use endpoints published by data-layer discovery.                              |
| `vllm.endpointDiscovery.portRules` | empty             | Optional render port mappings; see [Endpoint discovery](#endpoint-discovery). |
| `vllm.endpointDiscovery.loadBalancer.type` | `round-robin` | Selection algorithm; `round-robin` is the only built-in algorithm. |
| `vllm.timeout`             | `5s`                    | Per-request timeout for text-only requests.                                  |
| `vllm.mmTimeout`           | `30s`                   | Per-request timeout for multimodal requests.                                 |
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
continues; downstream scorers fall back to their own paths.

## Deployment

The plugin calls `POST {http}/v1/completions/render` and
`POST {http}/v1/chat/completions/render`, both of which are exposed by
`vllm serve <model>` and by the GPU-less `vllm launch render <model>`.
Any reachable HTTP endpoint serving the same model the scheduler tokenizes
for will work — sidecar in the EPP pod (loopback) or a dedicated Service
shared by multiple EPP replicas. When the inbound request carries an
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
use `vllm.url` for an HTTPS endpoint.

Each rule's `basePort` is required and must be between 1 and 65535. An empty or
omitted `selector` matches all endpoints, so a final catch-all rule can provide
a default base port. Nonempty rule lists have no inference-port fallback:
unmatched endpoints and endpoints whose resolved ports are out of range are
excluded, and the data layer logs the error. Invalid selectors and base ports
reject plugin configuration at startup.

Transport failures, attempt timeouts, HTTP 408, HTTP 429, and HTTP 5xx permit
one retry on a different discovered URL. Other HTTP errors return immediately.
Both attempts share `vllm.timeout` or `vllm.mmTimeout`, capped by the caller's
deadline. When an alternate URL is available, the first attempt gets half the
remaining budget and the retry gets the remainder. A single endpoint uses the
full budget. With no discovered endpoints, rendering returns an error. Failed
URLs remain eligible for subsequent requests until discovery removes them;
there is no circuit breaker or separate render health probe.

Each named token producer maintains its own endpoint set and balancing state.
Its HTTP/1.1 transport retains up to 16 idle connections per endpoint, with no
global idle-connection cap; idle connections expire after 90 seconds.
Alternative algorithms can be implemented in the tokenizer package through
`endpointLoadBalancer` and registered in `endpointLoadBalancerFactories`.
The picker supplies an independent snapshot without holding its endpoint lock;
algorithms must support concurrent calls to `Pick`.

A complete sample config that pairs this with `precise-prefix-cache-producer` and `prefix-cache-scorer` is at [`deploy/config/sim-epp-tokenizer-vllm-http-config.yaml`](../../../../../../../deploy/config/sim-epp-tokenizer-vllm-http-config.yaml).

---

## Related Documentation
- [Precise Prefix Cache Scorer](../../../scheduling/scorer/preciseprefixcache/README.md)
- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
