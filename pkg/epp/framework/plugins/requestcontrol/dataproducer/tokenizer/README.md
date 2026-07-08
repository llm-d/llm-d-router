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
  `/v1/chat/completions/render` over plain HTTP (TLS is not supported). Future
  protocol fields (e.g. `grpc`) can be added alongside `url` under the same
  `vllm` block.
- **`udsTokenizerConfig`**: deprecated gRPC-over-UDS sidecar (see warning below).

> [!WARNING]
> The `estimate` backend approximates token boundaries (≈4 bytes/token); its
> token IDs do not correspond to engine tokens. The precise prefix-cache scorer
> requires real tokens — configure a `vllm` `token-producer` explicitly for it.
> If omitted, the auto-created `estimate` producer satisfies the dependency but
> silently degrades precise cache correlation.

> [!WARNING]
> The `udsTokenizerConfig` backend (gRPC-over-UDS sidecar) is **deprecated**
> and will be removed in a future release. Existing configs continue to work
> but emit a deprecation warning at startup. Migrate to `vllm.url`. See
> [Migration](#migration-from-udstokenizerconfig) below.

## Config

| Parameter        | Default                 | Description                                                       |
| ---------------- | ----------------------- | ----------------------------------------------------------------- |
| `modelName`      | – (required for `vllm`) | Model whose tokenizer should be loaded / sent in render requests. |
| `vllm.url`       | `http://localhost:8000` | Base URL of the vLLM render endpoint (no trailing slash).         |
| `vllm.timeout`   | `5s`                    | Per-request timeout for text-only requests.                       |
| `vllm.mmTimeout` | `30s`                   | Per-request timeout for multimodal requests.                      |

The `estimate` backend tunes multimodal image placeholder estimation (empty uses
the defaults below):

| Parameter                          | Default   | Description                                                                |
| ---------------------------------- | --------- | -------------------------------------------------------------------------- |
| `estimate.image.mode`              | `dynamic` | `dynamic` (width×height/factor) or `static` (a constant per-image count).  |
| `estimate.image.defaultResolution` | 640×360   | Dynamic-mode fallback when an image's dimensions can't be decoded.         |
| `estimate.image.dynamic.factor`    | `1024`    | Dynamic-mode pixels-per-placeholder-token divisor.                         |
| `estimate.image.static.staticToken`| –         | Static-mode per-image placeholder count.                                   |

## Failure mode

Per-request errors are returned to the Director, which currently logs and
continues; downstream scorers fall back to their own paths.

## Deployment

The plugin calls `POST {http}/v1/completions/render` and
`POST {http}/v1/chat/completions/render`, both of which are exposed by
`vllm serve <model>` and by the GPU-less `vllm launch render <model>`.
Any reachable HTTP endpoint serving the same model the scheduler tokenizes
for will work — sidecar in the EPP pod (loopback) or a dedicated Service
shared by multiple EPP replicas.

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

A complete sample config that pairs this with `precise-prefix-cache-producer` and `prefix-cache-scorer` is at [`deploy/config/sim-epp-tokenizer-vllm-http-config.yaml`](../../../../../../../deploy/config/sim-epp-tokenizer-vllm-http-config.yaml).

## Migration from `udsTokenizerConfig`

The legacy UDS backend ran a per-pod tokenizer sidecar and connected over a
shared Unix domain socket. Replace it with the vLLM HTTP /render backend,
which calls the same model-serving pods (or a co-located `vllm launch render`
sidecar) and removes the dedicated tokenizer image.

Before:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    udsTokenizerConfig:
      socketFile: /tmp/tokenizer/tokenizer-uds.socket
```

After:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      url: "http://localhost:8000"   # or a shared render Service
```

See the [Deployment](#deployment) section above for sidecar vs shared-Service
options.

---

## Related Documentation
- [Precise Prefix Cache Scorer](../../../scheduling/scorer/preciseprefixcache/README.md)
- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
