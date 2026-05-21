# Token Producer Plugin

**Type:** `token-producer`

`DataProducer` plugin that renders the request prompt and publishes
`TokenIDs` (and a flat sorted `MultiModalFeatures` list) on
`InferenceRequestBody.TokenizedPrompt` for downstream consumers (scorers,
filters, other data producers).

Implements `fwkrhapi.DataProducer` and runs in the `PrepareRequestData`
phase, before filters and scorers. The plugin is idempotent: if
`InferenceRequestBody.TokenizedPrompt` is already populated by an earlier
producer, tokenization is skipped. Multi-modal features are flattened into the
upstream list shape, sorted by placeholder offset.

> [!NOTE]
> Legacy alias `tokenizer` is still accepted but logs a deprecation warning at
> instantiation. Prefer `token-producer` in new configs.

## Backend

The plugin calls vLLM's `/v1/completions/render` and
`/v1/chat/completions/render` over HTTP. An empty configuration falls back
to `vllm` with `http://localhost:8000`. Future protocol fields (e.g. `grpc`)
can be added alongside `http` under the same `vllm` block.

## Config

| Parameter        | Default                 | Description                                                       |
| ---------------- | ----------------------- | ----------------------------------------------------------------- |
| `modelName`      | – (required)            | Model whose tokenizer should be loaded / sent in render requests. |
| `vllm.http`      | `http://localhost:8000` | Base URL of the vLLM render endpoint (no trailing slash).         |
| `vllm.timeout`   | `5s`                    | Per-request timeout for text-only requests.                       |
| `vllm.mmTimeout` | `30s`                   | Per-request timeout for multimodal requests.                      |

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
      http: "http://localhost:8000"       # optional; this is the default
```

Plugin config — dedicated render Service:

```yaml
- type: token-producer
  parameters:
    modelName: "${MODEL_NAME}"
    vllm:
      http: "http://vllm-render.default.svc.cluster.local:8000"
```

A complete sample config that pairs this with `precise-prefix-cache-scorer`
is at
[`deploy/config/sim-epp-tokenizer-vllm-http-config.yaml`](/deploy/config/sim-epp-tokenizer-vllm-http-config.yaml).

---

## Related Documentation
- [Precise Prefix Cache Scorer](../../../scheduling/scorer/preciseprefixcache/README.md)
- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
