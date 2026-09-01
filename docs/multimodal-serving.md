# Multimodal Serving

Serve text, audio, and image workloads through a single llm-d InferencePool
using the `modality-filter` EPP plugin for architecture-aware endpoint
selection.

## Overview

llm-d's EPP routes requests to model-serving pods through a
Filter-Score-Pick pipeline. The `modality-filter` plugin extends this
pipeline to support multimodal workloads: it reads the request path and
drops endpoints whose `llm-d.ai/model-arch` pod label is incompatible with
the requested modality.

A single InferencePool can contain pods serving different model
architectures (text LLMs, omni models, diffusion models, STT encoders).
The filter ensures each request only reaches pods that can handle it.

### Path-to-architecture mapping

| Request path               | Compatible `model-arch` values         |
|----------------------------|----------------------------------------|
| `/v1/audio/speech`         | `omni-llm`, `autoregressive-tts`       |
| `/v1/audio/transcriptions` | `encoder-decoder-stt`                  |
| `/v1/images/generations`   | `diffusion`                            |
| `/v1/embeddings`           | (no filter — all pool pods eligible)   |
| `/v1/chat/completions`     | (no filter — all pool pods eligible)   |

Paths not in this mapping pass all endpoints through unchanged.

## Prerequisites

- Kubernetes cluster with [Gateway API](https://gateway-api.sigs.k8s.io/)
  CRDs installed (`inference.networking.k8s.io/v1` `InferencePool`)
- A Gateway API `Gateway` resource (e.g. Istio, kGateway, Envoy Gateway)
- llm-d EPP image built with the `modality-filter` plugin registered
  (in-tree since the plugin was merged; `--allow-experimental-plugins=true`
  required)
- vLLM-omni container image for the target model

## Deploying an omni model

### 1. Label worker pods

Worker pods must carry two labels:

```yaml
labels:
  app: <pool-name>                  # InferencePool selector
  llm-d.ai/model-arch: omni-llm    # modality-filter selector
```

The `app` label matches the InferencePool's `spec.selector`. The
`model-arch` label tells the EPP which modalities this pod supports.

Valid `model-arch` values: `omni-llm`, `autoregressive-tts`,
`encoder-decoder-stt`, `diffusion`, `autoregressive-llm`.

### 2. Deploy the vLLM-omni worker

Use the component at `deploy/components/vllm-omni/`. Required variables:

| Variable                   | Description                          | Example                             |
|----------------------------|--------------------------------------|-------------------------------------|
| `POOL_NAME`                | InferencePool name                   | `multimodal-pool`                   |
| `VLLM_OMNI_IMAGE`         | vLLM-omni container image            | `quay.io/rh_et_wd/vllm-omni-cuda`  |
| `VLLM_OMNI_REPLICA_COUNT` | Number of worker replicas            | `1`                                 |
| `OMNI_MODEL_NAME`         | HuggingFace model ID                 | `Qwen/Qwen2.5-Omni-7B`             |
| `OMNI_SERVED_MODEL_NAME`  | Model name exposed via the API       | `omni-1`                            |
| `OMNI_GPU_COUNT`           | GPUs per pod                         | `1`                                 |
| `OMNI_MODEL_VOLUME`        | Volume spec for model weights        | `emptyDir: {}`                      |

### 3. Configure the EPP

The EPP must include `modality-filter` in its plugin list. Use the config
at `deploy/config/multimodal-epp-config.yaml` or add `modality-filter` to
an existing config:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: modality-filter
- type: prefix-cache-scorer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: modality-filter
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
```

### 4. Create the InferencePool

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: multimodal-pool
spec:
  selector:
    matchLabels:
      app: multimodal-pool
  endpointPickerRef:
    name: multimodal-epp
    kind: Service
    port:
      number: 9002
  targetPorts:
  - number: 8000
```

### 5. Create HTTPRoutes

Route each multimodal endpoint directly to the InferencePool by
exact-path match. See `deploy/environments/dev/multimodal-direct/httproutes.yaml`
for the full set of routes.

The key principle: exact-path matches take precedence over `PathPrefix: /`
catch-all routes per Gateway API's route-precedence rules, so these routes
can coexist with a coordinator catch-all if one is present.

## Validation

After deploying, verify each endpoint reaches the correct pod:

```bash
# TTS — should return audio content from an omni-llm pod
curl -s http://<gateway>/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"omni-1","input":"Hello from llm-d multimodal.","voice":"default"}' \
  --output /tmp/speech.wav
file /tmp/speech.wav  # expect: RIFF (little-endian) data, WAVE audio

# Text — should return a chat completion (unaffected by multimodal config)
curl -s http://<gateway>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"omni-1","messages":[{"role":"user","content":"Say hello."}]}'

# Mixed traffic — run both in parallel and confirm neither errors
for i in $(seq 1 5); do
  curl -s http://<gateway>/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"omni-1","input":"Test '"$i"'","voice":"default"}' \
    --output /dev/null -w "TTS $i: %{http_code}\n" &
  curl -s http://<gateway>/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"omni-1","messages":[{"role":"user","content":"Count to '"$i"'"}]}' \
    --output /dev/null -w "Text $i: %{http_code}\n" &
done
wait
```

The RFC's Milestone 1 gate is: audio routing works, text is unaffected,
mixed traffic is stable.

## Architecture

```
Client
  │
  ▼
Gateway (HTTPRoute exact-path match)
  │
  ▼
InferencePool / EPP
  │  modality-filter: reads :path, drops incompatible model-arch pods
  │  decode-filter / prefix-cache-scorer / max-score-picker: standard pipeline
  │
  ▼
vLLM-omni worker (llm-d.ai/model-arch: omni-llm)
```

One gateway hop. No coordinator involvement for Milestone 1 multimodal
paths. The coordinator remains available for paths that require pipeline
orchestration (e.g. disaggregated encode/prefill/decode in future
milestones).
