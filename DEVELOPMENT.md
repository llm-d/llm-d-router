# Development

Documentation for developing the inference scheduler.

## Table of Contents

- [Development](#development)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Kind Development Environment](#kind-development-environment)
    - [Accessing the Gateway](#accessing-the-gateway)
    - [Prometheus Monitoring](#prometheus-monitoring)
    - [Grafana Dashboard](#grafana-dashboard)
    - [Development Cycle](#development-cycle)
    - [Debugging](#debugging)
    - [Inference Disaggregation Modes](#inference-disaggregation-modes)
      - [1. EPD — No Disaggregation (default)](#1-epd--no-disaggregation-default)
      - [2. Prefill/Decode (P/D) Disaggregation](#2-prefilldecode-pd-disaggregation)
      - [3. Encode/Prefill-Decode (E/PD) Disaggregation](#3-encodeprefill-decode-epd-disaggregation)
      - [4. Encode/Prefill/Decode (E/P/D) Disaggregation](#4-encodeprefilldecode-epd-disaggregation)
      - [5. Disaggregated Setup Verification](#5-disaggregated-setup-verification)
    - [Combining Scenarios with Data Parallel and KV Cache](#combining-scenarios-with-data-parallel-and-kv-cache)
    - [Simulator vs Real vLLM](#simulator-vs-real-vllm)
      - [Deploying with Simulator (default)](#deploying-with-simulator-default)
      - [Deploying with Real vLLM](#deploying-with-real-vllm)
      - [Deployment Component Summary](#deployment-component-summary)
    - [Cleanup](#cleanup)
  - [Running Tests](#running-tests)
    - [Unit Tests](#unit-tests)
    - [Integration Tests](#integration-tests)
    - [Filtered Tests](#filtered-tests)
    - [End-to-End Tests](#end-to-end-tests)
    - [Coverage](#coverage)
  - [Tokenization Architecture](#tokenization-architecture)
  - [Kubernetes Development Environment](#kubernetes-development-environment)
    - [Infrastructure Setup](#infrastructure-setup)
    - [RBAC and Permissions](#rbac-and-permissions)
    - [Developer Setup](#developer-setup)
    - [Environment Configuration](#environment-configuration)
    - [Deploying Changes](#deploying-changes)
    - [Cleanup Environment](#cleanup-environment)
  - [Submitting Changes](#submitting-changes)
    - [Scope](#scope)
    - [Presubmit](#presubmit)

## Overview

This repo builds the **Endpoint Picker Plugin (EPP)**, the inference scheduling component
that routes requests to vLLM backends. The EPP runs alongside a Gateway API implementation
and picks backends based on KV cache state, prefill locality, and load. A second binary,
the **routing sidecar** (`cmd/pd-sidecar/`), handles disaggregation routing.

The KIND environment is the easiest way to get started: one command, no cloud account.
A real Kubernetes cluster setup is covered later for shared or production-like testing.

## Requirements

- [Make] `v4`+
- [Golang] `v1.25`+
- [Docker] (or [Podman])
- [Kubernetes in Docker (KIND)]
- [Kubectl] `v1.25`+

[Make]:https://www.gnu.org/software/make/
[Golang]:https://go.dev/
[Docker]:https://www.docker.com/
[Podman]:https://podman.io/
[Kubernetes in Docker (KIND)]:https://github.com/kubernetes-sigs/kind
[Kubectl]:https://kubectl.docs.kubernetes.io/installation/kubectl/

## Kind Development Environment

Deploys the EPP, vLLM simulator, and Gateway API implementation into a local KIND cluster:

```bash
make env-dev-kind
```

Creates a new `kind` cluster (or reuses an existing one) in the `default` namespace. The cluster name defaults to `KIND_CLUSTER_NAME` in `Makefile.kind.mk` (currently `$(PROJECT_NAME)-dev`), and the kubectl context is `kind-<cluster-name>`.

> [!NOTE]
> You can pre-pull external images to avoid slow downloads:
> ```
> docker pull ghcr.io/llm-d/llm-d-inference-sim:v0.8.2
> docker pull ghcr.io/llm-d/llm-d-uds-tokenizer:dev
> ```

### Accessing the Gateway

Use port-forward for local development:

```bash
kubectl --context kind-llm-d-router-dev \
  port-forward service/inference-gateway-istio 8080:80
```

The default model depends on the disaggregation scenario:
- **EPD / P/D** (no encoder): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **E/PD / E/P/D** (with encoder, `DISAGG_E=true`): `Qwen/Qwen3-VL-2B-Instruct`

To confirm what model is available:

```bash
curl -s http://localhost:8080/v1/models | jq
```

Make a text completion request:

```bash
curl -s -w '\n' http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"hi","max_tokens":10,"temperature":0}' | jq
```

For multimodal scenarios (`DISAGG_E=true`), send an image request:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
      {"type":"text","text":"What is in this image?"}
    ]}],
    "max_tokens": 50
  }' | jq
```
<details>
<summary>Alternative access methods (NodePort, LoadBalancer)</summary>

**NodePort**

The gateway is also exposed as a NodePort and is exposed on your development machine on port 30080 by default. Override this at
cluster creation time with any free port in the range 30000-32767:

```bash
KIND_GATEWAY_HOST_PORT=<selected-port> make env-dev-kind
```

The service is then accessible at `http://localhost:30080`.

**LoadBalancer**

```bash
# Install and run cloud-provider-kind:
go install sigs.k8s.io/cloud-provider-kind@latest && cloud-provider-kind &
kubectl --context kind-llm-d-router-dev get service inference-gateway-istio
# Wait for the LoadBalancer External-IP to become available.
# The service is accessible over port 80.
```

</details>

### Prometheus Monitoring

To deploy Prometheus alongside the dev environment:

```bash
PROM_ENABLED=true make env-dev-kind
```

Prometheus will be accessible at `http://localhost:30090`. To use a different host port:

```bash
PROM_ENABLED=true KIND_PROM_HOST_PORT=30091 make env-dev-kind
```

### Grafana Dashboard

The upstream [Inference Gateway dashboard] covers EPP, inference pool, and vLLM metrics.

Add a Prometheus datasource at `http://localhost:30090`, then import the JSON via
**Dashboards > New > Import**. See the
[Grafana installation docs](https://grafana.com/docs/grafana/latest/setup-grafana/installation/)
for setup.

[Inference Gateway dashboard]:https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/tools/dashboards/inference_gateway.json

> [!NOTE]
> For significant customization beyond the standard deployment, use the `deploy/components`
> directory with `kubectl kustomize`. The `deploy/environments/kind` deployment is a useful
> reference.

### Development Cycle

Edit your code, then rebuild and reload into the cluster:

```bash
make env-dev-kind
```

This rebuilds the EPP image (tagged `dev` by default) and loads it into the cluster.
To use a specific tag:

```bash
EPP_TAG=0.0.4 make env-dev-kind
```

Then restart the deployment to pick up the new image:

```bash
kubectl rollout restart deployment tinyllama-1-1b-chat-v1-0-endpoint-picker
```

> [!NOTE]
> Images are built with debug symbols stripped (`-s -w`) by default. To produce a
> debuggable image for use with `dlv`, override `LDFLAGS`:
> ```bash
> LDFLAGS="" make image-build-epp
> ```
> To load a different vLLM simulator tag, set `VLLM_SIMULATOR_TAG`:
> ```bash
> VLLM_SIMULATOR_TAG=<tag> make env-dev-kind
> ```

### Debugging

**Building a debug image**

Debug symbols are stripped by default (`-s -w`). To build an image with symbols preserved
(required for `dlv` or other debuggers), clear `LDFLAGS`:

```bash
LDFLAGS="" make image-build-epp
```

To use a non-default runtime base image (e.g. a UBI variant or a debug-capable image),
set `BASE_IMAGE`:

```bash
BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-micro:9.7 make image-build-epp
```

Both overrides can be combined:

```bash
LDFLAGS="" BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-micro:9.7 make image-build-epp
```

> [!NOTE]
> The default base image is `gcr.io/distroless/static:nonroot`. If you switch to `scratch`,
> you must copy CA certificates from the builder stage manually - see the comments in
> `Dockerfile.epp` for guidance.

**Attaching an ephemeral debug container**

The distroless runtime image has no shell. For ad-hoc inspection (filesystem, processes,
network), attach an ephemeral container without modifying the image:

```bash
kubectl debug -it <pod-name> -n <namespace> \
    --image=busybox \
    --target=epp \
    -- sh
```

This creates a throwaway container that shares the pod's PID/network/filesystem namespace.
Ephemeral container support requires Kubernetes 1.23+.

To connect `dlv` to a running EPP process, build and deploy a debug image first (see above),
then attach `dlv` via the ephemeral container:

```bash
# 1. Build and load the debug image
LDFLAGS="" make image-build-epp
kind load docker-image $(EPP_IMAGE) --name llm-d-router-dev

# 2. Restart the deployment to pick up the new image
kubectl rollout restart deployment tinyllama-1-1b-chat-v1-0-endpoint-picker

# 3. Attach dlv in an ephemeral container
kubectl debug -it <pod-name> -n <namespace> \
    --image=ghcr.io/go-delve/delve:latest \
    --target=epp \
    -- dlv attach 1
```

> [!NOTE]
> `dlv attach 1` assumes the EPP binary is PID 1. Confirm with `ps` in a `busybox`
> ephemeral container if the pod runs additional processes.

### Inference Disaggregation Modes

The deployment uses three atomic Kustomize components (`vllm-encode`, `vllm-prefill`,
`vllm-decode`) that compose to form any disaggregation scenario. Disaggregation is
controlled by two independent boolean flags:

| Flag | Default | Meaning |
|---|---|---|
| `DISAGG_E` | `false` | Deploy a separate **Encoder** pod |
| `DISAGG_P` | `false` | Deploy a separate **Prefill** pod |

The combination of these flags determines the scenario:

| `DISAGG_E` | `DISAGG_P` | Scenario | Components |
|---|---|---|---|
| `false` | `false` | EPD (default) | decode only |
| `false` | `true` | P/D | prefill + decode |
| `true` | `false` | E/PD | encode + decode |
| `true` | `true` | E/P/D | encode + prefill + decode |

Data parallel and KV cache are orthogonal options that can be combined with any scenario:

| Variable | Default | Description |
|---|---|---|
| `VLLM_DATA_PARALLEL_SIZE` | `1` | Number of data-parallel ranks per vLLM pod. Applies to ALL pod types (encode, prefill, decode). Set to `2`+ to enable |
| `KV_CACHE_ENABLED` | `false` | Enable KV cache-aware scheduling |
| `VLLM_EXTRA_ARGS_E` | _(empty)_ | Additional flags appended to the Encoder vLLM container args. Use `--flag=value` format. Example: `--mm-processor-kwargs={}` |
| `VLLM_EXTRA_ARGS_P` | _(empty)_ | Additional flags appended to the Prefill vLLM container args. Use `--flag=value` format. Example: `--gpu-memory-utilization=0.9` |
| `VLLM_EXTRA_ARGS_D` | _(empty)_ | Additional flags appended to the Decode vLLM container args. Use `--flag=value` format. Example: `--tensor-parallel-size=2` |

For technical details, refer to [docs/disaggregation.md](docs/disaggregation.md) and
[deploy/environments/dev/README.md](deploy/environments/dev/README.md).

#### 1. EPD — No Disaggregation (default)

Unified deployment handling all stages (encode, prefill, decode) in a single pod. No separate encoder or prefill pods:

```bash
make env-dev-kind
```

Verify:
```bash
curl -s http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"hi","max_tokens":10}' | jq
```

#### 2. Prefill/Decode (P/D) Disaggregation

Separate Prefill and Decode pods:

```bash
DISAGG_P=true make env-dev-kind
```

> **Note:** The legacy `PD_ENABLED=true` is deprecated. Use `DISAGG_P=true` instead.

Verify:
```bash
curl -s http://localhost:30080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"hi","max_tokens":10}' | jq
```

#### 3. Encode/Prefill-Decode (E/PD) Disaggregation

Separate Encoder pods; Prefill and Decode combined. Defaults to `Qwen/Qwen3-VL-2B-Instruct` for multimodal support:

```bash
DISAGG_E=true make env-dev-kind
```

Verify with an image request:
```bash
curl -s http://localhost:30080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
      {"type":"text","text":"What is in this image?"}
    ]}],
    "max_tokens": 50
  }' | jq
```

#### 4. Encode/Prefill/Decode (E/P/D) Disaggregation

Fully disaggregated — separate Encoder, Prefill, and Decode pods. Defaults to `Qwen/Qwen3-VL-2B-Instruct`:

```bash
DISAGG_E=true DISAGG_P=true make env-dev-kind
```

> **Note:** The legacy `EPD_ENABLED=true` is deprecated. Use `DISAGG_E=true DISAGG_P=true` instead.

Verify with an image request:
```bash
curl -s http://localhost:30080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
      {"type":"text","text":"What is in this image?"}
    ]}],
    "max_tokens": 50
  }' | jq
```

#### 5. Disaggregated Setup Verification

After deploying any disaggregation mode, verify with a basic request:

```bash
kubectl --context kind-llm-d-router-dev port-forward service/inference-gateway-istio 8080:80
```

For multimodal disaggregation (E/PD, E/P/D), test with an image request to verify the encoder stage is working:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "image_url", "image_url": { "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg" } },
          { "type": "text", "text": "What is in this image?" }
        ]
      }
    ],
    "max_tokens": 100
  }'
```

### Combining Scenarios with Data Parallel and KV Cache

```bash
# P/D with 2-rank data parallel decode
DISAGG_P=true VLLM_DATA_PARALLEL_SIZE=2 make env-dev-kind

# EPD with KV cache-aware scheduling
KV_CACHE_ENABLED=true make env-dev-kind

# Fully disaggregated E/P/D with data parallel and KV cache
DISAGG_E=true DISAGG_P=true VLLM_DATA_PARALLEL_SIZE=2 KV_CACHE_ENABLED=true make env-dev-kind
```

### Simulator vs Real vLLM

The `deploy/components/` directory contains all reusable Kustomize components:

**vLLM workload components** — the base pods, split into three atomic building blocks:
- `vllm-encode/` — Encoder pod (multimodal, `--mm-encoder-only`)
- `vllm-prefill/` — Prefill pod
- `vllm-decode/` — Decode pod with routing sidecar

**Deployment overlays** — applied on top of the base components:
- `overlays/simulator/` — adds `--mode=${VLLM_SIM_MODE}`, UDS tokenizer sidecar, KV cache args, and `--zmq-endpoint` on Decode. Included by default in all dev scenario overlays.
- `overlays/real-vllm/` — adds `--kv-events-config` on Decode, `--ec-transfer-config` on Encode, and a shared PVC for encoder embeddings.

**Infrastructure components** — shared cluster infrastructure:
- `inference-gateway/` — Endpoint Picker (EPP) deployment, services, RBAC, InferencePool, Gateway, and HTTPRoute
- `istio-control-plane/` — Istiod control plane (namespaces, configmaps, RBAC, webhooks)
- `monitoring/` — Prometheus ServiceMonitors for EPP and vLLM metrics
- `crds-gateway-api/` — Gateway API CRDs
- `crds-gie/` — Gateway API Inference Extension CRDs
- `crds-istio/` — Istio CRDs

#### Deploying with Simulator (default)

The dev scenario overlays include the simulator component by default. No extra flags needed:

```bash
# EPD — no disaggregation (default)
make env-dev-kind

# P/D — prefill + decode
DISAGG_P=true make env-dev-kind

# E/PD — encode + prefill-decode
DISAGG_E=true make env-dev-kind

# E/P/D — encode + prefill + decode (fully disaggregated)
DISAGG_E=true DISAGG_P=true make env-dev-kind

# Any mode with data parallel
VLLM_DATA_PARALLEL_SIZE=2 make env-dev-kind
DISAGG_P=true VLLM_DATA_PARALLEL_SIZE=2 make env-dev-kind
```

#### Deploying with Real vLLM

> [!NOTE]
> The section will be updated soon

The `deploy/components/overlays/real-vllm/` component is ready to use. It provides
all the real vLLM-specific configuration (KV events, EC transfer, shared PVC). To use
it, create a scenario overlay that includes it instead of the simulator overlay.
For example, to deploy P/D with real vLLM:

```yaml
# deploy/environments/prod/p-d/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- ../../../components/vllm-prefill/
- ../../../components/vllm-decode/

patches:
- path: ../../dev/p-d/patch-decode.yaml    # reuse scenario patches

components:
- ../../../components/overlays/real-vllm/  # real vLLM instead of simulator
```

Then deploy with:

```bash
VLLM_IMAGE=vllm/vllm-openai:v0.16.0 \
  kubectl kustomize deploy/environments/prod/p-d \
  | envsubst | kubectl apply -f -
```

For encode disaggregation scenarios (E/PD, E/P/D), the `real-vllm` overlay automatically
adds `--kv-events-config` to the Decode deployment (per-pod ZMQ publisher for KV cache
events), `--ec-transfer-config` (producer role) to the Encode deployment, and creates
a shared PVC (`ec-cache-pvc`) for encoder embeddings transfer.

#### Deployment Component Summary

| Component | What it adds | When to use |
|---|---|---|
| `overlays/simulator/` | `--mode=${VLLM_SIM_MODE}`, UDS tokenizer, KV cache args, `--zmq-endpoint` on Decode | Dev/test with simulator image |
| `overlays/real-vllm/` | `--kv-events-config` on Decode (per-pod ZMQ publisher), `--ec-transfer-config` on Encode, ec-cache PVC | Production with real vLLM image |

| Variable | Default | Description |
|---|---|---|
| `VLLM_IMAGE` | `ghcr.io/llm-d/llm-d-inference-sim:v0.8.2` | vLLM container image to deploy. Can be a simulator or a real vLLM image (e.g., `vllm/vllm-openai:v0.16.0`). Defaults to the simulator image. |
| `VLLM_SIM_MODE` | `echo` | Simulator response mode. `echo` returns the input prompt as the response (useful for routing validation). `random` returns random sentences from a pre-defined bank. Only applies when using the simulator overlay. |

### Cleanup

```bash
make clean-env-dev-kind
```

> [!NOTE]
> Port mappings (`KIND_GATEWAY_HOST_PORT`, `KIND_PROM_HOST_PORT`) are baked into the cluster
> at creation time. To change them, run `make clean-env-dev-kind` first, then recreate.

## Running Tests

### Unit Tests

Coverage and race detection are always enabled.

```bash
make test-unit          # run all unit tests (epp + sidecar)
make test-unit-epp      # epp only
make test-unit-sidecar  # sidecar only
```

### Integration Tests

Requires the KIND development environment to be running (`make env-dev-kind`).

```bash
make test-integration   # coverage and race detection always enabled
```

### Filtered Tests

```bash
make test-filter PATTERN=TestName           # epp tests matching pattern
make test-filter PATTERN=TestName TYPE=sidecar
```

### End-to-End Tests

```bash
make test-e2e
```

This creates a temporary Kind cluster named `e2e-tests`, runs the full test suite against it, and deletes the cluster on completion.

**Keeping the cluster on failure**

Set `E2E_KEEP_CLUSTER_ON_FAILURE=true` to preserve the cluster (and, when using a real cluster, all created Kubernetes objects) when any test fails. This is useful for inspecting pod logs, events, or cluster state after a failure.

```bash
E2E_KEEP_CLUSTER_ON_FAILURE=true make test-e2e
```

When set, a successful run still cleans up normally — the cluster is only kept if there is at least one test failure.

**Accessing the cluster after a failure**

E2E tests do not update the host's kubeconfig to point at the `e2e-tests` Kind cluster. After a preserved failure, export the kubeconfig manually:

```bash
# Merge into the default kubeconfig ($HOME/.kube/config or $KUBECONFIG)
kind export kubeconfig --name e2e-tests

# Or write to a specific file
kind export kubeconfig --name e2e-tests --kubeconfig /path/to/kubeconfig
```

Then use it as normal:

```bash
kubectl --context kind-e2e-tests get pods
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `E2E_KEEP_CLUSTER_ON_FAILURE` | `false` | Preserve the Kind cluster (or Kubernetes objects) when the suite fails |
| `E2E_PORT` | `30080` | Host port mapped to the gateway NodePort |
| `E2E_METRICS_PORT` | `32090` | Host port mapped to the EPP metrics NodePort |
| `K8S_CONTEXT` | _(empty)_ | Use an existing cluster context instead of creating a Kind cluster |
| `NAMESPACE` | `default` | Namespace to deploy test resources into |
| `CONTAINER_RUNTIME` | `docker` | Container runtime used to load images into Kind (`docker` or `podman`) |
| `READY_TIMEOUT` | `3m` | How long to wait for resources to become ready |
| `EPP_IMAGE` | `ghcr.io/llm-d/llm-d-inference-scheduler:dev` | EPP image loaded into the Kind cluster |
| `DISAGG_E` | `false` | Deploy a separate Encoder pod. See [Inference Disaggregation Modes](#inference-disaggregation-modes) |
| `DISAGG_P` | `false` | Deploy a separate Prefill pod. See [Inference Disaggregation Modes](#inference-disaggregation-modes) |
| `VLLM_DATA_PARALLEL_SIZE` | `1` | Number of data-parallel ranks per vLLM pod. Applies to all pod types. Set to `2`+ to enable multi-rank inference. See [Combining Scenarios with Data Parallel and KV Cache](#combining-scenarios-with-data-parallel-and-kv-cache) |
| `VLLM_EXTRA_ARGS_E` | _(empty)_ | Additional flags for the Encoder vLLM container (e.g. `--mm-processor-kwargs={}`) |
| `VLLM_EXTRA_ARGS_P` | _(empty)_ | Additional flags for the Prefill vLLM container (e.g. `--gpu-memory-utilization=0.9`) |
| `VLLM_EXTRA_ARGS_D` | _(empty)_ | Additional flags for the Decode vLLM container (e.g. `--tensor-parallel-size=2`) |
| `VLLM_IMAGE` | `ghcr.io/llm-d/llm-d-inference-sim:v0.8.2` | vLLM container image to deploy. Can be a simulator or a real vLLM image (e.g., `vllm/vllm-openai:v0.16.0`) |
| `VLLM_SIM_MODE` | `echo` | Simulator response mode. Supported values: `echo` (returns the input prompt as the response), `random` (returns a random sentence from a pre-defined bank) |
| `SIDECAR_IMAGE` | `ghcr.io/llm-d/llm-d-routing-sidecar:dev` | Routing sidecar image loaded into the Kind cluster |
| `UDS_TOKENIZER_IMAGE` | `ghcr.io/llm-d/llm-d-uds-tokenizer:dev` | UDS tokenizer image loaded into the Kind cluster |

### Coverage

Coverage profiles are written to `coverage/` (gitignored). To generate an HTML report:

```bash
make coverage-report
open coverage/epp.html
```

To compare coverage against `main`:

```bash
make test-unit          # run tests on your branch first
make coverage-compare   # builds a baseline from main in a temp worktree, then diffs
```

To compare against a different ref:

```bash
make coverage-compare BASE_REF=release-0.5
```

To compare against multiple baselines in one session:

```bash
make test-unit
make coverage-compare                                              # vs main
make coverage-compare BASE_REF=release-0.6 COVERAGE_LABEL=release-0.6
```

If a worktree for the target ref already exists locally it is reused and not removed afterwards. A newly created worktree is always cleaned up after the comparison.

> [!NOTE]
> CI runs the same comparison automatically on every PR: one report against `main`
> and one against the most recent `release-*` branch. Both appear in the GitHub
> Actions Job Summary for the run.

## Tokenization Architecture

> [!NOTE]
> **Python is NOT required**. Previous EPP versions (before v0.5.1) used embedded Python
> tokenizers.

The project uses **UDS (Unix Domain Socket)** tokenization. A separate UDS tokenizer
sidecar container handles tokenization; the EPP itself does not. Previous approaches
(daulet/tokenizers, direct Python/vLLM linking) are deprecated and no longer used.

The UDS tokenizer image is built and published by the
[llm-d-kv-cache](https://github.com/llm-d/llm-d-kv-cache) repository.
Published images are available at `ghcr.io/llm-d/llm-d-uds-tokenizer:<tag>`

- The `:dev` tag tracks the kv-cache `main` branch.
- To pin a specific release, set `UDS_TOKENIZER_TAG` (or `UDS_TOKENIZER_IMAGE` for a
  fully custom reference):

  ```bash
  UDS_TOKENIZER_TAG=v0.7.0 make env-dev-kind
  ```

- To use a different registry, set `IMAGE_REGISTRY` (shared with all other images):

  ```bash
  IMAGE_REGISTRY=quay.io/my-org make env-dev-kind
  ```

- To build the image from source, run `make image-build-uds` in the `llm-d-kv-cache` repo.

## Kubernetes Development Environment

A real Kubernetes cluster can be used for development and testing. Setup has two layers:

- **Cluster infrastructure**: CRDs and operators, installed once by a cluster admin.
- **Developer environment**: your namespace and workloads.

On a shared cluster, each developer uses a separate namespace. On a personal cluster,
`default` works fine.

### Infrastructure Setup

> [!CAUTION]
> Only run this if you are the cluster admin. Applying CRDs and operators can be disruptive
> to other developers sharing the cluster.

Install Gateway API and GIE CRDs:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/standard-install.yaml
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/manifests.yaml
```

Install kgateway:

```bash
KGTW_VERSION=v2.0.2
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION \
  kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION \
  kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
  --set inferenceExtension.enabled=true
```

For more details, see the Gateway API Inference Extension
[getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).

### RBAC and Permissions

EPP is namespace-scoped. Its `Role` grants `get/watch/list` on `inferencepools` and `pods`,
plus `create` on `tokenreviews`/`subjectaccessreviews` for metrics auth
(`--metrics-endpoint-auth=true`, the default). To disable metrics auth and avoid the
cluster-scoped RBAC requirement, use `--metrics-endpoint-auth=false`.

### Developer Setup

> [!NOTE]
> This setup requires building and pushing container images to your own private registry.

**1. Set your namespace.**

```bash
export NAMESPACE=your-dev-namespace
kubectl create namespace ${NAMESPACE}
kubectl config set-context --current --namespace="${NAMESPACE}"
```

> [!NOTE]
> If you are using OpenShift, use `oc project "${NAMESPACE}"` instead.

**2. Set your Hugging Face token.**

Required to pull model weights. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN="<your-token>"
```

**3. Clone the `llm-d-kv-cache` repository.**

The Makefile expects it as a sibling of this repo at `../llm-d-kv-cache`:

```bash
git clone git@github.com:llm-d/llm-d-kv-cache.git ../llm-d-kv-cache
```

If you clone it elsewhere, set:

```bash
export VLLM_CHART_DIR=<path>/llm-d-kv-cache/vllm-setup-helm
```

**4. Deploy:**

```bash
make env-dev-kubernetes
```

> [!NOTE]
> The model and images of each component can be replaced. See
> [Environment Configuration](#environment-configuration) for details.

**5. Test the deployment.**

Expose the gateway via port-forward:

```bash
kubectl port-forward service/inference-gateway 8080:80 -n "${NAMESPACE}"
```

Make a request:

```bash
curl -s -w '\n' http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"hi","max_tokens":10,"temperature":0}' \
  | jq
```

> [!NOTE]
> If the response is empty or contains an error, jq may output a cryptic message.
> Drop the `| jq` to see the raw response.

### Environment Configuration

**1. EPP image registry and tag:**

```bash
export IMAGE_REGISTRY="<YOUR_REGISTRY>"
export EPP_TAG="<YOUR_TAG>"
```

> [!NOTE]
> The full image reference is `${IMAGE_REGISTRY}/llm-d-inference-scheduler:${EPP_TAG}`.
> For example, with `IMAGE_REGISTRY=quay.io/<my-id>` and `EPP_TAG=v1.0.0`, the image
> will be `quay.io/<my-id>/llm-d-inference-scheduler:v1.0.0`.

**2. vLLM replica count:**

```bash
export VLLM_REPLICA_COUNT_D=2
```

**3. Model name:**

```bash
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

For larger models, set additional vLLM parameters:

```bash
export MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
export PVC_SIZE=200Gi
export VLLM_MEMORY_RESOURCES=100Gi
export VLLM_GPU_MEMORY_UTILIZATION=0.95
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_COUNT_PER_INSTANCE=2
```

**4. Additional settings:**

More environment variables are documented in `scripts/kubernetes-dev-env.sh`.

### Deploying Changes

> [!WARNING]
> This requires manual image builds and pushes to your private registry.

Build and push a new image:

```bash
export EPP_TAG=$(git rev-parse HEAD)
export IMAGE_REGISTRY="quay.io/<my-id>"
make image-build
make image-push
```

Redeploy:

```bash
make env-dev-kubernetes
```

Test with a request:

```bash
kubectl port-forward service/inference-gateway 8080:80 -n "${NAMESPACE}"
curl -s -w '\n' http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"hi","max_tokens":10,"temperature":0}' \
  | jq
```

### Cleanup Environment

Remove all deployed resources in your namespace:

```bash
make clean-env-dev-kubernetes
```

To remove the namespace too:

```bash
kubectl delete namespace ${NAMESPACE}
```

To uninstall the cluster infrastructure:

Uninstall GIE CRDs:

```bash
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/manifests.yaml \
  --ignore-not-found
```

Uninstall kgateway:

```bash
helm uninstall kgateway -n kgateway-system
helm uninstall kgateway-crds -n kgateway-system
```

For more details, see the Gateway API Inference Extension
[getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/).

## Submitting Changes

Read the [llm-d organization contributing guide](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md)
first — it covers project-wide guidelines, code of conduct, and community resources that apply across
all llm-d repositories. The sections below describe router-repo-specific expectations on top of that
baseline.

### Scope

Scoped changes and localized bug fixes can be submitted directly as a PR. 
For larger changes please [create an issue](https://github.com/llm-d/llm-d-router/issues/new)
first describing the change so the maintainers can do an assessment, and work on the details
with you. Getting alignment on the requirements and approach is critical for getting your
changes merged.

Please call out any user facing changes your change will introduce, including changes to
documentation, deployment guides, etc. If your changes replace and deprecate an existing feature,
please be sure to consider that in your design and implementation.
We follow an "N+2 deprecation" policy: features deprecated in release N must continue to work
without user impact (e.g., configuration changes) for 2 releases and can be fully removed
in release N+2. Use of a deprecated feature must produce a clear warning message in releases
N and N+1, providing users with a two release grace period to adjust before the feature is removed.

Please use the [template](.github/PULL_REQUEST_TEMPLATE.md) provided when creating a PR.
If using coding agents, please ensure that the agent uses the PR template format as well.
The template contains a `release-notes` section which must be filled for any change that has
user facing impact.

For additional information and context, please refer to the [llm-d contributing guide](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md)

### Presubmit

Before opening a PR, run:

```bash
make presubmit
```

This runs the same lint, vet, and test checks as the CI pipeline. Fixing failures locally
saves a round-trip through GitHub Actions.