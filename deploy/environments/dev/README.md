# Development Environment Overlays

Kustomize overlays for deploying vLLM inference in different disaggregation scenarios.
Each scenario directory selects which atomic components to deploy and applies
scenario-specific patches. The atomic components live in `deploy/components/`:

| Component | Description |
|-----------|-------------|
| `vllm-decode/` | Decode pod — always deployed, includes routing sidecar (removed in EPD scenario) |
| `vllm-prefill/` | Prefill pod — deployed when `DISAGG_P=true` |
| `vllm-encode/` | Encoder pod — deployed when `DISAGG_E=true` |
| `overlays/simulator/` | Adds `--mode=${VLLM_SIM_MODE}`, UDS tokenizer, KV cache and ZMQ args (included by all scenario overlays) |

These overlays are used by both `scripts/kind-dev-env.sh` (for local KIND clusters)
and e2e tests (via `kustomize build` + env var substitution).

## Scenario Directories

| Directory | `DISAGG_E` | `DISAGG_P` | Components | Description |
|-----------|-----------|-----------|------------|-------------|
| `epd/` | false | false | decode | Unified deployment (no disaggregation). No routing sidecar; vLLM handles all stages directly |
| `p-d/` | false | true | prefill + decode | Separate Prefill and Decode pods with KV cache transfer |
| `e-pd/` | true | false | encode + decode | Separate Encoder, combined Prefill-Decode with EC transfer |
| `e-p-d/` | true | true | encode + prefill + decode | Fully disaggregated: separate Encoder, Prefill, and Decode |

Data parallel (`VLLM_DATA_PARALLEL_SIZE`) and KV cache (`KV_CACHE_ENABLED`) are independent
options that combine with any disaggregation mode — they are not separate scenarios. Data
parallel uses the same overlay as the chosen disaggregation mode; set
`VLLM_DATA_PARALLEL_SIZE=2` (or higher) to enable multi-rank routing within any scenario.

## Shared Infrastructure

| Directory | Description |
|-----------|-------------|
| `base-kind-istio/` | Istio control plane + inference gateway (EPP, services, RBAC, InferencePool, Gateway, HTTPRoute). Applied separately by `kind-dev-env.sh` before the scenario overlay |
| `e2e-infra/` | Test-specific infrastructure: NodePort services for health/metrics and envoy proxy config used by e2e tests |
| `kubernetes-kgateway/` | Alternative gateway configuration using Kubernetes Gateway API instead of Istio |

## Scenario Selection

Use `DISAGG_E` and `DISAGG_P` boolean flags when running `kind-dev-env.sh`:

```bash
# Default (no disaggregation)
./scripts/kind-dev-env.sh

# Prefill/Decode
DISAGG_P=true ./scripts/kind-dev-env.sh

# Encode / Prefill-Decode
DISAGG_E=true ./scripts/kind-dev-env.sh

# Encode / Prefill / Decode (fully disaggregated)
DISAGG_E=true DISAGG_P=true ./scripts/kind-dev-env.sh

# Data parallel — combine with any disaggregation mode
VLLM_DATA_PARALLEL_SIZE=2 ./scripts/kind-dev-env.sh
DISAGG_P=true VLLM_DATA_PARALLEL_SIZE=2 ./scripts/kind-dev-env.sh
```

## Key Environment Variables

Variables substituted at deploy time via `envsubst` or Go test `substituteMany`:

| Variable | Description | Example |
|----------|-------------|---------|
| `DISAGG_E` | Deploy a separate Encoder pod (`true`/`false`) | `false` |
| `DISAGG_P` | Deploy a separate Prefill pod (`true`/`false`) | `false` |
| `VLLM_SIM_MODE` | Simulator response mode: `echo` (returns input) or `random` (random sentences) | `echo` |
| `VLLM_IMAGE` | vLLM container image (simulator or real) | `ghcr.io/llm-d/llm-d-inference-sim:v0.8.2` |
| `SIDECAR_IMAGE` | Routing sidecar image | `ghcr.io/llm-d/llm-d-router-disagg-sidecar:dev` |
| `UDS_TOKENIZER_IMAGE` | UDS tokenizer sidecar image | `ghcr.io/llm-d/llm-d-uds-tokenizer:dev` |
| `MODEL_NAME` | Model name passed to vLLM. Can be a real HuggingFace model (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `Qwen/Qwen3-VL-2B-Instruct`) or an arbitrary name when using the simulator (e.g. `food-review`) | `food-review` |
| `POOL_NAME` | InferencePool name | `food-review-inference-pool` |
| `VLLM_REPLICA_COUNT_E` | Encode deployment replicas | `1` |
| `VLLM_REPLICA_COUNT_P` | Prefill deployment replicas | `1` |
| `VLLM_REPLICA_COUNT_D` | Decode deployment replicas | `1` |
| `VLLM_DATA_PARALLEL_SIZE` | Data parallel rank count per vLLM pod — applies to ALL pod types (encode, prefill, decode) | `1` |
| `CONNECTOR_TYPE` | KV connector for P/D | `nixlv2` |
| `KV_CONNECTOR_TYPE` | KV connector type (derived from `CONNECTOR_TYPE`) | `nixlv2` |
| `EC_CONNECTOR_TYPE` | EC connector for E scenarios | `ec-example` |
| `KV_CACHE_ENABLED` | Enable KV cache scoring | `false` |
| `HF_TOKEN` | HuggingFace token for downloading real models (empty for simulator) | `` |
| `NAMESPACE` | Kubernetes namespace for all resources | `default` |
| `VLLM_EXTRA_ARGS_E` | Extra flags appended to Encoder vLLM args (`--flag=value` format) | `` |
| `VLLM_EXTRA_ARGS_P` | Extra flags appended to Prefill vLLM args (`--flag=value` format) | `` |
| `VLLM_EXTRA_ARGS_D` | Extra flags appended to Decode vLLM args (`--flag=value` format) | `` |
