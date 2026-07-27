# LLM-D Coordinator

A Go service that orchestrates multi-phase LLM inference pipelines (Encode/Prefill/Decode) across specialized worker pools. It exposes OpenAI-compatible APIs and routes requests through an Inference Gateway to disaggregated vLLM workers.

For the architecture, request lifecycle, EPP integration, and plugin API, see [docs/coordinator_architecture.md](docs/coordinator_architecture.md). For the exact per-stage wire formats, see [docs/communication.md](docs/communication.md).

## Table of Contents

- [LLM-D Coordinator](#llm-d-coordinator)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
  - [API Endpoints](#api-endpoints)
  - [Docker](#docker)
  - [Development](#development)
    - [Running Tests](#running-tests)
      - [Unit Tests](#unit-tests)
      - [End-to-End Tests](#end-to-end-tests)

## Quick Start

The coordinator targets live in `Makefile.coord.mk`, which the root `Makefile`
does not include, so pass it explicitly with `-f`:

```bash
# Build
make -f Makefile.coord.mk build

# Run with default config
make -f Makefile.coord.mk run

# Run with custom config
./bin/coordinator --config path/to/config.yaml

# Run tests
make -f Makefile.coord.mk test
```

## Configuration

Configuration is a YAML file passed via the `--config` flag. See `config/coordinator/coordinator.yaml` for the annotated default, and [Configuring the pipeline](docs/coordinator_architecture.md#configuring-the-pipeline) for the full reference (top-level structure, environment overrides, connector selection, and the built-in steps).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions API |
| POST | `/v1/completions` | OpenAI Completions API |
| GET | `/healthz` | Health check |
| GET | `/readyz` | Readiness check |

Both completion endpoints support `"stream": true` for Server-Sent Events streaming.

## Docker

```bash
docker build -t coordinator -f Dockerfile.coordinator .
docker run -p 8080:8080 -v $(pwd)/config/coordinator:/config/coordinator coordinator
```

## Development

The coordinator targets live in `Makefile.coord.mk`; pass it with `-f`:

```bash
make -f Makefile.coord.mk build   # Build binary to bin/coordinator
make -f Makefile.coord.mk lint    # Run golangci-lint
make -f Makefile.coord.mk tidy    # Run go mod tidy
make -f Makefile.coord.mk clean   # Remove build artifacts
```

### Running Tests

#### Unit Tests

```bash
make -f Makefile.coord.mk test   # run coordinator unit tests
```

#### End-to-End Tests

```bash
make -f Makefile.coord.mk test-e2e-coordinator
```

This creates a temporary Kind cluster named `e2e-coordinator-tests`, runs the coordinator e2e suite against it, and deletes the cluster on completion.

`test-e2e-coordinator` is the local entry point: it builds the builder, coordinator, and epp images first. CI invokes `test-e2e-coordinator-run` instead, which expects those images to be already built and loaded into the container daemon (`image-pull` fetches only the simulator). Running `test-e2e-coordinator-run` directly without prebuilt images will use stale or missing ones.

**Keeping the cluster on failure**

Set `E2E_KEEP_CLUSTER_ON_FAILURE=true` to preserve the cluster when any test fails. This is useful for inspecting pod logs, events, or cluster state after a failure.

```bash
E2E_KEEP_CLUSTER_ON_FAILURE=true make -f Makefile.coord.mk test-e2e-coordinator
```

When set, a successful run still cleans up normally: the cluster is only kept if there is at least one test failure.

**Accessing the cluster after a failure**

E2E tests do not update the host's kubeconfig to point at the `e2e-coordinator-tests` Kind cluster. After a preserved failure, export the kubeconfig manually:

```bash
# Merge into the default kubeconfig ($HOME/.kube/config or $KUBECONFIG)
kind export kubeconfig --name e2e-coordinator-tests

# Or write to a specific file
kind export kubeconfig --name e2e-coordinator-tests --kubeconfig /path/to/kubeconfig
```

Then use it as normal:

```bash
kubectl --context kind-e2e-coordinator-tests get pods
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `E2E_KEEP_CLUSTER_ON_FAILURE` | `false` | Preserve the Kind cluster when the suite fails |
| `E2E_GATEWAY_PORT` | `30080` | Host port mapped to the gateway NodePort |
| `E2E_PRINT_LOGS` | `false` | Print all pod logs (coordinator, EPPs, Envoy, workers) for every spec, not just on failure |
| `CONTAINER_RUNTIME` | `docker` | Container runtime used to load images into Kind (`docker` or `podman`) |
| `EPP_IMAGE` | `ghcr.io/llm-d/llm-d-router-endpoint-picker:dev` | EPP image loaded into the Kind cluster |
| `VLLM_IMAGE` | `ghcr.io/llm-d/llm-d-inference-sim:v0.10.2` | vLLM image loaded into the Kind cluster |
| `VLLM_RENDER_IMAGE` | same as `VLLM_IMAGE` | vLLM render image loaded into the Kind cluster |
| `VLLM_RENDER_PORT` | `8082` | Port the vllm-render service listens on |
| `COORDINATOR_IMAGE` | _(empty)_ | Coordinator image loaded into the Kind cluster |
| `MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` | Model name used by the test pools |
| `NAMESPACE` | `default` | Namespace to deploy test resources into |
| `K8S_CONTEXT` | _(empty)_ | Use an existing cluster context instead of creating a Kind cluster |
| `READY_TIMEOUT` | `10m` | How long to wait for resources to become ready |
