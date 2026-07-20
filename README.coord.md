# LLM-D Coordinator

A Go service that orchestrates multi-phase LLM inference pipelines (Encode/Prefill/Decode) across specialized worker pools. It exposes OpenAI-compatible APIs and routes requests through an Inference Gateway to disaggregated vLLM workers.

## Table of Contents

- [LLM-D Coordinator](#llm-d-coordinator)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
    - [Server](#server)
    - [Gateway](#gateway)
    - [Pipeline](#pipeline)
    - [Built-in Step Parameters](#built-in-step-parameters)
    - [Gateway Routing](#gateway-routing)
  - [Plugin API](#plugin-api)
    - [Step Interface](#step-interface)
    - [Writing a Custom Step](#writing-a-custom-step)
    - [Registering the Step](#registering-the-step)
    - [Dependency Injection](#dependency-injection)
    - [RequestContext](#requestcontext)
  - [API Endpoints](#api-endpoints)
  - [Docker](#docker)
  - [Development](#development)
    - [Running Tests](#running-tests)
      - [Unit Tests](#unit-tests)
      - [End-to-End Tests](#end-to-end-tests)

## Architecture

```
+--------+
| Client |
+--------+
    |
    v
+-------------------+       +-------------+
| Inference Gateway | <---> | Coordinator |
+-------------------+       +-------------+
    |
    v
+-----+
| EPP |
+-----+
    |
    v
+--------------+
| vLLM Workers |
+--------------+
```

The Coordinator processes each request through a configurable pipeline of steps:

1. **replace-media-urls** - Downloads image URLs and inlines them as base64
2. **render** - Sends request to an external rendering/tokenization service
3. **encode** - Parallel fan-out: one encode request per multimodal entry
4. **prefill** - Combines encode results, sends to prefill worker
5. **decode** - Forwards the final request to decode worker, streams response back

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

Configuration is a YAML file passed via the `--config` flag. See `config/coordinator/coordinator.yaml` for the default.

### Server

```yaml
server:
  listen_addr: ":8080"        # Address to listen on
  read_timeout: 30s           # HTTP read timeout
  write_timeout: 120s         # HTTP write timeout (long for streaming)
```

### Gateway

Connection settings for the Inference Gateway that routes to vLLM worker pools:

```yaml
gateway:
  address: "http://inference-gateway:80"
  max_idle_conns_per_host: 100   # Connection pool size
  idle_conn_timeout: 90s
  timeout: 60s                   # Per-request timeout
```

The rendering service address is not a top-level setting; it is the `address` parameter of the `render` pipeline step (see below).

### Pipeline

The pipeline is an ordered list of steps. Each step has a `type` (registered name) and optional `params`:

```yaml
pipeline:
  steps:
    - type: replace-media-urls
      params:
        download_timeout: 10s
        max_concurrent_downloads: 10
    - type: render
      params:
        address: "http://rendering-service:8080"
    - type: encode
      params:
        max_parallel: 8
    - type: prefill
    - type: decode
```

To remove a step, delete it from the list. To reorder, move entries up or down. Steps execute sequentially in the order listed.

### Built-in Step Parameters

| Step | Parameter | Default | Description |
|------|-----------|---------|-------------|
| replace-media-urls | `download_timeout` | `10s` | Timeout for each image download |
| replace-media-urls | `max_concurrent_downloads` | `10` | Max parallel downloads |
| render | `address` | (required) | Base URL of the rendering service |
| render | `timeout` | `30s` | Timeout for a single render call |
| render | `max_total_tokens` | `0` (unlimited) | Reject requests whose tokenized prompt exceeds this |
| render | `max_total_placeholder_tokens` | `0` (unlimited) | Reject requests whose summed image-placeholder length exceeds this |
| encode | `max_parallel` | `8` | Max parallel encode requests |

### Gateway Routing

The coordinator sends every sub-request to the same gateway address. It does not use phase-specific URL prefixes; instead it stamps an `EPP-Phase` header (`encode`, `prefill`, or `decode`) so the Endpoint Picker can route to the correct worker pool. The request path is chosen by the request format:

| Phase | Header | Path |
|-------|--------|------|
| Encode | `EPP-Phase: encode` | `/v1/completions` for completions requests; otherwise `/inference/v1/generate`, or `/v1/chat/completions` when `use_openai_format` is set |
| Prefill | `EPP-Phase: prefill` | same as encode |
| Decode | `EPP-Phase: decode` | original client request path (`/v1/chat/completions` or `/v1/completions`) |

The decode step preserves the original client request path so the gateway can route it to the correct OpenAI-compatible endpoint on the decode worker.

## Plugin API

Custom pipeline steps can be added by implementing the `Step` interface and registering a factory function.

### Step Interface

```go
package pipeline

type Step interface {
    Name() string
    Execute(ctx context.Context, reqCtx *RequestContext) error
}
```

### Writing a Custom Step

```go
package mystep

import (
    "context"
    "github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func init() {
    pipeline.Register("my-step", NewMyStep)
}

type MyStep struct {
    someParam string
}

func NewMyStep(params map[string]any) (pipeline.Step, error) {
    s := &MyStep{someParam: "default"}
    if v, ok := params["some_param"].(string); ok {
        s.someParam = v
    }
    return s, nil
}

func (s *MyStep) Name() string { return "my-step" }

func (s *MyStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
    // Access and modify the request context:
    // - reqCtx.Body              (parsed JSON body, mutable)
    // - reqCtx.TokenIDs          (token IDs from the render step)
    // - reqCtx.MultimodalEntries (multimodal content)
    // - reqCtx.ECTransferParams  (encoder cache transfer params, per encode response)
    // - reqCtx.KVTransferParams  (KV cache transfer params)
    // - reqCtx.Model             (model name)
    // - reqCtx.Stream            (whether client requested streaming)
    //
    // Return nil to continue, or an error to abort the pipeline.
    return nil
}
```

### Registering the Step

Import your step package in `cmd/coordinator/main.go`:

```go
import _ "github.com/llm-d/llm-d-router/pkg/coordinator/steps/mystep"
```

Then add it to the pipeline config:

```yaml
pipeline:
  steps:
    - type: my-step
      params:
        some_param: "value"
    - type: decode
```

### Dependency Injection

A step that needs the shared gateway HTTP client implements `gateway.ClientAware`. After building each step, the coordinator type-asserts it against this interface and calls `SetGatewayClient` when it matches:

```go
// gateway.ClientAware receives the shared gateway HTTP client.
type ClientAware interface {
    SetGatewayClient(*Client)
}
```

Step parameters from the YAML `params` map are the mechanism for everything else. For example, the render step reads its service address from `params.address` in its factory rather than through an injected interface. The render step does expose a `SetServiceAddress` method, but it is used only by tests to point the step at a local server and is not called in production.

### RequestContext

The `RequestContext` is the shared state passed between steps:

```go
type RequestContext struct {
    RequestID         string              // Unique request ID
    OriginalPath      string              // Client request path (e.g., /v1/chat/completions)
    OriginalHeaders   http.Header         // Inbound request headers (forwarded upstream, minus hop-by-hop)
    OriginalBody      []byte              // Raw request body
    Body              map[string]any      // Parsed/mutable JSON body
    Model             string              // Model name
    Stream            bool                // SSE streaming requested
    TokenIDs          []int               // Token IDs from the render step
    MultimodalEntries []MultimodalEntry   // Downloaded multimodal content
    ECTransferParams  []map[string]any    // Encode results, one entry per encode response (mm_hash -> descriptor)
    KVTransferParams  map[string]any      // Prefill KV-cache transfer hints, consumed by the KV connector at decode
    ResponseWriter    http.ResponseWriter // Client response writer; decode steps stream the final response to it
}
```

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
| `E2E_PRINT_COORDINATOR_LOGS` | `false` | Print coordinator pod logs during the run |
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
