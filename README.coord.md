# LLM-D Coordinator

A Go service that orchestrates multi-phase LLM inference pipelines (Encode/Prefill/Decode) across specialized worker pools. It exposes OpenAI-compatible APIs and routes requests through an Envoy Gateway to disaggregated vLLM workers.

## Architecture

```
Client -> Coordinator -> Envoy Gateway -> EPP (ext_proc) -> vLLM Workers
```

The Coordinator processes each request through a configurable pipeline of steps:

1. **replace-media-urls** - Downloads image URLs and inlines them as base64
2. **render** - Sends request to an external rendering/tokenization service
3. **encode** - Parallel fan-out: one encode request per multimodal entry
4. **prefill** - Combines encode results, sends to prefill worker
5. **decode** - Forwards the final request to decode worker, streams response back

## Quick Start

```bash
# Build
make build

# Run with default config
make run

# Run with custom config
./bin/coordinator --config path/to/config.yaml

# Run tests
make test
```

## Configuration

Configuration is a YAML file passed via the `--config` flag. See `configs/coordinator.yaml` for the default.

### Server

```yaml
server:
  listen_addr: ":8080"        # Address to listen on
  read_timeout: 30s           # HTTP read timeout
  write_timeout: 120s         # HTTP write timeout (long for streaming)
```

### Gateway

Connection settings for the Envoy Gateway that routes to vLLM worker pools:

```yaml
gateway:
  address: "http://envoy-gateway:80"
  max_idle_conns_per_host: 100   # Connection pool size
  idle_conn_timeout: 90s
  timeout: 60s                   # Per-request timeout
```

### Rendering Service

```yaml
rendering_service:
  address: "http://rendering-service:8080"
```

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
        endpoint: "/v1/chat/completions/render"
    - type: encode
      params:
        gateway_path: "/inference/v1/generate"
        max_parallel: 8
    - type: prefill
      params:
        gateway_path: "/inference/v1/generate"
    - type: decode
```

To remove a step, delete it from the list. To reorder, move entries up or down. Steps execute sequentially in the order listed.

### Built-in Step Parameters

| Step | Parameter | Default | Description |
|------|-----------|---------|-------------|
| replace-media-urls | `download_timeout` | `10s` | Timeout for each image download |
| replace-media-urls | `max_concurrent_downloads` | `10` | Max parallel downloads |
| render | `endpoint` | `/v1/chat/completions/render` | Rendering service endpoint path |
| encode | `gateway_path` | `/inference/v1/generate` | Path appended after `/encode` prefix |
| encode | `max_parallel` | `8` | Max parallel encode requests |
| prefill | `gateway_path` | `/inference/v1/generate` | Path appended after `/prefill` prefix |

### Gateway Routing

The coordinator sends requests to the gateway with phase-specific path prefixes:

| Phase | Gateway Path | Example |
|-------|-------------|---------|
| Encode | `/encode` + `gateway_path` | `/encode/inference/v1/generate` |
| Prefill | `/prefill` + `gateway_path` | `/prefill/inference/v1/generate` |
| Decode | `/decode` + original request path | `/decode/v1/chat/completions` or `/decode/v1/completions` |

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
    "github.com/llm-d/coordinator/pkg/pipeline"
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
    // - reqCtx.MultimodalEntries (multimodal content)
    // - reqCtx.ECTransferParams  (encoder cache transfer params)
    // - reqCtx.KVTransferParams  (KV cache transfer params)
    // - reqCtx.MMHashes          (multimodal content hashes)
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
import _ "github.com/llm-d/coordinator/pkg/steps/mystep"
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

Steps that need the gateway client or rendering service address can implement optional interfaces:

```go
// Receives the shared gateway HTTP client
type gatewayAware interface {
    SetGatewayClient(*gateway.Client)
}

// Receives the rendering service base URL
type renderAware interface {
    SetServiceAddress(string)
}
```

### RequestContext

The `RequestContext` is the shared state passed between steps:

```go
type RequestContext struct {
    RequestID         string                      // Unique request ID
    OriginalPath      string                      // Client request path (e.g., /v1/chat/completions)
    OriginalBody      []byte                      // Raw request body
    Body              map[string]any              // Parsed/mutable JSON body
    Model             string                      // Model name
    Stream            bool                        // SSE streaming requested
    MultimodalEntries []MultimodalEntry           // Downloaded multimodal content
    ECTransferParams  map[string]ECTransferParam  // Encode results (mm_hash -> peer)
    KVTransferParams  map[string]any              // Prefill results
    MMHashes          []string                    // Multimodal content UUIDs
    ResponseWriter    http.ResponseWriter         // Client response writer
    Flusher           http.Flusher                // For SSE streaming
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
docker build -t coordinator .
docker run -p 8080:8080 -v $(pwd)/configs:/configs coordinator
```

## Development

```bash
make build    # Build binary to bin/coordinator
make test     # Run all tests
make lint     # Run golangci-lint
make tidy     # Run go mod tidy
make clean    # Remove build artifacts
```
