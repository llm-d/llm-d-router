# Coordinator Architecture

The coordinator is a Go service that accepts inference requests and drives them through
a configurable pipeline of steps. It currently supports the OpenAI-compatible API
(`/v1/chat/completions`, `/v1/completions`), and the entry layer is designed to be
extended to other inference protocols. Each step performs one unit of
work (download media, tokenize, encode, prefill, decode). The pre-processing steps call
side services directly (media download, and the render service for tokenization); the
encode, prefill, and decode steps forward sub-requests to vLLM worker pools through an
Inference Gateway that conforms to Gateway API Inference Extension (GAIE). An Endpoint
Picker (EPP) behind the gateway selects the concrete pod for each phase, so the
coordinator orchestrates the phases without knowing pod addresses.

The coordinator is stateless: all per-request state lives on a `RequestContext` that
exists only for the lifetime of that request, and nothing is shared or persisted across
requests. Any instance can serve any request, so the service scales horizontally by
running more replicas behind a load balancer with no coordination between them.

The goals the design serves:

- Easy extensibility: steps are self-contained plugins registered by name, so new
  processing stages are added without touching the pipeline or other steps (see
  [Creating a new step](#creating-a-new-step-plugin)).
- Versatile request processing: an ordered pipeline of independent steps that can be
  combined or reordered per deployment.
- Flexible disaggregation: the same steps can run combined (decode alone, or
  prefill+decode on one worker) or fully disaggregated (encode, prefill, decode on
  separate pools), chosen by configuration.
- Request-processing optimization: skip work that does not apply (no media download for
  text-only requests, no encode without multimodal content) and short-circuit when a
  worker can serve directly (the conditional-decode fast path).
- Both deferred and non-deferred vLLM node selection: defer pod selection to the EPP one
  phase at a time, or let a worker serve a request directly when it already holds the
  needed state.
- Tokenize the prompt once (in the render step) and reuse the token IDs across encode,
  prefill, and decode, so workers never re-tokenize.
- Tokens-in / tokens-out operation: steps can exchange token IDs directly instead of
  raw text, cutting per-step tokenization to a single render pass. This is also
  beneficial for reinforcement learning (RL), where the training loop works in token
  space and avoids the detokenize/re-tokenize round-trips a text-only interface forces.
- Pluggable KV and EC data transfer: select push- or pull-based transfer protocols
  (NIXL, SGLang, shared storage) per deployment without changing the steps.

This is a non-exhaustive list; the pipeline/step and connector abstractions are meant to
absorb further processing modes as they are added.

## Table of Contents

- [High-level picture](#high-level-picture)
- [Components](#components)
- [Request lifecycle](#request-lifecycle)
  - [RequestContext](#requestcontext)
  - [EPP-Phase routing](#epp-phase-routing)
- [EPP integration](#epp-integration)
  - [Per-phase model](#per-phase-model)
  - [Decode disaggregation deciders](#decode-disaggregation-deciders)
  - [Conditional decode handshake](#conditional-decode-handshake)
  - [KV and EC transfer protocols](#kv-and-ec-transfer-protocols)
  - [Cross-phase metadata](#cross-phase-metadata)
- [Coordinator vs. the llm-d-router sidecar model](#coordinator-vs-the-llm-d-router-sidecar-model)
  - [The sidecar model (llm-d-router)](#the-sidecar-model-llm-d-router)
  - [The coordinator model](#the-coordinator-model)
  - [Side by side](#side-by-side)
  - [Why move orchestration out of the sidecar](#why-move-orchestration-out-of-the-sidecar)
- [Creating a new step (plugin)](#creating-a-new-step-plugin)
  - [The Step contract](#the-step-contract)
  - [Steps over five points](#steps-over-five-points)
  - [Example](#example)
  - [Parameter parsing notes](#parameter-parsing-notes)
  - [Dependency injection](#dependency-injection)
  - [Using a connector from a step](#using-a-connector-from-a-step)
  - [Tests](#tests)
- [Configuring the pipeline](#configuring-the-pipeline)
  - [Top-level structure](#top-level-structure)
  - [Environment overrides](#environment-overrides)
  - [Connector selection](#connector-selection)
  - [Should the coordinator use the tokens-in format?](#should-the-coordinator-use-the-tokens-in-format)
  - [The built-in steps](#the-built-in-steps)
  - [Adding a step to the pipeline](#adding-a-step-to-the-pipeline)
- [References](#references)

## High-level picture

A client sends an inference request to the coordinator. The coordinator pre-processes the
request (media download, tokenization) against side services, then makes one inference
call per phase to the Inference Gateway. The Gateway consults the per-phase EPP
(Encode / Prefill / Decode), which picks a vLLM pod from that phase's pool. State that
must survive across phases (token IDs, multimodal hashes, KV/EC transfer descriptors)
lives on a per-request context held by the coordinator, not in a sidecar on the decode
pod.

```
        Client
          |
          |  inference request (OpenAI-compatible API)
          v
   Coordinator Service  ------------>  side services
          |                            (media download, render / tokenize)
          |
          |  one call per phase, tagged EPP-Phase: encode | prefill | decode
          v
     Inference Gateway
          |  endpoint picker protocol
          +-------------------+-------------------+
          v                   v                   v
      Encode EPP          Prefill EPP         Decode EPP    (per-phase endpoint picker)
          |                   |                   |
          v                   v                   v
      Encode vLLM         Prefill vLLM        Decode vLLM   (worker pools)
        pool                pool                pool
```

In short:

```
                         one request+response per phase (encode, prefill, decode)
                        +---------------------------------------------------------+
                        |                                                         v
Client  <-->  Coordinator  -->  Inference Gateway  -->  EPP -->  vLLM worker pool
                        ^                                                         |
                        +---------------------------------------------------------+
                                          response streamed/returned
```

The client opens a single connection to the coordinator. Behind it, the coordinator
issues several requests to the Inference Gateway and consumes each response before issuing
the next: it is an active client of the Gateway, not a one-shot proxy. A single step
can also fan out into several parallel requests: `replace-media-urls` downloads media
concurrently (to the source URLs), and `encode` issues one Gateway request per
multimodal item in parallel. The number and shape of these requests depend on the input
(how many multimodal entries it carries) and on the coordinator configuration (which
steps are enabled and how they are parameterized); text-only requests need no media
download or encode at all. Across phases the coordinator sequences the round-trips
(prefill and decode are one each), threading state from each response into the next
request. Each Gateway call routes it by the `EPP-Phase` header to
the matching per-phase EPP and then to a pod in that phase's pool.

Steps are skipped at runtime when they do not apply (for example, `encode` is a no-op
when the request has no multimodal entries, and `render` short-circuits when the
completions prompt is already a token array). See
[communication.md](communication.md) for the per-endpoint flow variants.

## Components

| Component | Path | Responsibility |
| :---- | :---- | :---- |
| Entry server | [pkg/coordinator/server/](../pkg/coordinator/server/) | chi HTTP server. Accepts `/v1/chat/completions` and `/v1/completions`, builds the `RequestContext`, runs the pipeline, exposes `/healthz` and `/readyz`. |
| Pipeline | [pkg/coordinator/pipeline/](../pkg/coordinator/pipeline/) | The `Step` abstraction, the ordered executor, the step registry, and the `RequestContext`. |
| Steps | [pkg/coordinator/steps/](../pkg/coordinator/steps/) | The built-in steps. Each registers itself with the pipeline registry in an `init()` function. |
| Gateway client | [pkg/coordinator/gateway/](../pkg/coordinator/gateway/) | HTTP client with a keep-alive pool to the configured Inference Gateway, path/format helpers, and the `EPP-Phase` header constants. |
| Connectors | [pkg/coordinator/connectors/](../pkg/coordinator/connectors/) | KV and EC transfer protocols. Selected by name at config time; control the `kv_transfer_params` / `ec_transfer_params` wire shapes. |
| Config | [pkg/coordinator/config/](../pkg/coordinator/config/) | Viper-backed YAML + env loader. |
| Entrypoint | [cmd/coordinator/](../cmd/coordinator/) | Wires config to the pipeline: builds each step, merges connector defaults, injects the gateway client. |

## Request lifecycle

1. [pkg/coordinator/server/handlers.go](../pkg/coordinator/server/handlers.go) reads and size-limits the body,
   parses it into a `map[string]any`, extracts `model` and `stream`, assigns or reuses
   the request ID, and constructs a `RequestContext`. Malformed input is rejected before
   the pipeline runs: an unreadable body or invalid JSON returns `400 Bad Request`, and a
   body exceeding the fixed built-in size limit returns `413 Request Entity Too Large`.
2. The server calls `pipeline.Execute(ctx, reqCtx)`.
3. [pkg/coordinator/pipeline/pipeline.go](../pkg/coordinator/pipeline/pipeline.go) runs each step in order. A
   step returning an error aborts the request, and the server maps the error to a
   client status (`classifyPipelineError`):
   - a step error wrapping `ErrBadRequest` (invalid client input) returns `400 Bad Request`;
   - an `UpstreamError` carrying a `4xx` from an upstream service (render or a worker via
     the Gateway) is forwarded with that same `4xx`, since the request was the root cause;
   - any other error, including an upstream `5xx`, returns `502 Bad Gateway`.

   Upstream response bodies are logged server-side only and never returned to the client.
   A step returning `ErrPipelineDone` instead stops the pipeline and reports success, used
   by `conditional-decode` when the decode worker serves the request directly.
4. The final `decode` step proxies the worker response straight back to the client
   through `RequestContext.ResponseWriter`, passing both a streaming SSE response and a
   non-streaming (buffered JSON) response through unchanged. The worker decides which
   based on the request's `stream` flag.

### RequestContext

[pkg/coordinator/pipeline/context.go](../pkg/coordinator/pipeline/context.go) defines the per-request state that
steps read and mutate. The load-bearing fields:

| Field | Set by | Used by |
| :---- | :---- | :---- |
| `Body` | server (parsed JSON) | every step; mutated in place as the request is enriched |
| `OriginalPath`, `OriginalHeaders`, `OriginalBody` | server | format detection, header forwarding |
| `Model`, `Stream` | server | request construction, response handling |
| `TokenIDs` | `render` | `conditional-decode`, `encode`, `prefill`, `decode` |
| `MultimodalEntries` | `replace-media-urls` (seeded), `render` (enriched) | `encode`, `prefill`, `decode` |
| `ECTransferParams` | `encode` (via the EC connector) | `prefill` |
| `KVTransferParams` | `prefill` (via the KV connector) | `decode` |
| `ResponseWriter` | server | `conditional-decode`, `decode` |

`RequestContext.ForwardedHeaders()` returns the inbound headers with hop-by-hop headers,
`Host`, `Content-Length`, and `Content-Type` removed, normalized to lowercase. Steps use
it as the base header set, then stamp the request ID and `EPP-Phase`.

### EPP-Phase routing

Every coordinator-to-worker call carries an `EPP-Phase` header (`encode`, `prefill`, or
`decode`) so the gateway routes to the correct pool. The constants live in
[pkg/coordinator/gateway/paths.go](../pkg/coordinator/gateway/paths.go). The request path is either the client's
original OpenAI path or the internal `/inference/v1/generate` path, depending on
`use_openai_format` (see [Configuring the pipeline](#configuring-the-pipeline)). Other
paths can be added later as new protocols are supported.

## EPP integration

The Endpoint Picker (EPP) selects the concrete vLLM pod for a request through the GAIE
[endpoint picker protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/004-endpoint-picker-protocol). The coordinator never addresses pods directly: it sends each
phase call to the configured gateway, and the per-phase EPP picks the pod from that
phase's pool.

### Per-phase model

Each Gateway route maps to a dedicated EPP instance configured for a single phase. The
coordinator drives the cascade across phases; each EPP call is single-phase scheduling.

| Phase | EPP instance | Role |
| :---- | :---- | :---- |
| `decode` | EPP-D | Selects a decode pod. Also evaluates whether the request needs disaggregation (encode and/or prefill). |
| `prefill` | EPP-P | Selects a prefill pod. |
| `encode` | EPP-E | Load-balances across encoder pods. |

This is an alternative to the sidecar-based orchestration in llm-d-router; see
[Coordinator vs. the llm-d-router sidecar model](#coordinator-vs-the-llm-d-router-sidecar-model)
for the comparison.

### Decode disaggregation deciders

EPP-D runs a scheduling cycle that, in addition to picking a decode pod, decides whether
the request can be served by decode alone or needs earlier phases. The decision is made
by decider plugins (EPP-side configuration):

| Decider | Stage | Logic |
| :---- | :---- | :---- |
| `always-disagg-multimodal-decider` | Encode | Disaggregate to encode if the request contains any `image_url`, `video_url`, or `input_audio` content block. |
| `always-disagg-pd-decider` | Prefill | Always disaggregate prefill from decode. |
| `prefix-based-pd-decider` | Prefill | Disaggregate prefill when `(inputTokens - hitPrefixTokens) >= nonCachedTokens`. `nonCachedTokens = 0` disables it. |

The decode-before-prefill ordering matters: the encode and prefill deciders inspect the
chosen decode endpoint, so the decode pod is selected first.

### Conditional decode handshake

The coordinator's `conditional-decode` step is the implemented mechanism for the "can
decode serve this directly" question. The step sends the decode request with the HTTP
`Prefer: if-available` header:

- The decode worker serves the request directly when it can (cache available). The
  response (SSE or JSON) is streamed back to the client and the pipeline stops.
- When it cannot, the worker returns `412 Precondition Failed`. The coordinator treats
  the 412 as a cache miss, swallows it, and continues the pipeline to encode, prefill,
  and decode.

The 412 response may carry scheduling or cache-locality hints in headers or body; these
are not yet consumed by the coordinator.

### KV and EC transfer protocols

Because the coordinator builds the prefill and decode request bodies itself, it must
emit the `kv_transfer_params` / `ec_transfer_params` shape the vLLM pods expect. That
shape is owned by the configured connectors ([pkg/coordinator/connectors/](../pkg/coordinator/connectors/)),
selected deployment-wide and required to match the connector configured on the pods.

KV connector protocols ([pkg/coordinator/connectors/kv/](../pkg/coordinator/connectors/kv/)):

| Connector | llm-d-router sidecar equivalent | Protocol |
| :---- | :---- | :---- |
| `kv-nixl` | `nixlv2` | NIXL P2P RDMA: prefill stores KV for remote decode; decode pulls it via `remote_engine_id` / `remote_block_ids` / `remote_host` / `remote_port`. |
| `kv-sglang` | `sglang` | SGLang bootstrap: `bootstrap_host`, `bootstrap_port`, `bootstrap_room`. |
| `kv-shared-storage` | `shared-storage` | Shared filesystem / object store: prefill writes KV, decode reads it; no transfer descriptor on the wire. |

The `kv-sglang` `bootstrap_port` advertised to prefill pods defaults to 8998 and can be
overridden process-wide with the `SGLANG_BOOTSTRAP_PORT` environment variable. The value
is read once on the first prefill request that uses the connector; a non-integer value is
rejected in favor of the default and logged at error level.

EC connector protocols ([pkg/coordinator/connectors/ec/](../pkg/coordinator/connectors/ec/)) ship encoder
embeddings from encode pods to the prefill pod:

| Connector | llm-d-router sidecar equivalent | Protocol |
| :---- | :---- | :---- |
| `ec-nixl` | `ec-nixl` | Encoder registers embeddings in NIXL-mapped memory and returns a per-`mm_hash` descriptor (`peer_host`, `peer_port`, `size_bytes`, `nixl_agent_metadata_b64`); the coordinator merges these and forwards them as `ec_transfer_params` on the prefill request. |
| `ec-shared-storage` | `ec-example` | Encoder writes embeddings to shared storage keyed by `mm_hash`; the prefill pod reads them back; no descriptor on the wire. |

The KV and EC selections are independent. The exact request and response bodies for each
phase and format are in [communication.md](communication.md).

### Cross-phase metadata

State produced by one phase and consumed by a later one is carried on the
`RequestContext` (see [RequestContext](#requestcontext)):

| Produced by | Field | Consumed by |
| :---- | :---- | :---- |
| `render` | `TokenIDs`, multimodal hashes/placeholders | encode, prefill, decode (EPP uses token IDs for prefix matching) |
| `encode` | `ECTransferParams` (merged per `mm_hash`) | prefill |
| `prefill` | `KVTransferParams` (from the prefill response body) | decode |

## Coordinator vs. the llm-d-router sidecar model

The coordinator is an alternative to the disaggregation orchestration in
[llm-d-router](https://github.com/llm-d/llm-d-router). Both route inference through the
same Inference Gateway and the same EPP scheduling machinery; they differ in
**where disaggregation is orchestrated** and **where tokenization happens**.

### The sidecar model (llm-d-router)

In llm-d-router, orchestration lives in a **vLLM sidecar that runs only on the decode
worker**. No sidecar or coordination logic runs on the prefill or encode nodes. The flow:

1. A request reaches the Inference Gateway and the EPP runs its
   `disagg-profile-handler` plugin. In a **single scheduling cycle** it selects pods for
   every phase the request needs, in a fixed order: decode (always), then encode (if
   multimodal content is detected), then prefill (if the P/D decider judges it
   beneficial).
2. The EPP communicates the selected pods to the decode sidecar as **request headers**:
   `x-prefiller-host-port` (the selected prefill worker) and `x-encoder-hosts-ports` (one
   or more encode workers). The gateway then forwards the request to the decode pod.
3. The sidecar orchestrates the cascade: it dispatches multimodal content to the encode
   workers, sends a remote prefill request (`max_tokens=1`) to the prefill worker,
   collects the returned KV parameters, and finally launches the local decode, which
   reads the KV cache from the prefill pod and streams the response back.

Tokenization happens on the workers (the sidecar forwards prompts, not token IDs). The
KV connector protocol is selected on the sidecar with `--kv-connector` (`nixlv2`,
`shared-storage`, `sglang`, `mooncake`) and must match the vLLM-side `kv_connector`.

### The coordinator model

The coordinator removes the sidecar entirely and pulls orchestration out to a standalone
service in front of the Inference Gateway:

1. The client request reaches the **coordinator**.
2. The coordinator tokenizes once via the render service, then makes **one
   EPP-mediated call per phase** to the gateway, tagging each with the `EPP-Phase` header.
   Each EPP call is single-phase scheduling, not a one-cycle selection of all phases.
3. Cross-phase state (token IDs, multimodal hashes, EC and KV transfer descriptors)
   lives on the coordinator's per-request `RequestContext`, not in headers handed to a
   sidecar. The coordinator builds the prefill and decode request bodies itself,
   emitting the `kv_transfer_params` / `ec_transfer_params` shape via its configured
   connectors.

### Side by side

| Dimension | llm-d-router (sidecar) | Coordinator |
| :---- | :---- | :---- |
| Orchestration location | vLLM sidecar on the decode pod | Standalone coordinator service in front of the Inference Gateway |
| Sidecar required | Yes (decode pod only) | No |
| Pipeline versatility | Fixed E/P/D orchestration baked into the sidecar | Configurable pipeline of independent, reorderable plugin steps; new stages added without touching existing ones |
| EPP scheduling | One cycle selects all phases (`disagg-profile-handler`) | One EPP call per phase, coordinator drives the cascade |
| vLLM pod selection | All phase pods chosen up front in one scheduling cycle | Deferred per phase: each pod is selected only when that phase's call is made, at the point its destination becomes relevant |
| Phase selection signal | EPP request headers `x-prefiller-host-port`, `x-encoder-hosts-ports` read by the sidecar | `EPP-Phase` header per call; per-phase EPP picks the pod |
| Tokenization | On the workers | Once, in the coordinator's render step; token IDs reused downstream (experimental path) |
| Cross-phase state | Held by the sidecar | Held on the coordinator `RequestContext` |

---

## Creating a new step (plugin)

A step is the unit of extension. The pipeline knows nothing about concrete steps; it
calls the `Step` interface and looks steps up by name in a registry. To add behavior,
write a step, register it under a type name, and reference that name via the `type`
field in the config.

Before writing a new step, read the closest existing one. [render.go](../pkg/coordinator/steps/render.go)
is the reference for calling a side service; [encode.go](../pkg/coordinator/steps/encode.go) is the
reference for parallel fan-out to workers; [decode.go](../pkg/coordinator/steps/decode.go) and
[conditional_decode.go](../pkg/coordinator/steps/conditional_decode.go) are the reference for
proxying a streaming response back to the client. Follow the structure and naming of
whichever is closest to your task.

### The Step contract

A step implements [pkg/coordinator/pipeline/step.go](../pkg/coordinator/pipeline/step.go):

```go
type Step interface {
    Name() string
    Execute(ctx context.Context, reqCtx *RequestContext) error
}
```

- `Name()` returns the step's type name. Use the same constant you register under.
- `Execute` does the work: it reads and mutates `reqCtx`, and returns an error to abort
  the request or `nil` to continue. Return `pipeline.ErrPipelineDone` to stop the
  pipeline early and report success (the response must already have been written to
  `reqCtx.ResponseWriter`).
- A response-producing step writes the client response to `reqCtx.ResponseWriter`,
  typically by proxying the upstream response with `httputil.ReverseProxy` set to
  `FlushInterval: -1`, which forwards each write immediately (SSE chunks stream through; a
  non-streaming JSON response passes through as one body). A terminal step like `decode`
  returns `nil`, since it is the last step; a non-terminal step that has fully served the
  response returns `ErrPipelineDone` to skip the remaining steps (the `conditional-decode`
  cache hit). `decode` and `conditional-decode` are the reference implementations.

A step is constructed by a `StepFactory`:

```go
type StepFactory func(params map[string]any) (Step, error)
```

`params` is the `params:` block from the step's config entry. The factory validates and
typed-extracts the parameters it needs and returns the constructed step, or an error
that aborts startup.

### Steps over five points

1. Pick a type name and declare it as a constant.
2. Register the factory in `init()`.
3. Implement the constructor (`StepFactory`): parse `params`, validate, build the step.
4. Implement `Name()` and `Execute()`.
5. If the step needs the gateway client or other injected dependencies, expose a setter
   (see [Dependency injection](#dependency-injection)).

### Example

```go
package steps

import (
    "context"
    "fmt"

    "github.com/llm-d/coordinator/pkg/gateway"
    "github.com/llm-d/coordinator/pkg/pipeline"
)

const ExampleStepName = "example"

func init() {
    pipeline.Register(ExampleStepName, NewExampleStep)
}

type ExampleStep struct {
    threshold int
    gwClient  *gateway.Client
}

func NewExampleStep(params map[string]any) (pipeline.Step, error) {
    threshold := 0
    if v, ok := params["threshold"].(int); ok {
        if v < 0 {
            return nil, fmt.Errorf("threshold must be non-negative, got %d", v)
        }
        threshold = v
    }
    return &ExampleStep{threshold: threshold}, nil
}

// SetGatewayClient satisfies gateway.ClientAware; the entrypoint calls it after construction.
func (s *ExampleStep) SetGatewayClient(c *gateway.Client) { s.gwClient = c }

func (s *ExampleStep) Name() string { return ExampleStepName }

func (s *ExampleStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
    // read/mutate reqCtx; call s.gwClient as needed.
    return nil
}
```

Because the file lives in `pkg/steps`, its `init()` runs whenever the binary imports the
package (the entrypoint already does, via `cmd/coordinator/main.go`). No central list to
edit: registration is the only wiring step.

### Parameter parsing notes

- Config values arrive as `map[string]any` decoded by viper. Integers decode as `int`,
  booleans as `bool`, durations as `string` (parse with `time.ParseDuration`), strings
  as `string`. Type-assert defensively and supply a default when the key is absent.
- Connector names (`kv_connector`, `ec_connector`) and `use_openai_format` are injected
  into every step's `params` by the entrypoint before the factory runs (see below), so a
  step can read them without the operator repeating them per step. The constants are
  `steps.ParamKVConnector` and `steps.ParamECConnector`.
- Shared parsing helpers live in [pkg/coordinator/steps/utils.go](../pkg/coordinator/steps/utils.go)
  (`parseUseOpenAIFormat`, `resolveFormat`, `buildMMFeatures`, `copyBody`,
  `coerceParamsMap`). Reuse them rather than re-implementing.

### Dependency injection

The pipeline registry constructs a step only from its `params`. Dependencies that are not
config (the gateway client, a side-service address) are injected by the entrypoint after
construction, via optional setter interfaces it type-asserts against.

The gateway client is injected through the `gateway.ClientAware` interface
([pkg/coordinator/gateway/client.go](../pkg/coordinator/gateway/client.go)):

```go
// in pkg/gateway
type ClientAware interface {
    SetGatewayClient(*Client)
}
```

[cmd/coordinator/main.go](../cmd/coordinator/main.go) applies it after building each step:

```go
if ga, ok := step.(gateway.ClientAware); ok {
    ga.SetGatewayClient(gwClient)
}
```

If a step needs the gateway client, implement `gateway.ClientAware` (a
`SetGatewayClient(*gateway.Client)` method); the four Gateway-facing steps (`encode`,
`prefill`, `decode`, `conditional-decode`) do. To inject a different dependency, define an
analogous setter interface next to that dependency and add a type-assert block in
`buildPipeline`. Keep injection optional: a step that does not implement the interface is
left untouched.

### Using a connector from a step

KV and EC transfer formats are pluggable independently of steps. A step that builds
prefill or decode bodies obtains a connector by name in its factory and calls the
connector interface during `Execute`:

```go
kvConn, err := kv.Build(kvName) // kvName from params[ParamKVConnector]
ecConn, err := ec.Build(ecName) // ecName from params[ParamECConnector]
```

- `kv.Connector` ([pkg/coordinator/connectors/kv/kv.go](../pkg/coordinator/connectors/kv/kv.go)) shapes the
  `kv_transfer_params` written into the prefill and decode request bodies.
- `ec.Connector` ([pkg/coordinator/connectors/ec/ec.go](../pkg/coordinator/connectors/ec/ec.go)) merges encode
  responses and shapes `ec_transfer_params` for prefill.

To add a connector protocol, add a name constant and a case in the relevant `Build`
switch; the decode/prefill/encode steps pick it up by name with no further change.

### Tests

Tests in `pkg/steps` describe each step's contract. Add a table-driven test next to the
new step covering its parameter validation and its `Execute` behavior, mirroring the
existing `*_test.go` files. Run `make presubmit` (or the targeted package test) and
confirm the output before claiming the step works.

---

## Configuring the pipeline

Configuration is a YAML file passed with `--config` (default `config/coordinator/coordinator.yaml`).
[config/coordinator/coordinator.yaml](../config/coordinator/coordinator.yaml) is the annotated canonical
example: every recognized key is present, required keys uncommented, optional keys shown
commented with their defaults. The loader is [pkg/coordinator/config/config.go](../pkg/coordinator/config/config.go).

### Top-level structure

```yaml
log_level: 2          # 1=warn 2=info 3=verbose 4=debug 5=trace; CLI -v overrides

server:               # inbound HTTP listener
  listen_addr: ":8080"
  read_timeout: 30s
  write_timeout: 120s

gateway:              # outbound client to the Inference Gateway
  address: "http://inference-gateway:80"
  max_idle_conns_per_host: 100
  idle_conn_timeout: 90s
  timeout: 60s

pipeline:             # connector defaults + ordered steps
  kv_connector: kv-shared-storage
  ec_connector: ec-shared-storage
  use_openai_format: true
  steps:
    - type: <step-type>
      params: { ... }
```

`pipeline.steps` is an ordered list. Steps run top to bottom, exactly as written. The
order in the file is the execution order; there is no implicit reordering.

### Environment overrides

Any key can be overridden by an environment variable prefixed `COORDINATOR_`, with `.`
replaced by `_` (viper `AutomaticEnv`). The env value wins over the YAML value. The
documented example is `COORDINATOR_PIPELINE_USE_OPENAI_FORMAT`. The CLI flag `-v` overrides
`log_level`.

### Connector selection

`pipeline.kv_connector` and `pipeline.ec_connector` are deployment-wide defaults. The
entrypoint injects them into every step's `params` so each step constructs the matching
connector. They must agree with the connector configured on the vLLM pods.

| Key | Values | Default |
| :---- | :---- | :---- |
| `kv_connector` | `kv-nixl`, `kv-shared-storage`, `kv-sglang` | `kv-shared-storage` |
| `ec_connector` | `ec-nixl`, `ec-shared-storage` | `ec-shared-storage` |

KV and EC are independent: `ec-nixl` can pair with `kv-shared-storage`, and so on. A
single step may override the default in its own `params` (`kv_connector:` /
`ec_connector:`), which is rarely needed.

### Should the coordinator use the tokens-in format?

`pipeline.use_openai_format` selects the wire format for the encode and prefill steps
(the steps that choose their upstream endpoint by format). Decode and conditional-decode
always forward on the client's original OpenAI path and are unaffected by this setting:

- `true` (default): forward the client's original OpenAI path (`/v1/chat/completions`,
  `/v1/completions`).
- `false`: the tokens-in format. Rewrite to the internal `/inference/v1/generate`
  token-array endpoint, sending `token_ids` and `features` (including `kwargs_data`)
  directly in the body.

A step can override the global with `use_openai_format:` in its own `params`. The
exact bodies per format are in [communication.md](communication.md).

`false` requires a `render` step in the pipeline: render produces the token IDs the
tokens-in format sends, so the coordinator fails to start when `false` is set without a
`render` step configured.

The tokens-in path (`false`) is **experimental**. It is the intended direction (a native
tokens-in endpoint, no re-tokenization), but the `/inference/v1/generate` endpoint is
relatively new and, as shown in [Format tradeoff](#format-tradeoff), its request bodies
carry the full preprocessor output and grow sharply with image resolution. The
OpenAI-format path (`true`) remains the tested default until those limitations are
addressed.

#### Format tradeoff

The choice trades request size against worker recompute, and matters only for
multimodal requests. In both formats the added `tokens` / `token_ids` field prevents
re-tokenization on the worker; the difference is how the image is carried.

- `/v1/chat/completions` carries the image as a raw `data:` URL. The body stays small,
  but the worker re-runs the vision preprocessor from the image bytes.
- `/inference/v1/generate` carries the preprocessor output (`kwargs_data`: a
  msgpack blob of `pixel_values` + `image_grid_thw`). The worker consumes it directly
  with no preprocessing, but the body is dominated by that tensor blob.

For a single 200x300 JPEG (12,082 B on disk), the resulting bodies and their parts:

| Carrier | Size | Contents |
| :---- | :---- | :---- |
| `/v1/chat/completions` body | ~16 KB | `data:` URL: the JPEG plus base64 inflation (4/3) |
| `/inference/v1/generate` body | ~1.1 MB | `kwargs_data["image"][0]`: ~1.15 MB base64 (~860 KB decoded tensor) |

The `generate` body scales with pixel count, not file size, so it grows sharply with
resolution while the `chat/completions` body tracks the compressed image. `mm_placeholders`
length (the number of image placeholder tokens) grows with resolution in both formats:

| Image | JPEG (typical) | `/v1/chat/completions` body | `/inference/v1/generate` body | `mm_placeholders` length |
| :---- | :---- | :---- | :---- | :---- |
| 200x300 | ~12 KB | ~16 KB | ~1.1 MB | 70 |
| 720p (1280x720) | ~80-200 KB | ~110-270 KB | ~20 MB (clamped at `max_pixels`) | ~1,160 |
| 1080p (1920x1080) | ~200-500 KB | ~270-670 KB | ~21 MB (clamped at `max_pixels`) | ~1,280 |
| 4K (3840x2160) | ~1-4 MB | ~1.3-5.4 MB | ~21 MB (clamped at `max_pixels`) | ~1,280 |
| 1080p, `max_pixels`=1.8MP | same | same | ~38 MB | ~2,304 |

The `ec_transfer_params` shape and `size_bytes` are identical between the two formats;
only the request carrier differs.

### The built-in steps

| `type` | Purpose | Key params |
| :---- | :---- | :---- |
| `replace-media-urls` | Download `image_url` references, inline as base64 data URIs, seed `MultimodalEntries`. | `download_timeout`, `max_concurrent_downloads`, `max_multimodal_entries` |
| `render` | Tokenize via the render service; populate `TokenIDs` and per-image hash/placeholder/kwargs. | `address` (required), `timeout`, `max_total_tokens`, `max_total_placeholder_tokens` |
| `conditional-decode` | Optional fast path: attempt decode with `Prefer: if-available`; on 412 continue, otherwise stream the response and stop. | (none) |
| `encode` | Parallel fan-out, one request per multimodal entry; merge EC descriptors. | `max_parallel`, `use_openai_format`, `ec_connector` |
| `prefill` | Single prefill call with tokens + EC/KV hints; capture `kv_transfer_params`. | `use_openai_format`, `kv_connector`, `ec_connector` |
| `decode` | Stream the final completion to the client. | `kv_connector` |

Parameter semantics and defaults are documented inline in
[config/coordinator/coordinator.yaml](../config/coordinator/coordinator.yaml). The wire formats each step
produces are in [communication.md](communication.md).

### Adding a step to the pipeline

After registering a step (see [Creating a new step](#creating-a-new-step-plugin)), enable
it by adding an entry under `pipeline.steps` at the position where it should run:

```yaml
pipeline:
  steps:
    - type: render
      params:
        address: "http://rendering-service:8080"
    - type: example       # the new step
      params:
        threshold: 16
    - type: decode
```

An unknown `type` fails startup with `unknown step type: <type>` from the registry. A
factory that rejects its `params` fails startup with the factory's error.

## References

- [communication.md](communication.md): the per-stage wire-format reference, with exact
  request and response JSON for every step and both request formats.
- [llm-d](https://github.com/llm-d/llm-d): the umbrella project, covering overall goals,
  components, and deployment.
- [llm-d-router architecture](https://github.com/llm-d/llm-d-router/blob/main/docs/architecture.md):
  the Inference Gateway, the EPP, and the sidecar-based disaggregation model the
  coordinator is an alternative to.
- [llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler): the
  EPP scheduling implementation (profile handlers, filters, scorers, deciders).
