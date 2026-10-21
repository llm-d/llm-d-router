# Inference API Routing

This document lists, per HTTP method and path, how the sidecar and the
coordinator handle a request: serve it, pass it through unparsed, or answer it
itself.

## Layers

1. The gateway HTTPRoute selects the backend, the coordinator or the
   InferencePool, before any llm-d code runs. It is deployment configuration.
   [httproutes.yaml](../deploy/components/inference-gateway/httproutes.yaml)
   sends every path to the InferencePool. The
   [coord-disaggregation guide](https://github.com/llm-d/llm-d/tree/main/guides/coord-disaggregation)
   sends `/v1/completions`, `/v1/chat/completions` and `/inference/v1/generate`
   to the coordinator with `Exact` matches and every other path, including
   `/v1/responses` and `/v1/messages`, to the InferencePool.
2. The [agentic-api](https://github.com/vllm-project/agentic-api) gateway, when
   deployed in front of llm-d, holds Responses and Conversations state. It
   answers stateful requests itself and forwards the rest to the router.
3. The sidecar and coordinator route tables, described here. They apply whether
   or not the agentic-api gateway is deployed.

## Outcomes

- **served**: the route's handler runs. Both components refuse a Responses
  body that `RejectStatefulResponsesFields` in
  [tokens.go](../pkg/common/request/tokens.go) rejects with a 400. The sidecar
  runs disaggregated prefill when EPP set `x-llm-d-prefiller-host-port`, and
  disaggregated encode when EPP set `x-llm-d-encoder-hosts-ports` and an
  encoder connector is configured. Without either, the sidecar may forward the
  body to its decoder unread; it always reads a Responses body for the check.
  The coordinator parses the body and runs its pipeline.
- **passthrough**: the request is forwarded with its body unread. The sidecar
  forwards to its decoder. The coordinator forwards to the gateway with
  `x-llm-d-epp-profile: decode`, so the request reaches a decode pod's sidecar,
  which applies its own table.
- **405**: the coordinator answers `405 Method Not Allowed` with an `Allow`
  header listing the route's methods and an empty body, from chi's default
  handler. chi answers a method it does not know, such as `PROPFIND`, with a 405
  and no `Allow` header on every path.
- **answered**: the component answers the request itself, without a backend.

When the sidecar reads a request body and refuses it, it answers with a
vLLM-style JSON 400, including on `/v1/messages`. The coordinator answers in the
error envelope of the request's API: Anthropic's for `/v1/messages` and
OpenAI's for the other paths.

## Path forms

Paths are case-sensitive. The coordinator runs before EPP and the sidecar runs
after it, so they accept different forms of the inference API paths in the
first rows below.

The coordinator cleans the path of a POST request with Go's `path.Clean`. When
the result is an inference API path it serves, the request is routed, checked
and sent upstream under that path, so a trailing slash, repeated slashes, dot
segments or an encoded slash do not change the outcome. Every other request is
forwarded as sent, so the 405 in the table applies to the exact path only.

EPP picks its request parser by matching the end of the raw path and ignores a
trailing slash. The sidecar therefore serves a POST to the six paths with or
without a trailing slash, and no other form of them. It answers a
percent-encoded form of the six paths, and its `ServeMux` answers a path with
repeated slashes or dot segments, with a 307 redirect to the clean path, for
example `/v1%2Fresponses` and `//v1/responses` to `/v1/responses`, so the
client sends the clean path back through EPP. It forwards any other path to the
decoder as sent.

A gateway can normalize repeated slashes, dot segments and encoded slashes
before the HTTPRoute, EPP and both components see the path. In Envoy these are
the `merge_slashes`, `normalize_path` and `path_with_escaped_slashes_action`
settings. [envoy.yaml](../deploy/environments/dev/coordinator-e2e-infra/envoy.yaml)
sets all three for the coordinator e2e environment, with `UNESCAPE_AND_REDIRECT`
answering an encoded slash with a redirect. Whether a gateway applies them
depends on its configuration, and none of them removes a trailing slash.

The coord-disaggregation guide's `Exact` matches see the path before the
coordinator cleans it. A request such as `POST /v1/completions/` matches the
guide's catch-all rule and is served by the sidecar without the coordinator
pipeline. In that topology, clients must send the exact path form.

## Routes

The agentic-api column reflects the gateway's route table at
`vllm-project/agentic-api@d162633`. "no route" means the path is not in that
table.

| Method | Path | Sidecar | Coordinator | agentic-api gateway |
|---|---|---|---|---|
| POST | `/v1/chat/completions` | served | served | forwards |
| POST | `/v1/completions` | served | served | forwards |
| POST | `/inference/v1/generate` | served | served | no route |
| POST | `/v1/responses` | served | served | forwards a stateless request |
| POST | `/v1/messages` | served | served | forwards |
| POST | `/generate` | served | passthrough [1] | no route |
| not POST | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/inference/v1/generate` | passthrough [3] | 405 | terminates `GET /v1/responses` (WebSocket) |
| not POST | `/generate` | passthrough [3] | passthrough [3] | no route |
| POST | `/v1/messages/count_tokens` | passthrough | passthrough | forwards |
| POST | `/v1/conversations` | passthrough | passthrough | forwards |
| GET | `/v1/models` | passthrough [4] | passthrough [4] | forwards |
| POST | `/v1/responses/compact` | passthrough | passthrough | terminates |
| GET | `/v1/responses/{response_id}` | passthrough [2] | passthrough [2] | terminates |
| GET, POST, DELETE | `/v1/conversations/{id}`, `/v1/conversations/{id}/items`, `/v1/conversations/{id}/items/{item_id}` | passthrough [2] | passthrough [2] | terminates |
| POST | `/v1/responses/input_tokens` | passthrough | passthrough | no route |
| DELETE | `/v1/responses/{response_id}` | passthrough [2] | passthrough [2] | no route |
| POST | `/v1/responses/{response_id}/cancel` | passthrough [2] | passthrough [2] | no route |
| GET | `/v1/responses/{response_id}/input_items` | passthrough [2] | passthrough [2] | no route |
| POST | `/v1/chat/completions/render`, `/v1/completions/render` | passthrough | passthrough | no route |
| GET, POST, DELETE | `/v1/chat/completions/{id}`, `/v1/chat/completions/{id}/messages` | passthrough [2] | passthrough [2] | no route |
| GET, POST, DELETE | `/v1/messages/batches` and its sub-paths | passthrough [2] | passthrough [2] | no route |
| GET | `/health` | answered | passthrough | terminates |
| any | `/healthz`, `/readyz` | passthrough | answered for GET; 405 for other methods | no route |
| GET, DELETE | `/v1/requests/{id}` | passthrough | answered by the `async-broker` step when enabled, which answers other methods with 405; otherwise passthrough | no route |
| any | `/v1/files` and its sub-paths, `/v1/embeddings`, any other path | passthrough | passthrough | no route |

1. The coordinator pipeline does not handle SGLang generate requests. The
   request reaches a decode pod's sidecar, which serves it.
2. Stateful. Without the agentic-api gateway, the request reaches whichever
   decode pod EPP selects, and any state the engine keeps exists on that pod
   only. The router does not support these requests without the gateway.
3. The engine answers by its own routes: vLLM and SGLang answer 405 for most
   methods, their CORS middleware answers `OPTIONS` preflights, and SGLang
   serves `PUT /generate`.
4. When EPP runs the `models-responder` plugin, it answers `GET /v1/models`
   with the pool's model list, and the request does not reach a pod.
