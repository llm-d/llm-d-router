# Inference API Routing

This document lists, per HTTP method and path, how the sidecar and the
coordinator handle a request: serve it, pass it through unparsed, or answer it
itself.

## Layers

Three layers decide where a request goes.

1. The gateway HTTPRoute selects the backend, the coordinator or the
   InferencePool, before any llm-d code runs. It is deployment configuration.
   [httproutes.yaml](../deploy/components/inference-gateway/httproutes.yaml)
   sends every path to the InferencePool. The
   [coord-disaggregation guide](https://github.com/llm-d/llm-d/tree/main/guides/coord-disaggregation)
   sends the coordinator's served paths to the coordinator with `Exact` matches
   and every other path to the InferencePool.
2. The [agentic-api](https://github.com/vllm-project/agentic-api) gateway, when
   deployed in front of llm-d, holds Responses and Conversations state. It
   answers stateful requests itself and forwards the rest to the router.
3. The sidecar and coordinator route tables, described here. They apply whether
   or not the agentic-api gateway is deployed.

## Outcomes

- **served**: the request body is parsed. The sidecar refuses a Responses body
  that `RejectStatefulResponsesFields` in
  [tokens.go](../pkg/common/request/tokens.go) rejects with a 400 and, when EPP
  set `x-prefiller-host-port`, runs disaggregated prefill. The coordinator runs
  its pipeline.
- **passthrough**: the request is forwarded with its body unread. The sidecar
  forwards to its decoder. The coordinator forwards to the gateway with
  `EPP-Profile: decode`, so the request reaches a decode pod's sidecar, which
  applies its own table.
- **405**: the coordinator answers `405 Method Not Allowed` with `Allow: POST`
  and an empty body, from chi's default handler.
- **answered**: the component answers the request itself, without a backend.

The sidecar answers a refused request body with a vLLM-style JSON 400 on every
path it serves, including `/v1/messages`. The coordinator's 400 is plain text.

## Path forms

Both components clean the request path with Go's `path.Clean` before routing.
When the result is one of the six inference API paths in the first rows below,
the request is routed, guarded, and forwarded upstream under that path, so a
trailing slash, repeated slashes, or dot segments do not change the outcome.
Paths are case-sensitive. The coordinator forwards every other path as sent.
The sidecar's `ServeMux` answers any other path that is not clean with a 307
redirect to its cleaned form, for example `/v1/responses/..` to `/v1`, and
forwards a clean one as sent.

The coord-disaggregation guide's `Exact` matches see the path before this
cleaning. A request such as `POST /v1/completions/` matches the guide's
catch-all rule and is served by the sidecar without the coordinator pipeline.
In that topology, clients must send the exact path form.

## Routes

The agentic-api column reflects the gateway's route table at
`vllm-project/agentic-api@d162633`. "no route" means the path is not in that
table.

| Method | Path | Sidecar | Coordinator | agentic-api gateway |
|---|---|---|---|---|
| POST | `/v1/chat/completions` | served | served | forwards |
| POST | `/v1/completions` | served | served | forwards |
| POST | `/inference/v1/generate` | served | served | no route |
| POST | `/v1/responses` | served | passthrough [1] | forwards a stateless request |
| POST | `/v1/messages` | served | passthrough [2] | forwards |
| POST | `/generate` | served | passthrough [2] | no route |
| not POST | `/v1/chat/completions`, `/v1/completions`, `/inference/v1/generate` | passthrough [4] | 405 | no route |
| not POST | `/v1/responses`, `/v1/messages`, `/generate` | passthrough [4] | passthrough [4] | terminates `GET /v1/responses` (WebSocket) |
| POST | `/v1/messages/count_tokens` | passthrough | passthrough | forwards |
| POST | `/v1/conversations` | passthrough | passthrough | forwards |
| GET | `/v1/models` | passthrough | passthrough | forwards |
| POST | `/v1/responses/compact` | passthrough | passthrough | terminates |
| GET | `/v1/responses/{response_id}` | passthrough [3] | passthrough [3] | terminates |
| GET, POST, DELETE | `/v1/conversations/{id}`, `/v1/conversations/{id}/items`, `/v1/conversations/{id}/items/{item_id}` | passthrough [3] | passthrough [3] | terminates |
| POST | `/v1/responses/input_tokens` | passthrough | passthrough | no route |
| DELETE | `/v1/responses/{response_id}` | passthrough [3] | passthrough [3] | no route |
| POST | `/v1/responses/{response_id}/cancel` | passthrough [3] | passthrough [3] | no route |
| GET | `/v1/responses/{response_id}/input_items` | passthrough [3] | passthrough [3] | no route |
| POST | `/v1/chat/completions/render`, `/v1/completions/render` | passthrough | passthrough | no route |
| GET, POST, DELETE | `/v1/chat/completions/{id}`, `/v1/chat/completions/{id}/messages` | passthrough [3] | passthrough [3] | no route |
| GET, POST, DELETE | `/v1/messages/batches` and its sub-paths | passthrough [3] | passthrough [3] | no route |
| GET | `/health` | answered | passthrough | terminates |
| GET | `/healthz`, `/readyz` | passthrough | answered | no route |
| GET, DELETE | `/v1/requests/{id}` | passthrough | answered by the `async-broker` step when enabled, which answers other methods with 405; otherwise passthrough | no route |
| any | `/v1/files` and its sub-paths, `/v1/embeddings`, any other path | passthrough | passthrough | no route |

1. The coordinator pipeline does not handle the Responses request shape. The
   request reaches a decode pod's sidecar, which serves it.
2. The coordinator pipeline does not handle Messages or SGLang generate
   requests. The request reaches a decode pod's sidecar, which serves it.
3. Stateful. Without the agentic-api gateway, the request reaches whichever
   decode pod EPP selects, and any state the engine keeps exists on that pod
   only. The router does not support these requests without the gateway.
4. The engine answers by its own routes: vLLM and SGLang answer 405 for most
   methods, their CORS middleware answers `OPTIONS` preflights, and SGLang
   serves `PUT /generate`.
