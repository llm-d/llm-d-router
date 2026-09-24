# Plan: OpenResponses Acceptance Tests → llm-d-router E2E

## Overview

The [openresponses/openresponses](https://github.com/openresponses/openresponses) repository is
the Open Responses specification project — a vendor-neutral, OpenAI-Responses-API-compatible
spec for structured LLM interfaces. It ships:

- An OpenAPI schema (`public/openapi/openapi.json`) for the `/v1/responses` and
  `/v1/responses/compact` endpoints.
- A TypeScript/Bun compliance test suite (`src/lib/compliance-tests.ts`) that runs
  HTTP and WebSocket acceptance tests against any conformant server.
- A CLI runner (`bin/compliance-test.ts`) that can be pointed at any base URL.

The llm-d-router already registers `POST /v1/responses` (see
`pkg/sidecar/proxy/chat_completions.go:ResponsesPath`) and proxies it via
`disaggregatedPrefillHandler(APITypeResponses)`.  No existing e2e test exercises that path.

The goal is to add a new Ginkgo test file (`test/e2e/responses_test.go`) that runs the
openresponses acceptance tests against the live cluster, using the same infrastructure
(Kind, Envoy, EPP) that the existing suite already provisions.

### Compliance test IDs (from `src/lib/compliance-tests.ts`)

| id | transport | streaming | notes |
|----|-----------|-----------|-------|
| `basic-response` | http | no | schema + output present + status=completed |
| `assistant-phase` | http | no | assistant phase labels in input |
| `response-output-phase-schema` | http | no | mock fixture — schema-only, no network call |
| `streaming-response` | http | yes | SSE events + schema + status=completed |
| `websocket-response` | ws | yes | WebSocket streaming |
| `websocket-sequential-responses` | ws | yes | two turns on one connection |
| `websocket-continuation` | ws | yes | `previous_response_id` continuation |
| `websocket-reconnect-store-false-recovery` | ws | yes | reconnect + `previous_response_not_found` |
| `websocket-previous-response-not-found` | ws | yes | missing id → error code |
| `websocket-failed-continuation-evicts-cache` | ws | yes | eviction after failure |
| `websocket-compact-new-chain` | ws | yes | `/responses/compact` + WS new chain |
| `system-prompt` | http | no | system role message |
| `tool-calling` | http | no | function_call output |
| `image-input` | http | no | base64 data URI image |
| `multi-turn` | http | no | conversation history |
| `compact-response` | http | no | `/responses/compact` endpoint |
| `compact-missing-model` | http | no | error case: 400/422 |

**Scope for this plan:** Tests exercising the router as a transparent proxy are in scope.
Tests that require full agentic state (`store`, `previous_response_id`, WebSocket session
affinity, `/responses/compact`) depend on backend support that the vLLM simulator does not
currently provide; those are added as pending/skipped specs with a clear skip reason.

---

## Sub-tasks

---

### Sub-task 1 — Understand what the simulator returns for `/v1/responses`

**Status:** `[ ] pending`

**Intent**

Before writing assertions, verify exactly what the vLLM simulator (`llm-d-inference-sim`)
returns when `POST /v1/responses` is called, so the tests are grounded in observed behaviour
rather than assumptions.

**Expected Outcomes**

- A documented list of which openresponses test IDs pass/fail/skip against the simulator.
- A note confirming whether the simulator echoes the `ResponseResource` schema or converts to
  chat-completions format.

**Todo List**

1. Run the existing e2e suite against a local Kind cluster (`make test-e2e`) and add a
   temporary `curl` call to `POST /v1/responses` inside a test to capture the raw response.
   Alternatively, start a single simulator pod (`docker run ...`) and probe it directly.
2. Compare the raw response body against the `responseResourceSchema` Zod definition
   (`/tmp/openresponses/src/generated/kubb/zod/responseResourceSchema.ts`).
3. Record findings in this file under a new "Simulator Notes" section before Sub-task 2.

**Relevant Context**

- Simulator image: `ghcr.io/llm-d/llm-d-inference-sim:v0.10.2` (see
  `test/e2e/e2e_suite_test.go:77`)
- Route registration: `pkg/sidecar/proxy/proxy.go:createRoutes()` — `POST /v1/responses`
  → `disaggregatedPrefillHandler(APITypeResponses)` which proxies to the decoder pod.

---

### Sub-task 2 — Add `/v1/responses` HTTP acceptance tests (non-streaming)

**Status:** `[ ] pending`

**Intent**

Add a new file `test/e2e/responses_test.go` with Ginkgo specs covering the HTTP
(non-WebSocket, non-streaming) openresponses tests that the simulator can satisfy.

**Expected Outcomes**

- `test/e2e/responses_test.go` compiles and passes `make presubmit`.
- The following test IDs pass against a deployed cluster with the simulator backend:
  - `basic-response`
  - `system-prompt`
  - `multi-turn`
  - `tool-calling` (may be skipped if simulator does not emit `function_call` output)
  - `image-input` (may be skipped if simulator does not process image content)
  - `compact-missing-model` (400/422 error path — router-level validation)
- Tests that cannot pass because the simulator does not implement the backend contract are
  wrapped in a `ginkgo.Skip(...)` with an explicit reason.

**Todo List**

1. Create `test/e2e/responses_test.go` in package `e2e`.
2. Follow the pattern from `test/e2e/generate_endpoint_test.go`: one `ginkgo.Describe`
   block, `ginkgo.Ordered`, wrapped in `testWrapper`.
3. For each applicable test ID above, add a `ginkgo.It` spec that:
   a. Builds the request body matching the openresponses `getRequest` definition.
   b. Posts to `/v1/responses` via `doPost` or `doPostWithError` (already in
      `requests_test.go`).
   c. Validates HTTP status and, where the simulator returns a parseable body, checks:
      - `"object"` field equals `"response"`.
      - `"status"` field equals `"completed"`.
      - `"output"` array is non-empty.
4. For `compact-missing-model`, post to `/v1/responses/compact` with `doPostWithError`
   and assert status is `400` or `422`.
5. Add a `runResponsesRequest` helper (analogous to `runChatCompletion`) to
   `requests_test.go`, or inline helpers directly in the new file.

**Relevant Context**

- `doPost` / `doPostWithError` — `test/e2e/requests_test.go:58-97`
- `testWrapper` — `test/e2e/setup_test.go`
- Existing non-PD single-node setup: `test/e2e/e2e_test.go:70` — the simplest reference
  to copy for infrastructure setup in a new `Describe` block.
- openresponses request shapes: `src/lib/compliance-tests.ts:testTemplates` — the Go test
  constructs equivalent JSON strings.

---

### Sub-task 3 — Add streaming `/v1/responses` acceptance test

**Status:** `[ ] pending`

**Intent**

Add a streaming SSE acceptance test for the `streaming-response` openresponses test ID.
This exercises `POST /v1/responses` with `"stream": true` and validates that the response
is a well-formed SSE stream terminating with a completed `ResponseResource`.

**Expected Outcomes**

- A `ginkgo.It("streaming responses endpoint returns SSE events")` spec in
  `test/e2e/responses_test.go` that passes or is skipped with a clear reason.
- The spec validates:
  - HTTP 200.
  - `Content-Type: text/event-stream`.
  - At least one `data:` line before `data: [DONE]`.
  - The final accumulated body deserialises to a `ResponseResource`-shaped JSON object
    (checking `"object"`, `"status"`, `"output"` fields only — not full schema validation,
    which is TypeScript-side).

**Todo List**

1. Add a `runStreamingResponsesRequest` helper that sends a streaming POST to
   `/v1/responses` and returns the raw SSE body (model after `runStreamingChatCompletion`
   in `requests_test.go`).
2. Parse the SSE body: split on `\n\n`, strip `data: ` prefixes, stop at `[DONE]`,
   JSON-unmarshal the last non-DONE event as the terminal response.  Reuse or generalise
   `extractFinishReasonFromStreaming` from `utils_test.go` as a pattern.
3. Assert at least one event was received and the terminal event contains `"object":"response"`.
4. Add the `ginkgo.It` spec.

**Relevant Context**

- `runStreamingChatCompletion` — `test/e2e/requests_test.go:342`
- `extractFinishReasonFromStreaming` — `test/e2e/utils_test.go`
- `parseSSEStream` in the openresponses repo (`src/lib/sse-parser.ts`) is the TypeScript
  reference for what constitutes a valid event sequence; the Go equivalent is simpler
  (no Zod validation needed).

---

### Sub-task 4 — Document skipped WebSocket tests and add placeholders

**Status:** `[ ] pending`

**Intent**

Add placeholder `ginkgo.It` specs for the WebSocket openresponses tests.  These cannot run
against the current simulator backend, but their presence documents the compliance gap and
makes it easy to enable them later.

**Expected Outcomes**

- One `ginkgo.It` per WebSocket test ID (see table above: 6 WebSocket tests) in
  `test/e2e/responses_test.go`, each calling `ginkgo.Skip(reason)` immediately.
- The skip reason references the openresponses test ID and names the missing capability
  (WebSocket session affinity, `previous_response_id` persistence, etc.).

**Todo List**

1. For each WebSocket test ID, add a `ginkgo.It` that calls `ginkgo.Skip(...)` with the
   reason `"openresponses/<id>: requires WebSocket session affinity not supported by simulator"`.
2. Follow the same `ginkgo.Ordered` block structure as the HTTP tests.
3. Run `make presubmit` and confirm all skipped specs show as `S` in the Ginkgo output.

---

### Sub-task 5 — Wire up the openresponses CLI in CI (optional, additive)

**Status:** `[ ] pending`

**Intent**

Allow the openresponses compliance CLI (`bin/compliance-test.ts`) to be run as an optional,
non-blocking step in CI by adding a Makefile target and documenting it in
`test/e2e/README.md`.  This is independent of the Ginkgo tests and provides a direct
comparison point against the openresponses web UI.

**Expected Outcomes**

- A `make test-openresponses-compliance` target that:
  - Requires `BASE_URL` and `MODEL` env vars.
  - Runs `bun run bin/compliance-test.ts --base-url $BASE_URL --model $MODEL --api-key dummy`
    from a checked-out copy of openresponses.
- A section in `test/e2e/README.md` describing when and how to run it.
- The target is not wired into `make presubmit` (it requires a live cluster and bun).

**Todo List**

1. Add the Makefile target (outside the `presubmit` dependency chain).
2. Add a "Running OpenResponses compliance tests" section to `test/e2e/README.md`.
3. Note the test IDs that currently pass vs. skip, pointing readers to the openresponses
   compliance table.

---

## Simulator Notes

_(Fill in after Sub-task 1 is complete.)_

---

## Dependencies and Constraints

- The vLLM simulator must respond on `POST /v1/responses`.  If it returns HTTP 404 or an
  unrecognised body shape, Sub-tasks 2–3 tests are all skipped rather than failed.
- WebSocket tests require either simulator support or a real vLLM backend; they are
  placeholders until that support lands.
- No new Go module dependencies are introduced; all request/response validation uses
  standard `encoding/json` and string comparison, matching the existing test style.
- `make presubmit` must pass after each sub-task before moving to the next.
