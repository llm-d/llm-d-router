# GenAI Payload Capture (opt-in)

The EPP can record the prompt of each scheduled request on its `gateway.request`
tracing span, following the upstream
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):
the span carries a `gen_ai.client.inference.operation.details` event with
`gen_ai.input.messages` and `gen_ai.system_instructions` attributes (both
`Opt-In` / `Development` stability upstream).

The feature is **off by default** and is Phase 1 of the
[GenAI payload events proposal](https://github.com/llm-d/llm-d/blob/main/docs/proposals/genai-payload-events.md)
in the llm-d repository — see the proposal for the full design (object-store
backends, redaction, vLLM-native capture) landing in later phases.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `LLMD_PAYLOAD_CAPTURE_ENABLED` | `false` | Master switch. Capture is opt-in. |
| `LLMD_PAYLOAD_BACKEND` | `noop` | `noop` emits nothing (secondary kill switch); `inline` attaches payloads to span events. `gcs`, `s3` and `filesystem` are reserved for Phase 2 and currently fall back to `noop` with a warning. |
| `LLMD_PAYLOAD_INLINE_THRESHOLD` | `4096` | Largest serialised payload (bytes) attached inline. Larger payloads are dropped and the event is marked `llm_d.payload.truncated: true`. |

Example (EPP container env):

```yaml
env:
  - name: LLMD_PAYLOAD_CAPTURE_ENABLED
    value: "true"
  - name: LLMD_PAYLOAD_BACKEND
    value: "inline"
```

Payload events are only recorded on sampled spans, so OTel tracing must be
enabled (`OTEL_TRACES_EXPORTER=otlp`) and the request must be sampled.

## What is captured

- **Chat completions** (`/v1/chat/completions`): messages as `role`/`parts`;
  `system` and `developer` messages are recorded as `gen_ai.system_instructions`.
- **Completions** (`/v1/completions`): the prompt as a single `user` message.
- **Anthropic messages** (`/v1/messages`): messages plus the top-level `system`
  field.
- Multimodal parts referencing external URLs are recorded as schema-standard
  `uri` parts. Raw inline bytes (data URLs, base64 audio/images) are **blob**
  parts and require an object-store backend (Phase 2); until then they are
  dropped and the event carries `llm_d.payload.truncated: true`. Other request
  types (embeddings, responses, generate) are not captured yet.

The `llm_d.payload.*` attributes are llm-d extensions defined by the proposal;
they are kept out of the reserved `gen_ai.*` namespace.

## Security

Captured payloads contain user data. Before enabling, ensure TLS on OTLP
endpoints, restrict collector access, and treat trace storage with the same
classification as the underlying request data. The redaction pipeline arrives
in Phase 3; until then do not enable capture on workloads carrying regulated
data unless the trace backend itself is appropriately controlled.
