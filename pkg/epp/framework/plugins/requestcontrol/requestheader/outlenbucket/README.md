# Output-Length Bucket

**Type:** `outlen-bucket`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`

Predicts the **output length** bin of a request from request-time
signals and stores it as a request attribute (`"outlen-bucket"`) for
output-length-aware scheduling. Consumers today: the in-flight token estimator
(`inflight-load-producer` / `token-load-scorer`). Planned consumers: flow-control
queue ordering (short-first) and KV-pressure gating.

## What It Does

The plugin runs after the request body is parsed and attached, but before
admission control. It classifies each request into one of three bins and
publishes the result via `request.PutAttribute("outlen-bucket", bucket)`:

| Bin | Meaning | Typical output |
|-----|---------|----------------|
| `LONG` | reasoning chain | >= 2,000 tokens |
| `SHORT` | tool-call / short response | < 500 tokens |
| `UNKNOWN` | no reliable signal | consumers apply their neutral middle estimate |

The plugin only classifies and publishes -- it does not itself change routing.
Separating production from consumption means the bin is a reliable shared signal
for any subsystem, exactly like `agent-identity`.

## How It Works

Classification:

1. `enable_thinking=true` -> **LONG** (reasoning mode).
2. `thinking_budget > 4000` (without explicit `enable_thinking`) -> **LONG**.
3. `has_tools=true` **and** `enable_thinking` false/absent -> **SHORT**
   (tool-call JSON). The `enable_thinking` guard matters: tools alone is *not*
   a SHORT signal when thinking is also on.
4. `max_output_tokens < 500` -> **SHORT** (explicit client cap).
5. Otherwise -> **UNKNOWN**.

The `enable_thinking`, `thinking_budget`, and tools signals are all read from the
chat-completions body (`chat_template_kwargs` and `tools`). Other body shapes
(Claude `/v1/messages`, OpenAI `/v1/responses`) are not inspected for these: their
thinking signals are not surfaced, so a tool-call there could not be told apart
from a reasoning request and is left UNKNOWN. `max_output_tokens` is the
normalized client cap and applies regardless of shape.

`UNKNOWN` is still published (as the zero value), so a missing attribute and an
explicit UNKNOWN read the same. The plugin is stateless and safe under
concurrent use.

## Inputs Consumed

- `request.Body.ChatCompletions.ChatTemplateKWArgs` -- `enable_thinking`,
  `thinking_budget` (populated by vLLM from the client's `extra_body`).
  Chat-completions only.
- `request.Body.ChatCompletions.Tools` -- presence implies `has_tools`.
- `request.Body.MaxOutputTokens` -- normalized client output cap.

Input length is intentionally *not* consumed -- it has no correlation with
output length and only adds noise.

## Outputs Produced

- `scheduling.InferenceRequest` attribute `"outlen-bucket"` (`outlenbucket.Bucket`).
  Read it with `scheduling.ReadRequestAttribute[outlenbucket.Bucket](req, outlenbucket.AttributeKey)`.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: outlen-bucket` entry to enable; the
runner discovers it as a `RequestHeaderProcessor` and wires it in. No parameters.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: outlen-bucket
  - type: inflight-load-producer
    parameters:
      addEstimatedOutputTokens: true
  - type: token-load-scorer
```

### Relationship to `addEstimatedOutputTokens`

This plugin only publishes the bucket; the `inflight-load-producer`'s
`addEstimatedOutputTokens` option decides whether the bucket is *used* to add
estimated output tokens to the in-flight load. They are independent switches:

| `outlen-bucket` | `addEstimatedOutputTokens` | Behavior |
|-----------------|----------------------------|----------|
| off | off | Output tokens not added to in-flight load. |
| off | on | Output added, but every request reads as UNKNOWN (flat estimate); the producer logs a one-time warning. |
| on | off | Bucket published for other consumers, not added to in-flight load. |
| on | on | Bucket published and used for the per-request output estimate. Intended setup. |

When the attribute is absent, it reads as its zero value, `UNKNOWN`, so an
explicitly-published UNKNOWN and a missing attribute produce the same estimate.

**Ordering is a phase guarantee, not a list-order requirement.** The producer
reads the bucket in its `Produce` / `PreRequest` hooks, which always run after
this plugin's `RequestHeader` hook -- regardless of config order. The guarantee is
the fixed call sequence in `Director.HandleRequest`
(`pkg/epp/requestcontrol/director.go`): `runRequestHeaderProcessors` before
`runDataProducerPlugins` and `runPreRequestPlugins`. It is *not* a declared
`Produce`/`Consume` dependency (`RequestHeaderProcessor` has none, and the
attribute API is untyped/string-keyed with no ordering check); `agentidentity`
relies on the same contract. If the plugin is not enabled -- or a future refactor
runs a reader before `runRequestHeaderProcessors` -- the read silently falls back
to `UNKNOWN` with no error, so anyone changing that call order must keep
`RequestHeader` first. (The producer logs a one-time warning on the not-enabled
case; see `warnMissingOutlenBucket`.)

## Limitations

- **Conservative by design.** A request with no reliable signal is left UNKNOWN
  rather than guessed, so some genuinely short requests get no routing benefit --
  but a long request is never labeled SHORT.
- **Signals must be present on the wire.** `enable_thinking` / `thinking_budget`
  only appear when the client sends them (via `extra_body`) and the server model
  supports them (e.g. GLM 5.2, Kimi K3).
