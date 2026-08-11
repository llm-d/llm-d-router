# OSL Bucket

**Type:** `osl-bucket`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`

Predicts the **output sequence length (OSL)** bin of a request from request-time
signals and stores it as a request attribute (`"osl-bucket"`) for
output-length-aware scheduling. Consumers today: the in-flight token estimator
(`inflight-load-producer` / `token-load-scorer`). Planned consumers: flow-control
queue ordering (short-first) and KV-pressure gating.

## What It Does

The plugin runs after the request body is parsed and attached, but before
admission control. It classifies each request into one of three bins and
publishes the result via `request.PutAttribute("osl-bucket", bucket)`:

| Bin | Meaning | Typical output |
|-----|---------|----------------|
| `LONG` | reasoning chain | ≥ 2,000 tokens |
| `SHORT` | tool-call / short response | < 500 tokens |
| `UNKNOWN` | no reliable signal | consumers fall back to their own estimate |

The plugin only classifies and publishes — it does not itself change routing.
Separating production from consumption means the bin is a reliable shared signal
for any subsystem, exactly like `agent-identity`.

## How It Works

Classification:

1. `enable_thinking=true` → **LONG** (reasoning mode).
2. `thinking_budget > 4000` (without explicit `enable_thinking`) → **LONG**.
3. `has_tools=true` **and** `enable_thinking` false/absent → **SHORT**
   (tool-call JSON). The `enable_thinking` guard matters: tools alone is *not*
   a SHORT signal when thinking is also on.
4. `max_output_tokens < 500` → **SHORT** (explicit client cap).
5. Otherwise → **UNKNOWN**.

`UNKNOWN` is still published (as the zero value), so a missing attribute and an
explicit UNKNOWN read the same. The plugin is stateless and safe under
concurrent use.

## Inputs Consumed

- `request.Body.ChatCompletions.ChatTemplateKWArgs` — `enable_thinking`,
  `thinking_budget` (populated by vLLM from the client's `extra_body`).
- `request.Body.ChatCompletions.Tools` — presence implies `has_tools`.
- `request.Body.MaxOutputTokens` — normalized client output cap.

ISL (input length) is intentionally *not* consumed — it has no correlation with
OSL and only adds noise.

## Outputs Produced

- `scheduling.InferenceRequest` attribute `"osl-bucket"` (`oslbucket.OSLBucket`).
  Read it with `scheduling.ReadRequestAttribute[oslbucket.OSLBucket](req, oslbucket.OSLBucketKey)`.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: osl-bucket` entry to enable; the
runner discovers it as a `RequestHeaderProcessor` and wires it in. No parameters.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: osl-bucket
  - type: inflight-load-producer
    parameters:
      addEstimatedOutputTokens: true
  - type: token-load-scorer
```

The token estimator degrades gracefully when `osl-bucket` is absent: with no
attribute set, every request reads as UNKNOWN and uses the flat UNKNOWN output
estimate (1,000 tokens).

## Limitations

- **Recall, not precision, is the tradeoff.** ~43% of genuinely short requests
  (no-signal chat traffic) fall through to UNKNOWN and get no routing benefit —
  acceptable, since they are not *harmed*. Precision on SHORT is 100%, so a long
  request is never misclassified as short.
- **Signals must be present on the wire.** `enable_thinking` / `thinking_budget`
  only appear when the client sends them (via `extra_body`) and the server model
  supports them (e.g. GLM 5.2, Kimi K3).
