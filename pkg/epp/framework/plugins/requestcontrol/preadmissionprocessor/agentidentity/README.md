# Agent Identity

**Type:** `agent-identity`
**Interfaces:** `requestcontrol.PreAdmissionProcessor`

Resolves a per-session identity from agent-specific HTTP headers and writes it into `InferenceRequest.FairnessID`, so every turn of an agent session lands in the same flow-control fairness queue.

## What It Does

The plugin runs after request assembly and before admission control. If the request does not already carry an explicit fairness ID (`x-gateway-inference-fairness-id`), it inspects a fixed set of agent session headers and copies the first non-empty value into `FairnessID`. The flow-control layer keys its queues on `FlowKey{ID: FairnessID, Priority}`, so this turns "all turns from one agent session" into "all turns share one queue."

Without it, every request from a given agent session falls into the default fairness queue alongside unrelated traffic, and per-session fairness, prefix-cache affinity, and per-tenant rate limiting all collapse to per-request granularity.

## How It Works

1. If `request.FairnessID` is already set to something other than `metadata.DefaultFairnessID`, return immediately — an explicit upstream `x-gateway-inference-fairness-id` always wins over a derived one.
2. Otherwise, walk the priority list of agent session headers and copy the first non-empty match into `request.FairnessID`. Operator-supplied entries from `additionalSessionHeaders` come first, followed by the built-in defaults in this order:
   1. `x-claude-code-session-id` (Claude Code)
   2. `x-session-affinity` (OpenCode)
   3. `session-id` (Codex)
   4. `session_id` (Codex, legacy underscored fallback)
3. If nothing matches, leave `FairnessID` as the default and return — the request is still admitted, just into the shared default queue.

The plugin is stateless and safe under concurrent use.

## Inputs Consumed

- `scheduling.InferenceRequest.Headers` — read-only lookup of the session headers above (built-in defaults plus any from `additionalSessionHeaders`). Keys are expected lowercase (Envoy normalizes inbound headers).
- `scheduling.InferenceRequest.FairnessID` — read to detect an upstream override; written when an agent header matches.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: agent-identity` entry to enable; the runner discovers it as a `PreAdmissionProcessor` and wires it in.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `additionalSessionHeaders` | `[]string` | No | `[]` | Extra header names to check before the built-in defaults. Order is preserved; the first non-empty match wins. Use this to support a new agent, or to track an upstream rename, without a code change. |

### Examples

Default configuration — no parameters, only the built-in headers are checked:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: agent-identity
```

With additional headers — checked before the built-in defaults (header names are arbitrary; substitute whatever the agent actually emits):

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: agent-identity
    parameters:
      additionalSessionHeaders:
        - x-my-agent-session
        - x-another-agent-id
```

### Per-agent client setup

The plugin only reads headers — getting them onto the wire is the agent's job. Each supported agent has different requirements.

#### Claude Code — **LiteLLM is required**

Claude Code speaks Anthropic's Messages API. llm-d's gateway exposes the OpenAI chat-completions wire format, so a translator is required in the path. LiteLLM works:

```yaml
# ~/.litellm/config.yaml
model_list:
  - model_name: <client-facing-model-name>
    litellm_params:
      model: hosted_vllm/<upstream-model-name>
      api_base: http://<llmd-gateway>/v1

litellm_settings:
  forward_client_headers_to_llm_api: true
```

`forward_client_headers_to_llm_api: true` is **required** — without it LiteLLM strips `x-claude-code-session-id` (and every other `x-*` header) on the way to the upstream, and the plugin sees nothing.

Then point Claude Code at LiteLLM and launch it:

```bash
export ANTHROPIC_BASE_URL=http://<litellm-host>:4000
export ANTHROPIC_AUTH_TOKEN=<litellm-master-key>
export ANTHROPIC_MODEL=<client-facing-model-name>
claude
```

`<client-facing-model-name>` must match the `model_name` declared in the LiteLLM `model_list` above. Claude Code emits `x-claude-code-session-id` automatically on every outbound request — no further client config needed.

#### OpenCode — **No LiteLLM required**

OpenCode uses Vercel's AI SDK with `@ai-sdk/openai-compatible` and speaks OpenAI chat-completions natively, so it talks to the llm-d gateway directly.

```json
// ~/.config/opencode/opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llmd-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llmd-local",
      "options": {
        "baseURL": "http://<llmd-gateway>/v1",
        "apiKey": "dummy"
      },
      "models": {
        "<upstream-model-name>": { "name": "<display-name>" }
      }
    }
  }
}
```

OpenCode emits `x-session-affinity` automatically on every outbound request.

#### Codex — **No LiteLLM required**

Codex emits a session header automatically on every outbound request. Current builds use the hyphenated `session-id` (no `x-` prefix); older builds use the underscored `session_id` form, which the plugin still recognizes as a fallback.

## Limitations

- **Default-queue fall-through is silent.** Requests from agents that don't match any of the configured headers land in the default fairness queue without any indication. This is by design (the plugin is non-fatal), but operators should not assume the absence of errors means every client is being identified.
- **Codex `previous_response_id` is not used.** It references the prior turn's response, not the chain root, so keying on it would shard one conversation across many queues. Correctly folding it back to the root requires a `ResponseBody` hook recording `response.id → root` mappings, which this plugin does not implement.

## Related Documentation
- Claude Code session header (official): <https://code.claude.com/docs/en/llm-gateway> — the `X-Claude-Code-Session-Id` row in "Request headers Claude Code includes."
- OpenCode session header (Cloudflare announcement, documents the `x-session-affinity` contract): <https://blog.cloudflare.com/workers-ai-large-models/>
- Codex session header (Codex CLI source — `build_session_headers` inserts `session_id` as an HTTP header on every outbound request; OpenAI does not document this in the public docs): <https://github.com/openai/codex/blob/d2e18246c96e8b440f9d97135356d37f3f3b4d63/codex-rs/codex-api/src/requests/headers.rs>