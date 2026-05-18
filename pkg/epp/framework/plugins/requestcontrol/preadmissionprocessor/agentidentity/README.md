# Agent Identity

**Type:** `agent-identity`
**Interfaces:** `requestcontrol.PreAdmissionProcessor`

Resolves a per-session identity from agent-specific HTTP headers and writes it into `InferenceRequest.FairnessID`, so every turn of an agent session lands in the same flow-control fairness queue.

## What It Does

The plugin runs after request assembly and before admission control. If the request does not already carry an explicit fairness ID (`x-gateway-inference-flow-fairness-id`), it inspects a fixed set of agent session headers and copies the first non-empty value into `FairnessID`. The flow-control layer keys its queues on `FlowKey{ID: FairnessID, Priority}`, so this turns "all turns from one agent session" into "all turns share one queue."

Without it, every request from a given agent session falls into the default fairness queue alongside unrelated traffic, and per-session fairness, prefix-cache affinity, and per-tenant rate limiting all collapse to per-request granularity.

## How It Works

1. If `request.FairnessID` is already set to something other than `metadata.DefaultFairnessID`, return immediately — an explicit upstream `x-gateway-inference-flow-fairness-id` always wins over a derived one.
2. Otherwise, walk the priority list of agent session headers and copy the first non-empty match into `request.FairnessID`:
   1. `x-claude-code-session-id` (Claude Code)
   2. `x-session-affinity` (OpenCode)
   3. `session_id` (Codex)
3. If nothing matches, leave `FairnessID` as the default and return — the request is still admitted, just into the shared default queue.

The plugin is stateless and safe under concurrent use.

## Inputs Consumed

- `scheduling.InferenceRequest.Headers` — read-only lookup of the three session headers above. Keys are expected lowercase (Envoy normalizes inbound headers).
- `scheduling.InferenceRequest.FairnessID` — read to detect an upstream override; written when an agent header matches.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: agent-identity` entry to enable; the runner discovers it as a `PreAdmissionProcessor` and wires it in.

### Parameters

The plugin takes no parameters. The factory accepts and ignores its `parameters` argument.

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| _(none)_ | — | — | — | — |

### Examples

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: agent-identity
```

### Per-agent client setup

The plugin only reads headers — getting them onto the wire is the agent's job. Each supported agent has different requirements.

#### Claude Code — **LiteLLM is required**

Claude Code speaks Anthropic's Messages API. llm-d's gateway exposes the OpenAI chat-completions wire format, so a translator is required in the path. LiteLLM works:

```yaml
# ~/.litellm/config.yaml
model_list:
  - model_name: claude-sonnet-4-5
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
      api_base: http://<llmd-gateway>:8080/v1

litellm_settings:
  forward_client_headers_to_llm_api: true
```

`forward_client_headers_to_llm_api: true` is **required** — without it LiteLLM strips `x-claude-code-session-id` (and every other `x-*` header) on the way to the upstream, and the plugin sees nothing.

Then point Claude Code at LiteLLM:

```bash
export ANTHROPIC_BASE_URL=http://<litellm-host>:4000
export ANTHROPIC_AUTH_TOKEN=<litellm-master-key>
```

Claude Code emits `x-claude-code-session-id` automatically on every outbound request — no further client config needed.

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
        "baseURL": "http://<llmd-gateway>:8080/v1",
        "apiKey": "dummy"
      },
      "models": {
        "meta-llama/Llama-3.1-8B-Instruct": { "name": "Llama 3.1 8B Instruct" }
      }
    }
  }
}
```

OpenCode emits `x-session-affinity` automatically on every outbound request.

#### Codex — **No LiteLLM required**

Codex emits `session_id` (literal underscore form, no `x-` prefix) automatically on every outbound request. Note that Envoy rejects underscore headers by default — the gateway must be configured with `headers_with_underscores_action: ALLOW` for `session_id` to reach the EPP.

## Limitations

- **Default-queue fall-through is silent.** Requests from agents that don't match any of the three headers land in the default fairness queue without any indication. This is by design (the plugin is non-fatal), but operators should not assume the absence of errors means every client is being identified.
- **Codex `previous_response_id` is not used.** It references the prior turn's response, not the chain root, so keying on it would shard one conversation across many queues. Correctly folding it back to the root requires a `ResponseBody` hook recording `response.id → root` mappings, which this plugin does not implement.
- **One header per agent.** Supports exactly the three agents above. Adding a new agent means adding to `priorityHeaders`.
- **Last-write-wins across multiple plugins.** If multiple `PreAdmissionProcessor` plugins are registered and write `FairnessID`, the order in which they run determines the result. The director runs them in registration order.

## Related Documentation
- Claude Code session header (official): <https://code.claude.com/docs/en/llm-gateway> — the `X-Claude-Code-Session-Id` row in "Request headers Claude Code includes."
- OpenCode session header (Cloudflare announcement, documents the `x-session-affinity` contract): <https://blog.cloudflare.com/workers-ai-large-models/>
- Codex session header (Codex CLI source — `build_session_headers` inserts `session_id` as an HTTP header on every outbound request; OpenAI does not document this in the public docs): <https://github.com/openai/codex/blob/d2e18246c96e8b440f9d97135356d37f3f3b4d63/codex-rs/codex-api/src/requests/headers.rs>