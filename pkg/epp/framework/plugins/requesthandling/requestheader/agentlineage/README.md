# Agent Lineage

**Type:** `agent-lineage`
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `plugin.ProducerPlugin`

Derives the program root of a request — the session ID of the root request of the agent tree the request belongs to — and stores it as a request attribute (`"derived-root"`) for the Director's fairness-key composition. Where `agent-identity` resolves one session per request, `agent-lineage` collapses a whole sub-agent fork into one identity, so an agent tree that spans many session IDs shares one flow-control queue.

## What It Does

The plugin runs after request assembly and before admission control. It resolves the request's own session ID and its parent ID (one hop up) from lineage headers, records `child -> parent` in a per-replica, tenant-scoped parent map, and publishes the root reached by walking that map. The Director reads the `"derived-root"` attribute when deriving `FairnessID` and prefers it over the `"agent-identity"` attribute.

## How It Works

1. Resolve the session ID:
   1. Parse the `x-codex-turn-metadata` JSON header and take its `session_id` field. Codex shares the root session across every thread of a tree, but the flat `session-id` header carries a derived cache-affinity key on root-thread requests, so the turn metadata is the authoritative source. Unparsable, oversized (over 8 KiB), or session-less values fall through.
   2. Else walk the session header list and take the first non-empty match. Operator-supplied entries from `additionalSessionHeaders` come first, followed by the built-in defaults in this order:
      1. `x-opencode-session-id` (opencode own session ID)
      2. `x-claude-code-session-id` (Claude Code)
      3. `session-id` (Codex)
      4. `session_id` (Codex, legacy underscored fallback)
      5. `x-session-affinity` (opencode cache affinity — the parent one hop up, not the session's own ID; a fallback for traffic without the own-ID header)
2. Resolve the parent ID from the parent header list: `x-parent-session-id` (shared convention) then `x-opencode-parent-session-id` (opencode's vendor form of the same link; opencode sends both). Codex's `x-codex-parent-thread-id` is deliberately not read: it names a thread ID while the parent map is keyed by session IDs, and Codex needs no map because its turn metadata carries the root on every request.
3. Derive the root:
   - Parent present: record the link in the parent map, then walk from the named parent until an ancestor has no live link. A parent absent from the map (a stray child after a replica affinity miss) is itself the root.
   - No parent: walk from the session; a session with no recorded link is its own root.
   - Neither: publish nothing.
4. Publish the root via `request.PutAttribute(DerivedRootKey, root)`.

The parent map is scoped by the fairness-ID header value (the tenant, `""` when absent), so links never cross tenants. Reading a link refreshes it, so an in-use program never expires; a link unused for `parentEntryTtlSeconds` is dropped. At `maxParentEntries` the least-recently-used link is evicted. A lineage cycle ends the walk at the current node with a stable result.

After the plugin returns, the Director derives `FairnessID` (`requestHandler.composeFairnessIDs` opt-in):

| Arrival | Flow key | Attribution source |
|---|---|---|
| Fairness header + derived root | `<tenant>/<root>` | `header` |
| Fairness header, no root or session | `<tenant>` | `header` |
| No fairness header + derived root | `default-tenant/<root>` when composing, `<root>` otherwise | `derived_root` |
| No fairness header, only agent identity | `<session>` | `agent_identity` |
| Neither | `default-flow` | `default` |

The tenant part is percent-encoded (`%` becomes `%25`, `/` becomes `%2F`) so the first segment is always the whole tenant; root and session IDs may contain the separator unencoded.

## Inputs Consumed

- `scheduling.InferenceRequest.Headers` — read-only lookup of the headers above (built-in defaults plus any from `additionalSessionHeaders` / `additionalParentHeaders`). Keys are expected lowercase (Envoy normalizes inbound headers).
- `scheduling.InferenceRequest.FairnessID` — read-only: the fairness-ID header value at classification time, used as the parent map's tenant scope. The plugin never rewrites it.

## Outputs Produced

- `scheduling.InferenceRequest` attribute `"derived-root"` (`string`) — set when the request carries a session or parent header.

The plugin does not read or write the `"agent-identity"` attribute: for opencode that attribute carries the parent one hop up (the cache-affinity value), which is not the session's own ID and would corrupt the parent map's child keys.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: agent-lineage` entry to enable; the runner discovers it as a `RequestHeaderProcessor` and wires it in.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `additionalSessionHeaders` | `[]string` | No | `[]` | Extra header names to check before the built-in session defaults. The first non-empty match wins. |
| `additionalParentHeaders` | `[]string` | No | `[]` | Extra header names to check before the built-in parent defaults. |
| `maxParentEntries` | `int` | No | `4096` | Bound on the per-replica parent map. The least-recently-used link is evicted at the bound. Interim value: the per-tenant program cap replaces it once it lands. |
| `parentEntryTtlSeconds` | `float` | No | `1800` | How long a parent link survives unused. Interim value: the program-end window replaces it once sized from trace idle-gap distributions. |

### Example

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: agent-lineage
    parameters:
      additionalParentHeaders:
        - x-my-harness-parent
      maxParentEntries: 8192
      parentEntryTtlSeconds: 3600
requestHandler:
  composeFairnessIDs: true
```

## Harness Wire Shapes

The header names come from the harness sources:

- **opencode** sends `x-opencode-session-id` (own session), `x-opencode-parent-session-id` and `x-parent-session-id` (parent one hop up, sent together), and `x-session-affinity` (the parent one hop up, or the fork source's session, or the own ID for roots).
- **Codex** sends `x-codex-turn-metadata` (JSON with the tree's root `session_id`), `session-id` (the root session for sub-agent threads, a derived cache key for root threads), `thread-id`, `x-codex-parent-thread-id` (thread-level parent), and `x-openai-subagent` (workload kind).
- **Claude Code** sends one session ID (`x-claude-code-session-id`) shared across the whole tree.

Client setup for each harness is the same as for `agent-identity`; see that plugin's README.

## Limitations

- **Per-replica map.** A program split across EPP replicas derives independently on each: a stray child roots at the parent ID it names. There is no cross-replica parent store by design (it would put a network round trip on the admission path).
- **Interim bounds.** `maxParentEntries` and `parentEntryTtlSeconds` are per-replica stand-ins for the per-tenant program cap and program-end window from the Flow Control: Workload design.
- **Workload kind is not read.** `x-openai-subagent` and similar kind headers are left to the eviction-class step of the design.

## Related Documentation

- Tracking issue: [llm-d/llm-d-router#3934](https://github.com/llm-d/llm-d-router/issues/3934) — workload classification: request lineage and (tenant, root) fairness key
- Umbrella: [llm-d/llm-d-router#2794](https://github.com/llm-d/llm-d-router/issues/2794) — make fairness and ordering agentic-aware
- `agent-identity` plugin README — session resolution and per-harness client setup
- opencode lineage headers (opencode source): `packages/core/src/session/model-request.ts` and `packages/core/src/session/affinity.ts`
- Codex session headers (Codex CLI source): `codex-rs/codex-api/src/requests/headers.rs` and `codex-rs/core/src/responses_metadata.rs`
