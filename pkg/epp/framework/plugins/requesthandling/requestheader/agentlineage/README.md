# Agent Lineage

**Type:** `agent-lineage`
**Interfaces:** `requesthandling.RequestHeaderProcessor`, `plugin.ProducerPlugin`

Derives the program root of a request — the session ID of the root request of the agent tree the request belongs to — and the request's resolved priority band, and stores both as request attributes (`"derived-root"`, `"resolved-band"`) for the Director. Where `agent-identity` resolves one session per request, `agent-lineage` collapses a whole sub-agent fork into one identity, so an agent tree that spans many session IDs shares one flow-control queue, and a forked child runs at its parent's band instead of whatever its own headers select.

## What It Does

The plugin runs after request assembly and before admission control. It resolves the request's own session ID and its parent ID (one hop up) from lineage headers, records a per-session node — the parent link and the resolved band — plus a live-child count per parent in a per-replica, tenant-scoped map, and publishes the root reached by walking that map together with the resolved band. The Director reads `"derived-root"` when deriving `FairnessID` (preferred over `"agent-identity"`) and applies `"resolved-band"` as the request's priority before admission.

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
3. Resolve the band:
   - A parented child whose parent has a live node inherits the parent's recorded band under a stamped tenant (a non-empty fairness-ID header).
   - With an explicit `x-llm-d-inference-objective` header the child resolves to `min(explicit, parent band)` — never above its parent, lower only by explicit opt-in.
   - An unstamped child resolves to `min(seeded, parent band)`: it can be lowered but not raised, because the untrusted-priority ceiling is not enforced yet. The clamp is safe in every population and stays unconditional.
   - Every other request keeps the seeded band. A declared workload kind is not a band input.
   - A stray child whose parent has no live node falls back to its own seeded band.
   - An objective header that names no known `InferenceObjective` still counts as explicit and resolves at the default priority, so the child runs at `min(default, parent band)`.
4. Derive the root:
   - With a session: walk from the session through its recorded links. While a link is live it is authoritative — a session's parent is fixed when the child is spawned, so a declaration naming a different parent is ignored until the entry expires.
   - Without a session but with a parent: walk from the named parent; a parent absent from the map (a stray child after a replica affinity miss) is itself the root.
   - Neither: publish nothing.
5. Publish the root and the resolved band via `request.PutAttribute`.

The node map is scoped by the request's fairness ID (the tenant, `""` when absent), so records never cross tenants. A request without a fairness-ID header takes the tenant a RequestHeader processor produced through `request-tenant`, if any. Reading a record refreshes it — including the terminal root of a walk, so an idle root with active children stays alive — so an in-use program never expires; a record unused for `parentEntryTtlSeconds` is dropped. At `maxParentEntries` the insert scan deletes every expired node it passes. If the map is still full, it evicts the oldest standalone node (no parent and no live child), then the oldest leaf link (a linked node with no live child), and finally the oldest node with a live child, whether a root or an interior node. The arriving request's own parent is never evicted. A parentless arrival is dropped only when every slot holds a live linked node or a node with a live child. A child counts toward its parent only while its node exists; the count drops when the child expires or is evicted. A key already in the map is refreshed rather than inserted, since harnesses resend both headers every turn. A lineage cycle ends the walk at the current node with a stable result, and the walk follows at most 64 links, so a client chaining links up to the map bound cannot extend the walk's cost.

After the plugin returns, the Director applies the resolved band as the request's priority once, before admission: bands resolve once and queued requests never migrate across them.

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
- `scheduling.InferenceRequest.FairnessID` — read-only: the fairness-ID header value, or the `request-tenant` a RequestHeader processor produced for a request without one, used as the node map's tenant scope. The plugin never rewrites it.
- `scheduling.InferenceRequest.Objectives.Priority` — read-only: the seeded objective priority the Director resolved before request header processors run, or the `request-priority` a RequestHeader processor that ran before this plugin produced.
- `scheduling.InferenceRequest` attribute `"request-tenant"` (`string`), optional — declared so the plugin runs after any processor that produces it; the Director has already copied it into `FairnessID`.
- the `x-llm-d-inference-objective` header (or its deprecated alias `x-gateway-inference-objective`) — read-only: its presence marks an explicit objective, which selects `min(explicit, parent band)` for a parented child.

## Outputs Produced

- `scheduling.InferenceRequest` attribute `"derived-root"` (`string`) — set when the request carries a session or parent header.
- `scheduling.InferenceRequest` attribute `"resolved-band"` (`int`) — set when lineage resolves a band different from the seeded priority; the Director applies it as the request's priority before admission.

The plugin does not read or write the `"agent-identity"` attribute: for opencode that attribute carries the parent one hop up (the cache-affinity value), which is not the session's own ID and would corrupt the parent map's child keys.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: agent-lineage` entry to enable; the runner discovers it as a `RequestHeaderProcessor` and wires it in.

This plugin does not replace `agent-identity`: consumers of the `agent-identity` attribute (`session-state-producer`, `thunder-agent`, session-affinity attribute sources) still require it, so run both plugins together when those are in use.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `additionalSessionHeaders` | `[]string` | No | `[]` | Extra header names to check before the built-in session defaults. The first non-empty match wins. The Codex turn-metadata header is parsed before every session header, including these. |
| `additionalParentHeaders` | `[]string` | No | `[]` | Extra header names to check before the built-in parent defaults. |
| `maxParentEntries` | `int` | No | `4096` | Bound on the per-replica node map. At the bound an expired node is reclaimed first, then the oldest standalone node is evicted. Interim value: the per-tenant program cap replaces it once it lands. |
| `parentEntryTtlSeconds` | `float` | No | `1800` | How long a node survives unused. Interim value: the program-end window replaces it once sized from trace idle-gap distributions. |

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
- **Bound sizing.** Size `maxParentEntries` to cover the sessions seen in the longest gap between a root's request and its first fork. Once the map is full, every new session scans it under the plugin mutex, and a root evicted before its first child arrives leaves that child on its seeded band.
- **Raise trusts the fairness-ID header.** A non-empty `x-llm-d-inference-fairness-id` lets a child inherit a higher parent band. A deployment that limits the raise to trusted callers must set or strip that header at the gateway until Seam A provides a trusted tenant identity.
- **Bound pressure under flooding.** A caller flooding distinct linked sessions can evict live links once no standalone node remains to displace. The map size bounds the damage and affected programs degrade to their seeded bands (queue keys survive through the named-parent fallback). The per-tenant program cap and the SessionStateRegistry fold bound this per tenant.
- **Workload kind is not read.** `x-openai-subagent` and similar kind headers are left to the eviction-class step of the design.

## Related Documentation

- Tracking issue: [llm-d/llm-d-router#3934](https://github.com/llm-d/llm-d-router/issues/3934) — workload classification: request lineage and (tenant, root) fairness key
- Umbrella: [llm-d/llm-d-router#2794](https://github.com/llm-d/llm-d-router/issues/2794) — make fairness and ordering agentic-aware
- `agent-identity` plugin README — session resolution and per-harness client setup
- opencode lineage headers (opencode source): `packages/core/src/session/model-request.ts` and `packages/core/src/session/affinity.ts`
- Codex session headers (Codex CLI source): `codex-rs/codex-api/src/requests/headers.rs` and `codex-rs/core/src/responses_metadata.rs`
