---
type: Open Question
title: Identifier hashing vs cleartext in DumpState debug endpoints
description: Whether stateful plugin DumpState endpoints hash tenant and program identifiers by default or emit cleartext values behind network access controls.
tags: ["security-and-trust-boundaries", "open-question", "unr-05"]
status: draft
sources:
  - { id: "pr-2008-rev-4759516825", resource: "https://github.com/llm-d/llm-d-router/pull/2008#pullrequestreview-4759516825", author: "human:LukeAVanDrie", last_modified: "2026-07-22T23:07:12Z" }
  - { id: "pr-2008-ic-5073212799", resource: "https://github.com/llm-d/llm-d-router/pull/2008#issuecomment-5073212799", author: "human:liu-cong", last_modified: "2026-07-24T18:33:04Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether `DumpState` output hashes tenant and program identifiers by default[^pr-2008-rev-4759516825] or emits them in cleartext because hashed identifiers carry little debugging value.[^pr-2008-ic-5073212799]

# Context

The `/debug/plugins/state` endpoint exposes bounded snapshots of stateful plugin registries for live operational diagnosis, trading tenant identifier privacy against operator debuggability. (Conflict audit ID: `UNR-05`.)

Attached review rule: [Enforce TLS minimums, namespace-scoped watches, and payload redaction](../rules/security_and_trust_boundaries.md#enforce-tls-minimums-namespace-scoped-watches-and-payload-redaction).

# Positions

- **Position A**: Hash program and tenant identifiers with SHA-256 by default in `DumpState` output so debug captures do not leak raw caller identifiers.[^pr-2008-rev-4759516825]
- **Position B**: Emit cleartext identifiers behind localhost or RBAC listener controls because irreversible hashes prevent operators from correlating debug state with specific tenants.[^pr-2008-ic-5073212799]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-2008-rev-4759516825]: [#2008](https://github.com/llm-d/llm-d-router/pull/2008#pullrequestreview-4759516825) (human:LukeAVanDrie)
[^pr-2008-ic-5073212799]: [#2008](https://github.com/llm-d/llm-d-router/pull/2008#issuecomment-5073212799) (human:liu-cong)
