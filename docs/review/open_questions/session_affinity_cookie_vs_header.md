---
type: Open Question
title: "Session affinity transport via Set-Cookie vs request header"
description: "Whether session affinity state is carried by standard HTTP Set-Cookie headers or by explicit session-token headers."
tags: ["security-and-trust-boundaries", "open-question", "unr-01"]
status: draft
sources:
  - id: "issue-28"
    resource: "https://github.com/llm-d/llm-d-router/issues/28"
    author: "human:elevran"
    last_modified: "2025-05-07T17:56:35Z"
  - id: "pr-600-r3156539950"
    resource: "https://github.com/llm-d/llm-d-router/pull/600#discussion_r3156539950"
    author: "human:zetxqx"
    last_modified: "2026-04-28T19:04:29Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether session affinity should be carried by a standard `Set-Cookie` response[^issue-28] or by a request header such as `x-session-token`.[^pr-600-r3156539950]

# Context

Session-affinity plugins route multi-turn requests back to warm KV-cache pods, requiring a wire mechanism that works across SDK clients while letting stateless callers opt out. (Conflict audit ID: `UNR-01`.)

Attached review rule: [Namespace new router headers under `x-llm-d-` and scrub internal headers at ingress and egress](../rules/security_and_trust_boundaries.md#namespace-new-router-headers-under-x-llm-d--and-scrub-internal-headers-at-ingress-and-egress).

# Positions

- **Position A**: Use standard HTTP `Set-Cookie` and `Cookie` headers so HTTP clients and browsers relay affinity tokens without custom header logic.[^issue-28]
- **Position B**: Carry affinity in an explicit header (`x-session-token` or an `x-llm-d-` successor) so stateless callers are not pinned automatically when the plugin is enabled.[^pr-600-r3156539950]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^issue-28]: [#28](https://github.com/llm-d/llm-d-router/issues/28) (human:elevran)
[^pr-600-r3156539950]: [#600](https://github.com/llm-d/llm-d-router/pull/600#discussion_r3156539950) (human:zetxqx)
