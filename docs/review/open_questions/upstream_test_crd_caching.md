---
type: Open Question
title: Remote fetch vs repository cache for upstream test CRDs
description: Whether unmodified upstream Gateway API and GIE CRDs used by verification and test scripts are fetched remotely on every run or cached in the repository.
tags: ["project-direction", "open-question", "unr-03"]
status: draft
sources:
  - { id: "pr-1482-ic-4633716490", resource: "https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490", author: "human:liu-cong", last_modified: "2026-06-05T16:59:52Z" }
  - { id: "pr-1482-ic-4661288603", resource: "https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4661288603", author: "human:elevran", last_modified: "2026-06-09T15:23:11Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether unmodified upstream CRDs used by tests are fetched from GitHub on every run[^pr-1482-ic-4633716490] or cached in the repository to avoid rate-limited fetch failures.[^pr-1482-ic-4661288603]

# Context

Verification scripts (`hack/verify-helm.sh`, `hack/verify-manifests.sh`) fetch upstream Gateway API and GIE CRDs from GitHub tags, exposing CI runs to transient HTTP rate limits. (Conflict audit ID: `UNR-03`.)

Attached review rule: [Consume versioned upstream CRDs and protobufs without vendoring](../rules/project_direction.md#consume-versioned-upstream-crds-and-protobufs-without-vendoring).

# Positions

- **Position A**: Fetch unmodified upstream CRDs from version-pinned release URLs rather than checking third-party YAML copies into the repository.[^pr-1482-ic-4633716490]
- **Position B**: Cache pinned upstream test CRDs locally or via CI artifact caching so GitHub rate limits do not cause spurious presubmit failures.[^pr-1482-ic-4661288603]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-1482-ic-4633716490]: [#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490) (human:liu-cong)
[^pr-1482-ic-4661288603]: [#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4661288603) (human:elevran)
