---
type: Open Question
title: Design proposal storage in repository vs issues
description: "Whether accepted design proposals belong in an in-repo proposals directory or in GitHub issues and linked design documents."
tags: ["scope-and-process", "open-question", "unr-07"]
status: draft
sources:
  - { id: "pr-2376-r3832021146", resource: "https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3832021146", author: "human:ahg-g", last_modified: "2026-08-21T16:42:09Z" }
  - { id: "pr-2376-r3837894575", resource: "https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3837894575", author: "human:elevran", last_modified: "2026-08-23T06:44:56Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether accepted design proposals live in an in-repo proposals directory[^pr-2376-r3832021146] or only in issues and linked design documents.[^pr-2376-r3837894575]

# Context

Architectural changes require public discussion before pull request iteration, but maintainers have not settled where accepted long-form RFCs are archived after discussion. (Conflict audit ID: `UNR-07`.)

Attached review rule: [Settle design tradeoffs on public issues and scope auto-close keywords](../rules/scope_and_process.md#settle-design-tradeoffs-on-public-issues-and-scope-auto-close-keywords).

# Positions

- **Position A**: Create an in-repo `proposals/` directory alongside `CONTRIBUTING.md` so accepted designs are versioned with code.[^pr-2376-r3832021146]
- **Position B**: Keep design proposals in GitHub issues and linked external documents so stale proposal files in the tree are not mistaken for active code specifications.[^pr-2376-r3837894575]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-2376-r3832021146]: [#2376](https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3832021146) (human:ahg-g)
[^pr-2376-r3837894575]: [#2376](https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3837894575) (human:elevran)
