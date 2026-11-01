---
type: Open Question
title: Plugin panic isolation at the invocation boundary
description: "Whether the EPP plugin framework wraps extension-point calls in recover guards per request or lets panicking plugins terminate the process."
tags: ["error-handling-and-boundaries", "open-question", "unr-06"]
status: draft
sources:
  - { id: "issue-2105", resource: "https://github.com/llm-d/llm-d-router/issues/2105", author: "human:LukeAVanDrie", last_modified: "2026-07-21T00:25:41Z" }
  - { id: "issue-2098-ic-5032265086", resource: "https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5032265086", author: "human:shmuelk", last_modified: "2026-07-21T09:21:03Z" }
  - { id: "issue-2098-ic-5028980468", resource: "https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5028980468", author: "human:LukeAVanDrie", last_modified: "2026-07-21T00:57:58Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether the plugin framework recovers plugin panics at the invocation boundary and reports them per request[^issue-2105] or lets a panicking plugin stop the process.[^issue-2098-ic-5032265086]

# Context

No extension point in the EPP plugin framework wraps plugin execution in `recover()`. A nil dereference or slice bounds fault in any filter, scorer, data producer, or flow-control policy terminates the EPP process and resets all in-flight requests. (Conflict audit ID: `UNR-06`.)

Attached review rule: [Propagate errors explicitly without fabricated fallbacks](../rules/error_handling_and_boundaries.md#propagate-errors-explicitly-without-fabricated-fallbacks).

# Positions

- **Position A**: Wrap plugin invocations (or shared background processor goroutines) in `recover()`, increment a `plugin_panics_total` counter labeled by extension point and plugin, and fail only the affected request.[^issue-2105]
- **Position B**: Let panics crash the process so Kubernetes restarts the replica into a clean initial state rather than continuing execution after a panic unwinds locks mid-mutation and leaves shared accounting corrupt, while converting recoverable domain faults into explicit `error` returns.[^issue-2098-ic-5032265086][^issue-2098-ic-5028980468]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^issue-2105]: [#2105](https://github.com/llm-d/llm-d-router/issues/2105) (human:LukeAVanDrie)
[^issue-2098-ic-5032265086]: [#2098](https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5032265086) (human:shmuelk)
[^issue-2098-ic-5028980468]: [#2098](https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5028980468) (human:LukeAVanDrie)
