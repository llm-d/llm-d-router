---
type: Open Question
title: "Multi-tenant cost attribution on Prometheus labels vs logs and traces"
description: "Whether tenant, workload, and user cost-attribution identifiers belong on opt-in Prometheus metric labels or exclusively in structured logs and traces."
tags: [observability, "open-question", "unr-08"]
status: draft
sources:
  - { id: "pr-2834", resource: "https://github.com/llm-d/llm-d-router/pull/2834", author: "human:simanadler", last_modified: "2026-09-14T07:55:53Z" }
  - { id: "pr-2834-r4032724626", resource: "https://github.com/llm-d/llm-d-router/pull/2834#discussion_r4032724626", author: "human:gyliu513", last_modified: "2026-09-17T03:07:51Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether per-tenant, per-workload, and per-user cost attribution belongs on Prometheus labels that operators opt into[^pr-2834] or only in structured logs and traces because header-derived values have no closed set.[^pr-2834-r4032724626]

# Context

Operators want token usage and latency attributed to business tenants, workloads, and users, whereas Prometheus time-series cardinality scales multiplicatively with unbounded request-header values. (Conflict audit ID: `UNR-08`.)

Attached review rule: [Register metrics under canonical component subsystems with bounded label cardinality](../rules/observability.md#register-metrics-under-canonical-component-subsystems-with-bounded-label-cardinality).

# Positions

- **Position A**: Allow operators to opt into `tenant_id`, `workload_id`, and `user_id` Prometheus labels in deployments where upstream gateways enforce closed tenant sets.[^pr-2834]
- **Position B**: Keep unbounded caller-supplied identity attributes out of Prometheus metric labels and emit per-tenant attribution through structured logs or OpenTelemetry span attributes.[^pr-2834-r4032724626]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-2834]: [#2834](https://github.com/llm-d/llm-d-router/pull/2834) (human:simanadler)
[^pr-2834-r4032724626]: [#2834](https://github.com/llm-d/llm-d-router/pull/2834#discussion_r4032724626) (human:gyliu513)
