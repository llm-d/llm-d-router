---
type: Open Question
title: "Non-Kubernetes InferenceObjective file schema"
description: Whether standalone deployments load InferenceObjective definitions from Kubernetes CRD manifests with ignored fields or from a dedicated strict file schema.
tags: ["project-direction", "open-question", "unr-12"]
status: draft
sources:
  - { id: "pr-3502", resource: "https://github.com/llm-d/llm-d-router/pull/3502", author: "human:aistackdev", last_modified: "2026-10-19T04:45:22Z" }
  - { id: "pr-3502-rev-5649007338", resource: "https://github.com/llm-d/llm-d-router/pull/3502#pullrequestreview-5649007338", author: "human:elevran", last_modified: "2026-10-26T08:45:37Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether non-Kubernetes deployments load `InferenceObjective` definitions from Kubernetes manifests with unused fields ignored[^pr-3502] or from a separate strict file schema.[^pr-3502-rev-5649007338]

# Context

Bare-metal and Slurm deployments without Kubernetes discover endpoints from static files and need a file-based path to configure per-tenant priority objectives. (Conflict audit ID: `UNR-12`.)

Attached review rule: [Model endpoints with orchestrator-neutral domain types](../rules/project_direction.md#model-endpoints-with-orchestrator-neutral-domain-types).

# Positions

- **Position A**: Parse standard Kubernetes `InferenceObjective` YAML manifests via `--objectives-file` and ignore Kubernetes-only fields such as `poolRef` so operators reuse one schema.[^pr-3502]
- **Position B**: Define a minimal orchestrator-neutral YAML schema decoded with `DisallowUnknownFields()` so silent field-ignoring does not mask typos.[^pr-3502-rev-5649007338]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-3502]: [#3502](https://github.com/llm-d/llm-d-router/pull/3502) (human:aistackdev)
[^pr-3502-rev-5649007338]: [#3502](https://github.com/llm-d/llm-d-router/pull/3502#pullrequestreview-5649007338) (human:elevran)
