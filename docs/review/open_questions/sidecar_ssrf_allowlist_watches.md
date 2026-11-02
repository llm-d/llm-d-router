---
type: Open Question
title: Disaggregation sidecar Kubernetes watches for SSRF allowlists
description: "Whether every decode pod disaggregation sidecar should watch InferencePool and Pod resources directly or avoid workload-pod API-server watches."
tags: ["project-direction", "open-question", "mvm-08"]
status: draft
sources:
  - { id: "pr-1145-ic-4479522771", resource: "https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771", author: "human:elevran", last_modified: "2026-05-18T16:05:58Z" }
  - { id: "pr-1145-ic-4489027953", resource: "https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953", author: "human:ahg-g", last_modified: "2026-05-19T14:55:39Z" }
---

# Question

Whether the disaggregation sidecar should watch `InferencePool` and `Pod` objects from every decode pod to build its SSRF allowlist[^pr-1145-ic-4479522771] or avoid API-server watches from workload pods.[^pr-1145-ic-4489027953]

# Context

The disaggregation sidecar validates prefill target addresses against known pool endpoints when `--enable-ssrf-protection` is enabled, creating a tension between SSRF defense and Kubernetes API-server watch fan-out. (Conflict audit ID: `MVM-08`.)

Attached review rule: [Cluster orchestration in pkg/coordinator and pod proxy in pkg/sidecar](../rules/settled_architectural_boundaries.md#cluster-orchestration-in-pkgcoordinator-and-pod-proxy-in-pkgsidecar).

# Positions

- **Position A**: Maintain a namespace-scoped `InferencePool` and `Pod` informer in the sidecar so decode pods reject forged prefill URLs even if upstream header scrubbing is bypassed.[^pr-1145-ic-4479522771]
- **Position B**: Avoid Kubernetes API-server watches from every model-serving workload pod and rely on ingress header scrubbing and network policy at the gateway boundary.[^pr-1145-ic-4489027953]

# Resolution criteria

Record a maintainer decision on the linked tracking issue, move this file from `open_questions/` to `decisions/`, update `type` to `Decision` and `status` to `stable`, add `# Outcome` and `verified`, append an entry to `log.md`, and run `python3 hack/render_playbook.py`.

[^pr-1145-ic-4479522771]: [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771) (human:elevran)
[^pr-1145-ic-4489027953]: [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953) (human:ahg-g)
