---
type: Decision
title: "Helm charts for end-user releases and test/e2e/artifacts Kustomize manifests for local kind and E2E"
description: "End-user deployments ship through config/charts Helm charts while Kustomize overlays under test/e2e/artifacts remain scoped to local development and E2E suites."
tags: ["maintainer-vs-maintainer", "mvm-09"]
status: stable
sources:
  - { id: "issue-1177", resource: "https://github.com/llm-d/llm-d-router/issues/1177", author: "human:ahg-g", last_modified: "2026-05-17T21:01:28Z" }
  - { id: "issue-1177-ic-4478503514", resource: "https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514", author: "human:shmuelk", last_modified: "2026-05-18T14:08:19Z" }
---

# Context

Maintainers discussed deleting `deploy/` Kustomize manifests entirely in favor of Helm guides versus retaining `deploy/` for local `kind` environments and E2E suites. (Conflict audit ID: `MVM-09`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #1177.[^issue-1177]
- **Position B (human:shmuelk)**: Counter-position evaluated in #1177.[^issue-1177-ic-4478503514]

# Outcome

Kept `config/charts/` as the canonical end-user packaging surface while retaining the Kustomize manifests, located in `test/e2e/artifacts/`, for local `kind` workflows and E2E tests.

Related rule: [Helm charts in config/charts/ and local Kustomize in test/e2e/artifacts/](../rules/governance_and_compatibility_policy.md#helm-charts-in-configcharts-and-local-kustomize-in-teste2eartifacts).

[^issue-1177]: [#1177](https://github.com/llm-d/llm-d-router/issues/1177) (human:ahg-g)
[^issue-1177-ic-4478503514]: [#1177](https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514) (human:shmuelk)
