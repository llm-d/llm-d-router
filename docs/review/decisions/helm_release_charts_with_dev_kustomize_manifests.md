---
type: Decision
title: "Helm charts for end-user releases and deploy Kustomize manifests for local kind and E2E"
description: "End-user deployments ship through config/charts Helm charts while Kustomize overlays under deploy remain scoped to local development and E2E suites."
tags: ["maintainer-vs-maintainer", "mvm-09"]
status: stable
sources:
  - id: "issue-1177"
    resource: "https://github.com/llm-d/llm-d-router/issues/1177"
    author: "human:ahg-g"
    last_modified: "2026-05-17T21:01:28Z"
  - id: "issue-1177-ic-4478503514"
    resource: "https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514"
    author: "human:shmuelk"
    last_modified: "2026-05-18T14:08:19Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers discussed deleting `deploy/` Kustomize manifests entirely in favor of Helm guides versus retaining `deploy/` for local `kind` environments and E2E suites. (Conflict audit ID: `MVM-09`.)

# Positions considered

- **Position A (human:ahg-g)**: Proposed in #1177.[^issue-1177]
- **Position B (human:shmuelk)**: Counter-position evaluated in #1177.[^issue-1177-ic-4478503514]

# Outcome

Kept `config/charts/` as the canonical end-user packaging surface while retaining `deploy/` for local `kind` workflows and E2E tests.

Related rule: [Deploy through Helm charts and configure components via structured files](../rules/project_direction.md#deploy-through-helm-charts-and-configure-components-via-structured-files).

[^issue-1177]: [#1177](https://github.com/llm-d/llm-d-router/issues/1177) (human:ahg-g)
[^issue-1177-ic-4478503514]: [#1177](https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514) (human:shmuelk)
