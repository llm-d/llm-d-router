---
type: Review Rule Set
title: Documentation and comments
description: "Colocated plugin READMEs, review-round documentation synchronization, and non-obvious invariant comments on locks, aliasing, and engine quirks."
tags: ["documentation-and-comments", "failure-class", "mvm-18", "nvd-03", "mvm-17"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "pr-820-ic-4398304029", resource: "https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4398304029", author: "human:ahg-g", last_modified: "2026-05-07T15:04:42Z" }
  - { id: "pr-888-r3143455213", resource: "https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143455213", author: "human:elevran", last_modified: "2026-04-26T12:13:42Z" }
  - { id: "pr-959-r3177858398", resource: "https://github.com/llm-d/llm-d-router/pull/959#discussion_r3177858398", author: "human:vMaroon", last_modified: "2026-05-03T08:40:19Z" }
  - { id: "pr-891-r3236365197", resource: "https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236365197", author: "human:LukeAVanDrie", last_modified: "2026-05-13T17:50:22Z" }
  - { id: "pr-933-ic-4864629122", resource: "https://github.com/llm-d/llm-d-router/pull/933#issuecomment-4864629122", author: "human:roytman", last_modified: "2026-07-02T10:17:17Z" }
  - { id: "pr-1112-r3242601033", resource: "https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242601033", author: "human:elevran", last_modified: "2026-05-14T15:57:20Z" }
  - { id: "pr-1115-r3306897196", resource: "https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3306897196", author: "human:hexfusion", last_modified: "2026-05-26T21:16:06Z" }
  - { id: "pr-1135-r3255416032", resource: "https://github.com/llm-d/llm-d-router/pull/1135#discussion_r3255416032", author: "human:ahg-g", last_modified: "2026-05-17T20:15:06Z" }
  - { id: "pr-1207-r3267187365", resource: "https://github.com/llm-d/llm-d-router/pull/1207#discussion_r3267187365", author: "human:ahg-g", last_modified: "2026-05-19T14:41:46Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Documentation and comments

Colocated plugin READMEs, review-round documentation synchronization, and non-obvious invariant comments on locks, aliasing, and engine quirks.

## Colocate plugin READMEs and link to canonical upstream guides

### Rule

Place each plugin's parameter schema, code-verified defaults, and minimal reference YAML example in a colocated `README.md` inside its package under `pkg/epp/framework/plugins` (see [AGENTS.md](../../../AGENTS.md#code-style) for single-source prose rules). Link to upstream `llm-d/llm-d` architecture guides or foundation policies rather than duplicating them, and use file or symbol links without fragile source line numbers.

### Scope and consequence

- **Applies to**: `docs/**`, `pkg/**/README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, Go doc comments (referenced symbols: `README.md`, `pkg/epp/framework/plugins`, `llm-d/llm-d`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Duplicated prose, stale configuration examples after review rebases, or undocumented lock and aliasing invariants mislead operators and future contributors.
- **Related decisions**: [Component metrics reference in llm-d-router paired with user guides in llm-d/llm-d](../decisions/local_metrics_docs_paired_with_umbrella_user_guides.md).

### Evidence

Observed across 167 pull requests and 17 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-820-ic-4398304029][^pr-888-r3143455213][^pr-959-r3177858398]

## Synchronize docstrings, diagrams, and configuration examples across review rounds

### Rule

Re-verify affected package docstrings, architecture diagrams, and configuration examples whenever code or types change during review or rebase, and avoid hardcoded plugin or feature counts that drift as plugins are added.

### Scope and consequence

- **Applies to**: `docs/**`, `pkg/**/README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, Go doc comments; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Duplicated prose, stale configuration examples after review rebases, or undocumented lock and aliasing invariants mislead operators and future contributors.

### Evidence

Distilled from 2 review clusters (54 pull requests / 10 maintainers, 53 pull requests / 12 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-891-r3236365197][^pr-933-ic-4864629122][^pr-1112-r3242601033]

## Document lock ordering, shallow-copy safety, and engine protocol quirks in code comments

### Rule

Follow [AGENTS.md](../../../AGENTS.md#code-style) for comment style, recording lock ordering preconditions, shallow-copy aliasing constraints, upstream engine protocol quirks, and deliberate interface contract overrides directly on the affected type or method.

### Scope and consequence

- **Applies to**: `docs/**`, `pkg/**/README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, Go doc comments; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Duplicated prose, stale configuration examples after review rebases, or undocumented lock and aliasing invariants mislead operators and future contributors.
- **Related decisions**: [Per-request endpoint attribute isolation with dynamic metric read-through](../decisions/per_request_endpoint_attribute_isolation_with_dynamic_readthrough.md).

### Evidence

Observed across 53 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1115-r3306897196][^pr-1135-r3255416032][^pr-1207-r3267187365]

[^pr-820-ic-4398304029]: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4398304029) (human:ahg-g)
[^pr-888-r3143455213]: [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143455213) (human:elevran)
[^pr-959-r3177858398]: [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3177858398) (human:vMaroon)
[^pr-891-r3236365197]: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236365197) (human:LukeAVanDrie)
[^pr-933-ic-4864629122]: [#933](https://github.com/llm-d/llm-d-router/pull/933#issuecomment-4864629122) (human:roytman)
[^pr-1112-r3242601033]: [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242601033) (human:elevran)
[^pr-1115-r3306897196]: [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3306897196) (human:hexfusion)
[^pr-1135-r3255416032]: [#1135](https://github.com/llm-d/llm-d-router/pull/1135#discussion_r3255416032) (human:ahg-g)
[^pr-1207-r3267187365]: [#1207](https://github.com/llm-d/llm-d-router/pull/1207#discussion_r3267187365) (human:ahg-g)
