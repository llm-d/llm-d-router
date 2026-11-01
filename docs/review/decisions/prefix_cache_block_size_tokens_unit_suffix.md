---
type: Decision
title: "Prefix-cache block size parameter includes Tokens unit suffix"
description: "Prefix-cache scorer configuration names the block-size field blockSizeTokens rather than unitless blockSize."
tags: ["maintainer-vs-maintainer", "mvm-01"]
status: stable
sources:
  - { id: "issue-747-ic-4326491096", resource: "https://github.com/llm-d/llm-d-router/issues/747#issuecomment-4326491096", author: "human:roytman", last_modified: "2026-04-27T11:13:27Z" }
  - { id: "issue-747-ic-4344420419", resource: "https://github.com/llm-d/llm-d-router/issues/747#issuecomment-4344420419", author: "human:ahg-g", last_modified: "2026-04-29T13:58:46Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Context

Maintainers debated reverting `blockSizeTokens` to `blockSize` to match internal variable names versus retaining an explicit unit suffix on user-facing configuration. (Conflict audit ID: `MVM-01`.)

# Positions considered

- **Position A (human:roytman)**: Proposed in #747.[^issue-747-ic-4326491096]
- **Position B (human:ahg-g)**: Counter-position evaluated in #747.[^issue-747-ic-4344420419]

# Outcome

Merged as `blockSizeTokens` across prefix-cache scorers so operators see tokens rather than ambiguous byte or block counts.

Related rule: [Name plugin packages with concise domain nouns and qualify config units](../rules/plugin_framework_contracts.md#name-plugin-packages-with-concise-domain-nouns-and-qualify-config-units).

[^issue-747-ic-4326491096]: [#747](https://github.com/llm-d/llm-d-router/issues/747#issuecomment-4326491096) (human:roytman)
[^issue-747-ic-4344420419]: [#747](https://github.com/llm-d/llm-d-router/issues/747#issuecomment-4344420419) (human:ahg-g)
