# Contributing to llm-d-router

Please access the
[llm-d organization contributing guide](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md) for overall contributing guidelines. This document covers only what is different or specific to this repository.

See [DEVELOPMENT.md](DEVELOPMENT.md) for how to build, run, and test the codebase, [docs/review_playbook.md](docs/review_playbook.md) for the compiled human-readable review guide, and [AGENTS.md](AGENTS.md) when working with coding or review agents.

## Finding an issue

- [Starter issues (`good first issue`)](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22+label%3Atriage%2Faccepted+no%3Aassignee+-linked%3Apr):
  self-contained tasks that run locally without GPUs via `make test` (inside the
  builder container) or the CPU-only KIND simulator in
  [DEVELOPMENT.md](DEVELOPMENT.md).
- [Scoped subsystem work (`help wanted`)](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3Atriage%2Faccepted+no%3Aassignee+-linked%3Apr):
  accepted issues that require familiarity with a specific subsystem.

See [docs/area_taxonomy.md](docs/area_taxonomy.md) for per-area issue queues,
Slack channels, and code owners.

## Claiming an issue

Comment `/assign` on the issue to assign it to yourself. The assignee is what
marks the issue as taken, so check for an existing assignee before you start
work. Comment `/unassign` to release an issue you no longer plan to work on.

An assigned issue is expected to get a pull request. See
[docs/issue_triage.md](docs/issue_triage.md#assigned-issues) for when an
assignment is released.
