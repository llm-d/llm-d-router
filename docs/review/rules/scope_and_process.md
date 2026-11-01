---
type: Review Rule Set
title: Scope and process
description: "Pull request slicing, public design issues, release-note hygiene, and fork-branch review iteration."
tags: ["scope-and-process", process, "apw-18", "unr-07"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "pr-707-ic-4642288302", resource: "https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4642288302", author: "human:ahg-g", last_modified: "2026-06-07T10:59:53Z" }
  - { id: "pr-860-r3274050146", resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3274050146", author: "human:elevran", last_modified: "2026-05-20T12:53:56Z" }
  - { id: "pr-1018-r3270503284", resource: "https://github.com/llm-d/llm-d-router/pull/1018#discussion_r3270503284", author: "human:liu-cong", last_modified: "2026-05-20T00:22:32Z" }
  - { id: "pr-860-r3178005111", resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3178005111", author: "human:vMaroon", last_modified: "2026-05-03T11:00:35Z" }
  - { id: "pr-1071-r3262248799", resource: "https://github.com/llm-d/llm-d-router/pull/1071#discussion_r3262248799", author: "human:ahg-g", last_modified: "2026-05-18T21:22:38Z" }
  - { id: "pr-1115-r3374238301", resource: "https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3374238301", author: "human:sagearc", last_modified: "2026-06-08T15:01:22Z" }
  - { id: "pr-707-ic-4541930806", resource: "https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4541930806", author: "human:ahg-g", last_modified: "2026-05-26T08:07:14Z" }
  - { id: "issue-857", resource: "https://github.com/llm-d/llm-d-router/issues/857", author: "human:elevran", last_modified: "2026-04-21T09:21:57Z" }
  - { id: "issue-1090-ic-5347088232", resource: "https://github.com/llm-d/llm-d-router/issues/1090#issuecomment-5347088232", author: "human:LukeAVanDrie", last_modified: "2026-08-19T19:37:33Z" }
  - { id: "issue-2590-ic-5499142962", resource: "https://github.com/llm-d/llm-d-router/issues/2590#issuecomment-5499142962", author: "human:roytman", last_modified: "2026-09-01T19:18:50Z" }
  - { id: "pr-1206-ic-4647621711", resource: "https://github.com/llm-d/llm-d-router/pull/1206#issuecomment-4647621711", author: "human:elevran", last_modified: "2026-06-08T09:55:57Z" }
  - { id: "pr-1657-ic-4708030072", resource: "https://github.com/llm-d/llm-d-router/pull/1657#issuecomment-4708030072", author: "human:shmuelk", last_modified: "2026-06-15T12:42:14Z" }
  - { id: "pr-1939-ic-4925378102", resource: "https://github.com/llm-d/llm-d-router/pull/1939#issuecomment-4925378102", author: "human:ahg-g", last_modified: "2026-07-09T13:11:00Z" }
  - { id: "pr-1172-ic-4477576689", resource: "https://github.com/llm-d/llm-d-router/pull/1172#issuecomment-4477576689", author: "human:elevran", last_modified: "2026-05-18T12:18:52Z" }
  - { id: "pr-2228-ic-5409613155", resource: "https://github.com/llm-d/llm-d-router/pull/2228#issuecomment-5409613155", author: "human:ahg-g", last_modified: "2026-08-25T11:19:38Z" }
  - { id: "pr-2823-ic-5668357219", resource: "https://github.com/llm-d/llm-d-router/pull/2823#issuecomment-5668357219", author: "human:zetxqx", last_modified: "2026-09-14T17:57:53Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Scope and process

Pull request slicing, public design issues, release-note hygiene, and fork-branch review iteration.

## Separate mechanical moves from behavioral changes

### Rule

Place mechanical package moves, file renames, linter rollouts, and build-tool updates in standalone pull requests separate from behavioral changes so collateral diffs do not obscure logic defects, and split large test additions by tier across unit, integration, and benchmark suites. See [AGENTS.md](../../../AGENTS.md#pull-requests) for pull request scope and template mechanics.

### Scope and consequence

- **Applies to**: Pull request metadata, `.github/**`, `Makefile*`, and multi-package diffs; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bundled mechanical moves, unaligned architectural changes, inaccurate release notes, or force-pushed review branches obscure logic defects and break incremental review diffs.

### Reviewers accept

Deferring cross-package migrations, sibling plugin gaps, RPC batching follow-ups, and broad documentation rewrites to linked issues to keep the active diff focused.[^pr-860-r3178005111][^pr-1071-r3262248799][^pr-1115-r3374238301]

### Evidence

Distilled from 2 review clusters (89 pull requests / 16 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-ic-4642288302][^pr-860-r3274050146][^pr-1018-r3270503284]

## Settle design tradeoffs on public issues and scope auto-close keywords

### Rule

Discuss architecture changes and behavior-changing requirements on public GitHub issues rather than private trackers before iterating on pull requests, so the community can evaluate the requirement. Use auto-closing issue keywords only when a pull request resolves the entire local issue, leaving multi-slice tracking issues open until the final slice lands.

### Scope and consequence

- **Applies to**: Pull request metadata, `.github/**`, `Makefile*`, and multi-package diffs; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bundled mechanical moves, unaligned architectural changes, inaccurate release notes, or force-pushed review branches obscure logic defects and break incremental review diffs.
- **Conflict audit references**: `APW-18`, `UNR-07`.

### Reviewers accept

Narrow wire-formatting adjustments motivated by an external conformance verifier when the issue and pull request state that compatibility scope explicitly.[^issue-2590-ic-5499142962]

### Open question

See [Design proposal storage in repository vs issues](../open_questions/design_proposals_location.md).

### Evidence

Distilled from 3 review clusters (74 pull requests / 14 maintainers, 49 pull requests / 13 maintainers, 14 pull requests / 9 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-ic-4541930806][^issue-857][^issue-1090-ic-5347088232]

## Use `NONE` release notes for internal changes and keep PR metadata synchronized

### Rule

Mark the `release-note` block as `NONE` for internal refactors, tests, CI edits, and fixes to unshipped features on `main`, and write operator-focused notes when defaults, flags, metrics, headers, or upgrade steps change. Update the pull request title, body, and release note whenever review iterations rename symbols or narrow scope.

### Scope and consequence

- **Applies to**: Pull request metadata, `.github/**`, `Makefile*`, and multi-package diffs; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bundled mechanical moves, unaligned architectural changes, inaccurate release notes, or force-pushed review branches obscure logic defects and break incremental review diffs.

### Evidence

Observed across 49 pull requests and 13 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1206-ic-4647621711][^pr-1657-ic-4708030072][^pr-1939-ic-4925378102]

## Iterate in place from personal fork branches with incremental commits

### Rule

Open pull requests from a personal fork instead of pushing feature branches to the upstream repository, update active pull requests in place, and close superseded pull requests with a link to the replacement. Push new commits during review rounds rather than force-pushing rebased history so reviewers can inspect incremental diffs.

### Scope and consequence

- **Applies to**: Pull request metadata, `.github/**`, `Makefile*`, and multi-package diffs; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bundled mechanical moves, unaligned architectural changes, inaccurate release notes, or force-pushed review branches obscure logic defects and break incremental review diffs.

### Evidence

Distilled from 3 review clusters (14 pull requests / 9 maintainers, 1 pull requests / 1 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1172-ic-4477576689][^pr-2228-ic-5409613155][^pr-2823-ic-5668357219]

[^pr-707-ic-4642288302]: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4642288302) (human:ahg-g)
[^pr-860-r3274050146]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3274050146) (human:elevran)
[^pr-1018-r3270503284]: [#1018](https://github.com/llm-d/llm-d-router/pull/1018#discussion_r3270503284) (human:liu-cong)
[^pr-860-r3178005111]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3178005111) (human:vMaroon)
[^pr-1071-r3262248799]: [#1071](https://github.com/llm-d/llm-d-router/pull/1071#discussion_r3262248799) (human:ahg-g)
[^pr-1115-r3374238301]: [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3374238301) (human:sagearc)
[^pr-707-ic-4541930806]: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4541930806) (human:ahg-g)
[^issue-857]: [#857](https://github.com/llm-d/llm-d-router/issues/857) (human:elevran)
[^issue-1090-ic-5347088232]: [#1090](https://github.com/llm-d/llm-d-router/issues/1090#issuecomment-5347088232) (human:LukeAVanDrie)
[^issue-2590-ic-5499142962]: [#2590](https://github.com/llm-d/llm-d-router/issues/2590#issuecomment-5499142962) (human:roytman)
[^pr-1206-ic-4647621711]: [#1206](https://github.com/llm-d/llm-d-router/pull/1206#issuecomment-4647621711) (human:elevran)
[^pr-1657-ic-4708030072]: [#1657](https://github.com/llm-d/llm-d-router/pull/1657#issuecomment-4708030072) (human:shmuelk)
[^pr-1939-ic-4925378102]: [#1939](https://github.com/llm-d/llm-d-router/pull/1939#issuecomment-4925378102) (human:ahg-g)
[^pr-1172-ic-4477576689]: [#1172](https://github.com/llm-d/llm-d-router/pull/1172#issuecomment-4477576689) (human:elevran)
[^pr-2228-ic-5409613155]: [#2228](https://github.com/llm-d/llm-d-router/pull/2228#issuecomment-5409613155) (human:ahg-g)
[^pr-2823-ic-5668357219]: [#2823](https://github.com/llm-d/llm-d-router/pull/2823#issuecomment-5668357219) (human:zetxqx)
