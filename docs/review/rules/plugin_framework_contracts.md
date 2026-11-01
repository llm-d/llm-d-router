---
type: Review Rule Set
title: Plugin framework contracts
description: "Core-vs-plugin import boundaries, extension-point separation, typed data-layer slots, factory validation, naming, and dead-code removal."
tags: ["plugin-framework-contracts", "failure-class", "mvm-02", "mvm-03", "apw-11", "mvm-01", "apw-01", "nvc-05", "mvm-12", "apw-08"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "issue-836", resource: "https://github.com/llm-d/llm-d-router/issues/836", author: "human:elevran", last_modified: "2026-04-16T14:18:07Z" }
  - { id: "issue-877", resource: "https://github.com/llm-d/llm-d-router/issues/877", author: "human:vMaroon", last_modified: "2026-04-23T17:23:11Z" }
  - { id: "pr-1115-r3261995399", resource: "https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3261995399", author: "human:liu-cong", last_modified: "2026-05-18T20:42:13Z" }
  - { id: "issue-545-ic-5194281537", resource: "https://github.com/llm-d/llm-d-router/issues/545#issuecomment-5194281537", author: "human:ahg-g", last_modified: "2026-08-05T16:11:12Z" }
  - { id: "issue-837", resource: "https://github.com/llm-d/llm-d-router/issues/837", author: "human:elevran", last_modified: "2026-04-16T16:22:26Z" }
  - { id: "issue-977", resource: "https://github.com/llm-d/llm-d-router/issues/977", author: "human:liu-cong", last_modified: "2026-05-03T18:45:41Z" }
  - { id: "pr-3145-ic-6088007878", resource: "https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6088007878", author: "human:ahg-g", last_modified: "2026-10-09T19:41:44Z" }
  - { id: "pr-3145-rev-5462400380", resource: "https://github.com/llm-d/llm-d-router/pull/3145#pullrequestreview-5462400380", author: "human:ahg-g", last_modified: "2026-10-08T20:34:52Z" }
  - { id: "issue-815-ic-4603503350", resource: "https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4603503350", author: "human:liu-cong", last_modified: "2026-06-02T14:38:03Z" }
  - { id: "pr-707-ic-4396004237", resource: "https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4396004237", author: "human:elevran", last_modified: "2026-05-07T09:44:18Z" }
  - { id: "pr-901-r3252785994", resource: "https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252785994", author: "human:vMaroon", last_modified: "2026-05-16T12:17:34Z" }
  - { id: "pr-1096-r3255496885", resource: "https://github.com/llm-d/llm-d-router/pull/1096#discussion_r3255496885", author: "human:ahg-g", last_modified: "2026-05-17T21:16:27Z" }
  - { id: "pr-901-r3268601757", resource: "https://github.com/llm-d/llm-d-router/pull/901#discussion_r3268601757", author: "human:liu-cong", last_modified: "2026-05-19T18:21:49Z" }
  - { id: "pr-959-r3196010531", resource: "https://github.com/llm-d/llm-d-router/pull/959#discussion_r3196010531", author: "human:hexfusion", last_modified: "2026-05-06T13:59:53Z" }
  - { id: "pr-1007-r3203628902", resource: "https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628902", author: "human:elevran", last_modified: "2026-05-07T17:55:06Z" }
  - { id: "pr-1396-ic-4642995683", resource: "https://github.com/llm-d/llm-d-router/pull/1396#issuecomment-4642995683", author: "human:roytman", last_modified: "2026-06-07T14:44:43Z" }
  - { id: "issue-747", resource: "https://github.com/llm-d/llm-d-router/issues/747", author: "human:roytman", last_modified: "2026-03-22T08:35:48Z" }
  - { id: "issue-813", resource: "https://github.com/llm-d/llm-d-router/issues/813", author: "human:ahg-g", last_modified: "2026-04-12T18:29:52Z" }
  - { id: "pr-888-r3143427635", resource: "https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143427635", author: "human:elevran", last_modified: "2026-04-26T11:49:25Z" }
  - { id: "pr-2036-ic-4999184223", resource: "https://github.com/llm-d/llm-d-router/pull/2036#issuecomment-4999184223", author: "human:LukeAVanDrie", last_modified: "2026-07-17T05:09:38Z" }
  - { id: "pr-2342-r3752875804", resource: "https://github.com/llm-d/llm-d-router/pull/2342#discussion_r3752875804", author: "human:ahg-g", last_modified: "2026-08-10T19:53:42Z" }
  - { id: "pr-683-r3143173577", resource: "https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143173577", author: "human:elevran", last_modified: "2026-04-26T07:52:42Z" }
  - { id: "pr-891-r3236349435", resource: "https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236349435", author: "human:LukeAVanDrie", last_modified: "2026-05-13T17:47:24Z" }
  - { id: "pr-1444-rev-4425939275", resource: "https://github.com/llm-d/llm-d-router/pull/1444#pullrequestreview-4425939275", author: "human:roytman", last_modified: "2026-06-04T08:15:00Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Plugin framework contracts

Core-vs-plugin import boundaries, extension-point separation, typed data-layer slots, factory validation, naming, and dead-code removal.

## Keep core runtime packages free of concrete plugin logic

### Rule

Do not import concrete plugin packages or add plugin-specific type switches inside `cmd/epp/runner`, `pkg/epp/datastore`, or `pkg/epp/framework/interface`. Expose shared runtime capabilities through `fwkplugin.Handle` and small optional behavioral interfaces verified with compile-time interface guards (`var _ Interface = (*Type)(nil)`), resolving capabilities once at startup registration.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `cmd/epp/runner`, `pkg/epp/datastore`, `pkg/epp/framework/interface`, `fwkplugin.Handle`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.

### Evidence

Observed across 47 pull requests and 13 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-836][^issue-877][^pr-1115-r3261995399]

## Separate data producers from scheduling filters and scorers

### Rule

Keep stateful indexing and raw endpoint extraction in data producers while placing policy evaluation in downstream scheduling plugins (see [docs/architecture.md](../../architecture.md#routing-flow)). Restrict `Filter` plugins to binary endpoint eligibility with fail-open bypass when optional telemetry is absent, express soft preferences as weighted `Scorer` plugins rather than positional filters, use pre-scheduling `Screener` plugins only for mandatory endpoint constraints that apply to every scheduling profile, and delegate load shedding to saturation detectors.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.
- **Related decisions**: [Session affinity as a shared stateless helper package](../decisions/session_affinity_shared_helper.md).

### Reviewers accept

Implementing strict single-endpoint request pinning as a pre-scheduling `Screener` once the tradeoff against a scheduling `Filter` is evaluated,[^pr-3145-ic-6088007878][^pr-3145-rev-5462400380] and a shared stateless helper package in place of a data producer when the extraction is header parsing with no I/O and no cross-plugin caching.[^issue-815-ic-4603503350]

### Evidence

Observed across 37 pull requests and 13 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-545-ic-5194281537][^issue-837][^issue-977]

## Exchange cross-plugin state through typed data-layer slots and keys

### Rule

Pass cross-plugin request and endpoint metadata through typed `datalayer.Slot[T]` handles (`pkg/epp/framework/interface/datalayer/slot.go`) or `Clone()`-safe values in `datalayer.AttributeMap` (`pkg/epp/framework/interface/datalayer/registry.go`) rather than global singletons or direct peer calls. Declare cross-plugin dependencies through `Produces()` and `Consumes()`, key instance-scoped attributes by non-empty plugin names, and validate producer-consumer DAG wiring at startup after default producers resolve.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `datalayer.Slot[T]`, `pkg/epp/framework/interface/datalayer/slot.go`, `datalayer.AttributeMap`, `pkg/epp/framework/interface/datalayer/registry.go`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.
- **Related decisions**: [Typed data-layer Slot handles over generic Extractor interfaces](../decisions/typed_datalayer_slots_over_generic_extractors.md).

### Evidence

Observed across 58 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-ic-4396004237][^pr-901-r3252785994][^pr-1096-r3255496885]

## Validate plugin parameters once in the factory constructor

### Rule

Validate parameter bounds, required handles, and forbidden system headers inside the factory constructor rather than exposing mutable post-construction setters or repeating checks per request. Register built-in plugins in the central catalog with an explicit stability tier and enforce experimental feature gates across explicit and default instances; see [pkg/epp/framework/plugins/README.md](../../../pkg/epp/framework/plugins/README.md) and [docs/create_new_filter.md](../../create_new_filter.md) for plugin registration and tier lifecycle rules.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.

### Reviewers accept

Documenting redundant plugin combinations in plugin READMEs rather than rejecting sibling plugins inside factory constructors.[^pr-1396-ic-4642995683]

### Evidence

Distilled from 2 review clusters (38 pull requests / 9 maintainers, 13 pull requests / 9 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-901-r3268601757][^pr-959-r3196010531][^pr-1007-r3203628902]

## Name plugin packages with concise domain nouns and qualify config units

### Rule

Name plugin packages with short lowercase domain nouns (such as `bylabel`, `queuedepth`, `tokenload`, or `sessionaffinity`), suffix newly registered plugin type strings with their extension-point role (`-filter`, `-scorer`, `-picker`, `-producer`, `-screener`, or `-decider`), and avoid repeating the package name in exported identifiers. Shipped type strings without a role suffix keep their names. Suffix quantitative configuration fields with explicit units or portable token counts rather than backend-dependent block counts, and accept human-readable quantity strings (`resource.Quantity`) for byte sizes.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `-filter`, `-scorer`, `-picker`, `-producer`, `-screener`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.
- **Related decisions**: [Prefix-cache block size parameter includes Tokens unit suffix](../decisions/prefix_cache_block_size_tokens_unit_suffix.md).

### Reviewers accept

Standard computer-science data-structure names on internal types and preserving established public constant identifiers across internal variable renames.[^pr-2036-ic-4999184223][^pr-2342-r3752875804]

### Evidence

Distilled from 2 review clusters (72 pull requests / 10 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-747][^issue-813][^pr-888-r3143427635]

## Share common helpers and delete dead code in the stranding change

### Rule

Decouple pure payload parsing from HTTP response writing, reuse shared utilities in `pkg/common` rather than duplicating logic across packages, and follow [AGENTS.md](../../../AGENTS.md#working-in-the-codebase) for canonical constant ownership across packages. Delete dead branches, write-only fields, unreferenced methods, and single-caller pass-through wrappers in the change that strands them.

### Scope and consequence

- **Applies to**: `cmd/epp/**`, `pkg/epp/framework/interface/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `pkg/common`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Coupling core runtimes to concrete plugins, mixing data production with scoring, or deferring parameter validation past factory construction breaks modular extensibility and startup safety.
- **Related decisions**: [Delete stranded deprecation helpers in the removal pull request](../decisions/delete_stranded_deprecation_helpers_immediately.md).

### Evidence

Distilled from 2 review clusters (56 pull requests / 10 maintainers, 42 pull requests / 12 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-683-r3143173577][^pr-891-r3236349435][^pr-1444-rev-4425939275]

[^issue-836]: [#836](https://github.com/llm-d/llm-d-router/issues/836) (human:elevran)
[^issue-877]: [#877](https://github.com/llm-d/llm-d-router/issues/877) (human:vMaroon)
[^pr-1115-r3261995399]: [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3261995399) (human:liu-cong)
[^issue-545-ic-5194281537]: [#545](https://github.com/llm-d/llm-d-router/issues/545#issuecomment-5194281537) (human:ahg-g)
[^issue-837]: [#837](https://github.com/llm-d/llm-d-router/issues/837) (human:elevran)
[^issue-977]: [#977](https://github.com/llm-d/llm-d-router/issues/977) (human:liu-cong)
[^pr-3145-ic-6088007878]: [#3145](https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6088007878) (human:ahg-g)
[^pr-3145-rev-5462400380]: [#3145](https://github.com/llm-d/llm-d-router/pull/3145#pullrequestreview-5462400380) (human:ahg-g)
[^issue-815-ic-4603503350]: [#815](https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4603503350) (human:liu-cong)
[^pr-707-ic-4396004237]: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4396004237) (human:elevran)
[^pr-901-r3252785994]: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252785994) (human:vMaroon)
[^pr-1096-r3255496885]: [#1096](https://github.com/llm-d/llm-d-router/pull/1096#discussion_r3255496885) (human:ahg-g)
[^pr-901-r3268601757]: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3268601757) (human:liu-cong)
[^pr-959-r3196010531]: [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3196010531) (human:hexfusion)
[^pr-1007-r3203628902]: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628902) (human:elevran)
[^pr-1396-ic-4642995683]: [#1396](https://github.com/llm-d/llm-d-router/pull/1396#issuecomment-4642995683) (human:roytman)
[^issue-747]: [#747](https://github.com/llm-d/llm-d-router/issues/747) (human:roytman)
[^issue-813]: [#813](https://github.com/llm-d/llm-d-router/issues/813) (human:ahg-g)
[^pr-888-r3143427635]: [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143427635) (human:elevran)
[^pr-2036-ic-4999184223]: [#2036](https://github.com/llm-d/llm-d-router/pull/2036#issuecomment-4999184223) (human:LukeAVanDrie)
[^pr-2342-r3752875804]: [#2342](https://github.com/llm-d/llm-d-router/pull/2342#discussion_r3752875804) (human:ahg-g)
[^pr-683-r3143173577]: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143173577) (human:elevran)
[^pr-891-r3236349435]: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236349435) (human:LukeAVanDrie)
[^pr-1444-rev-4425939275]: [#1444](https://github.com/llm-d/llm-d-router/pull/1444#pullrequestreview-4425939275) (human:roytman)
