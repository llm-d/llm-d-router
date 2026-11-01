---
type: Review Rule Set
title: "Configuration, Helm, and compatibility"
description: "Two-release deprecation windows, strict unknown-field decoding, safe defaults, shared Helm library templates, and lossless CRD conversion."
tags: ["configuration-helm-and-compatibility", "failure-class", "mvm-11", "apw-12", "nvc-11", "nvd-02", "mvm-23"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "pr-820-ic-4395649694", resource: "https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4395649694", author: "human:elevran", last_modified: "2026-05-07T08:56:21Z" }
  - { id: "pr-842-r3180806498", resource: "https://github.com/llm-d/llm-d-router/pull/842#discussion_r3180806498", author: "human:ahg-g", last_modified: "2026-05-04T10:14:01Z" }
  - { id: "pr-905-ic-4351689606", resource: "https://github.com/llm-d/llm-d-router/pull/905#issuecomment-4351689606", author: "human:elevran", last_modified: "2026-04-30T10:29:29Z" }
  - { id: "pr-894-r3151838341", resource: "https://github.com/llm-d/llm-d-router/pull/894#discussion_r3151838341", author: "human:ahg-g", last_modified: "2026-04-28T05:49:19Z" }
  - { id: "pr-1254-ic-4512813466", resource: "https://github.com/llm-d/llm-d-router/pull/1254#issuecomment-4512813466", author: "human:ahg-g", last_modified: "2026-05-21T21:12:06Z" }
  - { id: "pr-2365-r4241052101", resource: "https://github.com/llm-d/llm-d-router/pull/2365#discussion_r4241052101", author: "human:roytman", last_modified: "2026-10-11T09:26:35Z" }
  - { id: "pr-3511-r4379506333", resource: "https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379506333", author: "human:sagearc", last_modified: "2026-10-26T16:04:27Z" }
  - { id: "pr-683-r3143157150", resource: "https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143157150", author: "human:elevran", last_modified: "2026-04-26T07:36:32Z" }
  - { id: "pr-707-r3190715413", resource: "https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190715413", author: "human:roytman", last_modified: "2026-05-05T18:33:46Z" }
  - { id: "pr-919-r3161391196", resource: "https://github.com/llm-d/llm-d-router/pull/919#discussion_r3161391196", author: "human:hexfusion", last_modified: "2026-04-29T13:35:16Z" }
  - { id: "pr-1029-r3243916222", resource: "https://github.com/llm-d/llm-d-router/pull/1029#discussion_r3243916222", author: "human:ahg-g", last_modified: "2026-05-14T19:55:42Z" }
  - { id: "pr-1156-r3454102412", resource: "https://github.com/llm-d/llm-d-router/pull/1156#discussion_r3454102412", author: "human:bongwoobak", last_modified: "2026-06-22T17:11:01Z" }
  - { id: "issue-1187-ic-5139243431", resource: "https://github.com/llm-d/llm-d-router/issues/1187#issuecomment-5139243431", author: "human:LukeAVanDrie", last_modified: "2026-07-31T04:24:09Z" }
  - { id: "issue-924", resource: "https://github.com/llm-d/llm-d-router/issues/924", author: "human:bongwoobak", last_modified: "2026-04-29T18:21:03Z" }
  - { id: "pr-1040-ic-4415659669", resource: "https://github.com/llm-d/llm-d-router/pull/1040#issuecomment-4415659669", author: "human:ahg-g", last_modified: "2026-05-10T15:32:51Z" }
  - { id: "pr-1275-r3294949972", resource: "https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294949972", author: "human:elevran", last_modified: "2026-05-24T16:20:04Z" }
  - { id: "pr-1275-r3294998818", resource: "https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294998818", author: "human:ahg-g", last_modified: "2026-05-24T16:54:32Z" }
  - { id: "pr-2961-r4069441791", resource: "https://github.com/llm-d/llm-d-router/pull/2961#discussion_r4069441791", author: "human:yankay", last_modified: "2026-09-22T07:43:50Z" }
  - { id: "pr-972-r3178415392", resource: "https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178415392", author: "human:roytman", last_modified: "2026-05-03T16:33:29Z" }
  - { id: "pr-2925-r4144921130", resource: "https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4144921130", author: "human:ahg-g", last_modified: "2026-09-30T13:09:45Z" }
  - { id: "pr-2925-r4155389367", resource: "https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4155389367", author: "human:hexfusion", last_modified: "2026-10-01T12:34:46Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Configuration, Helm, and compatibility

Two-release deprecation windows, strict unknown-field decoding, safe defaults, shared Helm library templates, and lossless CRD conversion.

## Provide a two-release deprecation window for shipped user-facing surfaces

### Rule

Support released CLI flags, configuration fields, plugin types, HTTP headers, and metrics across at least two releases (`N+2`) before removal. Emit a prominent startup warning when a deprecated surface is used, migrate all in-tree manifests and tests to the replacement immediately, and sequence structural moves separately from renames so operators migrate once.

### Scope and consequence

- **Applies to**: `apix/**`, `cmd/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `pkg/epp/framework/plugins/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Removing shipped flags, YAML fields, plugin types, HTTP headers, or metrics without an `N+2` compatibility window or strict schema decoding breaks existing Helm deployments and silently drops operator typos.
- **Related decisions**: [Two-release deprecation shims required even during pre-1.0 config refactors](../decisions/two_release_deprecation_shims_for_pre_v1_config.md).

### Reviewers accept

Renaming or removing unshipped configuration keys, unreleased Helm charts, and internal headers without deprecation shims, and deleting shims once their deprecation window closes.[^pr-894-r3151838341][^pr-1254-ic-4512813466][^pr-2365-r4241052101][^pr-3511-r4379506333]

### Evidence

Observed across 72 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-820-ic-4395649694][^pr-842-r3180806498][^pr-905-ic-4351689606]

## Decode configuration with strict unknown-field rejection and startup validation

### Rule

Seed configuration structs from canonical default constructors, decode plugin and component configurations through `fwkplugin.StrictDecoder` (`DisallowUnknownFields()` in `pkg/epp/framework/interface/plugin/registry.go`), `yaml.UnmarshalStrict`, or `serializer.EnableStrict` (`pkg/epp/config/loader/configloader.go`) so unknown fields fail at startup, and validate cross-field and mode-scoped invariants deterministically before serving traffic. Use explicit mode enums over boolean flag combinations, and use pointer numeric fields with `omitempty` only when distinguishing an unset default from an explicit zero.

### Scope and consequence

- **Applies to**: `apix/**`, `cmd/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `fwkplugin.StrictDecoder`, `pkg/epp/framework/interface/plugin/registry.go`, `yaml.UnmarshalStrict`, `serializer.EnableStrict`, `pkg/epp/config/loader/configloader.go`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Removing shipped flags, YAML fields, plugin types, HTTP headers, or metrics without an `N+2` compatibility window or strict schema decoding breaks existing Helm deployments and silently drops operator typos.
- **Conflict audit references**: `NVC-11`, `NVD-02`.

### Evidence

Observed across 65 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-683-r3143157150][^pr-707-r3190715413][^pr-919-r3161391196]

## Ship safe out-of-the-box defaults and gate workload-sensitive features behind opt-in config

### Rule

Choose defaults that run within standard memory and CPU budgets without external sidecars or cross-component boot-order dependencies. Gate workload-sensitive heuristics, speculative estimators, and outbound payload rewrites behind explicit opt-in configuration so disabled features leave the default request path untouched.

### Scope and consequence

- **Applies to**: `apix/**`, `cmd/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `pkg/epp/framework/plugins/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Removing shipped flags, YAML fields, plugin types, HTTP headers, or metrics without an `N+2` compatibility window or strict schema decoding breaks existing Helm deployments and silently drops operator typos.
- **Related decisions**: [Flow control remains opt-in by default until standalone default-on graduation](../decisions/flow_control_opt_in_feature_gate_posture.md).

### Evidence

Observed across 35 pull requests and 9 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1029-r3243916222][^pr-1156-r3454102412][^issue-1187-ic-5139243431]

## Deduplicate Helm chart defaults through shared library templates

### Rule

Share common Helm logic in library templates while declaring values only in charts that render them, emit template `fail` errors when operators supply removed keys, and keep lifecycle hooks compatible with shell-less distroless images.

### Scope and consequence

- **Applies to**: `apix/**`, `cmd/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `pkg/epp/framework/plugins/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Removing shipped flags, YAML fields, plugin types, HTTP headers, or metrics without an `N+2` compatibility window or strict schema decoding breaks existing Helm deployments and silently drops operator typos.

### Reviewers accept

Repeating graduated stable Kubernetes API group literals directly in Helm templates and gating lifecycle features on supported Kubernetes versions with a documentation note rather than shell fallbacks.[^pr-1275-r3294998818][^pr-2961-r4069441791]

### Evidence

Observed across 52 pull requests and 15 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-924][^pr-1040-ic-4415659669][^pr-1275-r3294949972]

## Preserve lossless conversion across multi-version CRDs

### Rule

Serve multi-version CRDs through lossless conversion under a single storage version with codegen validation markers, and update all in-tree `config/charts`, `deploy`, and test manifests in the same change.

### Scope and consequence

- **Applies to**: `apix/**`, `cmd/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `pkg/epp/framework/plugins/**` (referenced symbols: `config/charts`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Removing shipped flags, YAML fields, plugin types, HTTP headers, or metrics without an `N+2` compatibility window or strict schema decoding breaks existing Helm deployments and silently drops operator typos.

### Evidence

Observed across 13 pull requests and 7 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-972-r3178415392][^pr-2925-r4144921130][^pr-2925-r4155389367]

[^pr-820-ic-4395649694]: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4395649694) (human:elevran)
[^pr-842-r3180806498]: [#842](https://github.com/llm-d/llm-d-router/pull/842#discussion_r3180806498) (human:ahg-g)
[^pr-905-ic-4351689606]: [#905](https://github.com/llm-d/llm-d-router/pull/905#issuecomment-4351689606) (human:elevran)
[^pr-894-r3151838341]: [#894](https://github.com/llm-d/llm-d-router/pull/894#discussion_r3151838341) (human:ahg-g)
[^pr-1254-ic-4512813466]: [#1254](https://github.com/llm-d/llm-d-router/pull/1254#issuecomment-4512813466) (human:ahg-g)
[^pr-2365-r4241052101]: [#2365](https://github.com/llm-d/llm-d-router/pull/2365#discussion_r4241052101) (human:roytman)
[^pr-3511-r4379506333]: [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379506333) (human:sagearc)
[^pr-683-r3143157150]: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143157150) (human:elevran)
[^pr-707-r3190715413]: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190715413) (human:roytman)
[^pr-919-r3161391196]: [#919](https://github.com/llm-d/llm-d-router/pull/919#discussion_r3161391196) (human:hexfusion)
[^pr-1029-r3243916222]: [#1029](https://github.com/llm-d/llm-d-router/pull/1029#discussion_r3243916222) (human:ahg-g)
[^pr-1156-r3454102412]: [#1156](https://github.com/llm-d/llm-d-router/pull/1156#discussion_r3454102412) (human:bongwoobak)
[^issue-1187-ic-5139243431]: [#1187](https://github.com/llm-d/llm-d-router/issues/1187#issuecomment-5139243431) (human:LukeAVanDrie)
[^issue-924]: [#924](https://github.com/llm-d/llm-d-router/issues/924) (human:bongwoobak)
[^pr-1040-ic-4415659669]: [#1040](https://github.com/llm-d/llm-d-router/pull/1040#issuecomment-4415659669) (human:ahg-g)
[^pr-1275-r3294949972]: [#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294949972) (human:elevran)
[^pr-1275-r3294998818]: [#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294998818) (human:ahg-g)
[^pr-2961-r4069441791]: [#2961](https://github.com/llm-d/llm-d-router/pull/2961#discussion_r4069441791) (human:yankay)
[^pr-972-r3178415392]: [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178415392) (human:roytman)
[^pr-2925-r4144921130]: [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4144921130) (human:ahg-g)
[^pr-2925-r4155389367]: [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4155389367) (human:hexfusion)
