---
type: Review Rule Set
title: Testing and CI
description: "Production factory tests with exact output assertions, colocated sleep-free unit tests, namespace-isolated E2E teardown, and presubmit CI parity."
tags: ["testing-and-ci", "failure-class", "nvc-07", "nvc-06", "unr-02"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "pr-916-r3215017290"
    resource: "https://github.com/llm-d/llm-d-router/pull/916#discussion_r3215017290"
    author: "human:elevran"
    last_modified: "2026-05-10T14:46:18Z"
  - id: "pr-972-rev-4216545185"
    resource: "https://github.com/llm-d/llm-d-router/pull/972#pullrequestreview-4216545185"
    author: "human:roytman"
    last_modified: "2026-05-03T17:01:56Z"
  - id: "pr-1007-r3226190634"
    resource: "https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3226190634"
    author: "human:ahg-g"
    last_modified: "2026-05-12T12:01:56Z"
  - id: "pr-1030-r3289321053"
    resource: "https://github.com/llm-d/llm-d-router/pull/1030#discussion_r3289321053"
    author: "human:ahg-g"
    last_modified: "2026-05-22T15:08:26Z"
  - id: "pr-1872-r3524933896"
    resource: "https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524933896"
    author: "human:roytman"
    last_modified: "2026-07-05T12:57:46Z"
  - id: "pr-1937-r3552704054"
    resource: "https://github.com/llm-d/llm-d-router/pull/1937#discussion_r3552704054"
    author: "human:shmuelk"
    last_modified: "2026-07-09T15:10:34Z"
  - id: "issue-1066"
    resource: "https://github.com/llm-d/llm-d-router/issues/1066"
    author: "human:yankay"
    last_modified: "2026-05-12T13:34:17Z"
  - id: "issue-1188"
    resource: "https://github.com/llm-d/llm-d-router/issues/1188"
    author: "human:LukeAVanDrie"
    last_modified: "2026-05-18T07:20:20Z"
  - id: "pr-1528-r3369900484"
    resource: "https://github.com/llm-d/llm-d-router/pull/1528#discussion_r3369900484"
    author: "human:hexfusion"
    last_modified: "2026-06-07T18:50:02Z"
  - id: "pr-2865-r4016088694"
    resource: "https://github.com/llm-d/llm-d-router/pull/2865#discussion_r4016088694"
    author: "human:elevran"
    last_modified: "2026-09-15T13:26:27Z"
  - id: "pr-1396-r3369095849"
    resource: "https://github.com/llm-d/llm-d-router/pull/1396#discussion_r3369095849"
    author: "human:roytman"
    last_modified: "2026-06-07T09:37:38Z"
  - id: "pr-1048-rev-4310234203"
    resource: "https://github.com/llm-d/llm-d-router/pull/1048#pullrequestreview-4310234203"
    author: "human:elevran"
    last_modified: "2026-05-18T13:02:07Z"
  - id: "pr-1481-r3372638867"
    resource: "https://github.com/llm-d/llm-d-router/pull/1481#discussion_r3372638867"
    author: "human:sagearc"
    last_modified: "2026-06-08T10:56:42Z"
  - id: "pr-1976-r3568707788"
    resource: "https://github.com/llm-d/llm-d-router/pull/1976#discussion_r3568707788"
    author: "human:roytman"
    last_modified: "2026-07-13T07:17:40Z"
  - id: "pr-2048-r3596786577"
    resource: "https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577"
    author: "human:shmuelk"
    last_modified: "2026-07-16T15:28:50Z"
  - id: "pr-1698-ic-4762216995"
    resource: "https://github.com/llm-d/llm-d-router/pull/1698#issuecomment-4762216995"
    author: "human:roytman"
    last_modified: "2026-06-21T14:03:30Z"
  - id: "issue-949-ic-4565428236"
    resource: "https://github.com/llm-d/llm-d-router/issues/949#issuecomment-4565428236"
    author: "human:shmuelk"
    last_modified: "2026-05-28T15:04:11Z"
  - id: "pr-960-r3182827088"
    resource: "https://github.com/llm-d/llm-d-router/pull/960#discussion_r3182827088"
    author: "human:elevran"
    last_modified: "2026-05-04T16:03:45Z"
  - id: "pr-1234-r3294875921"
    resource: "https://github.com/llm-d/llm-d-router/pull/1234#discussion_r3294875921"
    author: "human:elevran"
    last_modified: "2026-05-24T15:26:14Z"
  - id: "pr-1579-r3409481145"
    resource: "https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409481145"
    author: "human:ahg-g"
    last_modified: "2026-06-14T11:48:00Z"
  - id: "pr-3159-ic-6053625347"
    resource: "https://github.com/llm-d/llm-d-router/pull/3159#issuecomment-6053625347"
    author: "human:elevran"
    last_modified: "2026-10-08T06:13:22Z"
  - id: "pr-3130-r4166716568"
    resource: "https://github.com/llm-d/llm-d-router/pull/3130#discussion_r4166716568"
    author: "human:roytman"
    last_modified: "2026-10-02T14:34:06Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Testing and CI

Production factory tests with exact output assertions, colocated sleep-free unit tests, namespace-isolated E2E teardown, and presubmit CI parity.

## Exercise production factories and assert exact behavioral outputs

### Rule

Instantiate plugins and runners in tests through production factories and `fwkplugin.StrictDecoder` rather than constructing internal structs directly. In new and modified tests, assert exact output fields, error types, Prometheus metric values, and JSON wire payloads without masking tested fields via broad `cmpopts.IgnoreFields`, covering both states of feature gates, skipped disaggregated stages, and boundary inputs.

### Scope and consequence

- **Applies to**: `**/*_test.go`, `test/**`, `Makefile*`, `hack/**`, `.github/workflows/**` (referenced symbols: `fwkplugin.StrictDecoder`, `cmpopts.IgnoreFields`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bypassing production factories, sleeping in unit tests, or skipping presubmit wiring lets invalid defaults and flaky teardown reach the main branch.
- **Conflict audit references**: `NVC-07`.

### Reviewers accept

Relying on end-to-end flow tests over unreachable unit combinations and keeping pure parsing functions separate from environment-variable wrappers so table-driven unit tests can call the parser directly.[^pr-1872-r3524933896][^pr-1937-r3552704054]

### Evidence

Observed across 150 pull requests and 14 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-916-r3215017290][^pr-972-rev-4216545185][^pr-1007-r3226190634][^pr-1030-r3289321053]

## Colocate unit tests and synchronize without fixed sleeps

### Rule

Keep unit tests in colocated `*_test.go` files beside the source file they test, move paired test files whenever production types move, and place shared mocks or builders in leaf test packages that cannot cause import cycles. In new and modified tests, synchronize asynchronous startup and controller reconciliation with injectable clocks, lifecycle barriers, or `require.Eventually` rather than fixed `time.Sleep` calls, and bind ephemeral loopback listeners atomically without close-then-rebind port races.

### Scope and consequence

- **Applies to**: `**/*_test.go`, `test/**`, `Makefile*`, `hack/**`, `.github/workflows/**` (referenced symbols: `*_test.go`, `require.Eventually`, `time.Sleep`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bypassing production factories, sleeping in unit tests, or skipping presubmit wiring lets invalid defaults and flaky teardown reach the main branch.
- **Conflict audit references**: `NVC-06`, `UNR-02`.

### Reviewers accept

Calibrated sleep windows when testing refresh-on-read cache TTL expiration where polling the read method would reset the entry TTL.[^pr-1396-r3369095849]

### Evidence

Distilled from 2 review clusters (36 pull requests / 10 maintainers, 12 pull requests / 7 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1066][^issue-1188][^pr-1528-r3369900484][^pr-2865-r4016088694]

## Isolate parallel end-to-end suites with non-aborting teardown

### Rule

Run end-to-end tests against freshly built pull request images and shipping Helm charts, isolating parallel processes by namespace and by port (kind runs bind per-process NodePorts; existing-context runs use a supervised `kubectl port-forward`). Record created cluster resources before readiness waits, run non-aborting cleanup across all resources on teardown, and dump verbose pod logs only when a test fails.

### Scope and consequence

- **Applies to**: `**/*_test.go`, `test/**`, `Makefile*`, `hack/**`, `.github/workflows/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bypassing production factories, sleeping in unit tests, or skipping presubmit wiring lets invalid defaults and flaky teardown reach the main branch.

### Reviewers accept

Maintaining separate E2E runner scripts backed by shared helper libraries when cluster ownership, image sets, and environment contracts differ.[^pr-1698-ic-4762216995]

### Evidence

Observed across 15 pull requests and 6 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1048-rev-4310234203][^pr-1481-r3372638867][^pr-1976-r3568707788][^pr-2048-r3596786577]

## Wire new checks into `make presubmit` and pin third-party container image tags

### Rule

Wire every new linter, protobuf generator, and security check into `make presubmit` inside the version-pinned builder container so local runs match CI across architectures. `make presubmit` runs the format, lint, security, vulnerability, image-tag, and boilerplate checks; `make test-unit`, `test-integration-hermetic`, and `verify-helm-charts` run as separate CI jobs. Pin third-party container image tags in manifests and workflows rather than using `:latest` (`scripts/check-latest-tags.sh` lets images in the llm-d-owned `ghcr.io` and `quay.io` namespaces track `:latest` on `main` so cross-component regressions surface early), keep local developer tool configs out of version control, and hold dependency updates that introduce unfixed upstream regressions.

### Scope and consequence

- **Applies to**: `**/*_test.go`, `test/**`, `Makefile*`, `hack/**`, `.github/workflows/**` (referenced symbols: `scripts/check-latest-tags.sh`, `ghcr.io`, `quay.io`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Bypassing production factories, sleeping in unit tests, or skipping presubmit wiring lets invalid defaults and flaky teardown reach the main branch.

### Reviewers accept

Pinning CI container images to immutable version release tags rather than opaque SHA256 digests so workflow diffs remain readable,[^pr-3130-r4166716568] and E2E suites that build and run images under the local `:dev` tag so tests exercise the pull request's code.[^pr-2048-r3596786577]

### Evidence

Distilled from 2 review clusters (60 pull requests / 11 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-949-ic-4565428236][^pr-960-r3182827088][^pr-1234-r3294875921][^pr-1579-r3409481145][^pr-3159-ic-6053625347]

[^pr-916-r3215017290]: [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3215017290) (human:elevran)
[^pr-972-rev-4216545185]: [#972](https://github.com/llm-d/llm-d-router/pull/972#pullrequestreview-4216545185) (human:roytman)
[^pr-1007-r3226190634]: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3226190634) (human:ahg-g)
[^pr-1030-r3289321053]: [#1030](https://github.com/llm-d/llm-d-router/pull/1030#discussion_r3289321053) (human:ahg-g)
[^pr-1872-r3524933896]: [#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524933896) (human:roytman)
[^pr-1937-r3552704054]: [#1937](https://github.com/llm-d/llm-d-router/pull/1937#discussion_r3552704054) (human:shmuelk)
[^issue-1066]: [#1066](https://github.com/llm-d/llm-d-router/issues/1066) (human:yankay)
[^issue-1188]: [#1188](https://github.com/llm-d/llm-d-router/issues/1188) (human:LukeAVanDrie)
[^pr-1528-r3369900484]: [#1528](https://github.com/llm-d/llm-d-router/pull/1528#discussion_r3369900484) (human:hexfusion)
[^pr-2865-r4016088694]: [#2865](https://github.com/llm-d/llm-d-router/pull/2865#discussion_r4016088694) (human:elevran)
[^pr-1396-r3369095849]: [#1396](https://github.com/llm-d/llm-d-router/pull/1396#discussion_r3369095849) (human:roytman)
[^pr-1048-rev-4310234203]: [#1048](https://github.com/llm-d/llm-d-router/pull/1048#pullrequestreview-4310234203) (human:elevran)
[^pr-1481-r3372638867]: [#1481](https://github.com/llm-d/llm-d-router/pull/1481#discussion_r3372638867) (human:sagearc)
[^pr-1976-r3568707788]: [#1976](https://github.com/llm-d/llm-d-router/pull/1976#discussion_r3568707788) (human:roytman)
[^pr-2048-r3596786577]: [#2048](https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577) (human:shmuelk)
[^pr-1698-ic-4762216995]: [#1698](https://github.com/llm-d/llm-d-router/pull/1698#issuecomment-4762216995) (human:roytman)
[^issue-949-ic-4565428236]: [#949](https://github.com/llm-d/llm-d-router/issues/949#issuecomment-4565428236) (human:shmuelk)
[^pr-960-r3182827088]: [#960](https://github.com/llm-d/llm-d-router/pull/960#discussion_r3182827088) (human:elevran)
[^pr-1234-r3294875921]: [#1234](https://github.com/llm-d/llm-d-router/pull/1234#discussion_r3294875921) (human:elevran)
[^pr-1579-r3409481145]: [#1579](https://github.com/llm-d/llm-d-router/pull/1579#discussion_r3409481145) (human:ahg-g)
[^pr-3159-ic-6053625347]: [#3159](https://github.com/llm-d/llm-d-router/pull/3159#issuecomment-6053625347) (human:elevran)
[^pr-3130-r4166716568]: [#3130](https://github.com/llm-d/llm-d-router/pull/3130#discussion_r4166716568) (human:roytman)
