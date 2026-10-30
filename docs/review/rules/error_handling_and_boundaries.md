---
type: Review Rule Set
title: Error handling and boundaries
description: "Boundary input validation, explicit error propagation without fabricated fallbacks, unwrapped status-code mapping, and scoped linter suppressions."
tags: ["error-handling-and-boundaries", "failure-class", "nvc-08", "unr-06", "nvc-09"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "pr-683-r3143190014"
    resource: "https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143190014"
    author: "human:elevran"
    last_modified: "2026-04-26T08:08:19Z"
  - id: "pr-707-r3220172664"
    resource: "https://github.com/llm-d/llm-d-router/pull/707#discussion_r3220172664"
    author: "human:hexfusion"
    last_modified: "2026-05-11T15:37:47Z"
  - id: "pr-1147-ic-4461397416"
    resource: "https://github.com/llm-d/llm-d-router/pull/1147#issuecomment-4461397416"
    author: "human:ahg-g"
    last_modified: "2026-05-15T16:13:20Z"
  - id: "pr-1163-r3254240458"
    resource: "https://github.com/llm-d/llm-d-router/pull/1163#discussion_r3254240458"
    author: "human:roytman"
    last_modified: "2026-05-17T07:56:20Z"
  - id: "pr-901-r3192331180"
    resource: "https://github.com/llm-d/llm-d-router/pull/901#discussion_r3192331180"
    author: "human:liu-cong"
    last_modified: "2026-05-06T00:30:33Z"
  - id: "pr-944-r3435978187"
    resource: "https://github.com/llm-d/llm-d-router/pull/944#discussion_r3435978187"
    author: "human:shmuelk"
    last_modified: "2026-06-18T13:05:08Z"
  - id: "pr-3252-r4218084207"
    resource: "https://github.com/llm-d/llm-d-router/pull/3252#discussion_r4218084207"
    author: "human:roytman"
    last_modified: "2026-10-08T11:00:50Z"
  - id: "issue-876"
    resource: "https://github.com/llm-d/llm-d-router/issues/876"
    author: "human:vMaroon"
    last_modified: "2026-04-23T17:21:21Z"
  - id: "pr-1002-r3403097625"
    resource: "https://github.com/llm-d/llm-d-router/pull/1002#discussion_r3403097625"
    author: "human:ahg-g"
    last_modified: "2026-06-12T11:48:40Z"
  - id: "pr-1061-r3246024767"
    resource: "https://github.com/llm-d/llm-d-router/pull/1061#discussion_r3246024767"
    author: "human:liu-cong"
    last_modified: "2026-05-15T05:09:36Z"
  - id: "pr-1895-r3615464093"
    resource: "https://github.com/llm-d/llm-d-router/pull/1895#discussion_r3615464093"
    author: "human:elevran"
    last_modified: "2026-07-20T15:28:59Z"
  - id: "issue-1116-ic-4450544045"
    resource: "https://github.com/llm-d/llm-d-router/issues/1116#issuecomment-4450544045"
    author: "human:roytman"
    last_modified: "2026-05-14T12:12:12Z"
  - id: "pr-1603-r3399020119"
    resource: "https://github.com/llm-d/llm-d-router/pull/1603#discussion_r3399020119"
    author: "human:gyliu513"
    last_modified: "2026-06-11T20:36:53Z"
  - id: "pr-2052-r3597666475"
    resource: "https://github.com/llm-d/llm-d-router/pull/2052#discussion_r3597666475"
    author: "human:ahg-g"
    last_modified: "2026-07-16T17:40:06Z"
  - id: "pr-2215-r3698734025"
    resource: "https://github.com/llm-d/llm-d-router/pull/2215#discussion_r3698734025"
    author: "human:bongwoobak"
    last_modified: "2026-08-02T10:42:36Z"
  - id: "pr-3511-r4379124714"
    resource: "https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379124714"
    author: "human:sagearc"
    last_modified: "2026-10-26T15:30:40Z"
  - id: "issue-2995"
    resource: "https://github.com/llm-d/llm-d-router/issues/2995"
    author: "human:elevran"
    last_modified: "2026-09-23T16:11:23Z"
  - id: "issue-2997"
    resource: "https://github.com/llm-d/llm-d-router/issues/2997"
    author: "human:elevran"
    last_modified: "2026-09-23T16:11:27Z"
  - id: "pr-3048-rev-5330221439"
    resource: "https://github.com/llm-d/llm-d-router/pull/3048#pullrequestreview-5330221439"
    author: "human:roytman"
    last_modified: "2026-09-27T12:08:22Z"
  - id: "pr-3055-r4177698311"
    resource: "https://github.com/llm-d/llm-d-router/pull/3055#discussion_r4177698311"
    author: "human:roytman"
    last_modified: "2026-10-04T12:55:41Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Error handling and boundaries

Boundary input validation, explicit error propagation without fabricated fallbacks, unwrapped status-code mapping, and scoped linter suppressions.

## Validate untrusted inputs at ingress and configuration boundaries

### Rule

Concentrate nil, type-assertion, slice-bound, integer-overflow, content-type, and payload-size checks at network, parser, and configuration boundaries before buffering bodies or constructing domain types. Include field and expected-type context in boundary errors, and close caller-supplied listeners if early setup fails.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `pkg/common/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Swallowing errors, synthesizing fallback capacities or scores, or misclassifying client vs backend failures routes traffic on corrupt telemetry and masks operational faults.

### Reviewers accept

Omitting redundant defensive nil checks inside internal helpers whose callers guarantee valid inputs.[^pr-901-r3192331180][^pr-944-r3435978187][^pr-3252-r4218084207]

### Evidence

Observed across 52 pull requests and 11 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-683-r3143190014][^pr-707-r3220172664][^pr-1147-ic-4461397416][^pr-1163-r3254240458]

## Propagate errors explicitly without fabricated fallbacks

### Rule

Return producer, parser, and configuration failures as explicit errors instead of substituting fake placeholder counts, synthetic block sizes, or silent zero defaults that corrupt downstream routing decisions, and never call `panic` on a domain or invariant error in request or flow-control packages (`forbidigo` in `.golangci.yml` enforces the ban for `pkg/epp/flowcontrol`). Fail startup when TLS certificates, metrics listeners, or required plugin bindings are invalid.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `pkg/common/**` (referenced symbols: `.golangci.yml`, `pkg/epp/flowcontrol`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Swallowing errors, synthesizing fallback capacities or scores, or misclassifying client vs backend failures routes traffic on corrupt telemetry and masks operational faults.
- **Conflict audit references**: `NVC-08`, `UNR-06`.

### Open question

See [Plugin panic isolation at the invocation boundary](../open_questions/plugin_panic_isolation.md).

### Evidence

Observed across 44 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-876][^pr-1002-r3403097625][^pr-1061-r3246024767][^pr-1895-r3615464093]

## Map unwrapped error chains to canonical status codes

### Rule

Classify errors for retry, admission, and fail-open logic with `errors.Is` and `errors.As` across unwrapped chains rather than matching error message substrings, and format error responses to match the active API surface. Map client validation faults to 4xx, upstream worker protocol faults to 502, empty pools to 503, and saturation drops to 429.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `pkg/common/**` (referenced symbols: `errors.Is`, `errors.As`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Swallowing errors, synthesizing fallback capacities or scores, or misclassifying client vs backend failures routes traffic on corrupt telemetry and masks operational faults.

### Reviewers accept

Returning 502 Bad Gateway rather than 400 Bad Request when an upstream worker returns an invalid content type or malformed response for a valid client request.[^pr-3511-r4379124714]

### Evidence

Observed across 36 pull requests and 10 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1116-ic-4450544045][^pr-1603-r3399020119][^pr-2052-r3597666475][^pr-2215-r3698734025]

## Handle non-deferred close errors and scope linter suppressions to exact rules

### Rule

Handle or explicitly discard (`_ =`) non-deferred `Close()` errors on early-return paths and `t.Cleanup` callbacks (`errcheck` applies outside the `std-error-handling` exclusions in `.golangci.yml`; `.golangci-security.yml` excludes `gosec` from test code), and never use broad `//nolint:all` directives. When a verified invariant requires a suppression, scope it to the linter (`//nolint:<linter>`) or to the single `gosec` rule (`//#nosec` followed by the rule identifier) at that call site and include a comment stating why the invariant holds.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `pkg/common/**` (referenced symbols: `t.Cleanup`, `.golangci.yml`, `.golangci-security.yml`, `//nolint:all`, `//nolint:<linter>`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Swallowing errors, synthesizing fallback capacities or scores, or misclassifying client vs backend failures routes traffic on corrupt telemetry and masks operational faults.
- **Conflict audit references**: `NVC-09`.

### Evidence

Observed across 11 pull requests and 3 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-2995][^issue-2997][^pr-3048-rev-5330221439][^pr-3055-r4177698311]

[^pr-683-r3143190014]: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143190014) (human:elevran)
[^pr-707-r3220172664]: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3220172664) (human:hexfusion)
[^pr-1147-ic-4461397416]: [#1147](https://github.com/llm-d/llm-d-router/pull/1147#issuecomment-4461397416) (human:ahg-g)
[^pr-1163-r3254240458]: [#1163](https://github.com/llm-d/llm-d-router/pull/1163#discussion_r3254240458) (human:roytman)
[^pr-901-r3192331180]: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3192331180) (human:liu-cong)
[^pr-944-r3435978187]: [#944](https://github.com/llm-d/llm-d-router/pull/944#discussion_r3435978187) (human:shmuelk)
[^pr-3252-r4218084207]: [#3252](https://github.com/llm-d/llm-d-router/pull/3252#discussion_r4218084207) (human:roytman)
[^issue-876]: [#876](https://github.com/llm-d/llm-d-router/issues/876) (human:vMaroon)
[^pr-1002-r3403097625]: [#1002](https://github.com/llm-d/llm-d-router/pull/1002#discussion_r3403097625) (human:ahg-g)
[^pr-1061-r3246024767]: [#1061](https://github.com/llm-d/llm-d-router/pull/1061#discussion_r3246024767) (human:liu-cong)
[^pr-1895-r3615464093]: [#1895](https://github.com/llm-d/llm-d-router/pull/1895#discussion_r3615464093) (human:elevran)
[^issue-1116-ic-4450544045]: [#1116](https://github.com/llm-d/llm-d-router/issues/1116#issuecomment-4450544045) (human:roytman)
[^pr-1603-r3399020119]: [#1603](https://github.com/llm-d/llm-d-router/pull/1603#discussion_r3399020119) (human:gyliu513)
[^pr-2052-r3597666475]: [#2052](https://github.com/llm-d/llm-d-router/pull/2052#discussion_r3597666475) (human:ahg-g)
[^pr-2215-r3698734025]: [#2215](https://github.com/llm-d/llm-d-router/pull/2215#discussion_r3698734025) (human:bongwoobak)
[^pr-3511-r4379124714]: [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379124714) (human:sagearc)
[^issue-2995]: [#2995](https://github.com/llm-d/llm-d-router/issues/2995) (human:elevran)
[^issue-2997]: [#2997](https://github.com/llm-d/llm-d-router/issues/2997) (human:elevran)
[^pr-3048-rev-5330221439]: [#3048](https://github.com/llm-d/llm-d-router/pull/3048#pullrequestreview-5330221439) (human:roytman)
[^pr-3055-r4177698311]: [#3055](https://github.com/llm-d/llm-d-router/pull/3055#discussion_r4177698311) (human:roytman)
