---
type: Review Rule Set
title: Security and trust boundaries
description: "Header namespacing and ingress/egress scrubbing, InferencePool SSRF target allowlists, TLS minimums, namespace-scoped watches, and sensitive payload redaction."
tags: ["security-and-trust-boundaries", "failure-class", "unr-01", "mvm-08", "unr-05"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "issue-979-ic-5944114560"
    resource: "https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560"
    author: "human:LukeAVanDrie"
    last_modified: "2026-10-02T01:52:22Z"
  - id: "pr-1027-r3219971477"
    resource: "https://github.com/llm-d/llm-d-router/pull/1027#discussion_r3219971477"
    author: "human:ahg-g"
    last_modified: "2026-05-11T15:07:49Z"
  - id: "pr-1161-r3264933290"
    resource: "https://github.com/llm-d/llm-d-router/pull/1161#discussion_r3264933290"
    author: "human:ahg-g"
    last_modified: "2026-05-19T08:51:38Z"
  - id: "pr-2319-r3758140767"
    resource: "https://github.com/llm-d/llm-d-router/pull/2319#discussion_r3758140767"
    author: "human:roytman"
    last_modified: "2026-08-11T12:58:38Z"
  - id: "pr-3145-ic-6016728437"
    resource: "https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6016728437"
    author: "human:ahg-g"
    last_modified: "2026-10-06T12:55:54Z"
  - id: "pr-1872-r3524966490"
    resource: "https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524966490"
    author: "human:roytman"
    last_modified: "2026-07-05T13:22:43Z"
  - id: "pr-2716-r3955566592"
    resource: "https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955566592"
    author: "human:bongwoobak"
    last_modified: "2026-09-08T07:51:30Z"
  - id: "pr-3433-r4274645712"
    resource: "https://github.com/llm-d/llm-d-router/pull/3433#discussion_r4274645712"
    author: "human:ahg-g"
    last_modified: "2026-10-14T18:23:37Z"
  - id: "pr-1145-ic-4479522771"
    resource: "https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771"
    author: "human:elevran"
    last_modified: "2026-05-18T16:05:58Z"
  - id: "pr-1857-r3508246239"
    resource: "https://github.com/llm-d/llm-d-router/pull/1857#discussion_r3508246239"
    author: "human:vMaroon"
    last_modified: "2026-07-01T18:30:17Z"
  - id: "pr-2234-r3682907408"
    resource: "https://github.com/llm-d/llm-d-router/pull/2234#discussion_r3682907408"
    author: "human:nilig"
    last_modified: "2026-07-30T13:07:13Z"
  - id: "pr-1007-r3203628868"
    resource: "https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628868"
    author: "human:elevran"
    last_modified: "2026-05-07T17:55:06Z"
  - id: "pr-1350-r3310324692"
    resource: "https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310324692"
    author: "human:roytman"
    last_modified: "2026-05-27T10:50:04Z"
  - id: "pr-2837-r4225473483"
    resource: "https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473483"
    author: "human:kaushikmitr"
    last_modified: "2026-10-09T00:24:56Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Security and trust boundaries

Header namespacing and ingress/egress scrubbing, InferencePool SSRF target allowlists, TLS minimums, namespace-scoped watches, and sensitive payload redaction.

## Namespace new router headers under `x-llm-d-` and scrub internal headers at ingress and egress

### Rule

Prefix newly introduced router-owned HTTP headers with lowercase `x-llm-d-` (such as `x-llm-d-epp-profile` in `pkg/common/request` and the routing headers in `pkg/epp/metadata`), suffix headers that carry an address and port with `-host-port`, and normalize inbound header lookups to lowercase at ingress. Upstream Gateway API Inference Extension headers (`x-gateway-destination-endpoint`) and pre-convention names (`x-data-parallel-host-port`, `x-session-token`) keep their names. Strip client-supplied internal routing headers, deprecated header aliases, and untrusted transfer fields across all routes at ingress, and scrub internal pod topology headers before returning responses to clients.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `config/rbac/**` (referenced symbols: `x-llm-d-`, `x-llm-d-epp-profile`, `pkg/common/request`, `pkg/epp/metadata`, `-host-port`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Trusting client-supplied routing headers, dialing unverified target addresses, or logging raw prompts exposes backends to header spoofing, SSRF, and credential or prompt leakage.
- **Conflict audit references**: `UNR-01`.

### Reviewers accept

Passing allowlist-validated `x-request-id` correlation headers through verbatim, documenting when body stamping overwrites client fields that take precedence over headers, and scrubbing internal headers on the Envoy fallback route through `request_headers_to_remove` when early header mutation would strip the `ScreeningHeaders` the EPP reads.[^pr-1872-r3524966490][^pr-2716-r3955566592][^pr-3433-r4274645712]

### Open question

See [Session affinity transport via Set-Cookie vs request header](../open_questions/session_affinity_cookie_vs_header.md).

### Evidence

Observed across 34 pull requests and 8 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-979-ic-5944114560][^pr-1027-r3219971477][^pr-1161-r3264933290][^pr-2319-r3758140767][^pr-3145-ic-6016728437]

## Allowlist dynamic endpoint targets and reject control characters

### Rule

Validate peer endpoint URLs and KV-cache transfer targets against `InferencePool` host-and-port allowlists across all target ports and data-parallel ranks, blocking self-pulls and unverified internal cluster addresses. Reject whitespace and carriage-return or newline control characters in header-derived targets to prevent SSRF and request smuggling.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `config/rbac/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Trusting client-supplied routing headers, dialing unverified target addresses, or logging raw prompts exposes backends to header spoofing, SSRF, and credential or prompt leakage.
- **Conflict audit references**: `MVM-08`.

### Evidence

Distilled from 4 review clusters (34 pull requests / 8 maintainers, 26 pull requests / 8 maintainers, 10 pull requests / 10 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-979-ic-5944114560][^pr-1145-ic-4479522771][^pr-1857-r3508246239][^pr-2234-r3682907408]

## Enforce TLS minimums, namespace-scoped watches, and payload redaction

### Rule

Enforce minimum TLS versions and header-read timeouts on HTTP servers, trim trailing newlines when reading mounted secret files, and scope Kubernetes informers to the configured namespace rather than cluster-wide watches. Never log raw prompts, base64 media payloads, or end-user identifiers at debug or default verbosity.

### Scope and consequence

- **Applies to**: `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `config/rbac/**`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Trusting client-supplied routing headers, dialing unverified target addresses, or logging raw prompts exposes backends to header spoofing, SSRF, and credential or prompt leakage.
- **Conflict audit references**: `UNR-05`.

### Open question

See [Identifier hashing vs cleartext in DumpState debug endpoints](../open_questions/dump_state_identifier_hashing.md).

### Evidence

Observed across 26 pull requests and 8 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1007-r3203628868][^pr-1350-r3310324692][^pr-2837-r4225473483]

[^issue-979-ic-5944114560]: [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560) (human:LukeAVanDrie)
[^pr-1027-r3219971477]: [#1027](https://github.com/llm-d/llm-d-router/pull/1027#discussion_r3219971477) (human:ahg-g)
[^pr-1161-r3264933290]: [#1161](https://github.com/llm-d/llm-d-router/pull/1161#discussion_r3264933290) (human:ahg-g)
[^pr-2319-r3758140767]: [#2319](https://github.com/llm-d/llm-d-router/pull/2319#discussion_r3758140767) (human:roytman)
[^pr-3145-ic-6016728437]: [#3145](https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6016728437) (human:ahg-g)
[^pr-1872-r3524966490]: [#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524966490) (human:roytman)
[^pr-2716-r3955566592]: [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955566592) (human:bongwoobak)
[^pr-3433-r4274645712]: [#3433](https://github.com/llm-d/llm-d-router/pull/3433#discussion_r4274645712) (human:ahg-g)
[^pr-1145-ic-4479522771]: [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771) (human:elevran)
[^pr-1857-r3508246239]: [#1857](https://github.com/llm-d/llm-d-router/pull/1857#discussion_r3508246239) (human:vMaroon)
[^pr-2234-r3682907408]: [#2234](https://github.com/llm-d/llm-d-router/pull/2234#discussion_r3682907408) (human:nilig)
[^pr-1007-r3203628868]: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628868) (human:elevran)
[^pr-1350-r3310324692]: [#1350](https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310324692) (human:roytman)
[^pr-2837-r4225473483]: [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473483) (human:kaushikmitr)
