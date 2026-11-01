---
type: Review Rule Set
title: Observability
description: "Canonical metric subsystems and bounded label cardinality, single terminal request outcomes, package-scoped OpenTelemetry spans, and guarded debug logs."
tags: [observability, "failure-class", "mvm-20", "unr-08", "mvm-21", "nvc-13"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - { id: "pr-707-r3190768619", resource: "https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190768619", author: "human:roytman", last_modified: "2026-05-05T18:42:55Z" }
  - { id: "pr-793-r3324344552", resource: "https://github.com/llm-d/llm-d-router/pull/793#discussion_r3324344552", author: "human:ahg-g", last_modified: "2026-05-29T12:36:04Z" }
  - { id: "issue-1089-ic-4703238704", resource: "https://github.com/llm-d/llm-d-router/issues/1089#issuecomment-4703238704", author: "human:gyliu513", last_modified: "2026-06-14T22:12:32Z" }
  - { id: "pr-1429-rev-4493962366", resource: "https://github.com/llm-d/llm-d-router/pull/1429#pullrequestreview-4493962366", author: "human:liu-cong", last_modified: "2026-06-15T01:25:51Z" }
  - { id: "issue-1553", resource: "https://github.com/llm-d/llm-d-router/issues/1553", author: "human:liu-cong", last_modified: "2026-06-08T23:54:33Z" }
  - { id: "pr-1658-r3618566988", resource: "https://github.com/llm-d/llm-d-router/pull/1658#discussion_r3618566988", author: "human:liu-cong", last_modified: "2026-07-21T00:39:19Z" }
  - { id: "pr-1677-r3852749939", resource: "https://github.com/llm-d/llm-d-router/pull/1677#discussion_r3852749939", author: "human:roytman", last_modified: "2026-08-25T12:03:57Z" }
  - { id: "issue-708", resource: "https://github.com/llm-d/llm-d-router/issues/708", author: "human:gyliu513", last_modified: "2026-03-11T15:30:42Z" }
  - { id: "pr-922-r3178210848", resource: "https://github.com/llm-d/llm-d-router/pull/922#discussion_r3178210848", author: "human:ahg-g", last_modified: "2026-05-03T13:55:18Z" }
  - { id: "pr-1513-r3369205741", resource: "https://github.com/llm-d/llm-d-router/pull/1513#discussion_r3369205741", author: "human:shmuelk", last_modified: "2026-06-07T11:05:41Z" }
  - { id: "pr-891-r3189062435", resource: "https://github.com/llm-d/llm-d-router/pull/891#discussion_r3189062435", author: "human:elevran", last_modified: "2026-05-05T14:06:21Z" }
  - { id: "pr-901-r3252750496", resource: "https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252750496", author: "human:vMaroon", last_modified: "2026-05-16T11:43:30Z" }
  - { id: "pr-972-r3178428969", resource: "https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178428969", author: "human:roytman", last_modified: "2026-05-03T16:45:08Z" }
  - { id: "pr-2319-ic-5265371475", resource: "https://github.com/llm-d/llm-d-router/pull/2319#issuecomment-5265371475", author: "human:elevran", last_modified: "2026-08-12T10:15:46Z" }
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Observability

Canonical metric subsystems and bounded label cardinality, single terminal request outcomes, package-scoped OpenTelemetry spans, and guarded debug logs.

## Register metrics under canonical component subsystems with bounded label cardinality

### Rule

Register metrics under the component's canonical subsystem constant (`LLMDRouterEndpointPickerSubsystem` / `"llm_d_epp"`, `LLMDRouterCoordinatorSubsystem` / `"llm_d_coordinator"`, or the `llm_d_disagg_sidecar_` prefix; see [docs/metrics.md](../../metrics.md)), attach `plugin_name` and `plugin_type` labels to plugin-scoped metrics, and emit series only for configured plugins. Bound caller-influenced or pod-churn labels using closed enums, allowlists, or overflow buckets, expose live plugin state through scrape-time collectors, and delete series for removed endpoints so scaled-down pods do not leak stale metrics.

### Scope and consequence

- **Applies to**: `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Unbounded label values, duplicate terminal request observations, or unguarded per-endpoint debug logs exhaust Prometheus series limits and skew latency percentiles.
- **Related decisions**: [Disaggregation sidecar serves /metrics on its existing HTTP listener mux](../decisions/sidecar_metrics_on_shared_http_mux.md).

### Reviewers accept

Emitting per-request ratio histograms alongside raw query and hit counters when the precomputed distribution simplifies operator alerting.[^pr-1429-rev-4493962366]

### Open question

See [Multi-tenant cost attribution on Prometheus labels vs logs and traces](../open_questions/cost_attribution_prometheus_labels.md).

### Evidence

Observed across 57 pull requests and 12 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-707-r3190768619][^pr-793-r3324344552][^issue-1089-ic-4703238704]

## Record at most one terminal outcome per request and keep latency histograms unbiased

### Rule

Emit at most one terminal outcome observation per request across early-return, buffered, and error paths, matching histogram bucket units to emitted values. Separate router processing overhead from backend inference wait, and skip stage duration observations when a stage is bypassed so sub-microsecond no-ops do not skew low buckets.

### Scope and consequence

- **Applies to**: `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Unbounded label values, duplicate terminal request observations, or unguarded per-endpoint debug logs exhaust Prometheus series limits and skew latency percentiles.

### Evidence

Observed across 37 pull requests and 11 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1553][^pr-1658-r3618566988][^pr-1677-r3852749939]

## Acquire package-scoped OTel tracers and propagate spans end to end

### Rule

Acquire OpenTelemetry tracers via `tracing.Tracer(TracerScope)` using a package-scoped `TracerScope` constant (`llm-d.ai/...`), standard semantic-convention attribute keys, and route templates rather than raw URLs. Extract inbound W3C trace context at ingress and propagate spans across all outbound HTTP and gRPC calls.

### Scope and consequence

- **Applies to**: `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md` (referenced symbols: `tracing.Tracer(TracerScope)`, `llm-d.ai/...`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Unbounded label values, duplicate terminal request observations, or unguarded per-endpoint debug logs exhaust Prometheus series limits and skew latency percentiles.
- **Related decisions**: [Ratio-sampled per-plugin OpenTelemetry spans with high-overhead KV-event spans gated off](../decisions/sampled_per_plugin_otel_spans_and_opt_in_kv_event_spans.md).

### Evidence

Observed across 37 pull requests and 10 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-708][^pr-922-r3178210848][^pr-1513-r3369205741]

## Guard per-item debug loops and cap stateful plugin dumps

### Rule

Follow the `go-logr` verbosity conventions in [AGENTS.md](../../../AGENTS.md#logging), guarding per-candidate and per-block debug loops behind `v.Enabled()`. Summarize collections by count in `String()`, and implement `DumpState` only on stateful plugins with sanitized entries capped to fixed top-K bounds.

### Scope and consequence

- **Applies to**: `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md` (referenced symbols: `v.Enabled()`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Unbounded label values, duplicate terminal request observations, or unguarded per-endpoint debug logs exhaust Prometheus series limits and skew latency percentiles.
- **Conflict audit references**: `NVC-13`.

### Reviewers accept

Omitting debug log statements on routine plugin no-op branches when sibling plugins are silent and proxy header traces already record the state.[^pr-2319-ic-5265371475]

### Evidence

Observed across 57 pull requests and 10 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-891-r3189062435][^pr-901-r3252750496][^pr-972-r3178428969]

[^pr-707-r3190768619]: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190768619) (human:roytman)
[^pr-793-r3324344552]: [#793](https://github.com/llm-d/llm-d-router/pull/793#discussion_r3324344552) (human:ahg-g)
[^issue-1089-ic-4703238704]: [#1089](https://github.com/llm-d/llm-d-router/issues/1089#issuecomment-4703238704) (human:gyliu513)
[^pr-1429-rev-4493962366]: [#1429](https://github.com/llm-d/llm-d-router/pull/1429#pullrequestreview-4493962366) (human:liu-cong)
[^issue-1553]: [#1553](https://github.com/llm-d/llm-d-router/issues/1553) (human:liu-cong)
[^pr-1658-r3618566988]: [#1658](https://github.com/llm-d/llm-d-router/pull/1658#discussion_r3618566988) (human:liu-cong)
[^pr-1677-r3852749939]: [#1677](https://github.com/llm-d/llm-d-router/pull/1677#discussion_r3852749939) (human:roytman)
[^issue-708]: [#708](https://github.com/llm-d/llm-d-router/issues/708) (human:gyliu513)
[^pr-922-r3178210848]: [#922](https://github.com/llm-d/llm-d-router/pull/922#discussion_r3178210848) (human:ahg-g)
[^pr-1513-r3369205741]: [#1513](https://github.com/llm-d/llm-d-router/pull/1513#discussion_r3369205741) (human:shmuelk)
[^pr-891-r3189062435]: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3189062435) (human:elevran)
[^pr-901-r3252750496]: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252750496) (human:vMaroon)
[^pr-972-r3178428969]: [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178428969) (human:roytman)
[^pr-2319-ic-5265371475]: [#2319](https://github.com/llm-d/llm-d-router/pull/2319#issuecomment-5265371475) (human:elevran)
