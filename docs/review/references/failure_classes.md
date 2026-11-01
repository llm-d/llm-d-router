---
type: Reference
title: Failure classes and section taxonomy
description: "Taxonomy of the 12 review playbook sections (10 technical failure classes, Scope and process, and Project direction) with rule counts and concept file mappings."
tags: [taxonomy, "failure-classes", reference]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Section taxonomy

The review bundle organizes 58 normative entries across 12 section files distilled from 1809 pull requests (6824 maintainer pull request items) and 1090 issues (1575 maintainer issue items) between `2026-04-26` and `2026-10-26`.

| Order | Section | Kind | Entries | Bundle file | Defect and review focus |
| ---: | --- | --- | ---: | --- | --- |
| 1 | Scope and process | `process` | 4 | [`rules/scope_and_process.md`](../rules/scope_and_process.md) | Pull request slicing, public design issues, release-note hygiene, and fork-branch review iteration. |
| 2 | Plugin framework contracts | `failure-class` | 6 | [`rules/plugin_framework_contracts.md`](../rules/plugin_framework_contracts.md) | Core-vs-plugin import boundaries, extension-point separation, typed data-layer slots, factory validation, naming, and dead-code removal. |
| 3 | Project direction | `direction` | 8 | [`rules/project_direction.md`](../rules/project_direction.md) | Architectural boundaries across built-in runtimes, upstream signal producers, engine adapters, orchestrator-neutral endpoints, pool homogeneity, coordinator vs sidecar, Helm packaging, and upstream CRDs. |
| 4 | Configuration, Helm, and compatibility | `failure-class` | 5 | [`rules/configuration_helm_and_compatibility.md`](../rules/configuration_helm_and_compatibility.md) | Two-release deprecation windows, strict unknown-field decoding, safe defaults, shared Helm library templates, and lossless CRD conversion. |
| 5 | Concurrency and shared state | `failure-class` | 6 | [`rules/concurrency_and_shared_state.md`](../rules/concurrency_and_shared_state.md) | Mutex critical sections, bounded state with endpoint deletion cleanup, edge-triggered priority bands, drain-scoped contexts, per-replica state sync, and KV-event sequence recovery. |
| 6 | Error handling and boundaries | `failure-class` | 4 | [`rules/error_handling_and_boundaries.md`](../rules/error_handling_and_boundaries.md) | Boundary input validation, explicit error propagation without fabricated fallbacks, unwrapped status-code mapping, and scoped linter suppressions. |
| 7 | Protocol parsing and wire fidelity | `failure-class` | 6 | [`rules/protocol_parsing_and_wire_fidelity.md`](../rules/protocol_parsing_and_wire_fidelity.md) | Passthrough JSON fidelity, declarative route claims, complete SSE frame buffering, multi-API multimodal parity, host-port-rank endpoint keys, and disaggregated stage ordering. |
| 8 | Performance on the request path | `failure-class` | 5 | [`rules/performance_on_the_request_path.md`](../rules/performance_on_the_request_path.md) | Zero avoidable hot-path allocations, pooled transports and multiplexed event sockets, orthogonal load signals, contiguous prefix-cache scoring, and reproducible benchmarks. |
| 9 | Observability | `failure-class` | 4 | [`rules/observability.md`](../rules/observability.md) | Canonical metric subsystems and bounded label cardinality, single terminal request outcomes, package-scoped OpenTelemetry spans, and guarded debug logs. |
| 10 | Testing and CI | `failure-class` | 4 | [`rules/testing_and_ci.md`](../rules/testing_and_ci.md) | Production factory tests with exact output assertions, colocated sleep-free unit tests, namespace-isolated E2E teardown, and presubmit CI parity. |
| 11 | Security and trust boundaries | `failure-class` | 3 | [`rules/security_and_trust_boundaries.md`](../rules/security_and_trust_boundaries.md) | Header namespacing and ingress/egress scrubbing, InferencePool SSRF target allowlists, TLS minimums, namespace-scoped watches, and sensitive payload redaction. |
| 12 | Documentation and comments | `failure-class` | 3 | [`rules/documentation_and_comments.md`](../rules/documentation_and_comments.md) | Colocated plugin READMEs, review-round documentation synchronization, and non-obvious invariant comments on locks, aliasing, and engine quirks. |

# Section kinds

- **`failure-class` (10 sections, 46 rules)**: Recurring defect categories where maintainer reviews block merge until fixed or bounded by an explicit `Reviewers accept` clause.
- **`process` (1 section, 4 rules)**: Pull request slicing, public design issue alignment, release-note accuracy, and fork-branch review hygiene.
- **`direction` (1 section, 8 statements)**: Architectural invariants governing where new capabilities live across core runtimes, plugins, the coordinator, the disaggregation sidecar, and Helm charts.
