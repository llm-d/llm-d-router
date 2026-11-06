---
type: Reference
title: Failure classes and section taxonomy
description: "Taxonomy of the 4 Stage-1 review playbook sections (External wire and engine contracts, Unencoded runtime hazards, Governance and compatibility policy, and Settled architectural boundaries) with 17 non-code rules."
tags: [taxonomy, "failure-classes", reference]
status: stable
---

# Section taxonomy

The review bundle organizes 17 `When:`-triggered non-code review invariants across 4 compact Stage-1 rule files distilled from 1809 pull requests (6824 maintainer pull request items) and 1090 issues (1575 maintainer issue items) between `2026-04-26` and `2026-10-26`, after excluding contracts inferable from repository code search, [AGENTS.md](../../../AGENTS.md), [docs/architecture.md](../../architecture.md), `.golangci.yml`, and `Makefile`.

| Order | Section | Kind | Entries | Bundle file | Defect and review focus |
| ---: | --- | --- | ---: | --- | --- |
| 1 | External wire and engine contracts | `failure-class` | 4 | [`rules/external_wire_and_engine_contracts.md`](../rules/external_wire_and_engine_contracts.md) | Anthropic additive cache tokens, vLLM `continuous_usage_stats`, JSON passthrough, `Content-Length` invalidation, RFC 9110 `Connection: upgrade`, native engine `/render` passthrough, stateful `/v1/responses` rejection, and ZMQ KV-event sequence recovery. |
| 2 | Unencoded runtime hazards | `failure-class` | 4 | [`rules/unencoded_runtime_hazards.md`](../rules/unencoded_runtime_hazards.md) | Unjoined timed-out `DataProducer` goroutines and `sync.Pool` buffer lifetimes, Envoy `ext_proc` early header-skip latency metrics and decode-pod filtering, stateless `ParseResponse` vs `mergeUsage` and `ResponseWriter` `Unwrap`, and `(host, port, rank)` endpoint keys. |
| 3 | Governance and compatibility policy | `process` | 4 | [`rules/governance_and_compatibility_policy.md`](../rules/governance_and_compatibility_policy.md) | `N+2` deprecation shims for shipped surfaces vs zero shims for unshipped surfaces, plugin `Alpha`/`Beta`/`Stable` graduation and frozen `v1` fixtures, holding PRs coupled to unmerged engine contracts, versioned GIE CRDs, and `config/charts/` Helm vs `deploy/` Kustomize. |
| 4 | Settled architectural boundaries | `direction` | 5 | [`rules/settled_architectural_boundaries.md`](../rules/settled_architectural_boundaries.md) | Overload shedding in `SaturationDetector`/`AdmissionPlugin`, single base model and role per `InferencePool`, external conversation history vs bounded routing state, `pkg/coordinator` vs `pkg/sidecar` responsibilities, and ingress-only header stripping. |

# Section kinds

- **`failure-class` (2 sections, 8 rules)**: External wire contracts and non-local runtime execution hazards that cannot be inferred from local Go signatures alone.
- **`process` (1 section, 4 rules)**: Release compatibility windows, plugin tier graduation, upstream coupling policy, and shipped Helm packaging boundaries.
- **`direction` (1 section, 5 statements)**: Maintainer-settled architectural boundaries, explicit `- Accept:` exceptions, and `- Open:` fault lines across router components.
