---
type: Playbook
title: Review a pull request
description: "Agent and human procedure for loading targeted review rule sets by diff path, checking Reviewers accept bounds, and citing canonical concept anchors."
tags: [playbook, "code-review", routing]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Procedure

1. **Check scope and metadata first**: Load [`rules/scope_and_process.md`](../rules/scope_and_process.md) for every pull request to verify mechanical-move separation, issue-closing keyword scope (`Fixes #` vs `Part of #`), and `release-note` accuracy.
2. **Route changed file paths to rule sections**: Match `git diff --name-only` against the path table in `# Diff-path to rule-section routing` and load only the matching `Review Rule Set` files under `rules/` for touched subsystems rather than reading all 12 section files at once.
3. **Check `### Reviewers accept` before flagging**: Every documented exception won in prior reviews is recorded under `### Reviewers accept` on the rule block (for example, [Provide a two-release deprecation window for shipped user-facing surfaces](../rules/configuration_helm_and_compatibility.md#provide-a-two-release-deprecation-window-for-shipped-user-facing-surfaces) allows removing unshipped keys without shims, and [Session affinity as a shared stateless helper](../decisions/session_affinity_shared_helper.md) documents when a shared helper replaces a `DataProducer`).
4. **Check `open_questions/` before blocking on unsettled policy**: If a rule links to an `### Open question` concept with `status: draft` (such as [Plugin panic isolation at the invocation boundary](../open_questions/plugin_panic_isolation.md)), surface the open tradeoff and link the tracking issue rather than enforcing one unratified side as settled policy.
5. **Cite the concept permalink**: Link directly to `docs/review/rules/<section>.md#<rule-anchor>` (or the rendered section in `docs/review_playbook.md`) when posting a review comment.

# Diff-path to rule-section routing

Derived from package, flag, manifest, and workflow references across the 12 sections in [Failure classes and section taxonomy](../references/failure_classes.md):

| Touched repository paths | Primary rule files to load | Additional checks |
| --- | --- | --- |
| `cmd/epp/**`, `pkg/epp/datastore/**`, `pkg/epp/framework/interface/**` | [`rules/plugin_framework_contracts.md`](../rules/plugin_framework_contracts.md), [`rules/project_direction.md`](../rules/project_direction.md), [`rules/concurrency_and_shared_state.md`](../rules/concurrency_and_shared_state.md) | Verify zero imports of concrete packages under `pkg/epp/framework/plugins/**`. |
| `pkg/epp/framework/plugins/**` | [`rules/plugin_framework_contracts.md`](../rules/plugin_framework_contracts.md), [`rules/configuration_helm_and_compatibility.md`](../rules/configuration_helm_and_compatibility.md), [`rules/performance_on_the_request_path.md`](../rules/performance_on_the_request_path.md), [`rules/documentation_and_comments.md`](../rules/documentation_and_comments.md) | Check `fwkplugin.StrictDecoder`, stability tier registration, and colocated `README.md`. |
| `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**` | [`rules/protocol_parsing_and_wire_fidelity.md`](../rules/protocol_parsing_and_wire_fidelity.md), [`rules/error_handling_and_boundaries.md`](../rules/error_handling_and_boundaries.md), [`rules/security_and_trust_boundaries.md`](../rules/security_and_trust_boundaries.md) | Check unknown JSON field preservation, SSE frame buffering, and multimodal parity. |
| `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**` | [`rules/concurrency_and_shared_state.md`](../rules/concurrency_and_shared_state.md), [`rules/performance_on_the_request_path.md`](../rules/performance_on_the_request_path.md), [`rules/error_handling_and_boundaries.md`](../rules/error_handling_and_boundaries.md) | Check lock scope across I/O, capacity bounds, `ctx.Err()` drain, and benchmark gates. |
| `cmd/coordinator/**`, `pkg/coordinator/**`, `cmd/pd-sidecar/**`, `pkg/sidecar/**` | [`rules/project_direction.md`](../rules/project_direction.md), [`rules/protocol_parsing_and_wire_fidelity.md`](../rules/protocol_parsing_and_wire_fidelity.md), [`rules/security_and_trust_boundaries.md`](../rules/security_and_trust_boundaries.md), [`rules/error_handling_and_boundaries.md`](../rules/error_handling_and_boundaries.md) | Check stage ordering, SSRF allowlists, header scrubbing, and status-code mapping. |
| `apix/**`, `client-go/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `deploy/**` | [`rules/configuration_helm_and_compatibility.md`](../rules/configuration_helm_and_compatibility.md), [`rules/project_direction.md`](../rules/project_direction.md), [`rules/security_and_trust_boundaries.md`](../rules/security_and_trust_boundaries.md) | Check `N+2` deprecation windows, shared Helm library templates, and CRD conversion. |
| `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md` | [`rules/observability.md`](../rules/observability.md), [`rules/security_and_trust_boundaries.md`](../rules/security_and_trust_boundaries.md) | Check subsystem constants, label cardinality caps, `V(logging.DEFAULT)` verbosity, and OTel spans. |
| `test/**`, `**/*_test.go`, `Makefile*`, `hack/**`, `scripts/**`, `.github/**` | [`rules/testing_and_ci.md`](../rules/testing_and_ci.md), [`rules/scope_and_process.md`](../rules/scope_and_process.md) | Check factory-based tests, zero new `time.Sleep`, namespace isolation, and `make presubmit` wiring. |
| `docs/**`, `**/README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md` | [`rules/documentation_and_comments.md`](../rules/documentation_and_comments.md), [`rules/scope_and_process.md`](../rules/scope_and_process.md) | Check ASCII-only atemporal prose (`AGENTS.md`), colocated plugin docs, and `make verify-review-playbook`. |
