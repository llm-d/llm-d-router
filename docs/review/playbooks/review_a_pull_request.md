# Review a Pull Request

## Procedure

1. Follow [AGENTS.md](../../../AGENTS.md) for repository-wide coding, prose, logging, constant-reuse, and pull request rules.
2. Always load [scope_and_process.md](../rules/scope_and_process.md), then match `git diff --name-only` against `# Diff-path to rule-section routing` and load only matching rule files under `docs/review/rules/`.
3. Never load `docs/review_playbook.md` or `docs/review/provenance.json`, and do not glob `docs/review/**` or open unreferenced files under `decisions/` or `open_questions/`.
4. Do not flag patterns allowed by a rule's `- Accept:` line, and do not enforce unsettled tradeoffs on `- Open:` lines as policy (surface them at most as a neutral note).
5. Cite `docs/review/rules/<section>.md#<rule-anchor>` when reporting a rule violation.

## Diff-path to rule-section routing

| Touched repository paths | Rule files to load |
| --- | --- |
| `cmd/epp/**`, `pkg/epp/datastore/**`, `pkg/epp/framework/interface/**` | [plugin_framework_contracts.md](../rules/plugin_framework_contracts.md), [project_direction.md](../rules/project_direction.md), [concurrency_and_shared_state.md](../rules/concurrency_and_shared_state.md) |
| `pkg/epp/framework/plugins/**` | [plugin_framework_contracts.md](../rules/plugin_framework_contracts.md), [configuration_helm_and_compatibility.md](../rules/configuration_helm_and_compatibility.md), [performance_on_the_request_path.md](../rules/performance_on_the_request_path.md), [documentation_and_comments.md](../rules/documentation_and_comments.md) |
| `pkg/epp/framework/plugins/requesthandling/**`, `pkg/common/request/**` | [protocol_parsing_and_wire_fidelity.md](../rules/protocol_parsing_and_wire_fidelity.md), [error_handling_and_boundaries.md](../rules/error_handling_and_boundaries.md), [security_and_trust_boundaries.md](../rules/security_and_trust_boundaries.md) |
| `pkg/epp/scheduling/**`, `pkg/epp/flowcontrol/**`, `pkg/epp/statesync/**`, `pkg/kvcache/**`, `pkg/kvevents/**` | [concurrency_and_shared_state.md](../rules/concurrency_and_shared_state.md), [performance_on_the_request_path.md](../rules/performance_on_the_request_path.md), [error_handling_and_boundaries.md](../rules/error_handling_and_boundaries.md) |
| `cmd/coordinator/**`, `pkg/coordinator/**`, `cmd/pd-sidecar/**`, `pkg/sidecar/**` | [project_direction.md](../rules/project_direction.md), [protocol_parsing_and_wire_fidelity.md](../rules/protocol_parsing_and_wire_fidelity.md), [security_and_trust_boundaries.md](../rules/security_and_trust_boundaries.md), [error_handling_and_boundaries.md](../rules/error_handling_and_boundaries.md) |
| `apix/**`, `client-go/**`, `config/charts/**`, `config/crd/**`, `config/manifests/**`, `deploy/**` | [configuration_helm_and_compatibility.md](../rules/configuration_helm_and_compatibility.md), [project_direction.md](../rules/project_direction.md), [security_and_trust_boundaries.md](../rules/security_and_trust_boundaries.md) |
| `pkg/*/metrics/**`, `pkg/common/observability/**`, `config/observability/**`, `docs/metrics*.md`, `docs/telemetry.md` | [observability.md](../rules/observability.md), [security_and_trust_boundaries.md](../rules/security_and_trust_boundaries.md) |
| `test/**`, `**/*_test.go`, `Makefile*`, `hack/**`, `scripts/**`, `.github/**` | [testing_and_ci.md](../rules/testing_and_ci.md), [scope_and_process.md](../rules/scope_and_process.md) |
| `docs/**`, `**/README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md` | [documentation_and_comments.md](../rules/documentation_and_comments.md), [scope_and_process.md](../rules/scope_and_process.md) |
