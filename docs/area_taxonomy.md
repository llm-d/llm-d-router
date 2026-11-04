# Area Taxonomy

`area/*` labels route issues and pull requests to the right reviewers by component. Each area maps to one or more directories, so it can be applied either manually (via an `/area` directive on a PR or issue body) or automatically from the paths a change touches.

`.github/labeler.yml` is the canonical source for the path globs each `area/*` label maps to.

A change can span multiple areas. Apply every label that fits rather than picking one. Areas nest by path rather than exclude one another. For example, a change under `pkg/epp/scheduling/**` matches both `area/epp` and `area/scheduling` and both labels apply.

## Applying a label

- Manually: add an `/area <name>` line to a PR or issue body (see `.github/workflows/pr-kind-label.yaml` and `.github/workflows/issue-kind-label.yaml`)
- By comment: a maintainer writes `/area <name>` in an issue comment (see [issue_triage.md](issue_triage.md))
- Automatically: `.github/labeler.yml` applies the matching `area/*` label(s) on every PR

## Area contacts and code owners

Per-component code owners are listed in the matching sections of [CODEOWNERS](../CODEOWNERS).

| Area | Issues | Slack | `CODEOWNERS` Section |
|---|---|---|---|
| `area/epp` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fepp) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fepp+-label%3Aarea%2Fscheduling+-label%3Aarea%2Fflowcontrol+-label%3Aarea%2Fdatalayer) | `#sig-router` | Request control and handling; EPP configuration; Default (`@llm-d/router-maintainers`) |
| `area/scheduling` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fscheduling) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fscheduling) | `#sig-router` | Request control and handling; P2P request routing; SLO + latency-prediction; Prefix cache; Tokenization |
| `area/flowcontrol` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fflowcontrol) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fflowcontrol) | `#sig-router` | Flow control framework + plugins |
| `area/kvcache` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fkvcache) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fkvcache) | `#sig-router`, `#sig-kv-disaggregation` | KV-cache subsystem |
| `area/coordinator` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fcoordinator) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fcoordinator) | `#sig-router`, `#sig-pd-disaggregation`, `#sig-agentic-inference` | Coordinator |
| `area/sidecar` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fsidecar) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fsidecar) | `#sig-router`, `#sig-pd-disaggregation` | Proxy sidecar |
| `area/datalayer` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fdatalayer) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fdatalayer) | `#sig-router` | Default (`@llm-d/router-maintainers`) |
| `area/telemetry` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Ftelemetry) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Ftelemetry) | `#sig-router`, `#sig-observability` | Default (`@llm-d/router-maintainers`) |
| `area/dev` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fdev) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fdev) | `#sig-router`, `#llm-d-dev` | Default (`@llm-d/router-maintainers`) |
| `area/docs` | [open](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fdocs) / [triage](https://github.com/llm-d/llm-d-router/issues?q=is%3Aopen+is%3Aissue+label%3Aneeds-triage+label%3Aarea%2Fdocs) | `#sig-router`, `#sig-docs` | Default (`@llm-d/router-maintainers`) |
