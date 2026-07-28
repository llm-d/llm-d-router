# Area taxonomy

`area/*` labels route issues and pull requests to the right reviewers by component. Each area maps to one or more directories, so it can be applied either manually (via an `/area` directive on a PR or issue body) or automatically from the paths a change touches.

| Label | Paths | Scope |
|---|---|---|
| `area/epp` | `pkg/epp/**`, `cmd/epp/**` | Endpoint Picker core |
| `area/scheduling` | `pkg/epp/scheduling/**`, `pkg/epp/requestcontrol/**` | Scheduling and request routing decisions |
| `area/flowcontrol` | `pkg/epp/flowcontrol/**` | Admission, queuing and flow control |
| `area/kvcache` | `pkg/kvcache/**`, `pkg/kvevents/**` | KV cache indexing and events |
| `area/coordinator` | `pkg/coordinator/**`, `cmd/coordinator/**` | Disaggregated inference coordinator |
| `area/sidecar` | `pkg/sidecar/**`, `cmd/pd-sidecar/**` | PD sidecar proxy |
| `area/datalayer` | `pkg/epp/datalayer/**`, `pkg/epp/datastore/**` | EPP data layer and datastore |
| `area/telemetry` | `pkg/telemetry/**`, `pkg/common/observability/**` | Telemetry and observability |
| `area/dev` | `.github/**`, `hack/**`, `scripts/**`, `build/**`, `Makefile` | Dev tooling and CI/CD |
| `area/docs` | `docs/**`, top-level `*.md` | Documentation |

A change can span multiple areas. Apply every label that fits rather than picking one.

## Applying a label

- Manually: add an `/area <name>` line to a PR or issue body (see `.github/workflows/pr-kind-label.yaml` and `.github/workflows/issue-kind-label.yaml`).
- Automatically: path based labeling from this table is tracked separately against [#956](https://github.com/llm-d/llm-d-router/issues/956).
