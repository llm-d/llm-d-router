# Area taxonomy

`area/*` labels route issues and pull requests to the right reviewers by component. Each area maps to one or more directories, so it can be applied either manually (via an `/area` directive on a PR or issue body) or automatically from the paths a change touches.

`area/epp` and `area/dev` are labeled automatically by `.github/labeler.yml`, which is the canonical source for their path globs.

| Label | Paths | Scope |
|---|---|---|
| `area/scheduling` | `pkg/epp/scheduling/**`, `pkg/epp/framework/plugins/scheduling/**`, `pkg/epp/requestcontrol/**`, `pkg/epp/framework/plugins/requestcontrol/**` | Scheduling and request routing decisions |
| `area/flowcontrol` | `pkg/epp/flowcontrol/**`, `pkg/epp/framework/plugins/flowcontrol/**` | Admission, queuing and flow control |
| `area/kvcache` | `pkg/kvcache/**`, `pkg/kvevents/**` | KV cache indexing and events |
| `area/coordinator` | `pkg/coordinator/**`, `cmd/coordinator/**`, `test/coordinator/**` | Disaggregated inference coordinator |
| `area/sidecar` | `pkg/sidecar/**`, `cmd/pd-sidecar/**`, `test/sidecar/**` | PD sidecar proxy |
| `area/datalayer` | `pkg/epp/datalayer/**`, `pkg/epp/framework/plugins/datalayer/**`, `pkg/epp/datastore/**` | EPP data layer and datastore |
| `area/telemetry` | `pkg/common/observability/**` | Telemetry and observability |
| `area/docs` | `docs/**`, top-level `*.md` | Documentation |

These eight labels do not exist in the repository yet. Once created, their glob blocks move into `.github/labeler.yml` alongside `area/epp` and `area/dev`, and their rows drop from this table.

A change can span multiple areas. Apply every label that fits rather than picking one. Areas nest by path rather than exclude one another: a change under `pkg/epp/scheduling/**` matches both `area/epp` and `area/scheduling` and both labels apply.

## Applying a label

- Manually: add an `/area <name>` line to a PR or issue body (see `.github/workflows/pr-kind-label.yaml` and `.github/workflows/issue-kind-label.yaml`).
- Automatically: `area/epp` and `area/dev` are applied from `.github/labeler.yml` on every PR. The remaining eight areas above are pending label creation, tracked against [#956](https://github.com/llm-d/llm-d-router/issues/956).
