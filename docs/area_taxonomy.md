# Area Taxonomy

`area/*` labels route issues and pull requests to the right reviewers by component. Each area maps to one or more directories, so it can be applied either manually (via an `/area` directive on a PR or issue body) or automatically from the paths a change touches.

`.github/labeler.yml` is the canonical source for the path globs each `area/*` label maps to.

A change can span multiple areas. Apply every label that fits rather than picking one. Areas nest by path rather than exclude one another. For example, a change under `pkg/epp/scheduling/**` matches both `area/epp` and `area/scheduling` and both labels apply.

## Applying a label

- Manually: add an `/area <name>` line to a PR or issue body (see `.github/workflows/pr-kind-label.yaml` and `.github/workflows/issue-kind-label.yaml`)
- Automatically: `.github/labeler.yml` applies the matching `area/*` label(s) on every PR
