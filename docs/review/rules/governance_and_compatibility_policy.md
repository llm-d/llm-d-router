# Governance and compatibility policy

## N+2 deprecation for shipped surfaces and zero shims for unshipped surfaces
When renaming or removing CLI flags, config fields, plugin types, HTTP headers, or metrics: keep `N+2` two-release deprecation shims with startup warnings for surfaces shipped in a prior release tag while migrating in-tree configs and tests immediately, and modify or delete unshipped post-tag surfaces with zero deprecation shim.

## Plugin Alpha, Beta, and Stable graduation and frozen v1 fixtures
When registering plugins, changing plugin parameters, or editing `test/testdata/plugins/stable/v1/`: register new plugins at `Alpha` (`--allow-experimental-plugins`, no compat guarantee), preserve two minor versions at `Beta`, and freeze `Stable` schemas in `test/testdata/plugins/stable/v1/` without editing existing fixtures for breaking changes.

## Unmerged upstream engine contracts and versioned GIE CRDs
When targeting backend engine flags or endpoints, or updating GIE CRDs and protos: hold or close router PRs coupled to unmerged upstream engine flags or endpoints, and consume versioned upstream GIE CRDs and protos without vendoring (`test/e2e/artifacts/components/crds-istio` is the sole local-dev exception).
- Open: Whether unmodified upstream test CRDs should be fetched remotely on each run or cached in-repo.

## Helm charts in config/charts/ and local Kustomize in test/e2e/artifacts/
When changing deployment manifests or component flags under `config/charts/` or `test/e2e/artifacts/`: ship end-user deployment surfaces solely via `config/charts/` Helm charts and structured config, keeping `test/e2e/artifacts/` Kustomize for local kind and E2E only.
- Accept: Updating `test/e2e/artifacts/` Kustomize manifests for local kind and E2E suites.
