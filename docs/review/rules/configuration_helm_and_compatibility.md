# Configuration, Helm, and compatibility

## Provide a two-release deprecation window for shipped user-facing surfaces
Support shipped CLI flags, configuration fields, plugin types, HTTP headers, and metrics across at least two releases (`N+2`) before removal. Log a startup warning when a deprecated surface is used, migrate all in-tree manifests and tests to the replacement immediately, and separate structural moves from renames.
- Accept: Renaming or removing unshipped config keys, unreleased Helm charts, and internal headers without deprecation shims, or deleting shims after `N+2`.

## Decode configuration with strict unknown-field rejection and startup validation
Seed config structs from canonical default constructors, decode through `fwkplugin.StrictDecoder` (`DisallowUnknownFields()`), `yaml.UnmarshalStrict`, or `serializer.EnableStrict`, and validate cross-field and mode-scoped invariants before serving traffic. Prefer explicit mode enums over boolean flag combinations, and use pointer numeric fields with `omitempty` only when distinguishing unset defaults from explicit zero.

## Ship safe out-of-the-box defaults and gate workload-sensitive features behind opt-in config
Choose defaults that run within standard CPU and memory budgets without external sidecars or cross-component boot-order dependencies. Gate workload-sensitive heuristics, speculative estimators, and outbound payload rewrites behind opt-in config so disabled features leave the default request path untouched.

## Deduplicate Helm chart defaults through shared library templates
Share common Helm logic through library templates while declaring values only in charts that render them, fail template rendering with `fail` on removed keys, and keep lifecycle hooks compatible with shell-less distroless images.
- Accept: Repeating stable Kubernetes API group literals directly in Helm templates and gating lifecycle features on supported Kubernetes versions with docs rather than shell fallbacks.

## Preserve lossless conversion across multi-version CRDs
Serve multi-version CRDs through lossless conversion under a single storage version with codegen validation markers, updating `config/charts`, `deploy`, and test manifests in the same change.
