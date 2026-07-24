# Plugins

This directory contains the available plugins. For detailed information on individual plugins, please refer to the README.md located within each plugin's respective directory. To understand how they integrate into the incoming request processing lifecycle, see the [Endpoint Picker (EPP) design](https://github.com/llm-d/llm-d/tree/main/docs/architecture/core/router/epp) document.

## Plugin Stability Levels

Every plugin in `llm-d-router` is assigned a **Stability Level** upon registration (`cmd/epp/runner/runner.go` is the single source of truth for in-tree plugin stability):

| Stability Level | Lifecycle & Backwards Compatibility Guarantees | Feature Gate Requirement |
|---|---|---|
| **Alpha** | Experimental features under active development. No backwards-compatibility guarantees (parameters or behavior may change anytime). | Requires `experimentalPlugins` feature gate enabled. |
| **Beta** | Feature-complete and enabled by default. Backwards-compatible within current version; subject to a +2 minor version deprecation policy before removal. | Allowed by default (no feature gate required). |
| **Stable** | Production-grade and fully backwards-compatible across minor releases. Breaking changes only on major version bumps. | Allowed by default (no feature gate required). |

### Alpha Feature Gate (`experimentalPlugins`)

To ensure experimental plugins are only enabled intentionally, Alpha plugins require enabling the `experimentalPlugins` feature gate in the EPP configuration:

```yaml
featureGates:
  - experimentalPlugins
```

If an Alpha plugin is configured while `experimentalPlugins` feature gate is not enabled (the default), the EPP runner will fail initialization with an explicit error.

## Related Documentation

- [Architecture Overview](../../../../docs/architecture.md)
- [Creating a new Filter guide](../../../../docs/create_new_filter.md)

