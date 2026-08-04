# Plugins

This directory contains the available plugins. For detailed information on individual plugins, please refer to the README.md located within each plugin's respective directory. To understand how they integrate into the incoming request processing lifecycle, see the [Endpoint Picker (EPP) design](https://github.com/llm-d/llm-d/tree/main/docs/architecture/core/router/epp) document.

## Plugin Stability Levels

Every plugin in `llm-d-router` is assigned a **Stability Level** upon registration (`cmd/epp/runner/runner.go` is the single source of truth for in-tree plugin stability):

| Stability Level | Lifecycle & Backwards Compatibility Guarantees | Command Line Flag Requirement |
|---|---|---|
| **Alpha** | Experimental features under active development. No backwards-compatibility guarantees (parameters or behavior may change anytime). | Requires `--allow-experimental-plugins` CLI flag. |
| **Beta** | Feature-complete and enabled by default. Backwards-compatible within current version; subject to a +2 minor version deprecation policy before removal. | Allowed by default (no CLI flag required). |
| **Stable** | Production-grade and fully backwards-compatible across minor releases. Breaking changes only on major version bumps. | Allowed by default (no CLI flag required). |

> [!NOTE]
> Currently, all in-tree plugins are classified as **Alpha** or **Beta**. Plugins will be promoted to **Stable** as the project approaches its 1.0 release.

### Alpha Plugin CLI Flag (`--allow-experimental-plugins`)

To ensure experimental plugins are only enabled intentionally, Alpha plugins require passing the `--allow-experimental-plugins` command-line flag to the EPP runner:

```bash
epp --allow-experimental-plugins
```

If an Alpha plugin is configured while `--allow-experimental-plugins` is not set (the default), the EPP runner will fail initialization with an explicit error.

## Related Documentation

- [Architecture Overview](../../../../docs/architecture.md)
- [Creating a new Filter guide](../../../../docs/create_new_filter.md)

