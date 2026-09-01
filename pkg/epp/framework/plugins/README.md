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

## Multi-cluster variants

A few plugins ship a multi-cluster variant beside the stock plugin, for a cluster-scoped EPP whose candidate endpoints are clusters. The scheduling variants are `multicluster-kv-cache-utilization-scorer`, `multicluster-queue-scorer`, `multicluster-session-affinity-filter`, and `multicluster-prefix-cache-scorer`. The `multicluster-approx-prefix-cache-producer` is the matching request-control data producer. They rely on three datalayer variants: `multicluster-file-discovery` discovers peer clusters as endpoints, and `multicluster-metrics-data-source` with `multicluster-metrics-extractor` scrape each cluster and write its pool-aggregate metrics into the attributes the scorers read.

The pure-delegation variants (`multicluster-session-affinity-filter`, `multicluster-prefix-cache-scorer`, `multicluster-approx-prefix-cache-producer`) reuse the stock plugin's behavior and only rename the type, so their per-plugin Prometheus metrics report under the stock type label rather than the multicluster one.

### Dependencies

The metric scorers are not standalone. `multicluster-kv-cache-utilization-scorer` and `multicluster-queue-scorer` score a cluster from the pool-aggregate metrics that `multicluster-metrics-extractor` writes from what `multicluster-metrics-data-source` scrapes. Configure all three together in `dataLayer`. A config with the scorers but without the source and extractor fails to load.

### Example

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: multicluster-file-discovery
    name: discovery
    parameters:
      path: /etc/epp/clusters.yaml
  - type: multicluster-metrics-data-source
    name: metrics-source
    parameters:
      scheme: https
      caCertPath: /etc/epp/tls/ca.crt
  - type: multicluster-metrics-extractor
    name: metrics-extractor
  - type: multicluster-session-affinity-filter
    name: session-affinity
  - type: multicluster-kv-cache-utilization-scorer
    name: kv-cache
  - type: multicluster-queue-scorer
    name: queue
  - type: max-score-picker
  - type: single-profile-handler
dataLayer:
  discovery:
    endpoints:
      pluginRef: discovery
  sources:
    - pluginRef: metrics-source
      extractors:
        - pluginRef: metrics-extractor
```

`clusters.yaml` follows the same format as the pod file discovery, except an `address` may be a hostname.

## Related Documentation

- [Architecture Overview](../../../../docs/architecture.md)
- [Creating a new Filter guide](../../../../docs/create_new_filter.md)

