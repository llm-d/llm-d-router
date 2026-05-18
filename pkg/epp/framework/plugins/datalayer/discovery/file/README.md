# File Discovery Plugin

**Type:** `file-discovery`
**Interface:** `EndpointDiscovery`

Loads inference endpoints from a YAML or JSON file on the local filesystem,
optionally re-loading the file when it changes.

## What It Does

Provides an alternative to Kubernetes-based endpoint discovery for deployments
that run the EPP without a controller manager (bare metal, Slurm, Ray, local
development). The plugin reads a static endpoints file at startup, applies
each entry to the datastore via `DiscoveryNotifier`, and -- when configured to
do so -- watches the file for changes via fsnotify and reconciles the
datastore on each change.

## How It Works

- **Initial load.** On `Start`, the file is read once. Each entry is
  validated (address must be a parseable IP, port must be in `[1, 65535]`)
  and applied via `notifier.Upsert`. Per-entry validation errors are logged
  and the entry is skipped; file-level problems (open, parse, size > 1 MiB)
  abort startup.
- **Reload (optional).** When `watchFile: true`, fsnotify Write / Create /
  Remove events trigger a reload. After an atomic rename or ConfigMap-style
  symlink swap (which destroys the inode being watched), the watcher is
  re-attached so subsequent changes still fire. Reload semantics match the
  initial load: invalid entries are logged and skipped, valid entries are
  applied. Endpoints present in the previous load but absent from the new
  one are deleted via `notifier.Delete`.
- **Readiness.** The plugin closes its `Ready()` channel after the first
  successful load so callers can gate request-serving components on the
  datastore being populated.

## Inputs Consumed

A YAML or JSON file with the schema below. The path is supplied via the
plugin's `path` parameter.

```yaml
endpoints:
  - name: <string>              # required -- unique within the file
    namespace: <string>         # optional -- defaults to "default"
    address: <IPv4 or IPv6>     # required -- must be a valid IP address
    port: <string>              # required -- integer 1-65535 as a string
    metricsHost: <string>       # optional -- host:port for metrics scraping;
                                #             defaults to address:port
    labels:                     # optional -- arbitrary key/value labels
      <key>: <value>
```

## Configuration

**Location:** `dataLayer.discovery.pluginRef` referencing a plugin entry of
type `file-discovery` in `plugins`.
**Enabled by default:** No.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | `string` | yes | -- | Absolute path to the endpoints file. |
| `watchFile` | `bool` | no | `false` | When true, watch the file for changes via fsnotify and reload on Write / Create / Remove events. |

### Examples

```yaml
plugins:
  - type: file-discovery
    name: file-discovery
    parameters:
      path: /etc/epp/endpoints.yaml
      watchFile: true
dataLayer:
  discovery:
    pluginRef: file-discovery
```

A two-endpoint file referenced by the config above:

```yaml
endpoints:
  - name: vllm-0
    address: "10.0.0.1"
    port: "8000"
  - name: vllm-1
    address: "10.0.0.2"
    port: "8000"
    metricsHost: "10.0.0.2:9090"
```

## Limitations

- The endpoints file is capped at 1 MiB.
- `address` must be a literal IPv4 or IPv6 address. Hostnames are not
  resolved.
- File-discovery mode runs the EPP without a Kubernetes controller manager,
  so several K8s-only features are inactive: the `InferenceModelRewrite`
  and `InferenceObjective` reconcilers do not run, and any
  `k8s-notification-source` plugin in the data layer config will not bind.
  The runner emits a startup log naming the inactive features.
- A single bad entry on initial load is logged and skipped, not fatal. If
  the entire file is unreadable or unparseable, startup fails.

## Related Documentation

- [Plugins Index](../../../README.md)
