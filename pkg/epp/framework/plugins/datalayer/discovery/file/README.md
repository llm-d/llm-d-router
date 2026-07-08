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
  validated (address must be a valid IPv4 or RFC 1123 hostname, port must be
  in `[1, 65535]`) and applied via `notifier.Upsert`. Per-entry validation
  errors are logged and the entry is skipped; file-level problems (open,
  parse, size > 1 MiB) abort startup.
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
    address: <string>           # required -- IPv4 address or RFC 1123 hostname
    port: <string>              # required -- integer 1-65535 as a string
    metricsPort: <string>       # optional -- metrics scrape port (defaults to port)
    labels:                     # optional -- arbitrary key/value labels
      <key>: <value>
```

When `address` is an IPv4, the endpoint is treated as a pod (`PodName` is set
to the entry name). When `address` is a hostname, the endpoint is treated as a
cluster (`PodName` is empty).

### Endpoint Type Label

Each endpoint is tagged with the label `llm-d.ai/endpoint-type` so downstream
plugins can distinguish pod endpoints from cluster endpoints explicitly. The
label can be set in the endpoints file; when omitted, the plugin auto-detects
it from the address format:

- IPv4 address → `llm-d.ai/endpoint-type: pod`
- Hostname → `llm-d.ai/endpoint-type: cluster`

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

A two-endpoint file with pod IPs:

```yaml
endpoints:
  - name: vllm-0
    address: "10.0.0.1"
    port: "8000"
  - name: vllm-1
    address: "10.0.0.2"
    port: "8000"
```

A cluster endpoints file with host names instead of IP addresses:

```yaml
endpoints:
  - name: cluster-us-east
    address: spoke-us-east.example.com
    port: "443"
    metricsPort: "9090"
    labels:
      region: us-east
  - name: cluster-eu-west
    address: spoke-eu-west.example.com
    port: "443"
    metricsPort: "9090"
    labels:
      region: eu-west
```

## Limitations

- The endpoints file is capped at 1 MiB.
- `address` must be a valid IPv4 address or an RFC 1123 hostname. IPv6 is
  not supported. Hostnames are not resolved by the plugin; DNS resolution
  happens at scrape/connect time.
- Metrics are scraped from `address:metricsPort` (or `address:port` when
  `metricsPort` is not set).
- File-discovery mode runs the EPP without a Kubernetes controller manager,
  so several K8s-only features are inactive: the `InferenceModelRewrite`
  and `InferenceObjective` reconcilers do not run, and any
  `k8s-notification-source` plugin in the data layer config will not bind.
  The runner emits a startup log naming the inactive features.
- A single bad entry on initial load is logged and skipped, not fatal. If
  the entire file is not readable or fails to parse, startup fails.

## Related Documentation

- [Plugins Index](../../../README.md)
