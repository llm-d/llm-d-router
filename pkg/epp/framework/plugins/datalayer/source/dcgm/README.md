# DCGM Data Source

**Type:** `dcgm-data-source`

The DCGM Data Source polls NVIDIA DCGM Exporter for GPU hardware metrics and
passes the response to a paired `dcgm-extractor`.

## What it does

1. Iterates over every ready endpoint associated with the `InferencePool`.
2. Issues a `GET <scheme>://<host>:<port>/<path>` request to the DCGM Exporter.
3. Parses the Prometheus text response.
4. Returns the parsed metric families to the datalayer runtime, which forwards
   them to any extractors wired to this source via `dataLayer: sources:`.

The scrape host is either the pod IP (sidecar) or the node IP (DaemonSet),
controlled by `useNodeAddress`. In both cases `port` overrides the inference
server port via `HTTPDataSource` options.

## Inputs consumed

- Pod list from the `InferencePool` (polled individually on each scheduling cycle).
- When `useNodeAddress` is true, each endpoint's `NodeAddress` (`pod.Status.HostIP`).

## Configuration

- `scheme` (string, optional, default: `"http"`): Protocol scheme: `"http"` or `"https"`.
- `path` (string, optional, default: `"/metrics"`): URL path for the DCGM Exporter metrics endpoint.
- `port` (int, optional, default: `9400`): Port where the DCGM Exporter listens.
- `insecureSkipVerify` (bool, optional, default: `true`): Skip TLS certificate verification.
- `useNodeAddress` (bool, optional, default: `false`): When true, scrape
  `NodeAddress:port` (DaemonSet). When false, scrape `PodIP:port` (sidecar).

DaemonSet deployments require `DCGM_EXPORTER_KUBERNETES=true` on the exporter
so metrics include a `pod` label (GPU Operator sets this by default).

```yaml
# Sidecar (default)
- type: dcgm-data-source
  name: my-dcgm-source
  parameters:
    port: 9400

# DaemonSet
- type: dcgm-data-source
  name: my-dcgm-source
  parameters:
    port: 9400
    useNodeAddress: true
```

## Complete Configuration Example

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: dcgm-data-source
  name: dcgm-source
  parameters:
    port: 9400
    useNodeAddress: true
- type: dcgm-extractor
  name: dcgm-extractor
# ... other plugins (filters, scorers, profile handler, picker) ...
dataLayer:
  sources:
  - pluginRef: dcgm-source
    extractors:
    - pluginRef: dcgm-extractor
```
