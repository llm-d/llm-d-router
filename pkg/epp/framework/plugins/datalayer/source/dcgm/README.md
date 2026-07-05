# DCGM Data Source

**Type:** `dcgm-data-source`

The DCGM Data Source polls NVIDIA DCGM Exporter sidecar containers for GPU hardware metrics and passes the response to a paired `dcgm-extractor` extractor.

## What it does

1. Iterates over every ready endpoint associated with the `InferencePool`.
2. Issues a `GET <scheme>://<pod-ip>:<port>/<path>` request to each endpoint's DCGM Exporter sidecar.
3. Parses the Prometheus text response.
4. Returns the parsed metric families to the datalayer runtime, which forwards them to any extractors wired to this source via `data: sources:`.

The source uses `portOverride` on the underlying `HTTPDataSource` to reach the DCGM Exporter on a different port than the inference server.

## Inputs consumed

- Pod list from the `InferencePool` (polled individually on each scheduling cycle).

## Configuration

- `scheme` (string, optional, default: `"http"`): Protocol scheme: `"http"` or `"https"`.
- `path` (string, optional, default: `"/metrics"`): URL path for the DCGM Exporter metrics endpoint.
- `port` (int, optional, default: `9400`): Port where the DCGM Exporter sidecar listens.
- `insecureSkipVerify` (bool, optional, default: `true`): Skip TLS certificate verification.

```yaml
- type: dcgm-data-source
  name: my-dcgm-source
  parameters:
    port: 9400
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
- type: dcgm-extractor
  name: dcgm-extractor
# ... other plugins (filters, scorers, profile handler, picker) ...
data:
  sources:
  - pluginRef: dcgm-source
    extractors:
    - pluginRef: dcgm-extractor
```
