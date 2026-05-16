# Models Data Source

**Type:** `models-data-source`

The Models Data Source polls inference server pods for model information and passes the response to a paired `models-data-extractor` extractor.

## What it does

1. Iterates over every ready endpoint associated with the `InferencePool`.
2. Issues a `GET <scheme>://<endpoint-ip>:<port>/<path>` request to each endpoint.
3. Parses the `/v1/models` response.
4. Returns the parsed response to the datalayer runtime, which forwards it to any extractors wired to this source via `data: sources:`.

## Inputs consumed

- Pod list from the `InferencePool` (polled individually on each scheduling cycle).

## Configuration

- `scheme` (string, optional, default: `"http"`): Protocol scheme: `"http"` or `"https"`.
- `path` (string, optional, default: `"/v1/models"`): URL path for the models API endpoint.
- `insecureSkipVerify` (bool, optional, default: `true`): Skip TLS certificate verification.

```yaml
- type: models-data-source
  name: my-models-source
  parameters:
    scheme: "http"
    path: "/v1/models"
    insecureSkipVerify: true
```

The data source expects responses in the OpenAI-compatible format:

```json
{
  "object": "list",
  "data": [
    { "id": "llama-3-8b", "parent": "llama-3" },
    { "id": "mistral-7b", "parent": "mistral" }
  ]
}
```

## Complete Configuration Example

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: models-data-source
  name: vllm-models-source
  parameters:
    scheme: "https"
    insecureSkipVerify: false
- type: models-data-extractor
  name: vllm-models-extractor
# ... other plugins (filters, scorers, profile handler, picker) ...
data:
  sources:
  - pluginRef: vllm-models-source
    extractors:
    - pluginRef: vllm-models-extractor
```
