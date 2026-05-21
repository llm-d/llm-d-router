# Parsers

This directory contains parser plugins used to parse and understand the payloads of requests and responses. This understanding is key for empowering features like prefix-cache aware request scheduling and response usage tracking.

## Supported Parsers

*   **`openai-parser`**: The default parser, supporting the [OpenAI API](https://developers.openai.com/api/reference/overview). This is used when no parser is explicitly specified in the `EndpointPickerConfig`.
*   **`vllmgrpc-parser`**: A parser designed to handle requests specifically for the [vLLM gRPC API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/grpc_server/).
*   **`vertexai-parser`**: A parser designed to handle requests for the Vertex AI gRPC API, specifically supporting [PredictionService/ChatCompletions](https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L235). For unsupported Vertex AI APIs, it skips parsing and lets the request pass through without interpretation resulting in routing to a random endpoint.
*   **`passthrough-parser`**: A model-agnostic parser that supports any request format by passing the request body through without interpretation.
    *   **Drawback**: EPP cannot parse the payload, so payload-related scheduling scorers (e.g., `prefix-cache-scorer`) are not supported.

## Configuration

Parsers are configured via the `parser` section in the `EndpointPickerConfig` YAML file. You must first instantiate the parser plugin in the `plugins` section, and then reference its name in the `parser` section. 

If no parser is specified, `openai-parser` is used as the fallback.

Here is an example configuration using the `vllmgrpc-parser`:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: vllmgrpcParser
  type: vllmgrpc-parser
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
parser:
  pluginRef: vllmgrpcParser
```
