# Parsers

This directory contains parser plugins used to parse and understand the payloads of requests and responses. This understanding is key for empowering features like prefix-cache aware request scheduling and response usage tracking.

## Supported Parsers

*   **`openai-parser`**: A parser supporting the [OpenAI API](https://developers.openai.com/api/reference/overview). Along with `anthropic-parser` and `vllmhttp-parser`, it is registered by default if no parsers are explicitly specified.
*   **`anthropic-parser`**: A parser designed to handle requests for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages). It supports both standard JSON and streaming SSE responses.
*   **`vllmgrpc-parser`**: A parser designed to handle requests specifically for the [vLLM gRPC API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/grpc_server/).
*   **`vllmhttp-parser`**: A parser for vLLM HTTP endpoints that are not part of the OpenAI-compatible API surface — specifically `/inference/v1/generate` (which accepts pre-tokenized prompts and multimodal features). Because it only handles this path, you must also configure `openai-parser` if you want to support OpenAI-compatible paths on the same route.
*   **`vertexai-parser`**: A parser designed to handle requests for the Vertex AI gRPC API, specifically supporting [PredictionService/ChatCompletions](https://github.com/googleapis/googleapis/blob/89c3153888201c9e80bc5ec78d6ffca0debe6b52/google/cloud/aiplatform/v1beta1/prediction_service.proto#L235). For unsupported Vertex AI APIs, it skips parsing and lets the request pass through without interpretation resulting in routing to a random endpoint.
*   **`passthrough-parser`**: A model-agnostic parser that supports any request format by passing the request body through without interpretation.
    *   **Drawback**: EPP cannot parse the payload, so payload-related scheduling scorers (e.g., `prefix-cache-scorer`) are not supported.

### Serving mixed vLLM-specific and OpenAI-compatible traffic

`vllmhttp-parser` only parses the `/inference/v1/generate` path. To serve both vLLM-specific and OpenAI-compatible traffic on the same route, configure both `vllmhttp-parser` and `openai-parser` under the `requestHandler.parsers` list.

## Configuration

Parsers are configured via the `requestHandler.parsers` list in the `EndpointPickerConfig` YAML file. You must first instantiate the parser plugin in the `plugins` section, and then reference its name in the `requestHandler.parsers` list.

The EPP resolves incoming request paths to the matching parser using suffix matching. Suffixes are defined by each parser plugin's claims, and the first matching parser in the list is selected. If a parser does not define specific paths, it acts as a fallback for any unmatched traffic.

If no parsers are specified, `openai-parser`, `anthropic-parser`, and `vllmhttp-parser` are used by default.

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
requestHandler:
  parsers:
  - pluginRef: vllmgrpcParser
```

Configuration using both `vllmhttp-parser` and `openai-parser` (enables `/inference/v1/generate` while keeping OpenAI-compatible paths working):

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: vllmhttpParser
  type: vllmhttp-parser
- name: openaiParser
  type: openai-parser
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
requestHandler:
  parsers:
  - pluginRef: vllmhttpParser
  - pluginRef: openaiParser
```

Configuration with multiple parsers (e.g. OpenAI and Anthropic API support on the same route):

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: openaiParser
  type: openai-parser
- name: anthropicParser
  type: anthropic-parser
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
requestHandler:
  parsers:
  - pluginRef: openaiParser
  - pluginRef: anthropicParser
```

### Custom path claims for `openai-parser`

`openai-parser` accepts an `additionalPathClaims` parameter listing extra path suffixes to route to it, for OpenAI-compatible providers that expose sub-paths or prefixed variants of the canonical endpoints. Each claim must sit under a canonical OpenAI endpoint (`chat/completions`, `completions`, `embeddings`, `responses`, `conversations`) as a complete run of path segments; the request body is then parsed as that endpoint's API type. Matching is segment-aware, so `chat/completions/pipeline` is parsed as `chat/completions` rather than bare `completions`. A claim that sits under no canonical endpoint fails configuration loading.

```yaml
plugins:
- name: openaiParser
  type: openai-parser
  parameters:
    additionalPathClaims:
    - chat/completions/pipeline
    - myservice/embeddings
```
