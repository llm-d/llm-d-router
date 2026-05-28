# Vertex AI Parser Plugin

The Vertex AI Parser plugin (`vertexai-parser`) implements the `fwkrh.Parser` interface for the Vertex AI gRPC API. It enables the `llm-d-router` to parse and route incoming Vertex AI gRPC requests and extract metrics from their responses.

## How it Works

Vertex AI's flexible prediction services wrap standard OpenAI-compatible payloads inside gRPC protobuf envelopes. Instead of fully re-implementing the JSON parsing, the Vertex AI parser acts as a wrapper that:
1. **Parses the outer gRPC frame** to extract the raw protobuf payload.
2. **Unmarshals the protobuf request** into the corresponding Vertex AI message type.
3. **Extracts the inner JSON payload** from the embedded `HttpBody`.
4. **Delegates to the OpenAI parser** by cloning the HTTP headers, setting the path to an OpenAI-compatible route, and passing the extracted JSON.
5. **Packages the results**, including the original protobuf request in the `Payload` metadata so downstream plugins can access the full gRPC context.

## Supported gRPC Methods

The parser automatically matches incoming requests based on the `:path` header suffix:

| gRPC Method Suffix | Protocol Message | Inner OpenAI Path | Description |
| :--- | :--- | :--- | :--- |
| `PredictionService/ChatCompletions` | `aiplatformpb.ChatCompletionsRequest` | `/chat/completions` | OpenAI-compatible Chat Completions service. |
| `PredictionService/StreamRawPredict` | `aiplatformpb.StreamRawPredictRequest` | `/responses` | Streaming raw prediction service. |
| `PredictionService/RawPredict` | `aiplatformpb.RawPredictRequest` | `/responses` | Non-streaming raw prediction service. |

## Response Parsing

For responses, the parser:
1. Extracts the gRPC payload and unmarshals it into `httpbody.HttpBody`.
2. Extracts the raw JSON data from the body.
3. Delegates to the OpenAI parser to extract token usage metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) which are then recorded by the router.

## Configuration

To enable the Vertex AI parser, configure it in your `EndpointPickerConfig` under the `requestHandler` section:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
requestHandler:
  parser:
    pluginRef: vertexai-parser
plugins:
  - name: vertexai-parser
    type: vertexai-parser
```

No additional parameters are required.
