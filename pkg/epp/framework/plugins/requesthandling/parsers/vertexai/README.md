# Vertex AI Parser Plugin

The Vertex AI Parser plugin (`vertexai-parser`) implements the `fwkrh.Parser` interface for the Vertex AI prediction API. It enables the `llm-d-router` to parse and route incoming Vertex AI gRPC requests, as well as the REST requests a Vertex AI dedicated endpoint forwards, and extract metrics from their responses.

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

## Supported REST Paths

A Vertex AI dedicated endpoint forwards requests over HTTP with the payload as the request body, e.g. `POST /v1/projects/P/locations/L/endpoints/E:rawPredict`. Paths are matched by suffix, so every API version (`v1`, `v1beta1`, `ui`) and both the project-ID and project-number forms are covered; the query string is ignored.

| Path suffix | Inner OpenAI Path |
| :--- | :--- |
| `/chat/completions` | `/chat/completions` |
| `:predict`, `:rawPredict`, `:streamRawPredict` | Chosen from the payload: `messages` → `/chat/completions`, `prompt` → `/completions`, `input` → `/responses` |
| `/invoke/<container path>` | As above |

Invoke is claimed with the registry's segment-wildcard form `invoke/*`, which matches any path containing an `/invoke/` segment.

Parsing REST payloads is best effort: a dedicated endpoint may front any container, so a payload that is not OpenAI-compatible (for example a TensorFlow `{"instances": [...]}` body) is forwarded unchanged and still scheduled, rather than rejected.

The parser advertises both the `kubernetes.io/h2c` and `http` application protocols, so it can serve an `InferencePool` whose backends speak HTTP/1.1.

## Response Parsing

For gRPC responses, the parser:
1. Extracts the gRPC payload and unmarshals it into `httpbody.HttpBody`.
2. Extracts the raw JSON data from the body.
3. Delegates to the OpenAI parser to extract token usage metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) which are then recorded by the router.

REST responses are the model server's JSON or SSE stream as is, so they are passed straight to the OpenAI parser. The two are told apart by the `content-type` header (`application/grpc*`), or by the body when the header is absent.

## Configuration

To enable the Vertex AI parser, configure it in your `EndpointPickerConfig` under the `requestHandler` section:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
requestHandler:
  parsers:
  - pluginRef: vertexai-parser
plugins:
  - name: vertexai-parser
    type: vertexai-parser
```

No additional parameters are required.
