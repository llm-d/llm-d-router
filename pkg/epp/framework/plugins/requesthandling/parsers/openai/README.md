# OpenAI Parser Plugin

**Type:** `openai-parser`

Parses HTTP/H2C requests and responses in the OpenAI API format.

> [!NOTE]
> This plugin is enabled by default if no other parser is specified in `EndpointPickerConfig`. You do not need to explicitly declare it in your configuration.

Supports OpenAI-compatible completions, chat/completions, conversations, responses, embeddings, images/generations, and audio/speech endpoints. The fields parsed out vary by endpoint: the request's input content (prompt, messages, or input), the streaming mode, and token usage from responses that report it.

For `POST /v1/audio/speech`, the parser extracts only the text `input`; model, streaming mode, and payload forwarding use the existing generic paths. Successful binary audio responses are passed through without JSON parsing. vLLM-Omni token usage is read from response headers or the terminal `speech.audio.done` SSE event when available.

**Parameters:** None.

---

## Related Documentation
- [Parsers Index](../README.md)
