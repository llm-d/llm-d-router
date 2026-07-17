# OpenAI Parser Plugin

**Type:** `openai-parser`

Parses HTTP/H2C requests and responses in the OpenAI API format. 

> [!NOTE]
> This plugin is enabled by default if no other parser is specified in `EndpointPickerConfig`. You do not need to explicitly declare it in your configuration.

Supports all standard OpenAI-compatible endpoints: completions, chat/completions, conversations, responses, embeddings, and images/generations. The fields parsed out vary by endpoint: the request's input content (prompt, messages, or input), the streaming mode, and token usage from responses that report it.

**Parameters:** None.

---

## Related Documentation
- [Parsers Index](../README.md)
