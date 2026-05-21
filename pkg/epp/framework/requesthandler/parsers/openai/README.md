# OpenAI Parser Plugin

**Type:** `openai-parser`

Parses HTTP/H2C requests and responses in the OpenAI API format. 

> [!NOTE]
> This plugin is enabled by default if no other parser is specified in `EndpointPickerConfig`. You do not need to explicitly declare it in your configuration.

Supports all standard OpenAI-compatible endpoints: completions, chat/completions, conversations, responses, and embeddings. Extracts model name, prompt content, token counts, and streaming mode from the request body and response.

**Parameters:** None.

---

## Related Documentation
- [Parsers Index](../README.md)
