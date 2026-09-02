# OpenAI Parser Plugin

**Type:** `openai-parser`

Parses HTTP/H2C requests and responses in the OpenAI API format. 

> [!NOTE]
> This plugin is enabled by default if no other parser is specified in `EndpointPickerConfig`. You do not need to explicitly declare it in your configuration.

Supports all standard OpenAI-compatible endpoints: completions, chat/completions, conversations, responses, embeddings, images/generations, and images/edits. 
The fields parsed out vary by endpoint: the request's input content (prompt, messages, or input), the streaming mode, and token usage from responses that report it. 
The images/edits endpoint accepts multipart/form-data.

**Parameters:**

- `propagatePriority` (bool, default: `false`): When enabled, injects the EPP-resolved request priority into the outgoing request body's `priority` field. Client-supplied `priority` is always removed first so backend-native priority scheduling is governed by EPP.

---

## Related Documentation
- [Parsers Index](../README.md)
