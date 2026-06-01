# Anthropic Parser Plugin

**Type:** `anthropic-parser`

Parses HTTP/H2C requests and responses in the Anthropic Messages API format. Use this parser when the EPP fronts endpoints serving the Anthropic API.

Supports the Messages API (`/v1/messages`) endpoint. Extracts message content and streaming mode from the request body. Tracks token usage including prompt tokens, completion tokens, and cached tokens (via `cache_read_input_tokens`) from both standard JSON responses and server-sent events (SSE) for streaming responses.

**Parameters:** None.

---

## Related Documentation
- [Parsers Index](../README.md)
