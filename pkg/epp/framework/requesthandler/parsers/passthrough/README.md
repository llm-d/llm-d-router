# Passthrough Parser Plugin

**Type:** `passthrough-parser`

A model-agnostic parser that passes requests through without interpreting the payload. Use this parser when the request format is not supported by the OpenAI or vLLM gRPC parsers.

**Limitation:** Because the EPP cannot parse the request payload, scheduling plugins that depend on prompt content (e.g., `prefix-cache-scorer`, `precise-prefix-cache-scorer`) will not function. Only load-based and metric-based schedulers are effective with this parser.

**Parameters:** None.

---

## Related Documentation
- [Parsers Index](../README.md)
