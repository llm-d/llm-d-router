# External wire and engine contracts

## Anthropic split prompt tokens and vLLM streaming usage flags
When parsing or forwarding token usage and `stream_options`: sum Anthropic `/v1/messages` prompt tokens as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (unlike OpenAI nested `cached_tokens`), and treat vLLM `stream_options.continuous_usage_stats` (per-chunk cumulative usage) as separate from one-shot `include_usage` without injecting `include_usage` beside it.

## Client JSON passthrough, Content-Length, and Connection upgrade
When decoding JSON bodies, rewriting HTTP payloads, or checking proxy upgrade headers: unmarshal only router-inspected keys so unrecognized client fields and uninspected or malformed sub-objects pass through unchanged when not rewriting. Delete or recompute `Content-Length` on body rewrite while preserving it when untouched, and scan all `Connection` header lines and comma-separated tokens case-insensitively per RFC 9110 for `upgrade`.

## Native engine render passthrough and stateful Responses rejection
When integrating inference APIs, prompt rendering, or `/v1/responses`: forward raw client payloads to native engine `/render` routes (`/v1/responses/render`, `/v1/messages/render`) instead of translating schemas in the router, and reject `/v1/responses` `previous_response_id` and `conversation` fields with an explicit error.
- Accept: Contained interim translation while the backend engine lacks a native `/render` endpoint.

## ZMQ KV-events sequence gaps and worker epoch resets
When consuming ZMQ KV-event streams in `pkg/kvevents` or `pkg/kvcache`: track per-endpoint sequence numbers to detect gaps (`seq > lastSeq + 1`) and worker-restart epoch resets (`seq < lastSeq`), purging stale index entries and loading a base snapshot before applying deltas.
