# External wire and engine contracts

## Anthropic split prompt tokens and vLLM streaming usage flags
When parsing or forwarding token usage and `stream_options`: sum Anthropic prompt tokens as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (OpenAI nests `cached_tokens`), and treat vLLM `stream_options.continuous_usage_stats` (cumulative usage per chunk) as separate from one-shot `include_usage` without injecting `include_usage` beside it.

## Client JSON passthrough, Content-Length, and Connection upgrade
When decoding JSON bodies, rewriting payloads, or copying client headers: unmarshal only router-inspected keys so other fields pass through, recompute or drop `Content-Length` on rewrite, and scan `Connection` headers case-insensitively (RFC 9110) for `upgrade`. Do not forward client `Accept-Encoding` when the router decodes responses; Go `http.Transport` only decompresses gzip it requested itself.

## Native engine render passthrough and stateful Responses rejection
When integrating inference APIs, prompt rendering, or `/v1/responses`: forward raw client payloads to native engine `/render` routes (`/v1/responses/render`, `/v1/messages/render`) instead of translating schemas in the router, and reject `previous_response_id` and `conversation` fields with an explicit error.
- Accept: Interim translation while the engine lacks a native `/render` endpoint.

## Router-built P/D transfer params
When building or forwarding decode-leg `kv_transfer_params`: set keys the engine connector reads unconditionally (vLLM NIXL: `remote_request_id`), never mix cross-engine keys, and drop encoder `ec_transfer_params`. When forwarding `prompt_token_ids`, omit IDs from legacy translations lacking token parity, and exclude `echo`, `return_prompt_text`, or multimodal requests.

## ZMQ KV-events sequence gaps and worker epoch resets
When consuming ZMQ KV-event streams: track per-endpoint sequence numbers to detect gaps (`seq > lastSeq + 1`) and worker-restart epoch resets (`seq < lastSeq`), purging stale index entries and loading a base snapshot before applying deltas.
