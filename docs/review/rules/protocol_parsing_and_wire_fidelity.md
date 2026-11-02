# Protocol parsing and wire fidelity

## Preserve unrecognized client request fields and wire fidelity
Decode only router-inspected fields into typed structs while preserving unrecognized client JSON fields, 64-bit integer precision, key ordering, and explicit booleans on forwarded bodies. Normalize headers and paths once at ingress through read-only accessors without mutating forwarded client headers.

## Claim inference routes declaratively with unmatched-route fallthrough
Register plugin and endpoint route claims declaratively by HTTP method and path so unmatched routes fall through to downstream handlers without hardcoded paths in core dispatchers.
- Accept: Separate path and protocol slices when claimed paths apply uniformly, and forwarding native payloads directly across disaggregated stages running the same engine.

## Buffer streaming responses to complete SSE event boundaries and count terminal usage once
Buffer chunked streaming responses to complete SSE event delimiters before inspecting frames so mid-event TCP splits never drop `data:` lines. Preserve `http.Flusher` via `Unwrap()` on response writer wrappers, record token usage only once from the terminal chunk, and surface premature EOF or missing terminal events as stream errors.

## Maintain feature parity across inference APIs and validate multimodal items individually
Keep tokenization, tool-call rendering, usage extraction, and length classification consistent across chat completions, legacy completions, responses, messages, and generate endpoints. Group per-item fields into element structs rather than parallel slices, and validate multimodal metadata per item and modality rather than with request-wide booleans.
- Accept: Hashing multimodal items atomically as whole units when the backend inference engine hashes media items without chunking.

## Key endpoint state by host, port, and worker rank
Key per-endpoint state, cache indexes, eviction hooks, and event-socket ports by host, port, and worker rank rather than pod IP alone.

## Order disaggregated stages and isolate stage-specific payloads
Execute disaggregated encode and prefill stages before dependent decode stages, keep prefill-only payload mutations out of decode requests, and cancel downstream stages immediately when an upstream stage fails.
