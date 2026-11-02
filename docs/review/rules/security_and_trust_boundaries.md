# Security and trust boundaries

## Namespace new router headers under `x-llm-d-` and scrub internal headers at ingress and egress
Prefix new router-owned HTTP headers with lowercase `x-llm-d-`, suffix host-and-port headers with `-host-port`, and normalize inbound header lookups to lowercase at ingress (`x-gateway-destination-endpoint`, `x-data-parallel-host-port`, and `x-session-token` keep their names). Strip client-supplied internal routing headers, deprecated aliases, and untrusted transfer fields across all routes at ingress, and scrub internal pod topology headers before returning responses.
- Accept: Passing allowlist-validated `x-request-id` correlation headers verbatim, documenting body-over-header precedence, and scrubbing internal headers on Envoy fallback routes via `request_headers_to_remove` when early stripping would hide `ScreeningHeaders` from EPP.
- Open: Whether session affinity should be carried via standard HTTP `Set-Cookie` headers or via an explicit request header such as `x-session-token`.

## Allowlist dynamic endpoint targets and reject control characters
Validate peer endpoint URLs and KV-cache transfer targets against `InferencePool` host-and-port allowlists across all target ports and data-parallel ranks, blocking self-pulls and unverified cluster addresses. Reject whitespace and CR/LF control characters in header-derived targets.

## Enforce TLS minimums, namespace-scoped watches, and payload redaction
Enforce minimum TLS versions and header-read timeouts on HTTP servers, trim trailing newlines from mounted secret files, and scope Kubernetes informers to the configured namespace rather than cluster-wide watches. Never log raw prompts, base64 media payloads, or end-user identifiers at default or debug verbosity.
- Open: Whether `DumpState` debug endpoints should hash tenant and program identifiers by default or emit cleartext values behind listener access controls.
