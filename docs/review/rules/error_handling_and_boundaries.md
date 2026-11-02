# Error handling and boundaries

## Validate untrusted inputs at ingress and configuration boundaries
Validate nil pointers, type assertions, slice bounds, integer overflows, content types, and payload size limits at network, parser, and config boundaries before buffering bodies or constructing domain types. Include field and expected-type context in boundary errors, and close caller-supplied listeners when early setup fails.
- Accept: Omitting redundant defensive nil checks inside internal helpers whose callers already guarantee valid inputs.

## Propagate errors explicitly without fabricated fallbacks
Return producer, parser, and config failures as explicit errors instead of substituting placeholder counts, synthetic block sizes, or silent zero defaults, and never call `panic` on domain or invariant errors in request or flow-control paths. Fail startup when TLS certificates, metrics listeners, or required plugin bindings are invalid.
- Open: Whether the plugin framework should recover plugin panics per request at the invocation boundary or let panicking plugins terminate the process.

## Map unwrapped error chains to canonical status codes
Classify errors with `errors.Is` and `errors.As` across unwrapped chains rather than matching error strings, formatting error bodies for the active API surface. Map client validation faults to 4xx, upstream worker protocol faults to 502, empty pools to 503, and saturation drops to 429.
- Accept: Returning 502 Bad Gateway rather than 400 Bad Request when an upstream worker returns an invalid content type or malformed payload for a valid client request.
