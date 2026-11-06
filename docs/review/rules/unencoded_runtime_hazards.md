# Unencoded runtime hazards

## Unjoined timed-out DataProducer goroutines and buffer lifetimes
When writing `DataProducer` plugins or body handling around `dataProducerPluginsWithTimeout`: timeout returns without joining the producer goroutine, so check `ctx.Err()` before writes, do not mutate shared `*InferenceRequestBody` fields in place after timeout, and do not recycle `sync.Pool` buffers while an unjoined producer or snapshot reader aliases their bytes.
- Open: Whether the plugin framework should recover plugin panics per request or let panicking plugins terminate the process.

## Envoy ext_proc early header-skip metrics and decode-pod filtering
When modifying header-only or skipped-body exit paths in Envoy `ext_proc` handlers: record request latency and one terminal outcome on early-skip branches, and filter fallback endpoint selection in disaggregated pools to decode-only or decode-capable pods.

## Stateless ParseResponse, mergeUsage, and ResponseWriter Unwrap
When implementing streaming response parsers, usage folding, or `http.ResponseWriter` wrappers: keep `ParseResponse` stateless per SSE chunk while accumulating cross-chunk state in `mergeUsage` so later zeros do not overwrite reported counts, and implement `Unwrap() http.ResponseWriter` so `http.Flusher` works.

## Multi-worker DP endpoint keys by host, port, and rank
When keying per-endpoint maps, prefix-cache indexes, ZMQ sockets, or SSRF allowlists: key by `(host, port, rank)` rather than IP alone because multi-worker DP ranks share a single pod IP.
