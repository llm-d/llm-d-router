# Unencoded runtime hazards

## Unjoined timed-out DataProducer goroutines and buffer lifetimes
When writing `DataProducer` plugins or handling `dataProducerPluginsWithTimeout`: timeout returns without joining the producer goroutine, so check `ctx.Err()` before writes, and never mutate shared `*InferenceRequestBody` fields after timeout or recycle `sync.Pool` buffers while an unjoined producer aliases their bytes.
- Open: Whether the framework should recover plugin panics per request or let them terminate the process.

## Envoy ext_proc early header-skip metrics and decode-pod filtering
When changing header-only or skipped-body exits in Envoy `ext_proc` handlers: record request latency and one terminal outcome on early-skip branches, and restrict fallback endpoint selection in disaggregated pools to decode-capable pods.

## SSE chunk boundaries, usage folding, and ResponseWriter Unwrap
When parsing streamed responses, folding usage, or wrapping `http.ResponseWriter`: `ext_proc` chunks split SSE events at any byte, so every SSE parser carries partial lines per request, capping each assembled line, keep the last cumulative usage event in a chunk, and do not let later zeros overwrite counts. Implement `Unwrap() http.ResponseWriter` so `http.Flusher` works.

## Multi-worker DP endpoint keys by host, port, and rank
When keying per-endpoint maps, prefix-cache indexes, ZMQ sockets, or SSRF allowlists: key by `(host, port, rank)` because multi-worker DP ranks share one pod IP.

## Bounded metric labels and series cleanup
When adding Prometheus `*Vec` labels or configurable buckets: avoid unbounded per-pod or per-request series across autoscaling, wrap caller-controlled values (`model_name`, `fairness_id`) in `BoundedLabel`, delete series when endpoints or flows are removed, and cap configurable bucket counts.

## No locks held across network calls
When a mutex guards scheduling-path state: never hold it (read lock included) across Redis, HTTP, or gRPC calls. A waiting writer blocks new `RWMutex` readers, so a slow call becomes request latency; copy state under the lock and call after release with a bounded context.
