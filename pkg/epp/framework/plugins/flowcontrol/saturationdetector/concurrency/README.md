# Concurrency Detector Plugin

**Type:** `concurrency-detector`

Synchronous saturation detection and scheduling filter mechanism based on active in-flight request accounting.

## What it does

This plugin uses a two-tier approach to manage average pool load and protect individual endpoints.

### Role in Flow Control (The Gatekeeper)
The detector implements the `SaturationDetector` interface to provide a utilization gradient, allowing the Flow Controller to apply proportional backpressure.

    PoolSaturation = Aggregate Inflight Load / Aggregate Pool Capacity

In token mode, both numerator and denominator are evaluated in tokens: the aggregate inflight token count divided by the sum of all endpoints' MaxTokenConcurrency.

Hybrid mode is the exception: rather than one aggregate fraction, it evaluates each endpoint's saturation as the larger of its request and token ratios and reports the unweighted average across endpoints. This prevents distinct endpoints saturating on different dimensions from being masked by aggregate ratios that each remain low.

**Dispatch reservations:** The detector implements `DispatchReservationTracker`. Flow control reserves a slot when it dispatches a request and the director releases it after the `PreRequest` hooks have published the request's in-flight load, so a burst of dispatches cannot all read the same headroom. Pending reservations are added to the aggregate in-flight request count in `requests` mode. In `hybrid` mode the reported saturation is the larger of the per-endpoint average and the aggregate request ratio including reservations. `tokens` mode does not count reservations. Reservations are pool-wide and local to the EPP replica: a reservation counts against this detector whichever endpoint the request is later scheduled to, and it is not shared with other replicas.

**Heterogeneous Deployments:** Because this detector calculates saturation globally as a single aggregate fraction (in requests and tokens mode), it utilizes an aggregate queueing model. In deployments with heterogeneous compute (e.g., mixing H100 and L4 nodes), this heavily biases the pool saturation metric toward the state of the larger nodes. Contrast this with the Utilization Detector, which evaluates saturation as an unweighted average of individual endpoint scores.

### Role in Scheduling (The Traffic Shaper)
The detector implements the `Filter` interface to protect individual endpoints. It removes endpoints from candidate lists if their local inflight count exceeds the safety limit:

    EndpointLimit = Capacity * (1 + Headroom)

In tokens and hybrid modes the token check also counts the uncached tokens the incoming request would add to each endpoint (the `UncachedRequestTokens` attribute from the in-flight load producer), so an endpoint is removed when admitting the request would take it over the limit:

    InflightTokens + IncomingUncachedTokens <= TokenLimit

An endpoint with no in-flight tokens always passes the token check. A request larger than the limit can still be placed on an idle endpoint, which is the best placement the pool can offer, and the engine's own limits decide whether it runs.

This approach allows the Flow Controller to manage average pool load, while the Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.

**Fail-Open Fallback:** To prevent complete routing failure, if *all* candidate endpoints are filtered out (i.e., the entire cluster is over the safety limits), the filter softens and returns the original list of endpoints, allowing the scheduler's scorers to pick the least-bad option. With `failOpen: false` the filter returns no endpoints instead, so the profile finds no endpoint and the request fails rather than overloading one.

With flow control enabled, `failOpen: false` in tokens or hybrid mode also checks the queued request's projected cost before release. The built-in monolithic and P/D profile handlers keep the request queued when the required profiles have no capacity. The normal producer hooks and in-flight accounting run once after release. Without flow control, the filter still rejects a request when no endpoint fits.

Tokenization and repeatable prefix queries prepare admission data separately from ordinary producer hooks. An expired prefix estimate falls back to the undiscounted input/output cost; capacity sufficient for that cost does not wait for another prefix lookup. Approximate and precise prefix producers support repeatable queries. Other producers, including burst-prefix batching, run once on the ordinary path and do not provide a cache discount for admission.

Custom profile handlers and custom P/D role selection retain the aggregate saturation check and final scheduling filters. The projected check does not reserve capacity: load may change before scheduling, and the final filter remains authoritative. Request-only and fail-open configurations retain their existing admission path.

## Inputs consumed

The plugin internally tracks active concurrency by hooking into the request lifecycle (`PreRequest` and `ResponseBody`).
- **Requests Mode**: Maintains atomic counters of active requests per endpoint.
- **Tokens Mode**: Uses a `TokenEstimator` to estimate tokens from the incoming request and tracks the aggregate inflight tokens per endpoint.

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `concurrencyMode` (`string`): Evaluation mode. Valid values are `"requests"`, `"tokens"`, or `"hybrid"`. In `"hybrid"` mode both request and token accounting are evaluated. Pool saturation is computed per endpoint as the larger of that endpoint's request and token ratios, then averaged across endpoints, so an endpoint saturated on either dimension is reflected even when distinct endpoints saturate on different dimensions. An endpoint is filtered out when either its request load reaches the limit or its token load plus the request's uncached tokens exceeds the limit. (Default: `"requests"`)
- `maxConcurrency` (`int64`): Maximum requests in flight. Serves as the "ideal" request capacity for a single endpoint. Must be > 0. (Default: `100`)
- `maxTokenConcurrency` (`int64`): Maximum tokens in flight. The "tokens" mode equivalent of `maxConcurrency`. Must be > 0. (Default: `1000000`)
- `headroom` (`float64`): Allowed burst capacity above the ideal threshold, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)
- `failOpen` (`bool`): Whether the filter returns all candidates when every candidate is over its limit. When `false` it returns none. (Default: `true`)

## Trade-offs

Unlike the Utilization Detector, this approach reacts instantaneously to new requests, preventing sudden bursts from overwhelming an endpoint before telemetry updates. However, it suffers from two critical flaws:

1. **Open-Loop Divergence**: The detector operates as an open-loop controller, completely blind to actual hardware telemetry. While the internal counters are mathematically zero-sum and do not leak, consistent under- or over-estimations of token lengths will cause the view of pool saturation to systematically drift from the physical reality of the GPU workload.
2. **KV Cache Blindness**: Because the detector cannot observe true engine memory pressure, it is highly vulnerable to continuous-batching edge cases. If actual output generations exceed static estimates, the underlying KV cache will silently fill up. This forces the inference engine to preempt active requests and swap KV blocks to CPU memory, causing severe latency degradation (TPOT spikes) that remains completely invisible to this detector.
