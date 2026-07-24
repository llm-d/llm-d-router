# Utilization Detector Plugin

**Type:** `utilization-detector`

Reactive saturation detection and scheduling filter based on telemetry from LLM serving backends.

> [!NOTE]
> This plugin is enabled by default when flow control is enabled. You do not need to explicitly declare it in your configuration.

## What it does

This plugin uses a two-tier approach to manage average pool load and protect individual endpoints.

### Role in Flow Control (The Gatekeeper)
The detector implements the `SaturationDetector` interface to provide a utilization gradient, allowing the Flow Controller to apply proportional backpressure when the system is overloaded.

It relies on a "roofline model", evaluating both the queue depth and the KV cache utilization to find the most constrained resource for a given endpoint:

    EndpointScore = max(QueueDepth / QueueThreshold, KVCacheUsage / KVCacheThreshold)

The global pool saturation is the average of the endpoint scores:

    PoolSaturation = Average(EndpointScore)

*Note: Endpoints with missing or stale metrics are aggressively scored as 100% saturated.*

**Scrape-lag compensation:** `WaitingQueueSize` is scraped on a poller while the Flow Controller's dispatch loop runs far faster, so between two scrapes the queue term is stale-low and the gate would let the controller over-dispatch. When an `inflight-load-producer` is configured, the queue term is corrected by the in-flight requests the scrape does not yet reflect:

    Credit = max(0, InFlightRequests - (WaitingQueueSize + RunningRequestsSize))
    QueueScore = (WaitingQueueSize + Credit) / QueueThreshold

`InFlightRequests` is the producer's per-endpoint counter, incremented at dispatch and decremented on request completion, so the correction already accounts for requests that returned since the last scrape. Endpoints without the attribute contribute zero credit.

The correction rests on assumptions that hold for a single-writer, aggregated deployment:

- **`RunningRequestsSize` must be populated.** The subtraction of running requests is only sound when the metrics mapping defines `runningRequestsSpec` (defaults exist for vLLM and SGLang). A custom mapping without it leaves `RunningRequestsSize` at 0, and the credit over-counts by the running count.
- **Single EPP replica.** `InFlightRequests` is tracked per EPP replica, while `WaitingQueueSize + RunningRequestsSize` is engine-global. With multiple EPP replicas the credit floors at 0 and the compensation silently disappears (the same single-writer assumption the concurrency detector makes).
- **Aggregated (non-P/D) pools.** With the default producer config the prefill profile's request counter is held until end-of-stream, so under P/D disaggregation a prefill endpoint shows phantom credit for the full decode duration of every request whose prefill it served. Correcting this requires releasing the prefill request counter at start-of-stream in the producer.

### Role in Scheduling (The Traffic Shaper)
The detector implements the `Filter` interface to protect individual endpoints. It removes endpoints from candidate lists if their telemetry is stale, or if they exceed specific safety limits:

    MaxQueueLimit = QueueThreshold * (1 + Headroom)
    MaxKVCacheLimit = min(1.0, KVCacheThreshold * (1 + Headroom))

This approach allows the Flow Controller to manage average pool load, while the Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.

**Fail-Open Fallback:** To prevent complete routing failure, if *all* candidate endpoints are filtered out (i.e., the entire cluster is over the safety limits or stale), the filter softens and returns the original list of endpoints, allowing the scheduler's scorers to pick the least-bad option.

## Inputs consumed

The plugin consumes standard metrics from endpoints:
- `WaitingQueueSize` (Queue depth metric).
- `KVCacheUsagePercent` (KV cache utilization metric).
- `UpdateTime` (Timestamp used to calculate metric staleness).

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `queueDepthThreshold` (`int`): Target waiting queue depth limit. Serves as the "ideal" queue capacity for a single endpoint. Must be > 0. (Default: `5`)
- `kvCacheUtilThreshold` (`float64`): Target KV cache memory utilization limit, expressed as a fraction. Must be in `(0.0, 1.0]`. (Default: `0.8`)
- `metricsStalenessThreshold` (`string` / duration): Maximum age of metrics before an endpoint is considered stale. Stale endpoints are treated as 100% saturated. Must be > 0. (Default: `"200ms"`)
- `headroom` (`float64`): Allowed burst capacity above the ideal thresholds, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)

## Trade-offs

Unlike the Concurrency Detector, this approach operates as a closed-loop controller. It is immune to estimation divergence and reflects the actual performance limits of the continuous batching engine's memory manager. However, it suffers from two fundamental flaws of reactive systems:

1. **Telemetry Staleness (The Thundering Herd)**: Because it relies on asynchronous polling, the view of the endpoint state is perpetually delayed. A sudden burst of traffic can create a severe "thundering herd" condition, where the scheduler routes massive request volumes to a seemingly healthy endpoint before the next metric interval reveals it is completely saturated.
2. **Reactive Backpressure**: By definition, this detector only signals saturation after the inference engine is already under physical duress. It cannot preemptively shield an endpoint from an initial queue buildup; it can only throttle traffic after the physical limits have been breached and latency (TTFT/TPOT) has already degraded.
