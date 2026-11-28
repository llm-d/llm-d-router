# Weighted Shortest Processing Time (WSPT) Ordering Policy

**Type:** `wspt-ordering-policy`

The WSPT ordering policy favors requests with smaller estimated processing cost. By default it dispatches the smallest request first. With `maxDelaySeconds` set, it orders by arrival time delayed in proportion to cost, which limits how long larger requests can be passed (see [Bounded Delay](#bounded-delay)).

## Why Choose This Policy?

- **Lower Average Latency:** Serving short requests before long ones minimizes the mean time requests spend waiting in the queue.
- **Short Requests Are Not Blocked:** A small request can be dispatched ahead of a large request from the same flow that arrived earlier.

## What It Does

- A request's cost is its body size (`ByteSize`). The policy takes no per-request weight, so Smith's rule (`weight / cost`, highest first) reduces to ordering by size.
- By default, requests with **smaller byte sizes** are prioritized.
- If two requests have the same size, FCFS (First-Come, First-Served) is used as a tie-breaker based on logical enqueue time.
- The ordering key is fixed when a request is enqueued, so it does not change while the request waits.

### Bounded Delay

With `maxDelaySeconds` set, a request is ordered as if it arrived later by a delay proportional to its size:

`DelayedArrival = EnqueueTime + maxDelaySeconds * min(ByteSize / referenceBytes, 1)`

- Requests with **earlier delayed arrivals** are prioritized; ties fall back to FCFS.
- A small request can move ahead of a larger one that arrived up to its delay difference earlier.
- A request is never passed by a request that arrived more than `maxDelaySeconds` after it, so once `maxDelaySeconds` has elapsed since its arrival, no new request is dispatched ahead of it. Its added wait compared with FCFS is the time to dispatch the requests that arrived within that window, which can exceed `maxDelaySeconds`.
- All requests of `referenceBytes` or more receive the full delay, so they are served FCFS relative to each other.

## Inputs consumed

This policy inspects the following attributes of the request item:
- **Request Byte Size**: Used as the processing-cost estimate.
- **Logical Enqueue Time**: Used as a tie-breaker.

## Behavior and Queue Pairing

Like EDF, this policy relies on a heap-based priority queue that keeps items sorted by `Less`.

## Configuration

Without parameters, the policy orders by size alone.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `maxDelaySeconds` | float | `0` (unbounded) | Only requests that arrived within this many seconds after a request can be dispatched ahead of it. |
| `referenceBytes` | integer | none | Request size that receives the full `maxDelaySeconds`. Required when `maxDelaySeconds` is set. |

```yaml
plugins:
- type: wspt-ordering-policy
  parameters:
    maxDelaySeconds: 2
    referenceBytes: 65536
flowControl:
  priorityBands:
  - priority: 100
    orderingPolicyRef: wspt-ordering-policy
```

## Trade-offs

- **Starvation Risk:** Without `maxDelaySeconds`, if a flow continuously receives smaller requests, its larger requests wait until their queue TTL (`EffectiveTTL`) expires and are rejected. Set `maxDelaySeconds` to limit which requests can pass them. Ordering applies only within one flow, so other flows' requests cannot delay this flow's large requests.
- **Size Is an Estimate:** Byte size approximates prompt tokens. It does not account for prefix-cache hits or output length.
- **Computational Overhead:** Maintaining a priority heap incurs $O(\log n)$ cost for insertions and deletions.

## Related Documentation
*   [Ordering Overview](../README.md)
*   [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
