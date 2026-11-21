# Peer Demand Holdback Policy

**Type:** `peer-demand-holdback-policy` (alpha)

With several active EPP replicas, each replica dispatches from its own flow control queues. A replica holding
only low priority requests dispatches them into capacity that a single queue would have given to the high
priority requests queued on another replica. This policy closes that gap: it gates every band below the highest
band that has requests queued on a peer replica, on top of the ceilings its delegate computes.

The peer view comes from `flowControl.crossReplica`, which publishes each replica's queued requests per band
through the cross-replica syncer and refreshes the view after each publish. A replica's own queue is not part of
the view. A view older than five sync intervals is ignored, and the policy then returns the delegate's ceilings.
Without `flowControl.crossReplica` the policy returns the delegate's ceilings.

The highest band with peer demand keeps its delegate ceiling, so the pool's highest queued band always
progresses.

A peer counts a request it dispatched as band demand until the request's endpoint load is recorded. It publishes
its band demand and its in-flight load per endpoint as separate updates, so its band can read empty here while
the slot its last request took still reads free. The policy keeps the gate for `releaseDelay` after the peer band
empties, which lets the in-flight update arrive first.

## Configuration

```yaml
plugins:
- type: priority-holdback-policy
  name: shared-capacity
  parameters:
    shape: linear
    minCeiling: 0.75
- type: peer-demand-holdback-policy
  name: peer-aware
  parameters:
    delegate: shared-capacity
flowControl:
  usageLimitPolicyPluginRef: peer-aware
  crossReplica:
    syncerPluginRef: shared-load
    syncInterval: 20ms
```

| Parameter | Description |
| --- | --- |
| `delegate` | Usage limit policy whose ceilings this policy tightens. Optional: without it every band starts at 1.0. |
| `releaseDelay` | How long the lower bands stay gated after the peers' higher band empties. Optional: two `flowControl.crossReplica` sync intervals. Keep it above the `dataLayer.crossReplica` sync interval. `0s` releases at once. |

## Metrics

`llm_d_epp_flow_control_peer_queued_requests{priority, inference_pool}` reports the peer
requests queued per band, including the ones dispatched and not yet on an endpoint, as last read.

## Limits

- Bands are published by priority value, so a band exists in the view only when both replicas provision it. Bands
  come from InferenceObjective priorities, which every replica reconciles; a priority with no objective, such
  as one set per request, is visible to a peer only once the peer has the band too.
- A peer's queued count includes requests whose band is gated on that peer. A gated higher band gates the lower
  bands on every replica anyway, since ceilings are non-increasing.
