# Sojourn Time Observer Producer

**Type:** `sojourn-time-observer-hub`

A black-box observer that produces per-endpoint sojourn time statistics for the
[mrl scorer](../../../scheduling/scorer/mrlscorer/README.md). It splits each
request's sojourn time into two physically distinct intervals: TTFT (dispatch to first chunk) and decode (first chunk to end of stream) and maintains one t-digest per interval per endpoint.

It does not depend on polling data from the endpoints. Every sample comes from the request-control hooks the router already fires.

## Motivation

The minimal residual lifetime (mrl) scorer routes each incoming request to the endpoint with the smallest expected remaining in-flight work under a two-term mean-residual-life estimation. Computing that estimation needs two per-endpoint distributions:

- **TTFT** — waiting-plus-prefill time, so the scorer can account for requests that have been dispatched but not yet emitted tokens.
- **Decode** — post-first-chunk service time, so the scorer can account for requests that are emitting tokens.

The scorer also needs live, per-request dispatch and first-chunk timestamps to compute each in-flight request's **age**. The observer maintains both the two t-digests for distributions, and a fleet-wide in-flight index (per endpoint, per request) for requests' age.

The black-box approach supports cases where scraping metrics from the endpoints is not feasible.

> **Note:** intended for EPP hub deployments, which route across peer clusters rather than selecting model-serving endpoints.

## Architecture

The producer adopts the observer pattern, passively collecting statistics from request-control hooks and publishing them as a datalayer attribute:

| hook | what it does |
| --- | --- |
| `PreRequest` | Record `dispatchedAt` for the target endpoint in the fleet-wide in-flight index |
| `ResponseBody`, first chunk (`StartOfStream == true`) | Emit one queue+TTFT sample (`firstChunkAt - dispatchedAt`) into the endpoint's TTFT digest, and stamp `firstChunkAt` on the in-flight entry |
| `ResponseBody`, terminal chunk (`EndOfStream == true`) | Emit one decode sample (`endOfStreamAt - firstChunkAt`) into the endpoint's decode digest, and clear the in-flight entry |
| `Dispatch` | Every `intervalDuration`, serialize both digests and publish the paired snapshot when both are warm |

The **single-chunk (non-streaming) case** fires the first-chunk and end-of-stream
paths on the same event, in order: TTFT sample = `firstChunkAt - dispatchedAt`,
decode sample = 0. The decode digest correctly records 0 as a
distributional feature. Under a non-streaming-dominant workload, the decode digest will report `E[decode] ~ 0` and the two-term residual estimation collapses to TTFT-only, which is the right behavior for a workload where all the work
happens before the first (and only) chunk.

The snapshot is exposed through a `DynamicAttribute` attached once per endpoint, so each flush swaps a pointer rather than writing the AttributeMap.

### Fleet-wide in-flight index

The scorer needs to enumerate live per-request timestamps **per candidate endpoint**. `PluginState` is keyed by `RequestID` (even though it is a string, this is the framework contract), so an endpoint-indexed view of in-flight requests does not fit `PluginState` shape. The observer keeps its own index outside `PluginState`.

**Forward index.** The primary structure is `map[endpointID]map[requestID]*inflightEntry` under the observer's mutex. The outer key is the endpoint, the inner key is the request, and each entry carries `dispatchedAt` and `firstChunkAt`. This is the shape the scorer wants: `InFlightRequestsFor(endpointID)` iterates one inner map and returns a freshly-allocated `[]InFlightRequest` slice under a short read-lock. The map is bounded by the endpoint's concurrent in-flight count, which is a few tens, at most a few hundred fleet-wide (depending on the concurrency batching per endpoint).

**Reverse index.** The hooks that mutate the forward index are not symmetric. `PreRequest` knows the endpoint because the scheduler has just selected it, so writing the dispatched entry at `inflight[endpointID][requestID]` is direct. `onFirstChunk` and `onEndOfStream` fire from the response stream, which carries only the `requestID`, not the endpoint. The forward index's outer key is unknown at that point. Without a reverse index, finding the entry for a given `requestID` means scanning every endpoint's inner map: O(N × K) over N endpoints and K in-flight per endpoint, on every chunk event. The observer keeps a second map `map[requestID]endpointID` in lock-step with the forward index, so these hooks resolve the owning endpoint in O(1). The invariant: a `requestID` is present in exactly one `inflight[*]` inner map iff it is present in the reverse map with that endpoint as the value.

Endpoint delete purges both the digest state and the in-flight index for the deleted endpoint, along with every reverse-index entry pointing at it, so the paired-map invariant holds across endpoint lifecycle events. Requests already dispatched to a since-deleted endpoint complete normally but no longer show up in `InFlightRequestsFor`.

### What drives the flush

The producer implements `PollingDispatcher`. The datalayer's collector visits every endpoint on a tick, and calls `Dispatch` once per `intervalDuration`. The plugin owns no goroutine, no request pays for the recompute, and an endpoint that goes quiet is still visited so it neither leaks in-flight state nor freezes on a stale snapshot.

Nothing is scraped: `Dispatch` only means "your turn, for this endpoint, now", and `AppendExtractor` is rejected because this dispatcher publishes its own state rather than sourcing data for others.

> **Required:** a `PollingDispatcher` is only driven when it is listed under `dataLayer.sources`. Auto-creating the producer from the scorer's required data key wires the attribute but **not** the tick, so no snapshot would ever be published and every endpoint would read as cold. See [Configuration](#configuration).

## Warm-up gate

`Dispatch` publishes a snapshot only when both `TtftCount() >= minSamples` AND `DecodeCount() >= minSamples`. Before that, the endpoint's AttributeMap has no snapshot entry and the `DynamicAttribute` closure returns `nil` and the scorer
reads the endpoint as cold. The two counts are independent: decode is typically the gating digest during cold start, because a decode sample lands only after the first chunk has emitted a TTFT sample.

Both counts live inside the digests as `td.Count()` — no parallel scalar counter is maintained. That keeps producer memory bounded in value as well as in centroid count.

## Snapshot contents

The published `SojournEstimatorSnapshot` (see [sojourntime attribute README](../../../datalayer/attribute/sojourntime/README.md)) carries two serialized digest byte-slices and nothing else. All statistics(mean, MRL at age, tail quantiles) are computed from the digest at scoring time. Keeping the snapshot minimal has two purposes: bounded snapshot size (~1-4 KB per digest at default compression, **independent of stream length**), and no possibility of derived-field drift.

## Parameters

| Parameter | Default | Description | Tuning |
| --- | --- | --- | --- |
| `compression` | 200 | t-digest compression parameter, applied to both TTFT and decode digests. Must be `>= 1` | Raise for tighter tail-quantile accuracy at greater memory cost; a few hundred is typical |
| `minSamples` | 60 | Warm-up threshold, applied to each digest independently | Raise to demand more evidence before the scorer starts using the endpoint; lower to warm up faster with noisier data |
| `intervalDuration` | 5s | How often the snapshot is recomputed and published | Lower to react sooner; rounded to a multiple of the datalayer's tick |

## Configuration

The producer must be named and listed under `dataLayer.sources`, which is what drives its recompute. There is no other producer this observer depends on: it observes request-control hooks directly, so no auto-created upstream is required.

```yaml
plugins:
  - type: sojourn-time-observer-hub
    name: sojourn-observer
    parameters:
      compression: 200
      minSamples: 60
      intervalDuration: 5s
  - type: mrl-scorer-hub
    name: mrl
    parameters:
      sojournTimeObserverProducerName: sojourn-observer
dataLayer:
  sources:
    # Drives the periodic recompute. Scrapes nothing, binds no extractors.
    - pluginRef: sojourn-observer
```

`intervalDuration` is rounded to a multiple of the datalayer's base tick, and that base tick drops to the smallest interval any configured source asks for, so a fast scrape source elsewhere in the config can shift when this recompute lands.

Both this plugin and the scorer are Alpha, so the EPP must run with
`--allow-experimental-plugins`.

See the [mrl scorer README](../../../scheduling/scorer/mrlscorer/README.md) for how the two-term residual is computed from the snapshot and the in-flight index.
