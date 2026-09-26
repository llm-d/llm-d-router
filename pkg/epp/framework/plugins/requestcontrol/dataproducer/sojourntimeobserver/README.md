# Sojourn Time Observer Producer

**Type:** `sojourn-time-observer-hub`

A black-box observer that produces per-endpoint sojourn statistics for the
[mrl scorer](../../../scheduling/scorer/mrlscorer/README.md). It splits each
request's sojourn into two physically distinct intervals — TTFT (dispatch to
first chunk) and decode (first chunk to end of stream) — and maintains one
t-digest per interval per endpoint.

It does not depend on polling data from the endpoints; every sample comes from
the request-control hooks the router already fires.

## Motivation

The mrl scorer routes each incoming request to the endpoint with the smallest
expected remaining in-flight work under a two-term mean-residual-life residual.
Computing that residual needs two per-endpoint distributions:

- **TTFT** — waiting-plus-prefill time, so the scorer can price a request that
  has been dispatched but not yet emitted a token.
- **Decode** — post-first-chunk service time, so the scorer can price a
  request that is emitting tokens.

The scorer also needs live, per-request dispatch and first-chunk timestamps to
compute each in-flight request's age. The observer maintains both — the two
digests for distribution, and a fleet-wide in-flight index (per endpoint, per
request) for age.

The black-box approach supports cases where scraping metrics from the endpoints
is not feasible.

> **Note:** intended for EPP hub deployments, which route across peer clusters
> rather than selecting model-serving endpoints.

## Architecture

The producer adopts the observer pattern, passively collecting statistics from
request-control hooks and publishing them as a datalayer attribute:

| hook | what it does |
|---|---|
| `PreRequest` | Record `dispatchedAt` for the primary target endpoint in the fleet-wide in-flight index |
| `ResponseBody`, first chunk (`StartOfStream == true`) | Emit one TTFT sample (`firstChunkAt - dispatchedAt`) into the endpoint's TTFT digest, and stamp `firstChunkAt` on the in-flight entry |
| `ResponseBody`, terminal chunk (`EndOfStream == true`) | Emit one decode sample (`endOfStreamAt - firstChunkAt`) into the endpoint's decode digest, and clear the in-flight entry |
| `Dispatch` | Every `intervalDuration`, serialize both digests and publish the paired snapshot when both are warm |

The **single-chunk (non-streaming) case** fires the first-chunk and end-of-stream
paths on the same event, in order: TTFT sample = `firstChunkAt - dispatchedAt`,
decode sample = 0. The decode digest correctly records the zero as a
distributional feature; under a non-streaming-dominant workload the decode
digest will report `E[decode] ~ 0` and the two-term residual collapses to
TTFT-only, which is the right behavior for a workload where all the work
happens before the first (and only) chunk.

The snapshot is exposed through a `DynamicAttribute` attached once per
endpoint, so each flush swaps a pointer rather than writing the AttributeMap.

### Fleet-wide in-flight index

The scorer needs to enumerate live per-request timestamps per candidate
endpoint. This information does not fit in `PluginState` (which is keyed by
`RequestID`, not by endpoint), so the observer keeps its own
`map[endpointID]map[requestID]*inflightEntry` under its own mutex.
`PreRequest`, `onFirstChunk`, and `onEndOfStream` mutate this map; the
scorer's read path is `InFlightRequestsFor(endpointID)`, which returns a
freshly-allocated `[]InFlightRequest` slice under a short read-lock. The map
is bounded by the endpoint's concurrent in-flight count — a few tens at c=8
batching per pod, at most a few hundred fleet-wide.

Endpoint delete purges both the digest state and the in-flight index for the
deleted endpoint. Requests already dispatched to a since-deleted endpoint
complete normally but no longer show up in `InFlightRequestsFor`.

### What drives the flush

The producer implements `PollingDispatcher`. The datalayer's collector visits
every endpoint on a tick, and calls `Dispatch` once per `intervalDuration`. The
plugin owns no goroutine, no request pays for the recompute, and an endpoint
that goes quiet is still visited so it neither leaks in-flight state nor
freezes on a stale snapshot.

Nothing is scraped: `Dispatch` only means "your turn, for this endpoint, now",
and `AppendExtractor` is rejected because this dispatcher publishes its own
state rather than sourcing data for others.

> **Required:** a `PollingDispatcher` is only driven when it is listed under
> `dataLayer.sources`. Auto-creating the producer from the scorer's required
> data key wires the attribute but **not** the tick, so no snapshot would ever
> be published and every endpoint would read as cold. See
> [Configuration](#configuration).

## Warm-up gate

`Dispatch` publishes a snapshot only when both `TtftCount() >= minSamples` AND
`DecodeCount() >= minSamples`. Before that, the endpoint's AttributeMap has no
snapshot entry — the DynamicAttribute closure returns `nil` — and the scorer
reads the endpoint as cold. The two counts are independent: decode is typically
the gating digest during cold start, because a decode sample lands only after
the first chunk has emitted a TTFT sample.

Both counts live inside the digests as `td.Count()` — no parallel scalar
counter is maintained. That keeps producer memory bounded in value as well as
in centroid count.

## Snapshot contents

The published `SojournEstimatorSnapshot` (see
[sojourntime attribute README](../../../datalayer/attribute/sojourntime/README.md))
carries two serialized digest byte-slices and nothing else. All statistics
(mean, MRL at age, tail quantiles) are computed from the digest at scoring
time. Keeping the snapshot minimal has two purposes: bounded snapshot size
(~1-4 KB per digest at default compression, independent of stream length), and
no possibility of derived-field drift.

## Parameters

| Parameter | Default | Description | Tuning |
|---|---|---|---|
| `compression` | 200 | t-digest compression parameter, applied to both TTFT and decode digests. Must be `>= 1` | Raise for tighter tail-quantile accuracy at greater memory cost; a few hundred is typical |
| `minSamples` | 60 | Warm-up threshold, applied to each digest independently | Raise to demand more evidence before the scorer starts using the endpoint; lower to warm up faster with noisier data |
| `intervalDuration` | 5s | How often the snapshot is recomputed and published | Lower to react sooner; rounded to a multiple of the datalayer's tick |

## Configuration

The producer must be named and listed under `dataLayer.sources`, which is what
drives its recompute. There is no other producer this observer depends on: it
observes request-control hooks directly, so no auto-created upstream is
required.

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

`intervalDuration` is rounded to a multiple of the datalayer's base tick, and
that base tick drops to the smallest interval any configured source asks for,
so a fast scrape source elsewhere in the config can shift when this recompute
lands.

Both this plugin and the scorer are Alpha, so the EPP must run with
`--allow-experimental-plugins`.

See the [mrl scorer README](../../../scheduling/scorer/mrlscorer/README.md) for
how the two-term residual is computed from the snapshot and the in-flight
index.
