# Latency Observer Producer

**Type:** `latency-observer-producer-hub`

A black-box observer that produces latency data for a black-box latency scorer.
It tracks requests and responses and publishes a small latency snapshot of TTFT percentiles that the
scorer components use.
It does not depend on polling data from the endpoints.

## Motivation

Expected latency is an important factor in picking the endpoint for serving a request. To support
weighting latency in the decision, the
[latency-observation scorer](../../../scheduling/scorer/latencyobservation/README.md) provides a
relative score to every endpoint based on a TTFT latency prediction under its *current* load.
The scorer uses a TTFT latency curve that takes into account how many requests are already in flight
when a new one arrives, interpolating linearly between measured load points.
The producer's job is to find the anchor points based on the observed request traffic. It computes
three anchor points — a floor, a low-load and a typical-load latency data point — for the scorer to
interpolate between when predicting the current request's TTFT.

The black-box approach supports cases where scraping metrics from the endpoints is not feasible.

## Architecture

The data producer adopts an observer pattern, passively collecting metrics about the request/response
traffic. It captures latency in a sliding window and computes window-based percentiles for the
scorer.
It builds on the request-control hooks and publishes statistics as a datalayer attribute:

| hook | what it does |
|---|---|
| `Produce` | Read and pin each candidate's in-flight load, before this request joins it |
| `PreRequest` | Record the endpoint decision to enable capturing endpoint statistics |
| `ResponseBody`, first chunk | Capture TTFT and append it to that endpoint's observation window |
| `Dispatch` | Every `intervalDuration`, recompute the percentiles and publish |

The latency producer reads the in-flight request count from the `inflight-load-producer`'s counters.
The count is read in `Produce` rather than `PreRequest` so it reflects the load captured during the
decision (i.e., when the scorer runs).

The latency snapshot is exposed through a `DynamicAttribute` attached once per endpoint, so each
flush swaps a pointer rather than writing the attribute map.

### What drives the recompute

The producer implements `PollingDispatcher`, so the datalayer's collector — which already visits every
endpoint on a tick — calls `Dispatch` once per `intervalDuration`. The plugin owns no goroutine, no
request pays for the recompute, and an endpoint that goes quiet is still visited, so its window ages
out instead of freezing on a stale snapshot.

Nothing is scraped: `Dispatch` only means "your turn, for this endpoint, now", and `AppendExtractor`
is rejected because this dispatcher publishes its own state rather than sourcing data for others.
`cross-replica-publisher` is the same shape.

> **Required:** a `PollingDispatcher` is only driven when it is listed under `dataLayer.sources`.
> Auto-creating the producer from the scorer's required data key wires the attribute but **not** the
> tick, so its snapshot would never be published and every endpoint would read as cold. See
> [Configuration](#configuration).

## Streaming responses only

A time-to-first-token exists only when the response arrives in chunks. If the whole body comes back
at once its latency is end-to-end — prefill plus every decode step — which sits far above the real
TTFT, so it is discarded rather than recorded.

A deployment whose responses are not streamed therefore produces no observations, every endpoint
stays cold, and the scorer scores them all equally. That is correct, but it means the plugin
contributes nothing there.

## Percentile computation

The producer collects request latencies for each endpoint. It uses a **sliding window** to keep the
recent request TTFT latencies and the in-flight load associated with each of them.

The producer computes load metrics by scanning the sliding window and filtering requests by age
(configured by `maxObservationAge`) and by count (configured by `maxRequests`).
Together, both keep the metrics reflecting current behaviour rather than stale measurements.
The computed load metrics are a **low-load** latency (by default the P25 TTFT percentile) and a
**typical-load** latency (P50 by default). For each of them the producer also records the in-flight
load, averaged over the `[p-10, p+10]` percentile band around the percentile.

Those two describe the endpoint under load. The **floor** comes from long-term statistics instead of
recent ones. Once per `bucketDuration` the producer takes the P10 percentile of the observations in
that bucket and stores it in a **long-term history window**, which therefore reflects the endpoint's
latency when it is not under load. The history window is bound in size, and is trimmed by removing
its maximal values, not its oldest ones, when full. The P10 percentile over the history window is the
floor.

Until the history window holds an entry, the P10 of the short window stands in as the floor, so an
endpoint has a usable floor from its first flush.

All three are published as one snapshot once per `intervalDuration`, so no request pays for the work
and an endpoint that goes quiet ages its window out rather than freezing on stale anchors — while its
long-term baseline survives the lull untouched.

## Load confidence

A percentile over a handful of requests is noise, so `minRequests` gates the snapshot twice: the
floor on `Observations` (cumulative), the loaded anchors on `RecentRequestCount` (short window). Below
the threshold the floor reads as zero, marking the endpoint cold for the scorer exploration logic;
with a floor but too few recent requests the scorer predicts at the floor alone. The split lets a
calibrated endpoint keep its floor through an idle spell while it re-earns the loaded anchors. Both
counts and the threshold travel in the snapshot, so the scorer needs no parameter of its own.

## Published fields

| Field | Meaning |
|---|---|
| `FloorTTFT` | Load-invariant service floor: the P10 over the long-term history window |
| `WindowFloorTTFT` | P10 TTFT over the short window; the floor until the history window fills |
| `LowLoadTTFT` | TTFT at the low-load percentile (`lowPercentile`, default P25) |
| `TypicalLoadTTFT` | TTFT at the typical-load percentile (`typicalPercentile`, default P50) |
| `InflightAtLowLoad` | Average in-flight-at-dispatch in the band around `lowPercentile` |
| `InflightAtTypicalLoad` | Average in-flight-at-dispatch in the band around `typicalPercentile` |
| `RecentRequestCount` | Observation count in the capped short window |
| `Observations` | Cumulative observations that have fed the floor (gates `Floor()`) |
| `CalibrationThreshold` | Calibration threshold, copied from config so the scorer needs no separate param |

`CalibrationThreshold` separates the published field from the `minRequests` parameter it copies: the
parameter configures the threshold, the field carries it to consumers.

## Parameters

| Parameter | Default | Description | Tuning |
|---|---|---|---|
| `intervalDuration` | 1s | How often the snapshot is recomputed and published | Lower to react sooner; rounded to a multiple of the datalayer's tick |
| `maxObservationAge` | 3m | Age bound for the short window | Lower to follow load changes faster; raise if routing flaps |
| `maxRequests` | 100 | Count bound for the short window. Must be `<= windowSize` | Same direction as `maxObservationAge` |
| `minRequests` | 10 | Observations before an anchor is calibrated; published as `CalibrationThreshold` | Lower so new endpoints get traffic sooner, at the cost of noisier data |
| `lowPercentile` | 25 | Low-load anchor percentile → `LowLoadTTFT` / `InflightAtLowLoad` | Leave alone unless the latency curve bends somewhere unusual |
| `typicalPercentile` | 50 | Typical-load anchor percentile → `TypicalLoadTTFT` / `InflightAtTypicalLoad`. Must satisfy `0 < low < typical < 100` | As above |
| `windowSize` | 5000 | Ring buffer capacity per endpoint, allocated up front (~200 KB) | Lower to cut per-endpoint memory; keep `>= maxRequests` |
| `bucketDuration` | 1m | Window for each floor-history entry's P10 | Keep `<= maxObservationAge`; with `bucketHistorySize`, sets how far back the floor remembers |
| `bucketHistorySize` | 1000 | Per-bucket P10s kept for the floor. Must be >= 2 | Lower to let the floor forgive an endpoint that became permanently slower |
| `inFlightLoadProducerName` | "" | Which `inflight-load-producer` instance to read. Empty uses the default | — |


## Configuration

The producer must be named and listed under `dataLayer.sources`, which is what drives its recompute.
Only `inflight-load-producer` is auto-created, from this producer's own required data key.

```yaml
plugins:
  - type: latency-observer-producer-hub
    name: ttft-observer
    parameters:
      intervalDuration: 1s
      minRequests: 10
      bucketDuration: 1m
      bucketHistorySize: 1000
  - type: latency-observation-scorer-hub
    name: ttft
    parameters:
      ttftPercentilesProducerName: ttft-observer
dataLayer:
  sources:
    # Drives the periodic recompute. Scrapes nothing, binds no extractors.
    - pluginRef: ttft-observer
```

`intervalDuration` is rounded to a multiple of the datalayer's base tick, and that base tick drops to
the smallest interval any configured source asks for — so a fast scrape source elsewhere in the
config can shift when this recompute lands.

See the [scorer README](../../../scheduling/scorer/latencyobservation/README.md) for the prediction model and
an end-to-end pipeline.
