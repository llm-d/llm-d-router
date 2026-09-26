# Sojourn Time Attributes

This package defines the data structure the sojourn-time observer publishes for
the mrl scorer.

## `SojournEstimatorSnapshot`

Carries a serialized copy of the per-endpoint TTFT and decode t-digests. A
snapshot is present in the endpoint's AttributeMap only when both digests are
warm; a cold endpoint has no entry, so the scorer's `nil`-check reads it as
uncalibrated.

- **Key**: `SojournEstimatorSnapshotDataKey`
- **Fields**:
  - `TtftDigest`: `caio/go-tdigest/v5` serialization of the TTFT samples
    (`firstChunkAt - dispatchedAt`).
  - `DecodeDigest`: `caio/go-tdigest/v5` serialization of the decode samples
    (`endOfStreamAt - firstChunkAt`).

`Clone` deep-copies both byte slices so a subsequent producer flush that
overwrites the observer's snapshot pointer does not race with in-flight
consumer reads.

## Query helpers

The snapshot exposes three query helpers the scorer calls at scoring time. All
inputs and outputs are wall-clock seconds.

| Helper | Returns |
|---|---|
| `MrlTtft(a)` | `E[TTFT - a \| TTFT > a]`, the expected remaining TTFT for a request that has been in the TTFT phase for `a` seconds without emitting a first token |
| `MrlDecode(a)` | `E[decode - a \| decode > a]`, the expected remaining decode for a request that has been decoding for `a` seconds without emitting the terminal chunk |
| `MeanDecode()` | `E[decode]`, the expected total decode time for a fresh request; the scorer uses this on the pre-first-chunk term as "expected decode phase after TTFT completes" |

Both `Mrl*` helpers implement the standard mean-residual-life tail average:
sum centroid means strictly above `a`, weighted by centroid count, divide by
total count, subtract `a`, clamp to zero. `MrlDecode(0)` equals `MeanDecode()`
in the limit of many samples.

An empty or corrupt digest reads as zero on all three helpers, so a garbled
snapshot reduces the endpoint to a cold-equivalent rather than crashing the
scorer.

## Producers

The following plugin produces this attribute:

- **`sojourn-time-observer-hub`** (Request Control): observes each request's
  dispatch → first-chunk → end-of-stream lifecycle, ingests one TTFT sample
  and one decode sample per completion into the paired digests, and publishes
  a serialized snapshot to the endpoint's AttributeMap on its own flush
  cadence.
