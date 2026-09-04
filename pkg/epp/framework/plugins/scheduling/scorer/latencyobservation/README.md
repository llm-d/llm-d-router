# Latency Observation Scorer

**Type:** `latency-observation-scorer-hub`

A black-box scorer that ranks endpoints by their predicted time-to-first-token.
It consumes the latency snapshot published by the
[latency observer](../../../requestcontrol/dataproducer/latencyobserver/README.md) together with the
live in-flight load, and turns them into a prediction and a score.
It does not depend on polling data from the endpoints.

Both inputs are declared `Required`, so the observer and `inflight-load-producer` are auto-created
from the scorer's data keys. The observer additionally needs an entry under `dataLayer.sources` to
drive its recompute. See [Configuration](#configuration).

## Motivation

The endpoint that is fastest when idle is not the endpoint that will answer this request fastest.
What matters is how each endpoint behaves at the load it is carrying *right now*, which is why the
scorer predicts rather than ranks on a measured latency directly. A busy endpoint that degrades
gently can still beat a lightly loaded one that saturates sharply, and only a curve can express that.

The numbers come from responses the EPP already receives, so nothing has to be scraped. Metrics can be
missing, since an endpoint may expose none or sit behind a gateway that hides them, and they can lag by
a poll interval, by the endpoint's own reporting delay, or by whatever the network adds in between.
Neither limits this scorer: any endpoint that answers requests can be scored, and its load is read at
the moment of scoring.

The observer measures the curve; this package reads it. Measurement and prediction stay in separate
packages, so this one carries no windows, percentiles or observation state of its own.

> **Note:** intended for EPP hub deployments, which route across peer clusters rather than selecting
> model-serving endpoints.

## Prediction

Every endpoint has a latency curve: how long a new request waits as a function of how many requests
are already in flight. The observer measures three anchor points on it and this scorer interpolates.

| anchor | coordinates | meaning |
|---|---|---|
| **A** | `(0, FloorTTFT)` | queue-free service time |
| **B** | `(InflightAtLowLoad, LowLoadTTFT)` | low-load anchor (default P25) |
| **C** | `(InflightAtTypicalLoad, TypicalLoadTTFT)` | typical-load anchor (default P50) |

**A** and **C** always define the curve. **B** is inserted between them only when
[admissible](#when-the-low-load-anchor-is-used), splitting it into two segments. Any single prediction
reads one segment, so it uses two of the three anchors. Every anchor is something the observer
measured.

```
if B admissible:
    if inflight < InflightAtLowLoad:             # segment A->B
        predictedTTFT = FloorTTFT + inflight * (LowLoadTTFT - FloorTTFT) / InflightAtLowLoad
    else:                                        # segment B->C, extended past C
        predictedTTFT = LowLoadTTFT + (inflight - InflightAtLowLoad) *
                        (TypicalLoadTTFT - LowLoadTTFT) / (InflightAtTypicalLoad - InflightAtLowLoad)
else:                                            # segment A->C, extended past C
    predictedTTFT = FloorTTFT + inflight * (TypicalLoadTTFT - FloorTTFT) / InflightAtTypicalLoad
```

`inflight` is read live from the `InFlightLoad` attribute at scoring time, not from the snapshot, so
the curve is always evaluated at the endpoint's load right now.

The curve is continuous (both branches give `LowLoadTTFT` at `InflightAtLowLoad`), equals `FloorTTFT`
at zero load, and is monotone non-decreasing given `FloorTTFT < LowLoadTTFT < TypicalLoadTTFT`, which
the admissibility checks enforce. `predictedTTFT` is clamped to `>= FloorTTFT` as a defensive guard.

**Why three anchors and not two.** TTFT rises *faster than linearly* with in-flight load as an
endpoint approaches saturation. A single chord from the floor to **C** cuts across that convex curve
and under-predicts in between; **B** lets the curve bend so the loaded segment follows the local slope
where the endpoint is actually operating.

**Why A-B is a segment of its own.** The B-C slope is measured where queueing dominates, so it is
steep. Running it backwards below **B** makes TTFT cross below the floor within a handful of
requests. A draining-but-still-loaded endpoint would be predicted at its idle latency and win every
decision it appeared in. Interpolating from `(0, FloorTTFT)` matches how TTFT flattens as the queue
drains.

### When the low-load anchor is used

**B** is admissible only when all of these hold. They are conditions on a measured anchor, not
tuning knobs:

| check | condition | why |
|---|---|---|
| separated in load | `InflightAtTypicalLoad - InflightAtLowLoad >= minInflightGap` | the B-C slope is `change in TTFT over change in inflight`; if both anchors sit at the same load that denominator is noise and the slope is meaningless |
| ordered in latency | `TypicalLoadTTFT > LowLoadTTFT` | TTFT must rise with the percentile. If noise inverts them the slope goes negative and the *most* loaded endpoint scores best |
| above the floor | `LowLoadTTFT > FloorTTFT` | the floor is a long-window statistic and `LowLoadTTFT` a recent one, so after a drain the recent P25 can fall below it, tilting the A-B segment downwards |
| positive in-flight | `InflightAtLowLoad > 0` | keeps the A-B divisor safe; the gap check alone does not imply it |

When **B** is dropped the curve is the single floor chord **A to C**, which is well defined at any
load.

## Endpoint states

The observer's snapshot carries its own counts and calibration threshold, so the scorer classifies
each endpoint without a parameter of its own:

- **cold**: `Floor() == 0` (never observed, or fewer observations than `CalibrationThreshold`, so the
  floor is not yet trustworthy). Seeded optimistically at the best observed TTFT.
- **seed**: has a floor but is not calibrated (`RecentRequestCount < CalibrationThreshold`, or no
  in-flight anchor). Predicts at the floor.
- **trusted**: `RecentRequestCount >= CalibrationThreshold`, `InflightAtTypicalLoad > 0`,
  `TypicalLoadTTFT > floor`. Uses the curve above.

An endpoint whose live in-flight load cannot be read is demoted to untrusted: the curve would be
evaluated at an unknown load, so exploration governs it instead.

## Score

```
score = (maxTTFT - predictedTTFT) / (maxTTFT - minTTFT)
```

Lowest predicted TTFT scores highest. Cold endpoints seed at `minTTFT`; if every endpoint is cold,
all score 1.0 and the picker's tie-break spreads the traffic.

### Exploration

An under-observed endpoint can be starved: competing against a calibrated one it may never win the
traffic it needs to calibrate. `explorationRate` breaks that loop. Each under-observed endpoint is
flipped independently per request, and with probability `explorationRate` its final score is forced
to `1.0` so the picker sends it a probe; otherwise it is suppressed to `0`, but only when a
calibrated endpoint exists to take the traffic. The override applies to the **final score only**, so
a probe never distorts the trusted endpoints' normalisation.

Together with the observer's load confidence gate this reproduces what a manual warmup would do: a
new endpoint reads as cold, receives probes, crosses the calibration threshold, then competes on its
true latency.

Note the default is non-zero. At `0` an uncalibrated endpoint is seeded at the best observed TTFT
and ties for the win, taking a share of all traffic before anything is known about it.

## Parameters

| Parameter | Default | Description | Tuning |
|---|---|---|---|
| `explorationRate` | 0.1 | Per-endpoint probability of probing an under-observed endpoint. Range `[0, 1]` | Raise to give new or recovered endpoints a better chance of being tried; `0` sends everything to the trusted winner |
| `minInflightGap` | 2.0 | In-flight separation the two load anchors need before the low-load one is used. Must be > 0 | Raise to be stricter about when the curve is trusted; lower to use it on endpoints whose load barely varies |
| `roundTTFTStep` | 0.0 | Rounding step for predictions, in seconds. `0` disables. Must be >= 0 | Set to about `0.01` (10 ms) when endpoints swap over differences too small to matter |
| `ttftPercentilesProducerName` | "" | Which `latency-observer-producer-hub` instance to read. Empty uses the default | n/a |
| `inFlightLoadProducerName` | "" | Which `inflight-load-producer` instance to read. Empty uses the default | n/a |


## Configuration

```yaml
plugins:
  - type: latency-observer-producer-hub
    name: ttft-observer
  - type: latency-observation-scorer-hub
    name: ttft
    parameters:
      explorationRate: 0.1
      minInflightGap: 2.0
      ttftPercentilesProducerName: ttft-observer
  - type: max-score-picker
  - type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: ttft
      - pluginRef: max-score-picker
dataLayer:
  sources:
    # Drives the observer's periodic recompute. Scrapes nothing.
    - pluginRef: ttft-observer
```

Both plugins are Alpha, so the EPP must run with `--allow-experimental-plugins`.

Debug logging at `--v 4` emits one `latency-observation score` line per candidate with its
`predictedTTFT`, `score` and `hasBaseline`.

See the [observer README](../../../requestcontrol/dataproducer/latencyobserver/README.md) for how the
three anchors are measured.
