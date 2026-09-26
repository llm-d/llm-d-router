# MRL Scorer

**Type:** `mrl-scorer-hub`

A Distribution-axis scorer that ranks endpoints by their expected remaining
in-flight work under a two-term mean-residual-life (MRL) residual formula. It
consumes the paired TTFT and decode t-digest snapshot published by the
[sojourn-time-observer-hub](../../../requestcontrol/dataproducer/sojourntimeobserver/README.md)
together with the observer's live per-request in-flight index, and turns them
into a residual and a score.

It does not depend on polling data from the endpoints.

The snapshot data key is declared `Required`, so the observer is auto-created
from the scorer's data key. The observer additionally needs an entry under
`dataLayer.sources` to drive its recompute. See [Configuration](#configuration).

## Motivation

Existing production Distribution-axis scorers price load by count
(`multicluster-queue-scorer`, `active-request-scorer`) or by predicted TTFT
under current load (`latency-observation-scorer-hub`). None of them price the
*residual* work still owed on requests already in flight.

Two endpoints with the same in-flight count can differ enormously in residual
work — an eight-slot pool running eight barely-started long requests has
vastly more residual work than an eight-slot pool running eight almost-done
short requests. This scorer reads the *ages* of each in-flight request and
weights each by its own mean-residual-life estimate, closing that blind spot.

> **Note:** intended for EPP hub deployments, which route across peer clusters
> rather than selecting model-serving endpoints.

## The two-term residual

Every dispatched-not-completed request on a candidate endpoint falls into
exactly one of two observable phases, indexed by whether the first response
chunk has arrived:

- **Pre-first-chunk**: the request has been dispatched but no token has come
  back. Its age is `age_ttft = now - dispatchedAt`. Its expected remaining
  time is:
  - `MrlTtft(age_ttft)` — expected remaining TTFT, plus
  - `MeanDecode()` — expected total decode phase that still lies ahead once
    TTFT completes.
- **Post-first-chunk**: the request is emitting tokens. Its age is
  `age_decode = now - firstChunkAt`. Its expected remaining time is:
  - `MrlDecode(age_decode)` — expected remaining decode. TTFT is done; no
    further TTFT term applies.

The residual is the sum over all in-flight requests on the endpoint:

```
residual_i = sum over req in inflight_i:
    if req.firstChunkAt is zero:      # TTFT phase
        MrlTtft(now - req.dispatchedAt) + MeanDecode()
    else:                             # decode phase
        MrlDecode(now - req.firstChunkAt)
```

The two sums are over disjoint sets, indexed by whether the first chunk has
arrived. A given request contributes exactly one term at any given moment. The
`+ MeanDecode` on the pre-first-chunk term is counting the two *sequential
phases* the request will still go through: first the remaining TTFT, then a
full decode phase. Once the request crosses into post-first-chunk, only the
remaining-decode term applies.

### Why two terms, and not one sum

An alternative would collapse both branches into a single sum over
dispatch-relative ages:

```
residual_i = sum over req in inflight_i:
    Mrl(now - req.dispatchedAt)
```

That was the primary porting design. In simulation, on burstier arrival
regimes, it over-attributes residual work to requests that have spent a long
time waiting on the pod's internal FIFO: `Mrl(age)` on a heavy-tailed sojourn
distribution grows with `age`, so a request stuck at a saturated endpoint gets
credited with a much larger expected remaining time than a request that has
actually started service. The two-term formula, by separating TTFT-phase from
decode-phase, avoids that misattribution.

The two-term formula is a direct structural analog of the ideal (unavailable)
MRL algorithm that would split on `waiting` vs `service`:

| Ideal formula (unavailable) | Two-term formula (production) |
|---|---|
| Waiting term: `mrl_wait(age_wait) + mean_service` | Pre-first-chunk term: `MrlTtft(age_ttft) + MeanDecode` |
| Service term: `mrl_service(age_service)` | Post-first-chunk term: `MrlDecode(age_decode)` |

### Where the proxy is loose

TTFT contains pod-side prefill in addition to pod-side waiting. In the ideal
formula the pre-first-chunk term's "expected further work after waiting"
would be `mean_service = mean_prefill + mean_decode`; we only have
`MeanDecode`. Under production LLM traffic, decode is typically much longer
than prefill, so `MeanDecode ~ mean_service` and the approximation is small.
Under heavy-prefill workloads (very long prompts, small models) the gap
grows. Since this is a shared multiplier across endpoints, argmin ordering is
largely preserved as long as `mean_prefill` is roughly uniform across the
fleet.

## Score

```
score = (max_r - residual_i) / (max_r - min_r)
```

Lowest residual scores highest. Cold endpoints (no snapshot in the
AttributeMap, because the observer has not yet flushed a warm snapshot)
contribute residual `0`, so they participate in the argmin normalization on
equal footing with a fully drained warm endpoint — new endpoints get traffic
during their warm-up window. Ties (all residuals equal, or a single candidate)
yield `1.0` for every endpoint, letting the picker's tie-break spread traffic.

## Parameters

| Parameter | Default | Description | Tuning |
|---|---|---|---|
| `sojournTimeObserverProducerName` | "" | Which `sojourn-time-observer-hub` instance to read. Empty selects the default producer (the type-named instance) | Set when multiple sojourn-time observer instances are configured |

## Configuration

```yaml
plugins:
  - type: sojourn-time-observer-hub
    name: sojourn-observer
  - type: mrl-scorer-hub
    name: mrl
    parameters:
      sojournTimeObserverProducerName: sojourn-observer
  - type: max-score-picker
  - type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: mrl
      - pluginRef: max-score-picker
dataLayer:
  sources:
    # Drives the observer's periodic recompute. Scrapes nothing.
    - pluginRef: sojourn-observer
```

Both plugins are Alpha, so the EPP must run with `--allow-experimental-plugins`.

See the [observer README](../../../requestcontrol/dataproducer/sojourntimeobserver/README.md)
for how the two digests are populated and when the snapshot flushes.
