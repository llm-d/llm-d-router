# MRL Scorer

**Type:** `mrl-scorer-hub`

A Distribution-axis scorer that ranks endpoints by their expected remaining in-flight work under a two-term Mean Residual Life (MRL) residual formula. It consumes the paired TTFT and decode t-digest snapshot published by the [sojourn-time-observer-hub](../../../requestcontrol/dataproducer/sojourntimeobserver/README.md)
together with the observer's live per-request in-flight index, and turns them into a residual and a score.

It does not depend on polling data from the endpoints.

The snapshot data key is declared `Required`, so the observer is auto-created
from the scorer's data key. The observer additionally needs an entry under
`dataLayer.sources` to drive its recompute. See [Configuration](#configuration).

## Motivation

Existing production Distribution-axis scorers estimate load by count
(`multicluster-queue-scorer`, `active-request-scorer`) or by predicted TTFT under current loa (`latency-observation-scorer-hub`). None of them estimate the
*residual* work still owed on requests already in flight.

Two endpoints with the same in-flight count can differ enormously in residual work. An eight-slot pool running eight barely started long requests has vastly more residual work than an eight-slot pool running eight almost-done short requests.

This scorer reads the *ages* of each in-flight request and
weights each one by its own *mean-residual-life* estimate, closing that blind spot.

> **Note:** intended for EPP hub deployments, which route across peer clusters
> rather than selecting model-serving endpoints.

## The two-term residual

Every dispatched-not-completed request on a candidate endpoint falls into exactly one of two observable phases, indexed by whether the first response chunk has arrived:

- **Pre-first-chunk**: the request has been dispatched but no token has come back. Its age is `age_ttft = now - dispatchedAt`. Its expected remaining time is:
  - `MrlTtft(age_ttft)` — expected remaining TTFT, plus
  - `MeanDecode()` — expected total decode phase that still lies ahead once TTFT completes.
- **Post-first-chunk**: the request is emitting tokens. Its age is
  `age_decode = now - firstChunkAt`. Its expected remaining time is:
  - `MrlDecode(age_decode)` — expected remaining decode. TTFT is done; no
    further TTFT term applies.

The residual is the sum over all in-flight requests on the endpoint:

```text
residual_i = sum over req in inflight_i:
    if req.firstChunkAt is zero:      # TTFT phase
        MrlTtft(now - req.dispatchedAt) + MeanDecode()
    else:                             # decode phase
        MrlDecode(now - req.firstChunkAt)
```

The two sums are over disjoint sets, indexed by whether the first chunk has arrived. A given request contributes exactly one term at any given moment. The `+ MeanDecode` on the pre-first-chunk term is counting the two *sequential phases* the request will still go through: first the remaining TTFT, then a full decode phase. Once the request crosses into post-first-chunk, only the remaining-decode term applies.

### Why two terms

The two-term formula is a direct structural analog of the ideal (unavailable) MRL algorithm that would split on `waiting` vs `service`:

| Ideal formula (unavailable) | Two-term formula (production) |
| --- | --- |
| Waiting term: `mrl_wait(age_wait) + mean_service` | Pre-first-chunk term: `MrlTtft(age_ttft) + MeanDecode` |
| Service term: `mrl_service(age_service)` | Post-first-chunk term: `MrlDecode(age_decode)` |

### Where the proxy is loose

TTFT contains pod-side prefill in addition to pod-side waiting. In the ideal
formula the pre-first-chunk term's "expected further work after waiting"
would be `mean_service = mean_prefill + mean_decode`; we only have
`MeanDecode`. Under production LLM traffic, decode is typically much longer
than prefill, so `MeanDecode ~ mean_service` and the approximation is small.
Under heavy-prefill workloads (very long prompts, small models) the gap might grow.

## Score

```go
score = (max_r - residual_i) / (max_r - min_r)
```

Lowest residual scores highest. The normalization span `[min_r, max_r]` is
taken over warm endpoints only — endpoints whose snapshot has been published by the observer. A cold endpoint (no snapshot in the AttributeMap yet) is seeded to `min_r` for the normalization, so it ties with the least-loaded warm endpoint rather than winning outright on a residual of 0. An independent per-cold exploration coin at probability `explorationRate` (default `0.1`) runs after normalization: on a probe the cold endpoint's final score is forced to `1.0`; otherwise it is forced to `0`. The coin never alters `min_r` or `max_r`, so warm endpoints' relative ranking is unaffected by probes.

When every candidate is cold there is nothing to rank — all endpoints tie
at `1.0` and the picker's tie-break spreads traffic. The same tie result
holds when all warm residuals are equal.

## Parameters

| Parameter | Default | Description | Tuning |
| --- | --- | --- | --- |
| `sojournTimeObserverProducerName` | "" | Which `sojourn-time-observer-hub` instance to read. Empty selects the default producer (the type-named instance) | Set when multiple sojourn-time observer instances are configured |
| `explorationRate` | `0.1` | Probability that a cold endpoint is probed on a scoring call. Range `[0, 1]`. `0` disables the coin; the cold endpoint then stays at the seed-at-`min_r` tie with the least-loaded warm | Raise to spread traffic to cold endpoints faster during warm-up; lower if cold endpoints should warm up strictly through tied-with-warm requests |

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
