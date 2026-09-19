# Session Inter-Turn Latency Producer

The `session-interturn-latency-producer` learns the distribution of a
multi-turn workload's inter-turn intervals online and publishes an
`InterTurnPrediction` attribute on the request attribute store, so consumers
can predict when the session's next turn is likely to arrive. It is the
timing model of SAECache ([arXiv:2605.18825](https://arxiv.org/pdf/2605.18825)):
inter-turn intervals of a multi-turn workload follow a log-normal distribution
whose parameters vary by deployment, so they are learned online rather than
fixed.

## How it works

1. **Session identity.** The producer consumes the `SessionID` attribute
   published by the `session-id-producer` and reads the workload type from a
   request header supplied by the orchestrator. It acts only on requests
   whose type matches `sessionType`; everything else gets no prediction.
2. **Online estimation.** For each matching session, the producer measures
   the idle gap between one turn's response completion (ResponseBody hook)
   and the next turn's arrival (Produce hook). Gaps feed a log-normal fit: a
   sliding window over `ln(gap)` provides the maximum-likelihood sample
   estimate (mean and standard deviation), blended into the running
   parameters with an exponential moving average. Gaps below `minInterval`
   (timestamp-precision artifacts) and above `maxIdle` (session boundaries)
   are discarded.
3. **Publication.** Each matching request gets an `InterTurnPrediction`
   attribute carrying the fitted parameters and the observation count.
   Consumers read it with `interturn.ReadInterTurnPrediction` and derive a
   time horizon with its `Quantile` method; the `kv-cache-retention` plugin
   turns it into a KV-cache retention directive.

The fitted parameters, observation count, and tracked-session count are
visible at `/debug/plugins/state`.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `sessionTypeHeader` | `x-session-type` | Header carrying the workload type. |
| `sessionType` | `agentic` | Workload type to act on. Empty acts on every request with a session identifier. |
| `initialLogMean` | `2.28` | Estimator seed, in log-seconds. The default is the CC-Bench agentic-trace fit from the SAECache paper. |
| `initialLogStd` | `1.34` | Estimator seed, in log-seconds. Same source as `initialLogMean`. |
| `emaFactor` | `0.1` | Blend weight of each new sample estimate. |
| `minSamples` | `20` | Observations required before the estimator updates. |
| `windowSize` | `200` | Sliding-window capacity of the sample estimate. |
| `minInterval` | `100ms` | Gaps below this are discarded. |
| `maxIdle` | `1h` | Gaps above this are discarded; idle sessions past this are swept. |
| `maxSessions` | `100000` | Soft cap on tracked sessions. |

## Example

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. The producer requires a `session-id-producer`
entry; the framework orders producers so the session identifier is published
first.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: session-id-producer
    parameters:
      headerName: x-session-id
  - type: session-interturn-latency-producer
    parameters:
      sessionType: agentic
```
