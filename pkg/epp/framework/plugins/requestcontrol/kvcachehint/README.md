# KV-Cache Hint

The `kv-cache-hint` plugin annotates requests from multi-turn sessions with
KV-cache retention hints for backends implementing the vLLM Context-Aware
KV-Cache Retention API
([vllm-project/vllm#38514](https://github.com/vllm-project/vllm/pull/38514)).
Retention policy is configured per workload type, so a machine-paced subagent
loop and a human-paced chat session get different protection horizons.

## How it works

1. **Inputs.** The plugin consumes the `SessionID` attribute published by the
   `session-id-producer` and the `InterTurnPrediction` attribute published by
   the `session-interturn-latency-producer`. Both are required dependencies,
   so missing producers fail at startup. Requests without either attribute
   pass through untouched.
2. **Queue matching.** Each configured queue names one workload type. The
   prediction's `SessionType` selects the queue; predictions matching no
   queue get no hint. An empty `queues` list configures a single catch-all
   queue applied to every request with a prediction.
3. **Retention duration.** The hint protects the session's KV blocks for
   `clamp(Quantile(retain.quantile), retain.minTTL, retain.maxTTL)` of the
   queue's predicted inter-turn interval: long enough that the next turn
   usually arrives inside the window, bounded so a tail estimate never pins
   blocks indefinitely.
4. **Emission.** The plugin writes the retention API's request-body fields:

   ```json
   "retention_directives": [{"start": 0, "end": null, "priority": 70, "duration": 12.8}],
   "retention_scope": "<session id>"
   ```

   Requests that already carry `retention_directives` pass through untouched:
   client-supplied directives win. The scope is the request's session
   identifier, unless the request names a parent session (see
   `parentSessionHeader`), in which case blocks are grouped under the parent's
   scope so a spawned session's cache expires with the session that spawned
   it.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `parentSessionHeader` | `x-parent-session-id` | Header carrying the identifier of the session that spawned this one, used as the retention scope when present. Empty disables parent scoping. |
| `queues` | one `agentic` queue | Per-workload-type retention policies. Each entry sets `sessionType` and optionally a `retain` block. Empty configures a single catch-all queue. |
| `retain.quantile` | `0.9` | Quantile of the predicted inter-turn distribution used as the retention duration, in (0, 1). |
| `retain.minTTL` | `1s` | Lower clamp on the retention duration. |
| `retain.maxTTL` | `10m` | Upper clamp on the retention duration. |
| `retain.priority` | `70` | Eviction priority (0-100) written into the directive. |

## Example

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. The plugin requires `session-id-producer` and
`session-interturn-latency-producer` entries; the framework orders producers
so both attributes are published first.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: session-id-producer
    parameters:
      headerName: x-session-id
  - type: session-interturn-latency-producer
    parameters:
      sessionTypeHeader: x-session-type
      queues:
        - sessionType: agentic
          initialLogMean: 2.28
          initialLogStd: 1.34
        - sessionType: agentic-subagent
          initialLogMean: 0.69
          initialLogStd: 1.13
  - type: kv-cache-hint
    parameters:
      parentSessionHeader: x-parent-session-id
      queues:
        # Human-paced main session: protect through the distribution's
        # long tail, capped so idle sessions release HBM.
        - sessionType: agentic
          retain:
            quantile: 0.9
            minTTL: 30s
            maxTTL: 10m
        # Machine-paced loop, seconds-scale gaps: short protection is
        # enough, and a tight cap keeps turnover fast.
        - sessionType: agentic-subagent
          retain:
            quantile: 0.95
            minTTL: 1s
            maxTTL: 10s
```
