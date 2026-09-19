# KV-Cache Retention Plugin

The `kv-cache-retention` plugin annotates requests from multi-turn sessions
with retention directives for backends implementing the vLLM Context-Aware
KV-Cache Retention API
([vllm#37003](https://github.com/vllm-project/vllm/issues/37003),
[vllm#38514](https://github.com/vllm-project/vllm/pull/38514)).

## How it works

The plugin consumes two request attributes: the session identifier published
by the `session-id-producer` and the `InterTurnPrediction` published by the
`session-interturn-latency-producer`. Each request carrying both gets a
single whole-prompt directive in the forwarded body:

```json
{
  "retention_directives": [
    {"start": 0, "end": null, "priority": 70, "duration": 54.5}
  ],
  "retention_scope": "<session id>"
}
```

`duration` is the configured `quantile` of the predicted inter-turn
distribution, clamped to `[minRetention, maxRetention]`: the session's KV
blocks stay protected while the next turn is likely to arrive and fall back
to LRU once the session is far into the distribution's tail. Requests that
already carry `retention_directives` are forwarded unchanged, as are bodies
that are not parsed JSON maps (raw or proto payloads) and requests without a
prediction (no session identifier or non-matching workload type; see the
producer's README for the gating rules).

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `priority` | `70` | Eviction priority (0-100) written into the directive. |
| `quantile` | `0.9` | Quantile of the predicted distribution used as the retention duration. |
| `minRetention` | `1s` | Lower clamp on the retention duration. |
| `maxRetention` | `10m` | Upper clamp on the retention duration. |

## Example

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. The plugin requires the `session-id-producer`
and `session-interturn-latency-producer` entries; the framework orders the
producers before the plugin's PreRequest hook.

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
  - type: kv-cache-retention
    parameters:
      priority: 70
      quantile: 0.9
```

## Scope

This plugin covers the agentic timing model of SAECache
([arXiv:2605.18825](https://arxiv.org/pdf/2605.18825)) only. The remaining
SAECache components — per-queue weights, token-type weights, and positional
decay for structural reuse — depend on eviction feedback that the retention
API does not expose and are out of scope.
