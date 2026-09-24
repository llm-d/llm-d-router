# KV-Cache Hint v2 (bounded)

The `kv-cache-hint-v2` plugin annotates requests from multi-turn sessions
with KV-cache retention hints for backends implementing the vLLM
Context-Aware KV-Cache Retention API
([vllm-project/vllm#38514](https://github.com/vllm-project/vllm/pull/38514)),
like [`kv-cache-hint`](../kvcachehint/README.md), but bounds the protection:

| | `kv-cache-hint` | `kv-cache-hint-v2` |
|---|---|---|
| Protected range | full context (`start: 0, end: null`) | context head only (`start: 0, end: min(prompt tokens, maxProtectedPrefixTokens)`) |
| Global limit | none | `protectionBudgetTokens` caps the sum of live protected tokens across scopes |

Both bounds address the failure mode of whole-context protection under high
session concurrency: when the protected set exceeds the backend KV pool,
eviction degrades to non-recency order and the prefix-cache hit rate drops
(measured in `deploy/poc/agentic-serving/benchmark/results/REPORT.md`:
16 concurrent 100-260K-token sessions against a 2.2M-token pool cost 14
points of hit rate and tripled TTFT p90). The head of the context, the
system prompt and the shared history head, is what later turns and spawned
subagents re-read, so protecting only that range keeps most of the benefit
at a fraction of the pinned tokens.

## How it works

1. **Inputs.** Same as `kv-cache-hint`: the `SessionID` attribute from the
   `session-id-producer` and the `InterTurnPrediction` attribute from the
   `session-interturn-latency-producer`, both required at startup. Requests
   without either attribute pass through untouched.
2. **Queue matching and retention duration.** Same as `kv-cache-hint`: the
   prediction's `SessionType` selects a queue, and the duration is
   `clamp(Quantile(retain.quantile), retain.minTTL, retain.maxTTL)` of the
   queue's predicted inter-turn interval.
3. **Protected range.** The directive covers the first
   `min(prompt tokens, maxProtectedPrefixTokens)` tokens. The prompt size
   comes from the tokenized prompt when the tokenizer plugin runs, otherwise
   from a bytes/4 estimate of the raw body; an unknown size charges the full
   bound so the budget never undercounts.
4. **Budget.** A tracker holds one live reservation per retention scope,
   expiring with the directive TTL. A reservation renews in place: renewal
   costs nothing while the head size is unchanged, growth beyond the budget
   falls back to the existing size, and a scope that cannot fit a first
   reservation gets no directive at all; those requests fall back to plain
   LRU until earlier protections expire.
5. **Emission.** Same request-body fields as `kv-cache-hint`
   (`retention_directives`, `retention_scope`), with a numeric `end`:

   ```json
   "retention_directives": [{"start": 0, "end": 16384, "priority": 70, "duration": 12.8}],
   "retention_scope": "<session id>"
   ```

   Client-supplied directives win; the scope is the session or, when
   `parentSessionHeader` names one, the parent session.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `parentSessionHeader` | `x-parent-session-id` | Header carrying the identifier of the session that spawned this one, used as the retention scope when present. Empty disables parent scoping. |
| `queues` | one `agentic` queue | Per-workload-type retention policies, as in `kv-cache-hint`. Empty configures a single catch-all queue. |
| `retain.quantile` | `0.9` | Quantile of the predicted inter-turn distribution used as the retention duration, in (0, 1). |
| `retain.minTTL` | `1s` | Lower clamp on the retention duration. |
| `retain.maxTTL` | `10m` | Upper clamp on the retention duration. |
| `retain.priority` | `70` | Eviction priority (0-100) written into the directive. |
| `maxProtectedPrefixTokens` | `16384` | Protected head size per session. Size it to cover the system prompt plus the shared history head. |
| `protectionBudgetTokens` | `1000000` | Cap on the sum of live protected tokens across scopes. Size it well below the backend KV pool (half is a reasonable start) so unprotected blocks always leave the evictor room to work with. Must be >= `maxProtectedPrefixTokens`. |

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
        - sessionType: main
  - type: kv-cache-hint-v2
    parameters:
      # Protect the system prompt + shared history head of main sessions,
      # bounded so 16 concurrent sessions pin at most ~262K of a 2.2M-token
      # KV pool.
      maxProtectedPrefixTokens: 16384
      protectionBudgetTokens: 1100000
      queues:
        - sessionType: main
          retain:
            quantile: 0.9
            minTTL: 10s
            maxTTL: 10m
```
