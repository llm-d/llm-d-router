# Program-Score Ordering Policy

**Type:** `program-score-ordering-policy`

The program-score ordering policy orders requests within a single queue by the originating
program/agent's decayed turns taken and tokens consumed, dispatching the least-served program's
request first.

## Why Choose This Policy?

- **Per-agent queue-per-tenant granularity problem:** the `program-aware-fairness` policy maps
  each agent identity to its own queue, so a tenant running 5 agents gets 5 queues and 5x the
  dispatch share of a tenant running 1 agent. Scoring within one shared queue removes that
  dependency on how many agents a tenant happens to run.
- **Program-aware without an extra queue per agent:** suited to configurations where many agents
  share a single `FlowKey` (e.g. one FairnessID-less priority band) but should still take turns
  fairly by their own consumption, not by arrival order alone.

## What It Does

Each item's program is identified by its request's FairnessID (falling back to the default flow ID
when unset, same as `program-aware-fairness`). Two signals are tracked per program, each decaying
independently with the same exponential half-life:

- **Turns taken**, incremented once per dispatched request.
- **Tokens consumed**, incremented by `PromptTokens + CompletionTokens` once a response completes.

The combined cost is `weightTurns*decayedTurns + weightTokens*decayedTokens`; the item belonging to
the program with the **lower** cost dispatches first. Equal cost falls back to FCFS (enqueue time).

This is a `ScoringOrderingPolicy`: because turns/tokens decay with elapsed time, the FlowController
periodically re-heapifies queues using this policy so an item that has been sitting in the queue
(while other programs are served) surfaces at the root as its relative cost falls, even though it
was never itself added or removed.

## Inputs consumed

- **FairnessID** (via the request's `InferenceRequest`), to identify the program.
- **Dispatch events**, via the `PreRequest` hook, to increment turns.
- **Response token usage**, via the `ResponseBody` hook (`response.Usage`, only on the final/only
  chunk), to increment tokens.
- **Logical Enqueue Time**, as the FCFS tie-breaker.

## Behavior and Queue Pairing

Requires the same heap-based priority queue capability as `edf-ordering-policy`.

## Configuration

```yaml
orderingPolicyRef: program-score-ordering-policy
```

| Field                     | Default | Description                                                  |
| ------------------------- | ------- | -------------------------------------------------------------|
| `weightTurns`              | `1`     | Weight applied to the decayed turn count.                    |
| `weightTokens`             | `0.01`  | Weight applied to the decayed token count.                   |
| `halfLifeSeconds`          | `60`    | Exponential decay half-life applied to both signals.          |
| `rescoreIntervalSeconds`   | `1`     | Minimum interval between forced queue re-heapifies.           |
| `evictionTtlSeconds`       | `3600`  | How long an idle program's state is kept. `0` disables eviction. |
| `evictionSweepSeconds`     | `300`   | Interval between idle-eviction sweeps.                        |

`weightTokens` defaults small relative to `weightTurns` since token counts typically run one to two
orders of magnitude higher than turn counts; tune both to your workload's actual token volume.

## Observability

| Metric                             | Type    | Labels       | Description                              |
| ----------------------------------- | ------- | ------------ | ----------------------------------------- |
| `program_score_decayed_turns`       | Gauge   | `program_id` | Current decayed turn count per program.   |
| `program_score_decayed_tokens`      | Gauge   | `program_id` | Current decayed token cost per program.   |

## Trade-offs

- **Rescore cost:** a full band rescore is `O(n log n)` in the queue's size, run at most once per
  `rescoreIntervalSeconds` (never finer than the FlowController's own expiry-sweep cadence).
- **Unbounded program cardinality:** per-program state grows with distinct FairnessIDs seen;
  `evictionTtlSeconds` bounds this by dropping idle programs, at the cost of a program's history
  resetting if it goes idle past the TTL.
- **Extensibility:** additional scoring signals (e.g. a future longest-prefix-match signal to
  prioritize cache-warm requests) are meant to be added as further independently-weighted,
  independently-decayed terms in the same cost function, not a new abstraction layer.

## Related Documentation

- [Ordering Overview](../README.md)
- [Program-Aware Fairness Policy](../../fairness/program-aware/README.md)
- [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
