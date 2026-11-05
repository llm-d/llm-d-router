# Group Scheduler

**Type:** `group-scheduler`

The Group Scheduler is a fairness policy that admits requests **per conversation**. It derives nothing itself: the thread-identity plugin reads every request's content, renders it through the model's own tokenizer and publishes a thread id, and this policy seats one *thread* per engine and charges that engine's KV budget with the thread's exact prefill blocks. A thread keeps its seat across turns, so a conversation's KV cache is not thrashed by the turns of other conversations; when several threads want the same engine, they rotate a fixed tenure rather than compete per request.

One plugin instance answers three questions: it is the band's **FairnessPolicy** (which request to dispatch), the **Filter** (which endpoint may serve it — the thread's seat) and the **SaturationDetector** (whether the pool is full). The three answers come from one state, so they cannot disagree.

Registered at **Alpha** stability, so the EPP requires `--allow-experimental-plugins`. Without the flag the EPP refuses to start.

## Why choose this policy?

Consider a pool of two model servers and a workload of multi-turn conversations — coding agents, chat sessions, anything where a later turn re-sends the history of the earlier ones.

**Under FCFS or a load-balancing policy**, consecutive turns of one conversation land wherever has room. Each turn's prompt contains the previous turns' tokens, so the KV blocks a turn just computed are useful to the next turn only if the next turn runs on the same engine. Spread across engines, a warm cache is hit at random, prefill is repeated, and TTFT carries the cost. The scheduler cannot see the difference: every request looks equally cheap.

**Under a prefix-cache or affinity score**, a request is placed where its prefix is likely cached, but placement is still decided per request and nothing bounds how many conversations share one engine. Under load the cache is larger than the KV budget, so every new turn evicts another conversation's blocks; the engines preempt, and the cache the scoring relied on is gone by the time it is used again.

**Under the Group Scheduler**, the unit is the conversation. A thread is seated once, its blocks are charged to the engine's budget for as long as it stays, and its later turns are pinned there. The seat budget (`groupBudgetFraction` of the engine's measured `num_gpu_blocks`) bounds how many conversations an engine holds, so the resident set fits the cache the engines actually have. A thread that has used its tenure (or gone quiet) leaves, and the next waiting thread takes the seat — so a burst of new conversations is admitted in a bounded number of rotations rather than by evicting everything at once.

## What it does

1.  **Tidy**: the pass reconciles its accounting with the requests it can see — a released turn whose window outlived the reap threshold is closed and counted (`anomalies_total{kind="reaped_window"}`), an in-flight turn likewise (`reaped_inflight`), and an engine whose endpoints are gone is torn down.
2.  **Queue**: the heads this pass was handed are reconciled with the shared admission queue. A seated thread's head stays on its engine; an unseated thread's head waits in **one** queue for any engine.
3.  **Derive the cold charges**: each seated thread's un-prefilled tokens (`ColdTokens`) are what it would have to prefill if served now. They feed the cold-start stagger.
4.  **Enforce the budget**: each engine's charged blocks (`ne_blocks`) are brought back under its seat budget by evicting residents in a fixed class order — a thread with no queued turn first, then one past its tenure, then the longest-waiting — with the engine's smallest-cost victim chosen inside each class. Exiting threads keep their blocks until their in-flight requests drain, and enforcement no longer evicts for them.
5.  **Seat waiters**: free budget admits waiting threads, newcomers before returners and then oldest first, with a cold-start stagger: an engine's outstanding cold-entry charge is capped at `(1 − groupBudgetFraction)` of its KV, so a burst of large prompts cannot make every resident prefill at once. **Rule O**: an empty engine admits one oversize head, charging its full exact chain, rather than refusing it — an over-budget counter is recoverable, an idle engine is not.
6.  **Dispatch**: among seated threads with a dispatchable head, the engine with the fewest queued turns per seat wins; ties go to the older head. The dispatched turn's chain is charged to that engine in the same critical section as the decision, and the turn is handed to the framework. A head with no engine to serve it — anonymous traffic, a thread whose render produced no tokens, a stack whose capacity has not been measured yet — is served last, in arrival order, unseated and uncharged.

**Rotation.** A seat is held for `quantum` completed busy periods (a busy period is one turn's work: released, run, settled) or until the thread has been idle for `idleSeconds`. Either makes the resident rotation-eligible; enforcement then releases the seat so a waiting thread can take it. A thread that returns finds its seat if it is still there, and re-enters as a returner if it is not.

**The observed engine wins.** If a request's `PreRequest` reports an endpoint other than the seat, the thread is moved to the observed engine and the blocks are re-seated there (`affinity_violations_total` counts the event). The framework's choice is authoritative about where the request ran; the policy's model follows it.

## Unit of Fairness

**The thread** — one conversation, from thread identity. A thread is one FIFO of turns on one engine, so fairness is per conversation, not per request: a conversation with many short turns does not get a larger share of an engine than one with few long turns, it just uses its seat more often.

*   **Newcomers before returners.** Within a waiter queue, a thread's first turn precedes a returning thread's next turn, and among equals the older arrival goes first. A newcomer is a conversation that has not been served; a returner already has a seat's worth of history on the fleet.
*   **The session is an attribute, not a unit.** Thread identity may publish a session id, and this policy carries it for the `SEAT`/`EXIT` events, but it never groups threads by it. One session's parallel threads are independent units.
*   **The engine is the resource.** A thread's seat is on one engine; the budget it draws from is that engine's. Two engines are two budgets, and a thread does not span them.

## Inputs consumed

*   **Thread identity's attributes**: the thread id (the queue key and the seat identity), the arrival record (first turn or later, and which turn), and the rendered block keys the thread's prompt hashes to.
*   **The rendered tokens**: the block keys are hashed once, at arrival, by thread identity; this policy charges exactly those blocks. It performs no rendering of its own.
*   **Endpoint metrics**: `num_gpu_blocks` and the block size, read from the engine's own metric endpoint, once per engine, at discovery or on a refresh — **the only two things this policy ever takes from vLLM**. No live metric (queue depth, KV utilization, TTFT) enters any decision.
*   **Lifecycle events**: an endpoint appearing or disappearing; a request's `PreRequest` (which engine actually served it), its first response chunk (prefill done) and its end of stream.

## Configuration

```yaml
- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95    # seat budget = fraction x the engine's num_gpu_blocks
    quantum: 4                   # completed busy periods a seat is held before rotation
    idleSeconds: 20              # silence after which a seat is rotation-eligible
    inFlightCap: 64              # how many of one thread's requests may be in flight
    blockSizeTokens: 64          # must equal vLLM's --block-size and thread-identity's
    inflightReapSeconds: 9300    # a tracked request older than this is closed by the reap
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `groupBudgetFraction` | float | `0.95` | Fraction of an engine's measured `num_gpu_blocks` that may be charged to seats. Must be in (0, 1]. |
| `quantum` | int | `4` | Q: completed busy periods a seat is held before it becomes rotation-eligible. Must be ≥ 1. |
| `idleSeconds` | float | `20` | S: silence after which a resident becomes rotation-eligible (it is not going to use its seat). Must be > 0. |
| `inFlightCap` | int | `64` | Maximum in-flight requests of one thread. Must be ≥ 1. |
| `blockSizeTokens` | int | `64` | The engine's KV block size in tokens. Must equal vLLM's `--block-size` and thread identity's `blockSizeTokens`; the startup refuses a disagreement, naming both numbers. |
| `inflightReapSeconds` | float | `9300` | Age beyond which a tracked request is closed by the reap. Must exceed the client's request timeout. |

The parameters are decoded **strictly**: an unknown key fails the startup instead of being ignored.

### Which parameters to set

Most deployments set **one** of these.

*   **Set `blockSizeTokens` whenever the engine's `--block-size` is not 64.** It is the only parameter that can silently corrupt every cost: the policy charges the block keys thread identity hashed, so a disagreement means every charge is wrong. Both plugins refuse to start if they disagree.
*   **Leave `groupBudgetFraction` alone** unless the engine is shared with something else. `0.95` leaves 5% of the KV for the engine's own working room; the remainder doubles as the cold-start stagger cap.
*   **Leave `inFlightCap` alone** unless a single conversation is expected to fan out wider than 64 parallel turns.
*   **`quantum` is a workload choice.** It trades locality for the wait of others: a larger Q keeps a conversation's cache resident longer (better hit rate) and makes a waiting conversation wait longer to be admitted. 4 fits conversational traffic; a fan-out agentic workload with many short conversations wants 1–2.
*   **`idleSeconds` is a workload choice.** Set it above the client's typical gap between turns — a tool call, a human reading — or a conversation's seat is released between its own turns. It is measured from the last activity, not from the last completion.
*   **`inflightReapSeconds` is a safety belt.** Set it above the client's request timeout. It exists for a termination no callback reported; a value below the timeout closes live requests.

### Required wiring

The policy is one plugin but it answers three interfaces, and it needs thread identity beside it. A configuration that omits any of these refuses to start rather than running a degraded policy.

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: thread-identity
  parameters:
    blockSizeTokens: 64
    tokenProducer:
      modelName: your-model
      vllm:
        url: http://localhost:8000
- type: group-scheduler
  name: group
  parameters:
    groupBudgetFraction: 0.95
    quantum: 4
    idleSeconds: 20
    inFlightCap: 64
    blockSizeTokens: 64
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: group
flowControl:
  defaultRequestTTL: 0s
  maxBytes: "1000000000"
  maxRequests: 5000
  priorityBands:
  - priority: 0
    maxBytes: "1000000000"
    maxRequests: 5000
    fairnessPolicyRef: group
  saturationDetector:
    pluginRef: group
```

*   **`thread-identity` must be in the same config.** Without it every request is anonymous, Pick serves every head through the pass-through, and the stack looks healthy in every outcome metric while nothing is seated.
*   **The band's `fairnessPolicyRef` and the `saturationDetector` slot both name the instance**, and the instance also appears in the scheduling profile as its Filter. One instance behind three references is the supported shape; the loader's DAG validation rejects a reference whose producer sits in a later layer.
*   **The queue limits belong at the maximum** (the defaults shown). This policy decides admission; a flow-control TTL or a byte cap that binds first would deny requests the policy had decided to serve, and the denial would be invisible to it.
*   **`defaultRequestTTL: 0s`** means no queue TTL: a request waits for a seat rather than being dropped, which is the policy's own decision to make.
*   **No `token-producer` plugin is needed.** Thread identity renders the request's tokens itself, through the same upstream tokenizer, at arrival; a second producer in the chain would find the tokens already present.

## Observability

All series are under `llm_d_epp_group_`. Gauges are rendered at scrape from one snapshot; each engine's counters keep their series after the engine is gone, because totals sum them.

| Metric | Type | Labels | Meaning | Healthy reading |
| --- | --- | --- | --- | --- |
| `seats` | gauge | `engine` | seated threads on this engine | between 1 and the budget's capacity; a floor of 0 on a busy stack means nothing is being admitted |
| `ne_blocks` | gauge | `engine` | distinct blocks charged to the engine's budget | ≤ `budget_blocks`; above it is a transient the enforcement pass is correcting |
| `budget_blocks` | gauge | `engine` | the seat budget: `groupBudgetFraction` × `num_gpu_blocks` | constant per engine |
| `exiting_threads` | gauge | `engine` | threads marked exiting that still hold their seat until their requests drain | returns to 0; a value that persists is a drain that is not happening |
| `exiting_blocks` | gauge | `engine` | the part of `ne_blocks` that belongs to exiting threads | bounded by `ne_blocks` |
| `exiting_oldest_age_seconds` | gauge | `engine` | how long the oldest exiting thread has been on its way out | stays small; only growing means a stuck drain |
| `cold_inflight_tokens` | gauge | `engine` | outstanding cold-entry charge against the stagger cap | ≤ `cold_cap_tokens` |
| `cold_cap_tokens` | gauge | `engine` | derived stagger cap: `floor((1 − groupBudgetFraction) × num_gpu_blocks) × blockSizeTokens` | constant per engine |
| `capacity_missing` | gauge | `engine` | 1 while the engine's `num_gpu_blocks` is unknown | 0 once the metrics arrive; 1 for long means the engine serves FCFS only |
| `waiters_shared` | gauge | — | heads waiting in the one shared admission queue | the queue depth the policy sees |
| `dispatches_total` | counter | `engine` | dispatches served by each engine's seats; `engine="nocapacity"` counts the pre-capacity pass-through | sum is the request rate |
| `exits_total` | counter | `engine`, `reason` | thread exits by reason (`rotation`, `ttl`, `engine-gone`, `exiting-drain`, …) | the mix shows whether tenure or silence is doing the releasing |
| `entry_wait_seconds` | histogram | `engine`, `kind` (`new`, `reentry`) | **seat** waits: how long admission made a thread wait, arrival to first seat | p99 well below the client's patience |
| `queue_wait_seconds` | histogram | `kind` (`new`, `returner`, `untracked`) | **queue** waits: dispatch time minus the head's arrival, per request — every dispatched turn, including a seated thread's turn waiting on the in-flight cap or the dispatch order | the wait a request actually experiences; its difference from `entry_wait_seconds` is what the seats cost |
| `dispatched_prompt_tokens_total` | counter | `engine` | prompt tokens of every dispatched turn, summed over a request's prompts | the denominator of the predicted hit rate |
| `predicted_cached_tokens_total` | counter | `engine` | prompt tokens the engine was predicted to hold as a resident prefix, measured read-only **before** the turn's own charge | ratio to the line above is the predicted prefix hit rate |
| `cold_valve_held_total` | counter | `engine` | Pick passes in which the head was held by this engine's cold-entry valve | a rate, not a count: Pick runs at ~1 kHz |
| `anomalies_total` | counter | `kind` | should-be-zero events the policy could not act on (below) | `no_tokens_head` 0, `no_request_id` 0 in practice, `reaped_*` 0 |
| `affinity_violations_total` | counter | `from`, `to` | requests that landed on an engine other than the thread's seat (the thread was rebound) | 0 is the invariant; a value means the placement and the seat disagreed |
| `engine_gone_total` | counter | — | engines torn down by an endpoint delete | follows endpoint churn |

**Anomaly kinds.** `no_tokens_head` — one Pick **pass** saw a thread head with no rendered tokens, so it could not be seated or charged; a render failure, and a rate rather than a request count. `no_request_id` — a turn was released for a request with no id; a boundary guard, 0 in practice because the router assigns every request one. `reaped_inflight` / `reaped_window` — a request (or a released window) outlived the reap and had to be closed by it; these are the leak signals.

**Span attributes** (set at `PreRequest`, only when the request's span is recording): `llm_d.epp.group.queue_wait_ms`, `llm_d.epp.group.predicted_cached_tokens`, `llm_d.epp.group.prompt_tokens`, `llm_d.epp.group.seat_engine`. Nothing from the response is read for them.

## Trade-offs

*   **A returner's tail can wait behind newcomers.** Newcomers go before returners by design — a fresh conversation should not starve behind a fleet of established ones — so under saturation a conversation's next turn may wait for an admission while newer conversations are admitted. That is the choice; if returner latency matters more than newcomer fairness, a smaller `quantum` releases seats sooner.
*   **Nothing is admitted until capacity is measured.** Before an engine reports `num_gpu_blocks`, every request is served FCFS and unseated (`dispatches_total{engine="nocapacity"}`, `capacity_missing=1`). The policy cannot size a budget it does not know.
*   **One KV domain per endpoint.** The budget counts blocks on the endpoint it measures. An endpoint whose blocks are shared with another replica, or a deployment where two endpoints front one engine, breaks the model: the policy will over-count and under-admit.
*   **Prefix caching must be on.** The policy charges a thread's full block set and relies on the engine's prefix cache to make a later turn's shared blocks cheap. With `--no-enable-prefix-caching` (or a KV-cache-aware-off engine) the blocks are charged but not reused, so seats cost more than they return.
*   **The state is in-process, one instance per EPP.** Seats, the queue and the block index live in this process; a second EPP replica beside the first keeps its own, so two replicas will make independent admission decisions over the same pool and can each admit up to the full budget. Run one EPP per pool, or size the budget for the number of replicas.
*   **The seat is a prediction, not a reservation.** The engine may preempt a thread, and the observed placement can differ from the seat; the policy follows the observation and counts the violation, but a preemption it cannot see costs a re-prefill it did not budget for.

## Related Documentation

*   [Thread Identity](../threadidentity/README.md): the producer this policy depends on — it defines what a thread is
*   [Fairness Overview](../../fairness/README.md)
*   [Weighted Fair Share](../../fairness/weightedfairshare/README.md): the other fairness policy that divides a band by a declared weight
*   [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
