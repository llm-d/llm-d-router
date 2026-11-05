# Thread Identity

**Type:** `thread-identity`

Thread Identity gives every request the identity of the **conversation it belongs to**. It renders the request through the model's own tokenizer, hashes the tokens into a block-key chain, and matches that chain against the chains of the threads it has seen: a prompt that extends a known thread continues it, a prompt that extends a **busy** thread forks a sibling, and anything else starts a new thread. The id is published as the request's flow key, so flow control queues one conversation per queue, and the same id is the key the Group Scheduler seats.

The identity is **content only**. No header, cookie, tenant id or session marker takes part: two clients whose conversations are identical so far are the same thread. That is the design's one rule, and it is what makes the identity survive clients that do not cooperate — an agent framework that sends no session header, a client that reconnects, a retry after a timeout.

Registered at **Alpha** stability, so the EPP requires `--allow-experimental-plugins`. Without the flag the EPP refuses to start.

## Why choose this plugin?

Flow control needs a queue key before it can queue anything, and a sensible key for conversational traffic is the conversation. The alternatives do not survive contact with real clients:

*   **A header.** Correct when the client sends one, absent when it does not — and the clients that matter (agent frameworks, SDK retries, browser tabs) frequently send none. An absent key collapses every conversation into one queue.
*   **The request id.** Every request gets its own, so every turn is its own flow: no conversation ever shares a queue, and nothing can be pinned to an engine across turns.
*   **A tenant id.** Right for isolation, wrong for the queue: every conversation of a tenant lands in one queue, so the session's turns serialise behind each other.

Content is the one signal every client provides, whether it knows it or not.

## What it does

1.  **Render.** At `RequestHeader`, the request body is rendered through the token producer's backend — the model's own `chat/completions/render` (or `completions/render`, `messages/render`, `responses/render`) endpoint — producing the exact token ids the engine would prefill. A body that already carries tokens (a gRPC parser, or client-supplied prompt token ids) is used as sent rather than re-rendered, so the identity keys on what the engine will actually consume.
2.  **Chain.** The tokens are hashed into a chain of block keys at `blockSizeTokens`, using the same prefix-hash function the prefix scorers use, so a block key means the same thing to the identity, the scheduler and the cache index.
3.  **Match.** The chain is matched against the resident tips by the **longest verified prefix**, descending over the arrival's own block hashes (a tip need not exist at every length). The match also scores how many full blocks were hit.
4.  **Resolve** one of three outcomes:
    *   **continue** — the tip is free, or the request provably follows it (below): the arrival joins the thread, its turn number increments;
    *   **fork** — the tip is busy and the request cannot be proven to follow it: a **new** thread is minted from the same content, so the two turns run in parallel rather than serialising;
    *   **new** — nothing matched.
5.  **Publish** the thread id, the arrival record, the block keys, an outstanding marker, and the thread id again as the flow key (see *Outputs produced*).
6.  **Release.** At `PreRequest` the arrival marker becomes an in-flight marker; at `ResponseBody` (end of stream) or `RequestAbandoned` the marker is removed, so the thread's tip becomes free for its next turn.

**Two guards decide "provably follows",** both on by default:

*   **Sticky-busy (H1)**: a request that extends the tip *and* begins with the tip's own assistant reply is a continuation even while the thread looks busy — the client is demonstrably reading the reply it just received.
*   **Retry (S2)**: a request byte-identical to the tip, whose thread has nothing actually in flight (only arrival markers), is a client retry of one request, not a sibling — the two merge in the queue instead of running twice.

**One thread per conversation, one FIFO per thread.** Turns of a thread never run concurrently unless the client's prompt forces a fork: a parallel sub-agent, a retried and a continued request, two branches of one history. A fork inherits nothing but the content it shares.

## Unit of Fairness

**The thread**, which for this plugin is the conversation. The id is an opaque 10-hex-character value derived from the content — the tip's block hash and the fork ordinal — so it is stable across restarts for the same conversation prefix and short enough to be a metric label.

The scheduler that consumes it treats one thread as one FIFO on one engine; see the [Group Scheduler](../groupsched/README.md) for how threads share an engine.

## Outputs produced

Per request, at `RequestHeader`:

| Key | Value | Consumed by |
| --- | --- | --- |
| `thread-identity` | the thread id | the Group Scheduler (queue key, seat identity) |
| `thread-arrival` | `{Outcome, Turn}`: continued / forked / new, and the turn number | the Group Scheduler (newcomer vs. returner) |
| `thread-blocks` | the chain's block keys, one entry per prompt, with its token count | the Group Scheduler (the exact prefill cost) |
| `thread-marker` | the request's outstanding marker | this plugin only |
| `agent-identity` | **the thread id again** | the director's fairness-ID derivation — this is the request's flow key |

**Why the fourth key.** The flow key is a *field* of the request, and the director derives it after the header processors run, from the `agent-identity` attribute — the one attribute it turns into a flow key. A request-header plugin cannot set that field directly: the director hands each plugin a scoped copy of the request whose attribute store is shared but whose struct fields are not, so a field write would be lost. The key's documented contract is "any plugin able to resolve an agent identity is an acceptable source", and this plugin resolves the conversation, which is the identity flow control should key on.

**The consequence, stated plainly:** the fairness-ID resolution source label for these requests reads `agent_identity`, not a name of their own, because the attribute is upstream's. It is the only place the borrowed name surfaces.

**This plugin owns that attribute.** Registering, it refuses to start if any other configured plugin declares `agent-identity` in its `Produces()` or its `Consumes()`. A second producer would silently overwrite the flow key, by configuration order, with no error from the framework; a reader would take thread ids for session ids. Today that covers `agent-identity`, `thunderagent` and `sessionstate`. A plugin that reads the attribute **without declaring it** cannot be detected — that is the check's one blind spot.

## Inputs consumed

*   **The request body** (messages, tools, system prompts, multimodal parts): rendered through the token producer.
*   **Client-supplied prompt token ids**, when the request carries them (`kv_transfer_params.prompt_token_ids`, token-array completions prompts): used as sent, because the engine consumes exactly those.
*   **The request's explicit fairness header** (`x-llm-d-inference-fairness-id`), only to count that an operator override is in effect. The identity is still derived and still published: with composition on, the director joins the tenant and the thread (`tenant/thread`); with it off, the tenant keys the flow as sent.
*   **Endpoint lifecycle events**, when the token producer's backend discovers render endpoints rather than using a fixed URL.

No response content is read except the end of stream, which releases the thread's marker.

## Configuration

```yaml
- type: thread-identity
  parameters:
    blockSizeTokens: 64
    tokenProducer:
      modelName: your-model
      vllm:
        url: http://localhost:8000
    markerTtlSeconds: 9300
    maxThreads: 100000
    forkOnBusy: true
    stickyBusyGuard: true
    retryRule: true
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `blockSizeTokens` | int | `64` | The identity's block length in tokens. Must equal the engine's `--block-size` and the Group Scheduler's `blockSizeTokens`; the scheduler refuses to start on a disagreement, naming both numbers. |
| `tokenProducer` | object | **required** | The render backend, in the token-producer plugin's own parameter language (`modelName` + `vllm.url`, or `vllm.endpointDiscovery`). A config without it, or one selecting the estimate backend, **refuses to start**: identity in byte space is not an identity. |
| `markerTtlSeconds` | float | `9300` | How long an outstanding marker that never settles keeps its thread busy. Must exceed the client's request timeout (the default is a 9000 s harness timeout plus headroom). `queuedTtlSeconds` is the deprecated spelling, still accepted. |
| `maxThreads` | int | `100000` | Cap on the thread table — the only eviction path; idle non-outstanding threads are evicted first. |
| `sweepSeconds` | float | `300` | The cap-enforcement tick. |
| `forkOnBusy` | bool | `true` | The continuation rule's busy half. `false` makes every match a continuation, which is a measurement arm, not a deployment. |
| `stickyBusyGuard` | bool | `true` | H1, above. |
| `retryRule` | bool | `true` | S2, above. |
| `logRequests` | bool | `false` | One JSON line per request at the default log level. A scoring instrument; it changes no state and no decision. |
| `logHeaders` | bool | `false` | Adds the client's own identity headers to that line (an observer only: no header here is ever read by the identity). |
| `logHeaderNames` | list | `[]` | Extra headers for that whitelist. |

The parameters are decoded **strictly**: an unknown key fails the startup. That matters here — the plugin's surface shrank when session scoping was removed, and a stale `sessionHeaders` key must fail loudly rather than do nothing.

### Which parameters to set

*   **Set `blockSizeTokens` whenever the engine's `--block-size` is not 64.** It must match the scheduler's and the engine's.
*   **Set `tokenProducer`'s URL** to the render sidecar the deployment runs (or use endpoint discovery on a cluster where the render endpoints are published).
*   **Leave the rest alone** unless the workload says otherwise. `markerTtlSeconds` must exceed the client's request timeout; `maxThreads` only binds a stack serving more than 100k concurrent conversations.
*   **`forkOnBusy: false` is a measurement arm.** It removes the fork rule, so concurrent turns of one conversation serialise; use it to measure what forking buys, not in production.

### Required wiring

```yaml
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
    blockSizeTokens: 64
    groupBudgetFraction: 0.95
```

*   **This plugin must come before admission.** It is a `RequestHeader` processor: the drawer runs it after body parsing and before `Admit`, so the flow key exists when the request enqueues.
*   **No `token-producer` plugin is needed beside it.** This plugin renders at arrival through the same upstream tokenizer; the dispatch-time producer, if present, finds `Body.TokenizedRequest` already populated and returns on its skip path. Both then read one render.
*   **The Group Scheduler requires this plugin.** Without it every request is anonymous and the scheduler serves everything through the pass-through.
*   **Do not also configure `agent-identity`, `thunderagent` or `sessionstate`.** Each declares the flow-key attribute in some direction and this plugin refuses to start beside them.

## Observability

All series are under `llm_d_epp_thread_`.

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `new_total` | counter | — | arrivals that started a new thread |
| `matched_total` | counter | — | arrivals that continued their thread |
| `forks_total` | counter | — | arrivals that forked a new thread off a busy one |
| `sticky_busy_total` | counter | — | continuations proven by H1 while the tip looked busy |
| `retry_total` | counter | — | arrivals resolved as client retries by S2 |
| `evicted_total` | counter | — | threads evicted by `maxThreads` |
| `render_failure_total` | counter | — | requests the render could not describe; each still gets its own thread |
| `multi_prompt_requests_total` | counter | — | requests whose body carried several prompts |
| `explicit_override_total` | counter | — | requests that carried an explicit fairness header (an operator override is in effect) |
| `records` | gauge | — | threads currently in the table |
| `render_ns` | histogram | — | time in the render |
| `chain_ns` | histogram | — | time hashing the tokens into a chain |
| `resolve_ns` | histogram | — | time in the match-and-mint decision |

The outcome counters should sum to the arrival rate. A `forks_total` that stays high on a workload with no parallel sub-agents means the continuation guards are not firing — check `retryRule` and `markerTtlSeconds` against the client's behaviour. `render_failure_total` above zero means the render sidecar is failing or the model name is wrong; the traffic still runs, one thread per request, which looks like a working stack in every latency metric.

## Trade-offs

*   **Content is the whole identity.** Two clients whose conversations are identical so far share one thread, and therefore one queue and one seat. In practice a prompt is a conversation's own text, so this is rare; when it happens the two turns continue each other and the second sees the first's queue position. That is the accepted cost of needing no client cooperation.
*   **A render on the arrival path.** Turning content into tokens costs a `/render` round trip per request (~a hundred milliseconds on a loaded sidecar). It is one render for the whole stack — the dispatch-time producer and every prefix scorer reuse it — but it is on the critical path of the request's first admission.
*   **The identity is only as good as the tokenizer agreement.** If the render backend's chat template differs from the engine's, the chain differs from what the engine prefills and the cache hit the scheduler charges for does not exist. Both must be the same model and template.
*   **A cap is an eviction, not a bound on memory.** `maxThreads` keeps the table finite by evicting idle threads, so a conversation that returns after eviction starts a new thread — with the correctness that its queue key is new, and the cost that its seat (if any) is re-decided.
*   **The state is in-process, one instance per EPP.** Two EPP replicas derive identities independently and correctly — the derivation is a pure function of content — but each keeps its own thread table and its own seats; there is no cross-replica coordination.

## Related Documentation

*   [Group Scheduler](../groupsched/README.md): the consumer that makes the thread a seat
*   [Token Producer](../../dataproducer/tokenizer/README.md): the render backend this plugin drives
*   [Prefix Hash](../../dataproducer/prefixhash/README.md): the block-key function
*   [Fairness Overview](../../fairness/README.md)
