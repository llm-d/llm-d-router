# Branch Identity

**Type:** `thread-identity`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.PreRequest`,
`requestcontrol.ResponseBodyProcessor`, `plugin.ProducerPlugin`, `plugin.StateDumper`,
`datalayer.Registrant`

Derives the **flow-control queue key from the request body**: at arrival the plugin renders the
request with the token producer, hashes it into the prefix scorer's own block chain, and matches
it against the live branch table before admission control, so the request's `FairnessID` is a
derived **branch id**.

**Glossary** (used throughout this package): a *request* is one call (one turn); a *branch* is a
chain of requests, each continuing the previous, and its id is the queue key; the *tip* is a
branch's latest request; the three arrival outcomes are *continue*, *fork* and *new*; a *session*
is the client's session header — an attribute, never a key; an *engine* is one vLLM pod; a *seat*
is the right to run on an engine, held by one branch; a *queue* is branches waiting for a seat.

## Why the queue key is a BRANCH, not a session

The fairness id *is* the flow-control queue key: one id = one FIFO queue, and the scheduler only
ever sees queue heads. Keying by session therefore serialises a session's siblings behind each
other by construction — measured as TTFT p50 ×6.1 and a −13.5pt main-chain cache-hit collapse
(`v2c-analysis.md`). A branch is the unit that can dispatch in parallel: the parent turn and each
sub-agent that inherits its context get their own queue.

The **session** is still available — as an attribute (`branch-session`), never as a queue key.
The stage-2 session fairness layer reads it to tell a newcomer from an incumbent.

Nothing in the scheduler changes: the policy seats whatever id it is given.

## The identity is the engine's token stream

A branch is a **KV-cache lineage**, so the identity is the tokens the engine would render, in the
**prefix scorer's own hash space** — not a normalized view of the body:

1. **One render per request, at arrival.** `RequestHeader` calls the token producer as a library
   (`tokenizer.Producer.Tokenize`), which writes `Body.TokenizedRequest` + `CacheSalt`. The
   dispatch-time `token-producer` plugin then finds the tokens present and returns on its own skip
   path, as does every later consumer: **exactly one render per request**, shared by identity and
   every prefix scorer. The render is *relocated*, not added — ~120ms leaves the dispatch path for
   the arrival path (measured: 1.42µs/token linear, 91% of EPP processing, 6–15% of TTFT; the
   totals are unchanged).
2. **Block chain** (`prefixhash.GetBlockHashesWithPromptTokens`): B-token blocks, chained, seeded
   with the target model + cache salt — the same hashes the prefix scorer and the ledger use, so
   "same prefix" means the same thing to identity and to the cache. `blockSizeTokens` is *the same
   knob* as the scorer's `B`. All prompts count: a multi-prompt request's per-prompt chains are
   **concatenated** (the faithful mapping of the byte-era `strings.Join`), counted as
   `multi_prompt_requests_total`.
3. **The continuation rule.** R continues T iff **T's full-block chain, except its last full
   block, is a prefix of R's chain.**
   - The tip is indexed by its *second-to-last* full-block hash; the lookup probes k = R's own
     full-block count descending to 1. The tip index is **not prefix-closed** (a tip may exist at
     k=7 with none at k=5), so the probe must not stop at the first miss.
   - **Tail tolerance.** The unverified tail of a tip is therefore its partial block plus one full
     block: for a tip of two or more full blocks, **always between B and 2B−1 tokens of slack**. A
     request that differs from the tip only there continues it — which absorbs end-of-tip rendering
     variance that the history does not reproduce (the generation prompt, a template's empty-think
     stub, a BPE merge at the boundary). One block of slack, rather than the partial block alone, is
     what makes the guarantee hold wherever the prompt ends: with the partial block alone the slack
     can be *zero*, and an end variance that crosses the block edge re-keys the branch.
   - **Guards.** A tip with exactly **one** full block gets no tolerance — its prefix would be
     empty and match every request — so that block must match and only its partial block is
     unverified. A tip **shorter than one block** has no block hash to index and is not continuable
     at all: it re-keys once on its next turn.
   - **S2's exact key** is separate: `(token count, last hash including the partial block)`, i.e.
     the whole rendered prompt. Two requests with the same pair rendered the same prompt.
   - The **longest verified prefix wins**; among equally long candidates a free tip beats a busy
     one, then the longer tip, then the most recently seen (the id breaks an exact tie, so a replay
     is deterministic).
4. **Continue or fork**:
   - tip free (nothing in flight or pending) → **continue** (same id, same queue);
   - **H1 reply guard**: a request that extends a tip AND begins with the tip's assistant reply
     contains that reply, which an in-flight request cannot have — proven completion, continue.
     Stage 0 measured 42 of 48 fork candidates to be exactly this case, so the guard — not a busy
     grace window — is what keeps the fork rate honest (`stickyBusyGuard`, on by default). The
     guard is sound only because a match proves the tip's token prefix: message index `msgCount` in
     the request IS the first message past the tip.
   - **S2 retry rule**: a request **identical** to the tip while the branch has nothing IN FLIGHT
     (only arrival markers) is a client retry of a still-queued turn, not a sibling → continue
     (`retryRule`, on by default; counted as `retry_total`). The queue key is fixed at arrival, so
     two identical prompts arriving while the tip is still queued MERGE into one queue entry and
     nothing re-keys them later. That is the cheaper error (identical bodies share their whole
     prefix, so the merge costs no cache) against forking a queue on what may be a pure retry. An
     identical prompt arriving after the first one was DISPATCHED is decided by the busy flag and
     forks normally;
   - tip outstanding → **fork** a new branch. The usual case is a sub-agent inheriting the parent's
     context.
5. **Opaque stable ids**: 10 hex chars derived from the creating request's tip commitment, the fork
   ordinal and the session scope. The same conversation derives the same id on any instance with
   the same model, salt and block size, and a continuation keeps it for the branch's whole life. An
   id held by a **live** branch is never reused: a re-minted seed (a client re-sending an earlier
   turn after the branch advanced, a look-alike fork) bumps the ordinal and gets a distinct branch.
   Overwriting would leave the old record *ghosted* — still matched by the tip index, no longer in
   the table — so it would keep advancing requests and would never be evictable.

`x-llm-d-inference-fairness-id` still wins when present (an explicit override) and is counted
(`explicit_override_total`). **A render failure never fails the request**: the request gets its own
branch (a NEW keyed by the request, never a continuation) and `render_failure_total` names it. The
arrival render carries **no time constant of its own**: it runs under the request's context, bounded
by the render backend's own `timeout` (the token producer's, the deployment value) and by the client
abandoning the request. A shorter identity-specific bound was tried and removed — under an arrival
burst or a loaded sidecar it fires on requests that are merely slow, and a timed-out render mints a
false NEW branch, i.e. an identity we invented because we gave up.

## The arrival record (a published output)

Every derived request carries `branch-arrival`: the **outcome** (`matched` = continued, `forked`,
`new`) and the arrival's **1-based turn number on its branch** (1 for a branch's first request, and
for a fork's first). It is a pure output — written after the resolve, read by the scheduler to tell
a newcomer from an incumbent — and nothing in this plugin reads it back.

```go
record, ok := fwksched.ReadRequestAttribute[threadidentity.ThreadArrival](request, threadidentity.ThreadArrivalKey)
```

## Hook timing

| hook | what it does | why here |
|---|---|---|
| `RequestHeader` (pre-admission) | render → chain → resolve → sets `FairnessID`; publishes `thread-identity`, `branch-session`, `branch-arrival` | the queue key must exist **before** the request enqueues, and the render has to happen once — here, where the parse is already done |
| `PreRequest` (post-scheduling) | moves the request's marker from the branch's `pending` set to its `inflight` set | **not** where busy is set: busy is set at arrival, because under load the queueing interval is most of a busy period (v2c median wait 22s) |
| `ResponseBody(EndOfStream)` | removes the request's marker from the branch's `inflight` set | forced by the server on abort/disconnect/panic for every request that picked a pod |

**Markers are per request id, not counters.** Each branch holds a `pending` and an `inflight` *set*
of request ids (`x-request-id`, or a per-request synthetic id when the client sent none, published
as the `branch-marker` attribute so all three hooks agree). The outstanding predicate is "my sets
are non-empty", so a cancellation, a rejection, a retry and a settle each add or remove exactly one
entry — a FIFO counter cannot say *which* request ended, and shifting it on the wrong one either
frees a branch that still has work queued or holds one that has none.

## Retention

Branches are **never GC'd by silence**. Stage 0 measured a p99.9 inter-turn idle gap of 3,036
seconds: any plausible `T_gc` drops dormant threads that then return, and a dropped branch is a
**false new branch** — a fresh fairness id, i.e. a newcomer, which is the unsafe direction. Branch
records are content-addressed and tiny, the flow-control registry already GCs idle *queues*, and
the only eviction path is the count cap (`maxBranches`, idlest non-outstanding branches first).

**Markers expire after `markerTtlSeconds` (default 3600s — the client's request timeout), not after
the flow queue's 5-minute GC.** Those are different clocks: on A0 16% of requests waited more than
600s for admission, so a 600s horizon would expire the markers of requests still legitimately
queued and hand their branch to a concurrent sibling as if it were free. Expiry is the only way a
rejected-before-dispatch request is ever observed, and pruning errs toward continuation.

## Configuration

```yaml
plugins:
  - type: thread-identity
    parameters:
      blockSizeTokens: 64           # = the prefix scorer's B; the output is in ITS hash space
      tokenProducer:                # REQUIRED — the SAME render backend the token-producer plugin uses
        modelName: meta-llama/Llama-3.1-8B-Instruct
        vllm:
          url: http://127.0.0.1:8000
      markerTtlSeconds: 3600        # marker horizon = the client request timeout (NOT the queue GC)
      sweepSeconds: 60              # cap enforcement tick
      maxBranches: 100000           # the only eviction path
      # sessionHeaders: [x-my-session]   # prepended to the built-in provider list
      # forkOnBusy: true                  # false = every match continues (measurement arm only)
      # stickyBusyGuard: true             # H1 reply guard
      # retryRule: true                   # S2: an exact duplicate of a still-queued tip is a retry
      # logRequests: false                # INSTRUMENT: one JSON line per request (see below)
```

**`tokenProducer` is the real-token requirement, and it FAILS CLOSED.** The identity is a
token-space key, so a plugin without this block — or with one that selects the tokenizer's `estimate`
backend (byte packing) — **refuses to start**, naming `tokenProducer` in the error. There is no
warning-and-continue path: the estimate backend would silently turn the identity back into a
byte-space key, which is the retired v1 behaviour, and a wave whose arm ran that way is
indistinguishable from a correct one in its artifacts. The effective config (the block included) is
echoed at startup so a run's arm is verifiable from its own log:

```
thread-identity: effective config {"blockSizeTokens":64,"tokenProducer":{...},...}
```

It must be the *same* backend (model, endpoint, salt) the `token-producer` plugin uses, or the
identity's hashes and the prefix scorer's diverge for the same request.

Built-in session headers (priority order): `x-claude-code-session-id`, `x-session-affinity`,
`session-id`, `session_id` — the same list `agent-identity` resolves.

## Metrics

| metric | meaning |
|---|---|
| `llm_d_epp_thread_matched_total` | continuations (`continue`: same id as the branch's previous turn) |
| `llm_d_epp_thread_forks_total` | forks (extension of an outstanding tip) |
| `llm_d_epp_thread_new_total` | no tip matched: a new branch |
| `llm_d_epp_thread_sticky_busy_total` | continuations proven by the H1 reply guard despite a busy tip |
| `llm_d_epp_thread_retry_total` | S2 continuations: identical duplicates of a still-queued tip |
| `llm_d_epp_thread_explicit_override_total` | requests that carried an explicit fairness id (a run of this plugin must keep it at 0) |
| `llm_d_epp_thread_render_failure_total` | arrivals with no identity (render error/timeout): each starts its own branch |
| `llm_d_epp_thread_multi_prompt_requests_total` | requests whose render produced more than one prompt |
| `llm_d_epp_thread_records` | live branches |
| `llm_d_epp_thread_render_ns` | arrival-time render latency per request (the producer, relocated from dispatch) |
| `llm_d_epp_thread_chain_ns` | block-chain hashing per request |
| `llm_d_epp_thread_resolve_ns` | resolve cost per request (inside the table lock: match + mint) |
| `llm_d_epp_thread_evicted_total` | cap evictions |

`DumpState` returns the branch table (full blocks, token count, message count, in-flight/queued,
forks, turns, idle/age; newest 200 rows).

## The `logRequests` instrument (off by default)

A **shadow arm** — the plugin armed on a stack whose client still sends the fairness header — keeps
the client's explicit id as the queue key, so the derived id reaches no span. `logRequests: true`
closes that gap with one stdout line per request:

```
thread-identity: request {"seq":1,"ts":"…","scope":"<session header>","derived":"8f21c0aa93",
                          "outcome":"matched","turn":7,"explicit":true,"blocks":12,
                          "tokens":347895,"marker":"…"}
```

`scope` is the client session header the arm configured, `seq` is arrival order, `outcome` and
`turn` are the arrival record. It is behaviour-neutral — no state, no decision.

Keep ids short: they can become metric label values (`--fairness-id-metric-label-limit`).

## The equivalence gate (v1 → v2)

The v2 identity is a **token-space** key, so its ids differ from v1's by construction (they are
minted from the tip commitment, which moved from the byte chain to the prefix-scorer chain). What
must NOT change is the **grouping**, so the gate is **partition equivalence**: build the
old-id ↔ new-id mapping over a recorded window and require it to be one-to-one, except for the
declared classes, each counted.

Both sides are replayed through the real hooks by `stage0c_replay_test.go`, on the rebased tree:

```sh
# old side: the shipped byte implementation
STAGE0C_WINDOW=/tmp/stage0c/a0.jsonl \
  STAGE0C_OUT=/tmp/stage0c/a0.old.jsonl \
  go test ./pkg/epp/framework/plugins/threadidentity/ -run TestStage0CReplay -v -timeout 3h

# new side, twice (the second run proves the ids are deterministic)
STAGE0C_WINDOW=/tmp/stage0c/a0.jsonl STAGE0C_TOKENS=<engine-render.jsonl> \
  STAGE0C_OUT=/tmp/stage0c/a0.new.jsonl \
  go test ./pkg/epp/framework/plugins/threadidentity/ -run TestStage0CReplay -v -timeout 3h
```

`STAGE0C_TOKENS` is a JSONL corpus of `{i, n, in_tokens, tokens}` rendered offline with the
engine's chat template (`transformers.apply_chat_template`); the harness injects those tokens and
**validates them against the recorded `in_tokens`**, so a drifted template cannot pass silently.

**Result, WEKA stage-0C (`a0`, 3,626 requests, Qwen3 template — the arm's model):**

| | count |
|---|---|
| requests | 3,626 |
| branches, v1 (bytes) | 981 |
| branches, v2 (tokens) | 943 |
| **old branches split into 2+ new** | **0** |
| **new branches merging 2+ old** | **37** (36 pairs, 1 triple) → 38 fewer branches |
| merge class | **the declared LAST-BLOCK MERGE class** — all 37 are the one-full-block tolerance continuing a request into a branch it used to start, NOT a split: every merge point's divergence is **68–71 tokens from the tip's end**, i.e. inside the tip's **last full block**, the block the tolerance absorbs and a partial-block-only tolerance would NOT |
| of the 37, same ground-truth thread | **37 / 37** — every merge stays inside one conversation (no cross-thread merge) |
| class (i) tool/thinking splits | 0 (the corpus has no tool blocks) |
| class (iii) sub-block tips | 0 (smallest request is 136 tokens) |
| class (iv) render failures | 0 |
| determinism: id mismatches across 2 replays | 0 |
| content continuity: exact extensions that continued | 2,577 / 2,577 = **100.00%** (v1: 100.00%) |

So on WEKA no conversation was ever re-fragmented (0 splits — the unsafe direction) and 37 tail
re-renders (the harness rewriting the last message's tail, 10 of them exactly one block shorter)
became continuations — the safe direction, and each one counted.

**Result, Exgentic stage-0B (13,429 requests, tool-call corpus):**

| | count |
|---|---|
| branches, v1 | 806 |
| branches, v2 | 928 |
| **old branches split** | **122** — every one class (iii): the conversation's first turn renders to 37 tokens (well under B), so v2 does not index that tip and the next turn re-keys ONCE; all later turns continue the new branch |
| merges | 0 |
| class (i) tool/thinking splits | 0 — in these bodies the tool block is constant per conversation, so there is no deep divergence at all |
| determinism mismatches | 0 |
| content continuity | 12,496 / 12,496 = **100.00%** |

That window records no `in_tokens`, so the render could not be cross-checked against the engine
there (the template is still applied to the real bodies); the a0 window, which does record them,
renders at a constant −2.

**Metrics, re-run after the F1–F3 fixes: unchanged.** The re-run reproduces the table above exactly
(same 0 splits, same 37 last-block merges, same 2,577/2,577 continuity, 0 determinism mismatches),
which is the expected outcome and a check on F1: the re-chain applies to multi-prompt requests only
(`multi_prompt_requests_total` is 0 on both windows), so a single-prompt corpus cannot move.

**Declared classes, for reference** — (i) tools/thinking now count, so a mid-thread tool-set change
or a re-rendered history splits the branch (exactly where the KV cache splits); (ii) the tip's
trailing partial block is not verified; (iii) a tip shorter than one full block is not continuable
and re-keys once; (iv) a render failure at arrival is a NEW branch plus a counter; (v) an
end-of-tip variance that crosses a block edge — closed *by construction* by the one-full-block
tolerance. **Plus one class that is the OPPOSITE of (v): the LAST-BLOCK MERGE** — a request that
used to start its own branch, whose divergence sits in its predecessor's last full block, is
CONTINUED into that branch. That is the tolerance working on its design target, it moves in the
safe direction (a shared queue, never a re-fragmented conversation), and it is what the 37 WEKA
differences are: classified, not a defect, and never a split.

## Known limits

- **State is per EPP instance.** Behind N replicas, the same conversation landing on another EPP
  re-derives its branch from content (same id, different table), which is fine for identity but not
  for the *outstanding* marks. The stage-1 campaign runs a single EPP.
- **Multi-prompt slack.** For a multi-prompt request the tolerance applies to the *concatenated*
  chain, so the unverified region can exceed 2B−1 tokens (an earlier prompt's blocks are interior
  and stay verified, the last prompt's tail is what is unverified). Single-prompt chat is unaffected;
  cross-protocol continuation is not a requirement. A later prompt's blocks are **re-chained** onto
  the chain before them (the prefix scorer seeds each prompt independently, which would make a
  prompt shared by two requests a set of shared keys — tips `[A,C]` and `[B,C]` would then continue
  one another). Single-prompt requests keep prefixhash's hashes verbatim, so their ids are unchanged
  by that re-chain.
- **A stale re-send is not a continuation... unless it is one block short of the tip.** A request
  that is a proper prefix of its branch's tip (a client re-sending an earlier turn after the branch
  advanced) matches only if the tip is at most one full block longer; then it continues the branch
  and the tip moves back to it. The next real turn finds the branch either way, so the branch id is
  unaffected — only the tip pointer regresses for one turn.
- **A fork's first new message that is an assistant message** (a framework that spawns a child
  *after* the parent's assistant tool-call turn puts that message in the child's history) is
  indistinguishable from an H1 continuation by construction. The design labels this H2 and says
  measure, don't fix.
- Without a client session header, look-alike traces of different sessions may share a branch (the
  accepted floor).

## Deployment preconditions (not enforced by this plugin)

- **The flow-control loop must actually reach `Pick`.** Identity's queue key only decides anything
  if the fairness policy is consulted. Confirmed for our config: one priority band, the noop
  saturation detector returns 0.0 — below the fail-open ceiling of 1.0 — so no band is gated and
  `Pick` runs every cycle while any queue holds a head.
- **The per-band caps are BOUNDED BY DEFAULT.** A band's `maxBytes` defaults to **1 GB** and its
  `maxRequests` to **5000**; `0` means *take the default*, **not** "unbounded" (per-*flow* maxima
  are the opposite: `0` disables them). Our core configs never set either, so the campaign ran on
  the defaults — they never bound in the measured ranges, but that is luck, not design. An arm that
  needs an effectively removed band bound must set an explicit large value.
- An arm must not send `x-llm-d-inference-fairness-id` from its harness
  (`--session-header=x-llm-d-inference-fairness-id`), or every request is an "explicit override" and
  the arm measures A0 with extra steps. Keep `llm_d_epp_thread_explicit_override_total == 0`.
