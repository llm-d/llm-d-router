# KV-Cache Retention POC benchmark: baseline vs main-session retention

Date: 2026-09-23/24. Raw data: the `baseline-*.json` / `treatment-*.json` /
`treatment-v2-*.json` / `lowconc-*.json` inference-perf reports and
`metrics-*.txt` vLLM counter snapshots in this directory. Five arms; the first
three at 16 concurrent sessions, the last two at 8:

- **baseline**: no session headers, `kv-cache-hint` inactive, plain LRU.
- **treatment (v1)**: `kv-cache-hint`, whole-context directives
  (`start: 0, end: null`) on main-agent turns.
- **treatment-v2**: `kv-cache-hint-v2`, directives bounded to the 16,384-token
  context head under a 1,100,000-token global protection budget
  (`epp-config-hint-v2.yaml`).
- **lowconc-baseline / lowconc-v2**: the baseline and v2 arms repeated at
  8 concurrent sessions with a 60s idle-gap cap (`inference-perf-lowconc.yaml`).

## Setup

- Serving: vLLM at [vllm-project/vllm#38514](https://github.com/vllm-project/vllm/pull/38514)
  (Context-Aware KV-Cache Retention API, branch `moreh-dev/vllm:retention-api-bench`),
  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, TP=2 on 2x H100-80GB,
  max-model-len 262144, fp8 KV cache (2,196,848 tokens), gpu-memory-utilization 0.88.
- Routing: GKE Inference Gateway -> EPP (branch `capri-xiyue/sea-cache-poc`) -> single vLLM replica.
- Load: inference-perf v0.7.0, `weka_trace_replay` of
  `semianalysisai/cc-traces-weka-with-subagents-060826-256k` (Claude Code
  agentic traces with subagent fan-out), 16 concurrent sessions, 32-session
  pool, `base_seed=20260910`, think-time gaps capped at 30s, no stage timeout.
- Arms (prefix cache reset before each):
  - **baseline**: no session headers; `kv-cache-hint` inactive.
  - **treatment**: patched client (`../patch/openai_client.py`) sends
    `x-session-id` on all requests and `x-session-type: main` on main-agent
    turns only. EPP `session-interturn-latency-producer` + `kv-cache-hint`
    track the `main` queue only: every main turn gets
    `retention_directives: [{start: 0, end: null, priority: 70, duration: ~29-54s}]`
    with the session id as scope; subagent requests get no directives.
    Predicted duration converged to ~28.8s (q0.9 of the 30s-capped gap
    distribution), confirming the online estimator works.

## Results

Deltas are relative to baseline.

| metric | baseline | treatment (v1) | treatment-v2 |
|---|---|---|---|
| benchmark time (s) | 15,525 | 16,480 (+6.2%) | 11,711 (-24.6%) |
| successful requests | 4,032 | 3,740 (-7.2%) | 3,511 (-12.9%) |
| TTFT mean (s) | 0.858 | 2.102 (+145%) | 1.126 (+31%) |
| TTFT p50 (s) | 0.442 | 0.503 (+14%) | 0.456 (+2.9%) |
| TTFT p90 (s) | 1.423 | 5.994 (+321%) | 2.094 (+47%) |
| TTFT p99 (s) | 8.909 | 22.803 (+156%) | 13.260 (+49%) |
| request latency mean (s) | 20.40 | 23.22 (+14%) | 22.38 (+9.7%) |
| request latency p99 (s) | 169.5 | 199.9 (+18%) | 195.0 (+15%) |
| TPOT p99 (s) | 0.048 | 0.158 (+230%) | 0.075 (+56%) |
| ITL p99 (s) | 0.053 | 0.302 (+473%) | 0.197 (+273%) |
| input tok/s | 25,288 | 22,280 (-12%) | 28,573 (+13%) |
| output tok/s | 300.6 | 235.0 (-22%) | 313.4 (+4.3%) |
| vLLM prefix-cache hit rate | 93.45% (367.7M/393.5M) | 79.18% (291.4M/368.0M) | 90.29% (302.1M/334.6M) |
| sessions succeeded / failed | 23 / 9 | 22 / 10 | 22 / 10 |
| events completed / cancelled | 4,028 / 1,898 | 3,735 / 2,190 | 3,505 / 2,420 |
| preemptions | 0 | 0 | 0 |

Failure composition is comparable across arms: 4-5 deterministic HTTP 400s
(input + max_tokens crossing the 262,144 window on the same traces) and 5
sessions lost to predecessor-wait timeouts in each arm, so the aggregate
deltas are not a composition artifact. The extra treatment timeouts landed on
the largest traces (35, 39), consistent with the slower serving path rather
than causing it.

## Reading

At this contention level the whole-context retention directive hurts every
headline metric. Two mechanisms fit the data:

1. **Over-pinning.** Each main turn protects its entire context
   (`start: 0, end: null`) for ~29s at priority 70. With 16 concurrent
   sessions of 100-260K-token contexts, the protected set alone
   oversubscribes the 2.2M-token KV pool. Once most resident blocks carry
   the same priority, the evictor degenerates: eviction order stops being
   recency-based, and the blocks it does evict (subagent tails, unprotected
   suffixes) are exactly the ones about to be re-read, driving the hit rate
   down 14 points.
2. **Evictor overhead.** A 200K-token context is ~12.5K blocks; stamping
   retention onto every block of every main turn adds scheduler-loop work.
   The ITL p99 (+473%) and TPOT p99 (+230%) degradation without any
   preemption points at per-step scheduler latency, not memory thrash alone.

## treatment-v2: bounded range + protection budget

`kv-cache-hint-v2` implements the first and third items of the v1 reading:
directives cover only the 16,384-token context head, and a 1,100,000-token
budget (~half the pool) caps the sum of live protections. Observed steady
state: ~6 concurrent scopes x 16,384 = ~98K protected tokens, ~4% of the
pool.

Relative to v1, every metric recovers: hit rate 79.2% -> 90.3%, TTFT p90
6.0s -> 2.1s, ITL p99 0.30s -> 0.20s, and throughput moves from -12%/-22%
below baseline to +13%/+4% above it. The over-pinning mechanism is
confirmed: bounding the protected set restores recency-based eviction for
the bulk of the pool.

Relative to baseline the picture is mixed. Throughput is higher and the run
finishes 25% sooner, but the hit rate stays 3 points below plain LRU and
the latency tails above it (TTFT p90 +47%, ITL p99 +273%). Two caveats on
cross-arm reads: treatment-v2 completed fewer events (3,505 vs 4,028;
one more giant trace hit the predecessor-wait timeout and cancelled its
remainder), so the request mixes differ; and the residual ITL tail is
consistent with per-block retention stamping in the scheduler loop, which
bounding shrinks (1,024 blocks per directive vs ~12,500) but does not
remove.

Conclusion: at this contention level (16 sessions, ~4x KV oversubscription,
30s-capped gaps) plain LRU remains the hit-rate winner and the retention
hint buys no clear end-to-end latency benefit. The v2 bounds are the right
shape, they turn a strict regression into roughly break-even, but the win
case still needs a workload where eviction pressure comes from session
turnover rather than steady-state oversubscription.

## Low-concurrency pair: 8 sessions, 60s gap cap

Configuration `inference-perf-lowconc.yaml`: 8 concurrent sessions from a
16-session pool, `trace_idle_gap_cap_seconds` 60 (was 30), otherwise
identical to the runs above; EPP on `kv-cache-hint-v2` for both arms, the
baseline arm simply sends no session headers. This is the regime the hint
targets: the live working set (~1.2M tokens) fits the 2.2M-token pool, so
eviction pressure comes from session turnover, and the longer idle windows
give LRU a real chance to evict an idle main session between turns.

The setup did what it was designed to do: baseline hit rate fell from
93.45% at 16 sessions to 89.90% here, so plain LRU is now evicting blocks
that get re-read. The two arms are also the best-matched pair of the
campaign: identical success and failure counts, the same three traces
failing on the 262,144 window and the same trace hitting the
predecessor-wait timeout, and run times within 0.4%.

| metric | lowconc-baseline | lowconc-v2 | delta |
|---|---|---|---|
| benchmark time (s) | 9,919 | 9,880 | -0.4% |
| successful requests | 2,093 | 2,095 | +0.1% |
| TTFT p50 (s) | 0.465 | 0.498 | +7% |
| TTFT p90 (s) | 1.507 | 2.183 | +45% |
| TTFT p99 (s) | 12.216 | 17.802 | +46% |
| request latency p50 (s) | 7.31 | 7.49 | +2.5% |
| request latency p99 (s) | 113.7 | 108.1 | -4.9% |
| TPOT p99 (s) | 0.040 | 0.050 | +26% |
| ITL p99 (s) | 0.0295 | 0.0301 | +2% |
| input / output tok/s | 20,295 / 240.6 | 20,455 / 242.6 | +0.8% |
| vLLM prefix-cache hit rate | 89.90% (181.0M/201.3M) | 87.71% (177.3M/202.1M) | -2.2pp |
| sessions succeeded / failed | 12 / 4 | 12 / 4 | |
| events completed / cancelled | 2,092 / 1,492 | 2,094 / 1,490 | |

With the workloads matched, the bounded hint still loses 2.2 points of hit
rate and adds ~45% to the TTFT tail, while ITL and throughput are
unchanged. Two things follow from the shape of that result:

1. **The protected range is the wrong range.** The 16,384-token head is
   the shared prefix every turn re-reads, so it is always hot and LRU keeps
   it anyway; protecting it is redundant. The blocks LRU actually evicts
   from an idle session are the cold middle and tail of a 100-260K-token
   context, which head-only protection leaves uncovered. Whole-context
   protection (v1) covers them but pins the pool; head-only protection (v2)
   spares the pool but covers nothing that was at risk.
2. **The residual cost sits on the admission path.** ITL is flat (the
   per-step scheduler penalty seen at 16 sessions is gone at 1,024 blocks
   per directive), but TTFT p90/p99 rise ~45% with only a 2-point hit-rate
   drop to explain them. The remaining cost is consistent with directive
   application at request admission, which lands before the first token.
   A 2-point hit-rate loss with only ~6% of the pool pinned also suggests
   the two-structure evictor does not preserve pure LRU order for
   unprotected blocks once any directive is live; that is a vLLM-side
   question.

Across five arms the retention hint never matched plain LRU on hit rate.

## Next steps worth testing

- **Whole-context range under the v2 budget.** Config-only on
  `kv-cache-hint-v2`: `maxProtectedPrefixTokens: 262144` with
  `protectionBudgetTokens` around 600,000, so two to three idle main
  sessions keep their cold middles while the pool stays mostly LRU. This
  directly tests point 1 above and is the one remaining directive shape
  not yet measured.
- Profile the vLLM retention path (`_apply_retention_to_block` and the
  priority evictor's ordering for unprotected blocks) under directive
  load; both the TTFT-side cost and the hit-rate loss at low pin ratios
  point there rather than at the EPP.
- Raise the gap cap further (120s or uncapped) only after the range
  question is settled; longer idles amplify whichever effect dominates.
