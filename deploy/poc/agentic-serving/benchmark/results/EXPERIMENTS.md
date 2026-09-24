# KV-Cache Retention POC: experiment matrix and analysis

Summary view of the five benchmark arms run on 2026-09-23/24. Full per-arm
numbers, raw inference-perf reports, and vLLM counter snapshots are in this
directory; the narrative write-up is `REPORT.md`.

Common to every arm: vLLM at vllm-project/vllm#38514 (Context-Aware KV-Cache
Retention API) serving Qwen3-Coder-30B-A3B-Instruct-FP8, TP=2 on 2x H100-80GB,
262,144 max context, fp8 KV cache with a 2,196,848-token pool; GKE Inference
Gateway -> EPP (`capri-xiyue/sea-cache-poc`) -> one vLLM replica; inference-perf
v0.7.0 replaying `semianalysisai/cc-traces-weka-with-subagents-060826-256k`
(Claude Code agentic traces with subagent fan-out), `base_seed=20260910`, no
stage timeout, prefix cache reset before each arm. Arms with the hint active
use a patched client that sends `x-session-id` on every request and
`x-session-type: main` on main-agent turns only, so the EPP tracks and protects
main sessions and ignores subagent requests.

## 1. What each experiment did

| Arm | Concurrency / pool | Idle gap cap | Session headers | EPP plugin | Directive shape per main turn | Global budget | Tokens pinned at steady state | Wall time |
|---|---|---|---|---|---|---|---|---|
| baseline | 16 / 32 | 30s | none | hint inert (plain LRU) | none | n/a | 0 | 4h19m |
| treatment (v1) | 16 / 32 | 30s | id + type | `kv-cache-hint` | `start: 0, end: null` (whole context), priority 70, duration | none | 2-4M (exceeds the 2.2M pool) | 4h45m |
| treatment-v2 | 16 / 32 | 30s | id + type | `kv-cache-hint-v2` | `start: 0, end: 16384` (context head), priority 70, duration | 1,100,000 | ~98K (~4% of pool) | 3h15m |
| lowconc-baseline | 8 / 16 | 60s | none | hint inert (plain LRU) | none | n/a | 0 | 2h45m |
| lowconc-v2 | 8 / 16 | 60s | id + type | `kv-cache-hint-v2` | `start: 0, end: 16384`, priority 70, duration | 1,100,000 | ~49-131K (~6% of pool) | 2h45m |

Duration is the 0.9 quantile of the per-session inter-turn distribution fitted
online by `session-interturn-latency-producer`, clamped to [1s, 10m]; with a
30s gap cap it converged to ~28.8s, with a 60s cap to ~20-55s depending on the
session.

## 2. Results

| Metric | baseline | treatment (v1) | treatment-v2 | lowconc-baseline | lowconc-v2 |
|---|---|---|---|---|---|
| vLLM prefix-cache hit rate | 93.45% | 79.18% | 90.29% | 89.90% | 87.71% |
| TTFT p50 (s) | 0.44 | 0.50 | 0.46 | 0.47 | 0.50 |
| TTFT p90 (s) | 1.42 | 5.99 | 2.09 | 1.51 | 2.18 |
| TTFT p99 (s) | 8.91 | 22.80 | 13.26 | 12.22 | 17.80 |
| ITL p99 (s) | 0.053 | 0.302 | 0.197 | 0.030 | 0.030 |
| TPOT p99 (s) | 0.048 | 0.158 | 0.075 | 0.040 | 0.050 |
| input tok/s | 25,288 | 22,280 | 28,573 | 20,295 | 20,455 |
| output tok/s | 300.6 | 235.0 | 313.4 | 240.6 | 242.6 |
| requests ok / failed | 4,032 / 5 | 3,740 / 5 | 3,511 / 4 | 2,093 / 3 | 2,095 / 3 |
| sessions ok / failed | 23 / 9 | 22 / 10 | 22 / 10 | 12 / 4 | 12 / 4 |
| preemptions | 0 | 0 | 0 | 0 | 0 |

Failures are the same in kind across arms: a fixed set of traces whose input
plus `max_tokens` crosses the 262,144 window (deterministic HTTP 400), and the
largest traces hitting inference-perf's predecessor-wait timeout. The
lowconc pair is fully matched: identical counts, identical failing traces,
wall time within 0.4%.

## 3. Per-experiment analysis

| Arm | Question it answered | Outcome vs its baseline | Mechanism | What it established |
|---|---|---|---|---|
| baseline | What does plain LRU do on this workload at 4x KV oversubscription? | reference | vLLM frees a request's blocks tail-first, so an idle session loses its context from the end; head blocks are evicted last | 93.45% hit rate: with 30s-capped gaps nothing idles long enough to be wrongly evicted, so LRU is near-optimal here |
| treatment (v1) | Does protecting a main session's whole context for its predicted idle window help? | hit rate -14.3pp, TTFT p90 +321%, ITL p99 +473%, throughput -12%/-22% | Over-pinning: 16 sessions x 100-260K tokens exceed the pool; once most resident blocks share priority 70 the evictor can only take unprotected material (subagent history) which is re-read within seconds. Plus ~12,500 block stamps per directive on the scheduler loop (ITL tail with zero preemptions) | Whole-context protection at this concurrency is unaffordable; the directive-apply cost is on the critical path |
| treatment-v2 | Does bounding the range (16K head) and adding a global budget fix v1? | vs v1: hit rate +11pp, all tails recover, throughput above baseline. vs baseline: hit rate -3.2pp, TTFT p90 +47%, ITL p99 +273% | Pinning drops to ~4% of the pool, restoring LRU order for the bulk; residual ITL tail consistent with 1,024 stamps per directive | Over-pinning was the dominant v1 effect; bounding turns a regression into rough break-even but not a win. Request mixes differ (3,511 vs 4,032 completed), so this comparison is the least clean |
| lowconc-baseline | Can we construct the regime the hint targets, where LRU evicts blocks that get re-read? | hit rate fell to 89.90% from 93.45% at 16 sessions | Working set (~1.2M) fits the pool, so eviction is driven by session turnover; 60s gaps let idle sessions age out | Yes: LRU is now measurably evicting reusable blocks. Headroom for the hint exists |
| lowconc-v2 | With headroom present and workloads matched, does head-only protection recover any of it? | hit rate -2.2pp, TTFT p90 +45%, ITL flat, throughput flat | See section 4 | No. The cleanest pair of the campaign is a clear negative. The head is the wrong range to protect |

## 4. Cross-cutting analysis

**Why head-only protection cannot gain anything.** Two properties of vLLM's
prefix cache combine here. First, when a request finishes vLLM returns its
blocks to the free queue in reverse order, so LRU evicts an idle session's
tail first and its head last; the 16K head v2 protects is what LRU keeps
longest anyway. Second, a prefix-cache hit is the contiguous prefix from
token 0: if the head is resident but block 16K+1 is gone, the hit is 16K and
the rest of a 200K context is recomputed regardless of what else happens to
be resident. So the most head-only protection can deliver on a returning
session is a 16K hit, while baseline LRU, evicting from the tail, often
leaves a much longer contiguous prefix intact. v2 pays a cost to protect a
range that was never at risk and does nothing for the range that was. v1 had
the right range and an unaffordable dose.

**Where the 2.2-point loss in lowconc-v2 likely comes from (inferred, not
measured).** Only ~6% of the pool was pinned, and the pinned blocks were hot,
so capacity cannot explain it; the evictor's behavior must have changed.
Ranked by fit to the data: (1) the PR's two-structure evictor runs a slow
path once any directive is live, and if it orders unprotected blocks by
`last_freed_time` rather than free-queue position, blocks freed in the same
batch tie and the tail-first/head-last property is lost for all unprotected
content, a small loss spread everywhere; (2) heads of finished or late
sessions stay pinned for the remainder of their TTL after LRU would have
aged them out; (3) the LRU re-entry position of expired blocks. All three
are vLLM-side.

**Why the TTFT tail moved far more than the hit rate.** The misses are not
spread evenly; they concentrate on idle long-context main sessions
returning. One such miss is a 100-200K-token re-prefill in a burst, which at
8,192 tokens per chunked-prefill step occupies 12-25 scheduler steps and
delays the first token of everything queued behind it. A small number of
large contiguous misses moves p90/p99 TTFT while leaving ITL untouched,
since requests already decoding are not behind the prefill queue. Directive
application also happens at admission, before the first token, adding a
minor TTFT-side cost that does not appear in ITL.

**Overall.** Across five arms the retention hint never matched plain LRU on
hit rate. Whole-context protection is unaffordable at scale; head-only
protection is redundant with what LRU already does. The one directive shape
not yet measured is whole-context range under a tight budget, so two or
three idle main sessions keep their full contiguous context while the pool
stays mostly LRU: `maxProtectedPrefixTokens: 262144` with
`protectionBudgetTokens` around 600,000 on the existing v2 plugin. If that
is also negative, the conclusion is that on this workload vLLM's LRU with
tail-first freeing already captures nearly all the information the EPP can
add, and the remaining work is on the vLLM side (profiling
`_apply_retention_to_block` and the evictor's ordering of unprotected
blocks).
