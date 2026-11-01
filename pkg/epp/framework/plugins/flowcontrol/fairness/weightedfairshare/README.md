# Weighted Fair Share Fairness Policy

**Type:** `weighted-fair-share-fairness-policy`

The Weighted Fair Share policy shares a priority band between tenants in proportion to weights the operator declares per tenant. While the band is contended, the policy drives each tenant's throughput toward a share proportional to its weight: at equal weights every tenant gets the same throughput, and at weights 7:3 one tenant gets 70% and the other 30%. Throughput here is measured in the configured cost, which by default is weighted tokens per second (prompt tokens plus twice the output tokens); with `costPerInputToken: 0, costPerOutputToken: 1` it is output tokens per second.

Each flow keeps a running total called its virtual service: the cost of its completed requests divided by its weight. The policy serves the flow with the smallest virtual service, which drives the flows' shares of cost toward the ratio of their weights. At equal weights it shares the band evenly by work done rather than by turn.

Registered at **Alpha** stability, so the EPP requires `--allow-experimental-plugins`. Without the flag the EPP refuses to start.

## Why choose this policy?

Consider two teams sharing one priority band on the same pool:

| Team | Prompt tokens | Output tokens | Cost per request (default `0 / 1 / 2`) |
| --- | --- | --- | --- |
| `team-a` | 1,000 | 300 | 1,600 |
| `team-b` | 40,000 | 1,000 | 42,000 |

Both teams send more requests than the pool can serve, so requests queue in the EPP.

**Requests of different sizes.** [Round Robin](../roundrobin/README.md) alternates turns: one `team-a` request, then one `team-b` request. Each `team-b` turn carries 26 times the work of a `team-a` turn, so `team-b` receives about 96% of the pool's work and `team-a` queues behind it. Weighted Fair Share charges each team for the work its requests consumed. At equal weights it serves about 26 `team-a` requests for each `team-b` request, and each team receives half of the work.

**Weighted shares.** If `team-a` should receive more of the pool, set `flowWeights` to `team-a: 7` and `team-b: 3`. Under contention `team-a` receives 70% of the work and `team-b` 30%. Round Robin and Global Strict cannot give two teams in one band different shares.

**Spare capacity.** A share applies only under contention. If `team-a` sends less than its share, `team-b` uses the remainder. When `team-a` sends again, it rejoins at its share without a burst to make up for the quiet period, and `team-b` is not penalized for having used the idle capacity.

"Work" is set by the cost coefficients. The default charges prompt and output tokens, which on long prompts is mostly prefill. To divide output tokens per second instead, use `0 / 0 / 1`; to divide request counts, use `1 / 0 / 0`. See [Choosing the cost coefficients](#choosing-the-cost-coefficients).

## What it does

1.  **Charging**: When a request completes (`ResponseBody`, at end of stream), its flow's virtual service advances by `cost / weight`, using the configured cost function over the server-reported prompt and completion token counts.
2.  **Picking**: `Pick` serves the active flow with the smallest virtual service. Among flows with equal virtual service, the one whose head request has waited longest wins. Head wait does not otherwise affect the ranking.
3.  **No idle credit**: Each band keeps a minimum virtual service, set to the virtual service of the flow it last picked (the smallest among the flows competing at that pick). Before ranking, any candidate below it is raised to it. A flow that stayed backlogged is already at or ahead of it. A flow that was idle, or is new, rejoins level with the band instead of arriving with credit for the time it was absent, so it cannot take every dispatch in a burst on return. A flow that used capacity while others were idle is not penalized for it afterwards.
4.  **Weight resolution**: The weight is looked up in the `flowWeights` table by the flow's fairness ID, falling back to `defaultFlowWeight` for any ID the table does not name. Requests carry no weight.
5.  **Pruning**: Flows idle beyond `idleTtlSeconds` are swept out, at most once per `sweepSeconds`, on the `Pick` path. The policy owns no goroutine. A pruned flow that returns starts at the band's minimum virtual service.

## Unit of Fairness

**Cost**, defined by the configured cost function:

```
cost = costPerRequest + costPerInputToken * promptTokens + costPerOutputToken * completionTokens
```

`completionTokens` is the model server's total of generated tokens, so it includes reasoning (thinking) tokens; they are charged at `costPerOutputToken` like any other output.

The default `0 / 1 / 2` weights an output token at twice an input token. The coefficients select which resource the weights divide. With `0 / 0 / 1` cost is completion tokens alone, so weights 7:3 target a 70:30 split of output tokens and prompt processing is free. With `1 / 0 / 0` cost is a flat charge per request, so the same weights target a 70:30 split of requests regardless of their size. Choose the coefficients from the SLO being divided; see [Choosing the cost coefficients](#choosing-the-cost-coefficients).

**Weights come from operator configuration.** A request carries only its fairness ID, and the weight is resolved against the `flowWeights` table. The router defines no weight header, so a caller cannot influence its own share.

**Virtual service is tracked per flow, and a flow is a fairness ID at one priority.** A team that sends requests at two priorities has two flows, each with its own virtual service. The two are kept apart because bands dispatch in strict priority order: if they were pooled, work the team did in one band would count against it in the other. The weight is looked up by fairness ID only, so both of the team's flows get the same weight.

## Inputs consumed

*   **Queue state**: Iterates active queues on the `PriorityBandAccessor`, reading each head item and its enqueue time.
*   **Minimum virtual service**: Reads and advances the minimum virtual service stored on the priority band, which a flow returning from idle is raised to.
*   **Request headers**: `x-llm-d-inference-fairness-id` only, which supplies the flow id and falls back to a single default id when absent. No weight is read from the request.
*   **Policy configuration**: the `flowWeights` table and `defaultFlowWeight`, which together resolve every flow's weight.
*   **Response usage**: `PromptTokens` and `CompletionTokens` from the final response chunk. A streaming response carries usage only if the request set `stream_options.include_usage`, or the model server forces it (vLLM: `--enable-force-include-usage`). Without usage, only `costPerRequest` is charged, so under the default coefficients the flow's virtual service never advances.

## Configuration

```yaml
- type: weighted-fair-share-fairness-policy
  name: weighted-fair-share-fairness-policy
  parameters:
    flowWeights:
      team-a: 7
      team-b: 3
    defaultFlowWeight: 1      # any fairness ID not named above
    costPerInputToken: 0      # share generation throughput instead of total work
    costPerOutputToken: 1
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `flowWeights` | `{}` | Weight per fairness ID. Values must be `> 0`; there is no upper bound. |
| `defaultFlowWeight` | `1` | Weight for any fairness ID absent from `flowWeights`, including the default tenant unstamped callers fall into. Must be `> 0`. |
| `idleTtlSeconds` | `3600` | Idle flows are pruned after this. `0` disables pruning. |
| `sweepSeconds` | `300` | Minimum interval between idle sweeps. |
| `costPerRequest` | `0` | Flat cost charged per completed request. |
| `costPerInputToken` | `1` | Cost per prompt token. |
| `costPerOutputToken` | `2` | Cost per completion token. |

All three cost coefficients must be `>= 0` and not all zero. With all-zero coefficients nothing is charged and every flow's virtual service stays at zero, so that configuration is rejected at startup.

A weight change takes effect for cost charged after the change. Virtual service already accumulated is not rescaled.

### Choosing the cost coefficients

On a long-context agentic corpus (ISL ~130k, OSL ~1k) input is about 98% of the default cost, so the default effectively shares prefill and generation throughput is almost unrepresented.

| `perRequest` / `perInput` / `perOutput` | Shares |
| --- | --- |
| `0 / 1 / 2` | default: roughly GPU work, output weighted 2x |
| `0 / 0 / 1` | output tokens per second |
| `0 / 1 / 0` | prefill only |
| `1 / 0 / 0` | request counts (turns) |
| `0 / 0.1 / 1` | mostly generation, still charging something for prefill |

**Charging nothing for input makes prefill free**, so a tenant can enlarge its prompts at no cost to its share. Use `0 / 0 / 1` when the SLO being divided is output tokens per second and prompt sizes are uniform or trusted.

## Observability

| Metric | Type | Labels |
| --- | --- | --- |
| `llm_d_epp_weighted_fair_share_virtual_service` | gauge | `flow_id`, `priority`, `policy` |
| `llm_d_epp_weighted_fair_share_flow_weight` | gauge | `flow_id`, `priority`, `policy` |

Under contention the virtual-service gauges of backlogged flows in one band stay close together; a persistent gap means a flow is not receiving its share. The gauge includes any raise to the band's minimum virtual service, so it is not a cumulative cost total.

The weight gauge shows the weight resolved for each flow. A fairness ID that does not match a `flowWeights` key silently takes `defaultFlowWeight`, and this gauge is how to detect that. Collectors are per-instance and registered through `handle.Metrics()`, so two instances can share one registry.

## Trade-offs

*   **The control loop is delayed.** Cost is charged at completion, which on long requests is tens of seconds after the dispatch decision it should have informed. Requests in flight are not yet reflected in their flow's virtual service, so with few dispatch slots and long requests the achieved split can fall short of the weight ratio. [Round Robin](../roundrobin/README.md) knows a turn's cost at selection time and has no such lag.
*   **Shares hold only while flows stay backlogged.** The policy is work-conserving: a flow that does not offer enough work to use its share leaves the remainder to the others, and a flow whose queue empties rejoins at the band's minimum virtual service without credit for the gap.

## Related Documentation

*   [Fairness Overview](../README.md)
*   [Round Robin](../roundrobin/README.md): the same idea with turns as the unit
*   [Program-Aware](../program-aware/README.md): its `las` strategy uses the same cost definition
*   [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
