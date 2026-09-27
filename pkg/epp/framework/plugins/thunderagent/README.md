# ThunderAgent Plugin

Session level admission control for agentic workloads, a minimal
implementation of ThunderAgent (arXiv 2602.13692). A session is an agent
trajectory, identified by the request FairnessID. Without an explicit
fairness header, the director fills it from the `agent-identity` plugin;
requests with neither are not tracked.

This package currently ships the session ledger: each session's KV token
footprint (the larger of the `usage.total_tokens` of its last completed turn
and the byte estimate of the turn in flight; a session is assumed to have at
most one request in flight) and the pod it is bound to, exposed through the
`llm_d_epp_thunder_agent_sessions`,
`llm_d_epp_thunder_agent_endpoint_working_set_tokens` and
`llm_d_epp_thunder_agent_endpoint_capacity_tokens` gauges. Comparing the working set against
the engine's own KV utilization shows the KV-thrashing signature: idle
sessions own their context in the prefix cache, but the engine reports those
blocks as free.

The admission gate (flow control fairness policy plus pause sweep) and the
placement scorer build on this ledger in follow-up changes.

## Configuration

```yaml
- type: thunder-agent
  name: thunder
  parameters:
    capacityTokens: 4194304        # fallback when cache_config_info is absent
    evictionTtlSeconds: 3600       # idle session state retention; the only release path
```
