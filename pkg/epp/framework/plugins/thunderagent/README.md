# ThunderAgent Plugin

Session level admission control for agentic workloads, a minimal
implementation of ThunderAgent (arXiv 2602.13692). A session is an agent
trajectory, identified by the session id the `agent-identity` plugin reads
from the session headers Claude Code, OpenCode and Codex send; other clients
send a header listed in its `additionalSessionHeaders`. The flow control
fairness ID is not used. Requests with no session id are not tracked.
`agent-identity` is a required dependency: configuration loading fails
without it.

This package currently ships the session ledger: each session's KV token
footprint (the larger of the `usage.total_tokens` of its last completed turn
and the sum of the byte estimates of its turns in flight) and the pod it is
bound to, exposed through the
`llm_d_epp_thunder_agent_sessions`,
`llm_d_epp_thunder_agent_endpoint_working_set_tokens` and
`llm_d_epp_thunder_agent_endpoint_capacity_tokens` gauges. Comparing the working set against
the engine's own KV utilization shows the KV-thrashing signature: idle
sessions own their context in the prefix cache, but the engine reports those
blocks as free.

The same values (session counts, per-pod working set and capacity) are
available from the `/debug/plugins/state` endpoint. Session ids never appear
in metrics or state dumps.

The admission gate (flow control fairness policy plus pause sweep) and the
placement scorer build on this ledger in follow-up changes.

## Configuration

```yaml
- type: thunder-agent
  name: thunder
  parameters:
    capacityTokens: 4194304        # fallback when cache_config_info is absent
    evictionTtlSeconds: 3600       # idle session state retention; the only release path
    evictionSweepSeconds: 10       # how often idle sessions are swept
```
