# ThunderAgent Plugin

Tracks the KV cache footprint of agentic sessions on each pod, based on
ThunderAgent (arXiv 2602.13692), and keeps each session on the pod that holds
its KV. It does not change admission.

A session is an agent trajectory, identified by the session id the
`agent-identity` plugin reads from the session headers Claude Code, OpenCode
and Codex send; other clients send a header listed in its
`additionalSessionHeaders`. The flow control fairness ID is not used. Requests
with no session id are not tracked. `agent-identity` is a required dependency:
configuration loading fails without it.

The plugin keeps a session ledger: each session's KV token footprint (the
larger of the `usage.total_tokens` of its last completed turn and the sum of
the byte estimates of its turns in flight) and the pod it is bound to, exposed
through the `llm_d_epp_thunder_agent_sessions`,
`llm_d_epp_thunder_agent_endpoint_working_set_tokens` and
`llm_d_epp_thunder_agent_endpoint_capacity_tokens` gauges. Comparing the
working set against the engine's own KV utilization shows the KV-thrashing
signature: idle sessions own their context in the prefix cache, but the engine
reports those blocks as free.

When a pod leaves the pool, the plugin drops it from the ledger; its sessions
keep their footprint and move to the pod that serves their next turn.

The same values (session counts, per-pod working set and capacity) are
available from the `/debug/plugins/state` endpoint. Session ids never appear
in metrics or state dumps.

## Scorer

Referenced as a scorer in a scheduling profile, the plugin gives 1.0 to the
pod a session is bound to in the ledger and 0.0 to the other candidates. It
returns no scores for requests with no session id, for sessions the ledger
does not know, and when the bound pod is not a candidate; the other scorers
then place the request, and `PreRequest` binds the session to the picked pod.
Its category is `Affinity`.

It does the same job as `session-affinity-scorer` with the `session_id`
strategy, with one difference: the binding comes from the ledger, so a
session stays on the pod whose KV the ledger counts for it. Do not configure
both in one profile.

## Configuration

```yaml
plugins:
- type: agent-identity
- type: thunder-agent
  name: thunder
  parameters:
    capacityTokens: 4194304        # fallback when cache_config_info is absent
    evictionTtlSeconds: 3600       # idle session state retention; the only release path
    evictionSweepSeconds: 10       # how often idle sessions are swept
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: thunder
```
