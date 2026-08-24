# Async Broker Step

The async-broker step bridges the coordinator to the [llm-d-async](https://github.com/llm-d/llm-d-async) broker, giving standard OpenAI clients access to request-level queueing through the gateway they already use. Clients opt in per request with a mode header, and requests without the header pass through the step untouched. This also keeps AP-dispatched requests inert when they re-enter the pipeline, since dispatches never carry the mode header.

The step is optional and must run first in the pipeline when enabled. Queued requests re-enter the same pipeline on dispatch, so they stay eligible for everything the coordinator does for synchronous requests.

## Request modes

| Mode | Behavior | For |
| :---- | :---- | :---- |
| No header | Untouched, the normal request path | default behavior, AP dispatch re-entry |
| `X-AP-Mode: passthrough` | Forwarded live with quota classification and objective and fairness stamping | live traffic tied to async tenant quota and priority |
| `X-AP-Mode: enqueue` | Written to the broker queue, answers 202 plus id, result collected later by id | batch, deferred work |
| `X-AP-Mode: wait` | Written to the broker queue, connection held until the result lands | request and response semantics over the queue |

## Request contract

Everything is communicated through headers on a standard OpenAI request, and payloads are not parsed. The step resolves the tenant from a header, classifies the request reserved or overflow against Redis quota counters using the same key scheme as the AP's redis-quota gate (one quota account per tenant across all modes when both sides use the same attribute), and expresses priority as InferenceObjective names the EPP understands. Objective and fairness headers are always stamped server side, so clients cannot self-assign priority.

```
POST http://gateway:8081/v1/chat/completions
Content-Type: application/json
X-Team: premium                    # tenant (quota account, fairness id)
X-AP-Mode: wait                    # passthrough | enqueue | wait
X-Request-Id: job-4217             # optional, enables retry and fetch by id
X-Request-Timeout-Seconds: 30      # optional deadline

{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Summarize this."}]}
```

**Enqueue** returns immediately and the completion is collected later by id:

```
HTTP/1.1 202 Accepted
{"id": "job-4217", "status": "pending"}

GET http://gateway:8081/v1/requests/job-4217
X-Team: premium                    # must match the enqueueing tenant

HTTP/1.1 200 OK                    # the model's response, upstream status mirrored
{"id": "chatcmpl-...", "object": "chat.completion", "choices": [...]}

# still queued or executing:  202 {"id": "job-4217", "status": "pending"}
# wrong tenant, expired TTL, or deleted:  410 Gone
```

After a successful fetch delivery the result's TTL is shrunk to a grace window (fetch_grace_seconds, 60s default), so a client that lost the response can re-fetch while unfetched results do not linger past the grace period.

**Wait** returns the model's response on the original connection with the upstream status mirrored, exactly as if the model server had answered directly, and the delivered result is deleted eagerly. Wake-up is a Redis keyspace notification on the result key, with a polling fallback when notifications are unavailable. The hold runs to the request deadline and answers 504 there, or ends early at `wait_cap_seconds` with the 202 response, leaving the request fetchable. If the client disconnects, the step cancels the request pre-dispatch.

**Passthrough** classifies and stamps, then lets the pipeline continue, so streaming and upstream errors behave exactly as they do without the step.

## Endpoints

The step registers three routes on the coordinator listener:

- `GET /v1/requests/{id}` fetches a queued result, tenant scoped and non-destructive
- `DELETE /v1/requests/{id}` cancels a still queued request and reclaims its result. A request already dispatched runs to completion, and its result then sits out the mailbox TTL
- `GET /v1/models` serves the model list derived from the configured routes

## Broker state

On a queue named `foo`, step traffic and raw producer traffic share one sorted set and are indistinguishable to the AP's gates, lanes, and dispatch. Each message carries its own result destination in its envelope:

```
foo                        request queue: shared, popped destructively in deadline order
foo-results                belt: raw producers' results, drained by their collector
results:req:acme:job-4217  mailbox: one step result, read in place, expires via TTL
request-active:acme:job-4217  in-flight marker: present means fetch answers pending
```

A mailbox is the same list structure as a belt, holding exactly one result under a key named by (tenant, id). The in-flight marker holds a random per-request token, and cleanup is a compare-and-delete on that token, so a stale replica finishing an old request cannot clobber newer state.

## The AP side

The AP protocol is unchanged. Dispatches carry no mode header, so they re-enter the coordinator as ordinary requests and get phased to the EPP like any synchronous call.

A request keeps one client-visible id for its whole life: the validated `x-request-id` (or a minted UUID) is the fetch id and names the mailbox. The envelope id is that id prefixed with the tenant, so every AP-side key derived from it (the in-flight marker and the cancellation key) is tenant scoped, and one tenant's id choices cannot collide with another's. The dispatch call itself carries no `x-request-id`, so that hop logs under a fresh UUID in the coordinator, and `traceparent` on the envelope metadata is the join key between the two. Results are written to the message's mailbox with the configured TTL, and the list push fires the keyspace notification that completes any held wait. The step depends on three AP-side features from llm-d-async (result TTLs on queue config, per-lane objective and fairness stamping, and DEADLINE_EXCEEDED classification for deadline-aborted sends), see [llm-d-async#394](https://github.com/llm-d/llm-d-async/pull/394).

## Configuration

To enable the step, add this block as the first entry under `steps:` in the coordinator's pipeline config, and point `redis_url` at the Redis your async processor uses.

```yaml
- type: async-broker
  params:
    redis_url: "redis://redis:6379"
    routes:
      - model: "my-model"
        queue: "team-a-queue"
        tier: "interactive"
    objectives:
      interactive:
        reserved: "interactive-reserved"
        overflow: "interactive-overflow"
    quota:
      limits:
        team-a: 8
```

| Param | Default | Description |
| :---- | :---- | :---- |
| `redis_url` | required | the Redis holding the async processor's queues |
| `mode_header` | `X-AP-Mode` | selects the serving mode per request |
| `tenant_header` | `X-Team` | resolves the tenant (quota account, fairness id) |
| `timeout_header` | `X-Request-Timeout-Seconds` | per-request deadline for queued modes |
| `routes` | none | selects queue and tier per (model, tenant), first match wins, empty fields match anything |
| `default_queue` | `request-sortedset` | queue for requests matching no route |
| `objectives` | none | InferenceObjective names stamped per tier, selected by quota classification |
| `quota` | prefix `quota:`, attribute `userid`, window 300s | reserved concurrency limits per tenant, counters shared with the AP's redis-quota gate. Tenants without an entry are always classified reserved |
| `timeouts` | wait 60s, enqueue 1h | deadline bounds per queued mode. `max_seconds` caps client requested deadlines |
| `wait_cap_seconds` | none | bounds held wait connections, ending the hold with the 202 response |
| `fetch_grace_seconds` | 60 | mailbox TTL applied after a delivered fetch. Zero deletes the result on delivery |
| `wakeup_mode` | `auto` | `notify`, `poll`, or `auto` which probes for keyspace notification support |
| `forward_headers` | SLO headers | allowlisted client headers forwarded on queued messages. The mode, objective, and fairness headers are rejected here |

All params and their defaults are documented in `pkg/coordinator/steps/async_broker_config.go`, and a commented example lives in `config/coordinator/coordinator.yaml`.

## Timeouts and TTLs

| Clock | Runs from → until | Default | Where / Key | When it fires |
| :---- | :---- | :---- | :---- | :---- |
| Wait deadline | request accepted → result written to Redis | 60s | step param `timeouts.wait.default_seconds`, `X-Request-Timeout-Seconds` per request | hold answers 504 DEADLINE_EXCEEDED |
| Enqueue deadline | request accepted (202) → result written to Redis | 1h | step param `timeouts.enqueue.default_seconds`, `X-Request-Timeout-Seconds` per request | fetch returns 504 DEADLINE_EXCEEDED |
| Deadline clamp | applied once at admission, not a running clock | wait 1h, enqueue none | step param `timeouts.<mode>.max_seconds` | silently caps the requested deadline |
| Wait hold cap | request accepted → result written to Redis or deadline | none | step param `wait_cap_seconds` | hold ends with 202 pending, still fetchable by id |
| Per-dispatch attempt | AP worker sends the request → full response read back | 5m | AP flag `--request-timeout` | 504 DEADLINE_EXCEEDED, not retried |
| Result TTL | result written to Redis → first fetch, expiry, or DELETE | none | AP queue config `result_ttl_seconds` | result deleted + fetch returns 410 Gone |
| Post-fetch grace | first delivered fetch → grace expiry or DELETE | 60s | step param `fetch_grace_seconds` | result deleted + fetch returns 410 Gone |

The three lifecycle clocks hand off without overlap: the deadline ends where the result TTL begins (result written), and the result TTL ends where the grace begins (first delivered fetch). Wait mode deletes the result on delivery, so the TTL and grace rows apply to enqueue results and to wait requests that fell back at the cap. Raw producers supply a deadline per message, and their results go to the shared belt, which is drained destructively, so the TTL and grace rows do not apply there.

## Deployment notes

- The gateway must route `GET/DELETE /v1/requests/*` and `GET /v1/models` to the coordinator. Stock llm-d routing forwards only the inference paths, so these need adding to the coordinator's HTTPRoute.
- The tenant header is trusted as asserted, the same as everywhere else on the llm-d serving path. Request id is the only secret protecting a stored result, so clients that need an unguessable handle should omit `X-Request-Id` and use the minted UUID.
- Set `result_ttl_seconds` on every AP queue the step feeds, or unfetched results never expire.
- Redis needs keyspace notifications enabled for the wait wake-up (`notify-keyspace-events Kl`). The step detects their absence and falls back to polling.
- Wait mode holds one gateway to coordinator connection per waiting client, so the gateway's circuit breaker limits on the coordinator cluster must be sized for held connections, not request rate. Envoy defaults are far too low.
- `preserve_external_request_id` should be set on the gateway so client supplied request ids survive the hop for retry and fetch by id.
- Delivery is at most once at any replica count. A message popped by an AP that then crashes is lost, and the client holds a pending id until its deadline expires. Delivery guarantees beyond this belong to client retries by id.
