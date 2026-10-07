# Endpoint Pin Screener

**Type:** `endpoint-pin-screener`
**Interfaces:** `requestcontrol.Screener`

Keeps only the endpoint that the request's `x-llm-d-pin-host-port` header names.

## What it does

Some requests must land on one known pod. For example, in SGLang
disaggregation the decode body names the prefill pod in `bootstrap_host`, so the
prefill request must go to that pod. The caller sends the `<ip:port>` of the pod
on `x-llm-d-pin-host-port`. Only the request that must be pinned carries the
header.

## How It Works

- A request without `x-llm-d-pin-host-port` keeps every endpoint.
- A request with the header keeps the one endpoint whose `<ip:port>` equals the
  header value. The value is parsed, so every spelling of an address matches.
  IPv6 addresses are in brackets, as `net.JoinHostPort` writes them.
- When that endpoint is not a candidate, the screener keeps no endpoint, and the
  director answers `503`. The screener does not fall back to another pod.

Screeners run after flow-control admission and before the scheduling profiles.
A pinned request can therefore wait in the queue and then get `503`, and a
profile filter can still remove the pinned endpoint, for example a prefill
filter on a decode pod, which also gives `503`.

EPP removes `x-llm-d-pin-host-port` from the request it forwards to the model
server.

A pinned request is scheduled, counted, and observed as a normal request. The
scheduling profile runs on the one candidate, so scorers cannot change the
outcome, but data producers and PreRequest plugins still run and record their
effects on the pinned endpoint for the request's lifetime. The request counts in
`request_total` and the other request metrics like any other request. The `503`
for a missing pinned endpoint carries the `rejected-no-endpoints` dropped
reason, the same as a request for which no endpoint exists at all.

## Inputs consumed

- The `x-llm-d-pin-host-port` request header.

## Configuration

The screener takes no parameters.

```yaml
plugins:
- type: endpoint-pin-screener
```

The Screener is discovered from the top-level `plugins` list and runs once per
request. Do not add it to a scheduling profile.

## Limitations

- A pin names an HTTP endpoint. With several data-parallel ranks behind one
  SGLang port, SGLang chooses the rank inside that endpoint.
- EPP cannot tell a trusted pin from a client pin. The coordinator drops a
  client's `x-llm-d-pin-host-port` on its pipeline and passthrough paths. A
  caller that reaches EPP without the coordinator can pin a request to any
  endpoint that the scheduling profile accepts.

## Related Documentation

- [Request control](../../../../../../../docs/architecture.md#request-control):
  where screeners run.
- [Coordinator architecture: RequestContext](../../../../../../../docs/coordinator_architecture.md#requestcontext):
  the client headers the coordinator drops.
