# Prefill Pin Screener

**Type:** `prefill-pin-screener`
**Interfaces:** `requestcontrol.Screener`

Keeps only the endpoint that the request's `x-prefill-pin` header names.

## What it does

The coordinator's `prefill-decode` step first asks EPP which prefill endpoint it
would pick (see [reserve-endpoint](../../reserveendpoint/README.md)), and writes
that endpoint into the SGLang decode body. The real prefill request is then
scheduled again, and can land on another pod. This screener makes the second
pick equal to the first: the coordinator sends the reserved `<ip:port>` on
`x-prefill-pin`, and the screener keeps only that endpoint.

## How It Works

- A request without `x-prefill-pin` keeps every endpoint, so all other traffic
  is not changed.
- A request with the header keeps the one endpoint whose `<ip:port>` equals the
  header value. IPv6 addresses are in brackets, as `net.JoinHostPort` writes
  them.
- When that endpoint is not a candidate, the screener keeps no endpoint, and the
  director answers `503` before any scheduling profile runs. It does not fall
  back to another pod: the decode request names the pinned pod, so a prefill on
  another pod would leave both requests waiting for the SGLang bootstrap
  timeout.

Screeners run before data producers, admission and the scheduling profiles, so
on a pinned request those work on one endpoint.

## Inputs consumed

- The `x-prefill-pin` request header.

## Output produced

- The candidate endpoints, narrowed to at most one.

## Configuration

The screener takes no parameters. See the configuration example in the
[reserve-endpoint README](../../reserveendpoint/README.md#configuration), which
shows both plugins in one EPP config.

The coordinator drops a client's `x-prefill-pin` header, so only the
coordinator can pin a request. Other callers of the EPP must not forward the
header from untrusted clients.

## Limitations

- A pin names an HTTP endpoint. With several data-parallel ranks behind one
  SGLang port, SGLang chooses the rank inside that endpoint.

## Related Documentation

- [Coordinator architecture: KV and EC transfer protocols](../../../../../../../docs/coordinator_architecture.md#kv-and-ec-transfer-protocols)
