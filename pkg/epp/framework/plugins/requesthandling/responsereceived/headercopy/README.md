# Request Header Copy Handler Plugin

**Type:** `request-header-copy-handler`

This plugin copies each request header named in its `headers` parameter onto the
response, under the same name, for a coordinator to read what EPP plugins set on the
request EPP forwarded.

## What it does

Scoped to coordinator deployments. A request header a PreRequest plugin sets reaches the
model server and stops there: the server builds its response from its own result, and
Envoy has no default rule that copies request headers onto a response. The coordinator
is the client of that exchange and never sees the request after EPP changed it.

Runs after response headers are received (`ResponseHeader`). For each name in the
`headers` parameter that is present on the request, it writes the same name and value
onto the response headers. The director sends every non-system-owned response header to
Envoy as a header mutation, so the copied headers reach the coordinator.

The value is the one the request carries when the response arrives. For the internal
routing headers, such as `x-llm-d-prefiller-host-port` and
`x-llm-d-kv-cache-prefill-p2p-host-port`, only EPP plugins can supply it, because EPP
drops client values on ingress. For any other name in the `headers` parameter, the value
can come from the client.

Sets nothing when the request or the response has no header map.

## Inputs consumed

The request headers named in the `headers` parameter, as the request EPP forwarded
carries them.

## Configuration

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `headers` | yes | none | Request header names to copy onto the response, each one when present on the request. Names are compared case-insensitively. A name listed more than once is copied once, and EPP logs a warning at startup. |

The list is explicit on purpose: the request header map holds every client header
(`authorization`, `cookie`, `host`, `content-type`, ...), so a copy of every request
header would echo client credentials and mislabel the response body.

**Configuration Example:**

The `kv-offloading` coordinator connector reads the prefill pod's OffloadingConnector
P2P tier address from the prefill response. `p2p-source-producer` sets it on the prefill
request as `x-llm-d-kv-cache-prefill-p2p-host-port`, and this plugin copies it onto the
response:

```yaml
plugins:
  - type: p2p-source-producer
    parameters:
      prefixMatchInfoProducerName: precise-cache
  - type: request-header-copy-handler
    parameters:
      headers:
        - x-llm-d-kv-cache-prefill-p2p-host-port
```

The EPP `ext_proc` filter must set `response_header_mode: SEND`, or EPP never sees the
response headers.

## See also

- [`p2p-source-producer`](../../dataproducer/p2psource/README.md) sets the header the
  example copies.
- [`topology-stamp-handler`](../topologystamp/README.md) writes a response header from
  an endpoint attribute instead of a request header.
