# Destination Endpoint Served

**Type:** `destination-endpoint-served`
**Stability:** Alpha
**Interfaces:** `ResponseHeaderProcessor`

Writes the identity of the endpoint the scheduler picked into the
response header `x-conformance-test-served-endpoint`. Downstream
consumers (harnesses, dashboards, log processors) join the header value
to a `clusters.yaml`-derived table and attribute per-request cost,
latency, or any other measured metric to the arm that served the
request.

## Why this exists

Envoy's response `filter_metadata["envoy.lb"]` map — the field
[`destination-endpoint-served-verifier`](../../test/responsereceived/destination_endpoint_served_verifier.go)
reads to identify the served endpoint — is written only when Envoy's
own load balancer selected the upstream. That covers `ROUND_ROBIN`,
`LEAST_REQUEST`, `RING_HASH`, and the other cluster load-balancing
policies.

An EPP whose data-plane cluster is `ORIGINAL_DST` with
`original_dst_lb_config.use_http_header: x-gateway-destination-endpoint`
does not use Envoy's load balancer at all: the endpoint identity is
carried in the request header the EPP sets, and Envoy dials that value
directly. No LB choice is made, so no `filter_metadata["envoy.lb"]` is
written on the response. The verifier under that data-plane emits
`fail: missing envoy lb metadata` for every request and attribution
downstream collapses to zero.

This plugin sidesteps Envoy's LB metadata entirely. The
`ResponseHeaderProcessor` extension point receives the picked
`EndpointMetadata` as its fourth argument, populated by the director
from `RequestContext.TargetPod` regardless of Envoy's data-plane
configuration. That `EndpointMetadata` is the same object CostGuard's
per-arm t-digest is keyed on, so the header value the plugin writes and
the arm the scorer accounts against are always the same identity.

## What it writes

- The bare `EndpointMetadata.Name` — the `name` field the discovery
  plugin (e.g. `multicluster-file-discovery`) assigned to the endpoint
  — e.g. `spoke-gemma-a`.
- On the rare path where `targetEndpoint` is nil (the extension point
  fired without a scheduler pick), a legible failure string
  (`fail: no target endpoint`) so a consumer can tell "attribution
  missing" apart from "attribution to an unknown arm".

`Name` is chosen over `<namespace>/<name>` or `<address>:<port>`
because it is the shortest human-readable identity and matches the
`name` key in `clusters.yaml`, which is how the pricing-reversal
harness keys its `clusters_table`. Any consumer preferring the
`<address>:<port>` form can switch to `EndpointMetadata.Address +
":" + EndpointMetadata.Port` in a small fork of this plugin without
changing the wire contract.

## Attribution without an upstream LB

Attribution normally works one of two ways:

1. **Upstream-LB attribution.** Envoy performs LB, records the
   selected host into `filter_metadata["envoy.lb"]`, and the EPP
   reads that on the response. The verifier plugin implements this
   path.
2. **Header-driven attribution.** The EPP writes the pick into the
   request header Envoy dials, and echoes the same identity into the
   response header. This plugin implements this path.

Path 2 is strictly cheaper: no LB metadata round-trip, no dependency
on the data-plane's LB policy, and the attribution is authoritative
(it is the pick, not a description of the pick). It is the correct
path for cluster-scoped EPPs whose data-plane is `ORIGINAL_DST`.

## Configuration

No parameters.

| Parameter | Type | Default | Description |
|---|---|---|---|

Example:

```yaml
- type: destination-endpoint-served
  name: costguard-attribution
```

## Category

`Attribution` — per-request response-header telemetry.

## Concurrency

Stateless. `ResponseHeader` is safe under concurrent requests; the only
shared state is `response.Headers`, which is per-request.

## Relation to `destination-endpoint-served-verifier`

The verifier
([test/responsereceived/destination_endpoint_served_verifier.go](../../test/responsereceived/destination_endpoint_served_verifier.go))
is retained unchanged for conformance suites that run Envoy with a
real upstream LB. The two plugins write to the same header key and
must not be enabled together. An EPP profile picks one:

- Upstream LB in Envoy (WRR / LEAST_REQUEST / RING_HASH) →
  `destination-endpoint-served-verifier`.
- `ORIGINAL_DST` with `use_http_header` in Envoy →
  `destination-endpoint-served` (this plugin).

## Source layout

- `plugin.go` — the plugin: `Plugin`, `New`, `WithName`, `TypedName`,
  `Factory`, `ResponseHeader`. Type constant and header-key constant
  live at the top.
- `plugin_test.go` — unit tests for the two branches of
  `ResponseHeader` and for `TypedName`.
