# Topology Stamp Handler Plugin

**Type:** `topology-stamp-handler`

This plugin stamps the topology of a scheduling profile's selected endpoint onto the
response, for a coordinator to forward from the prefill response to the decode request.

## What it does

Scoped to coordinator deployments, where prefill and decode are picked by separate EPPs
and the peer's topology must cross the wire rather than stay in a request attribute.

Runs after response headers are received (`ResponseHeader`). Reads the encoded topology
of the endpoint selected by `profileName` (the primary profile's endpoint when unset) and
writes it to `headerName` on the response headers, using the same `key=value` wire format
`topology-affinity-filter` and `topology-affinity-scorer` decode on the decode side.

Sets nothing when the profile did not run (e.g. a decode-only request that skipped
prefill), the selected endpoint has no `Topology` attribute, or the response has no
header map to write to.

## Inputs consumed

Reads the `Topology` attribute (`topology-extractor`) from the endpoint selected by
`profileName`, resolved from `request.SchedulingResult`.

## Configuration

| Parameter               | Required | Default            | Description                                                                  |
|--------------------------|----------|---------------------|--------------------------------------------------------------------------------|
| `headerName`             | no       | `x-peer-topology`   | Response header the encoded topology is written to. Must be `x-peer-topology`: the coordinator's request forwarding is not configurable, so any other value is rejected at startup. |
| `profileName`            | no       | primary profile      | Scheduling profile whose selected endpoint's topology is stamped.            |
| `topologyProducerName`   | no       | default producer     | `topology-extractor` instance to read the `Topology` attribute from.         |

**Configuration Example:**
```yaml
plugins:
  - type: topology-extractor
  - type: topology-stamp-handler
    name: prefill-topology-stamp
    parameters:
      profileName: prefill
```

## See also

`topology-affinity-filter` and `topology-affinity-scorer` decode `headerName` on the
decode side when configured with a matching `peerTopologyHeader` parameter.
