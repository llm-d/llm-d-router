# Topology Affinity Filter Plugin

**Type:** `topology-affinity-filter`

This plugin filters candidate endpoints by topology proximity to a peer endpoint selected
in an earlier scheduling phase.

## What it does

Scoped to single-EPP disaggregated deployments, where one EPP process runs both the
`decode` and `prefill` scheduling profiles for a request. Coordinator deployments, where
prefill and decode are picked by separate EPPs, are not yet supported.

For a disaggregated prefill/decode request, `disagg-profile-handler` selects the decode
endpoint first, then runs the `prefill` profile to select a prefill endpoint. This plugin
runs in the `prefill` profile and keeps only the candidates whose topology is co-located
with the decode pick at `minAffinity` or tighter (host, rack, zone, or region; same host
implies same rack, zone, and region).

Two rules make the filter fail open rather than restrict routing when locality is unknown
or unreachable:

- **No peer topology available** — the peer endpoint is unknown, or it has no non-empty
  topology field — returns all candidates unchanged.
- **No candidate meets `minAffinity`** — returns all candidates unchanged. Topology
  affinity is a preference; it must never make a request unroutable.

A missing value never matches, including empty against empty: an endpoint with no
`Hostname` never passes `minAffinity: host`, even against a peer that also has no
`Hostname`. A candidate endpoint entirely missing the `Topology` attribute is dropped,
not treated as a match.

`minAffinity: host` assumes a host is the NVLink boundary, true for switched 8-GPU
NVLink baseboards (HGX/DGX) but not for rack-scale NVLink domains (e.g. NVL72), where
the switched fabric spans many hosts and `Rack` is the tier that shares NVLink-class
bandwidth. On that hardware, use `minAffinity: rack`, if the extractor populates a
`Rack` value that reflects the NVLink domain rather than a physical enclosure.

## Inputs consumed

Reads the `Topology` attribute (`topology-extractor`) from the candidate endpoints and
from the peer endpoint. The peer endpoint is resolved from the `peer-endpoint` request
attribute, published by `disagg-profile-handler` before running the `prefill` profile.

Declares `Topology` as an optional data dependency: a config with no `topology-extractor`
logs a startup warning rather than an error, since the filter fails open when the
attribute is absent.

## Configuration

| Parameter               | Required | Default | Description                                                                          |
|--------------------------|----------|---------|----------------------------------------------------------------------------------------|
| `minAffinity`            | no       | `host`  | Tightest-to-loosest floor an endpoint must meet: `host`, `rack`, `zone`, or `region`.  |
| `topologyProducerName`   | no       | default producer | `topology-extractor` instance to read the `Topology` attribute from.        |

**Configuration Example:**
```yaml
plugins:
  - type: topology-extractor
  - type: topology-affinity-filter
    name: prefill-topology-affinity
    parameters:
      minAffinity: host
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-topology-affinity
```

## See also

The `topology-affinity-scorer` plugin grades the same candidates by proximity instead of
dropping them, and is the scoring counterpart of this filter.
