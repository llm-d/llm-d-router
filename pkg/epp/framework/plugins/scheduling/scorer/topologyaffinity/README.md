# Topology Affinity Scorer Plugin

**Type:** `topology-affinity-scorer`

**Category:** `Affinity`

This plugin scores candidate endpoints by topology proximity to a peer endpoint selected
in an earlier scheduling phase.

## What it does

Supports two deployment modes:

- **Single EPP**: one EPP process runs both the `decode` and `prefill` scheduling
  profiles for a request. `disagg-profile-handler` selects the decode endpoint first,
  then runs the `prefill` profile to select a prefill endpoint. This plugin runs in the
  `prefill` profile and grades candidates against the decode pick, resolved from a
  request attribute.
- **Coordinator, separate P/D EPPs**: the prefill EPP is picked first, in a different
  process. The prefill EPP's `topology-stamp-handler` plugin encodes its selected
  endpoint's topology onto a response header; the coordinator forwards that header onto
  the decode request. This plugin runs in the decode EPP's `decode` profile and resolves
  the peer topology from that header via `peerTopologyHeader`.

In both modes the plugin grades each candidate by the tightest topology level it shares
with the peer (host, rack, zone, or region; same host implies same rack, zone, and
region), using a fixed proximity curve:

| Common level | Score | KV-transfer path        |
|--------------|-------|--------------------------|
| host         | 1.00  | NVLink                   |
| rack         | 0.20  | NIC, single switch       |
| zone         | 0.05  | multiple switch hops     |
| region       | 0.02  | inter-datacenter         |
| none         | 0.00  |                          |

The curve is shaped by KV-cache transfer bandwidth, not evenly spaced: same host
dominates same rack 5:1 and same zone 20:1, so co-location wins outright while the
looser levels still break ties among non-colocated candidates. Approximate transfer
bandwidth by path, which the ratios above track: NVLink ~900 GB/s, a single NIC and
switch hop ~100-400 Gb/s (roughly two orders of magnitude below NVLink), multiple
switch hops within a zone lower still, and inter-datacenter links the tightest of all.

`Compare` returns exactly one `Level` per call (an ordered switch with early returns,
tightest tier first), so `Score` performs a single `levelScores` lookup per endpoint,
never a sum. Each score is already in `[0.00, 1.00]`; there is nothing to normalize
within this scorer. A total above 1.0 across a profile's scorers comes from
`runScorerPlugins` in the scheduler (`pkg/epp/scheduling/scheduler_profile.go`), which
clamps each scorer's output to `[0, 1]` and then sums `score * weight` across every
scorer configured in the profile — a property of profile weighting, not of this
scorer's own output.

The curve is hardcoded, not configurable per level, in this release. A missing value
never matches, including empty against empty: an endpoint with no `Hostname` never scores
`host` proximity against a peer that also has no `Hostname`.

Every candidate scores 0 when no peer topology is available (the peer endpoint is
unknown, or has no non-empty topology field) or when the candidate is missing the
`Topology` attribute. A zero score contributes nothing to the weighted sum rather than
skewing it.

The `Hostname` tier assumes a host is the NVLink boundary, true for switched 8-GPU
NVLink baseboards (HGX/DGX) but not for rack-scale NVLink domains (e.g. NVL72), where
the switched fabric spans many hosts and `Rack` is the tier that shares NVLink-class
bandwidth. On that hardware, filter or score on `Rack` rather than `Hostname`, if the
extractor populates a `Rack` value that reflects the NVLink domain rather than a
physical enclosure.

## Inputs consumed

Reads the `Topology` attribute (`topology-extractor`) from the candidate endpoints.

The peer topology is resolved first from the `peer-endpoint` request attribute (single
EPP, published by `disagg-profile-handler` before running the `prefill` profile), falling
back to the `peerTopologyHeader` request header (coordinator mode) when the attribute is
absent.

Declares `Topology` as an optional data dependency: a config with no `topology-extractor`
logs a startup warning rather than an error, since every candidate scores 0 when the
attribute is absent.

## Configuration

| Parameter               | Required | Default | Description                                                                  |
|--------------------------|----------|---------|--------------------------------------------------------------------------------|
| `topologyProducerName`   | no       | default producer | `topology-extractor` instance to read the `Topology` attribute from. |
| `peerTopologyHeader`     | no       | unset   | Request header carrying the peer topology in coordinator deployments. Set to `x-peer-topology` when running in a decode EPP behind the coordinator; unused in single-EPP deployments. No other value is accepted: the coordinator's forwarding is not configurable, so a different name is rejected at startup. |

The plugin trusts `peerTopologyHeader` without re-verifying its source. Only set it on a
profile reachable exclusively through the coordinator's forwarded header; a decode EPP
reachable directly by a client would let that client spoof its own peer topology.

**Configuration Example, single EPP:**
```yaml
plugins:
  - type: topology-extractor
  - type: topology-affinity-scorer
    name: prefill-topology-affinity
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-topology-affinity
        weight: 1
```

**Configuration Example, coordinator (decode EPP):**
```yaml
plugins:
  - type: topology-extractor
  - type: topology-affinity-scorer
    name: decode-topology-affinity
    parameters:
      peerTopologyHeader: x-peer-topology
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: decode-topology-affinity
        weight: 1
```

## See also

The `topology-affinity-filter` plugin drops candidates below a minimum affinity instead
of scoring them, and is the filtering counterpart of this scorer.
