# Topology Affinity Filter Plugin

**Type:** `topology-affinity-filter`

This plugin filters candidate endpoints by topology proximity to a peer endpoint selected
in an earlier scheduling phase.

## What it does

Scoped to single-EPP disaggregated deployments, where one EPP process runs both the
`decode` and `prefill` scheduling profiles for a request. Coordinator deployments, where
prefill and decode are picked by separate EPPs, are not yet supported.

The `disagg-profile-handler` publishes the endpoint selected in the earlier stage as
the peer. In prefill-first mode, this filter runs in the `decode` profile relative to
the selected prefiller. In decode-first mode, it runs in the `prefill` profile relative
to the selected decoder. Candidates match at `minAffinity` or tighter (host, rack,
zone, or region; same host implies same rack, zone, and region).

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

## Request-count allowance

An optional `loadAllowance` limits the extra in-flight requests tolerated on the
least-loaded matching endpoint relative to the least-loaded known non-matching
endpoint. When that difference exceeds the allowance, the filter returns its full
incoming candidate set for downstream scoring. At or below the allowance, it keeps
all matching candidates. Endpoints removed by earlier filters remain excluded.

The allowance is a nonnegative integer. Omission or `null` disables the load gate;
zero enables it with no allowance for extra local requests. Missing, nil, incorrectly
typed or negative request counts preserve the topology-only result. Incomplete
topology that cannot establish whether a candidate is outside the requested boundary
also preserves that result. If either group is absent, the topology-only behavior applies.

Counts describe router-observed outstanding work. They are not queueing milliseconds.
Calibrate the allowance for the hardware, model and workload. A downstream locality
scorer can override the intended escape; use the active-request scorer to select the
least-loaded survivor when evaluating this rule.

For example, the following parameters permit two extra local requests. The value is
illustrative and requires calibration on the deployment:

```yaml
parameters:
  minAffinity: host
  loadAllowance: 2
```

## Inputs consumed

Reads the `Topology` attribute (`topology-extractor`) from the candidate endpoints and
from the peer endpoint. The peer endpoint is resolved from the `peer-endpoint` request
attribute, published by `disagg-profile-handler` before running the later profile.

Declares `Topology` as an optional data dependency: a config with no `topology-extractor`
logs a startup warning rather than an error, since the filter fails open when the
attribute is absent.

When `loadAllowance` is configured, `InFlightLoad` is a required producer dependency.
`inFlightLoadProducerName` selects a named producer; omission uses the default one.
Runtime load observations can still be absent on individual endpoints, in which case
the filter preserves its topology-only result.

## Configuration

| Parameter               | Required | Default | Description                                                                          |
|--------------------------|----------|---------|----------------------------------------------------------------------------------------|
| `minAffinity`            | no       | `host`  | Tightest-to-loosest floor an endpoint must meet: `host`, `rack`, `zone`, or `region`.  |
| `topologyProducerName`   | no       | default producer | `topology-extractor` instance to read the `Topology` attribute from.        |
| `loadAllowance` | no | disabled | Maximum extra local in-flight requests; nonnegative integer. |
| `inFlightLoadProducerName` | no | default producer | Producer supplying the `InFlightLoad` attribute when the allowance is enabled. |

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
