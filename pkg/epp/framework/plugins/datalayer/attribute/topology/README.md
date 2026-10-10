# Topology Attributes

This package defines the data structures for endpoint topology information,
used by topology-aware routing plugins.

## `Topology`

Carries the locality of an endpoint. Populated once at endpoint creation.

- **Key**: `TopologyAttributeKey` (`Topology`)
- **Fields**:
  - `Hostname`: The host name of the endpoint. Sourced from `spec.hostname`
    on the Pod object, or from a user-configured pod label.
  - `Rack`: The failure domain rack of the endpoint. Sourced from a
    user-configured pod label.
  - `Zone`: The failure domain zone of the endpoint. Sourced from a
    user-configured pod label.
  - `Region`: The geographic region of the endpoint. Sourced from a
    user-configured pod label.

## Producers

The following plugins produce this attribute:

- **`topology-extractor`** (Data Layer): Sets the `Topology` attribute using
  `spec.hostname` from the Pod object, or the value of a configured endpoint label.

In coordinator deployments, where prefill and decode are picked by separate EPPs, the
prefill EPP's **`topology-stamp-handler`** (Request Control) reads this attribute off its
selected endpoint and encodes it onto the `x-peer-topology` response header, so the
decode EPP's `topology-affinity-filter`/`-scorer` can read the same locality data without
the `Topology` attribute itself crossing the wire.

## Consumers

The following plugins consume this attribute:

- **`topology-affinity-filter`** (Scheduling): Keeps candidate endpoints whose
  `Topology` is co-located with a peer endpoint at a configured minimum affinity.
- **`topology-affinity-scorer`** (Scheduling): Scores candidate endpoints by
  `Topology` proximity to a peer endpoint.
