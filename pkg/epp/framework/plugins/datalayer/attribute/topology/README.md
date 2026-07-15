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
