# Topology Attributes

This package defines the data structures for endpoint topology information,
used by topology-aware routing plugins.

## `Topology`

Carries the locality of an endpoint. Populated once at endpoint creation.

- **Key**: `TopologyAttributeKey` (`Topology`)
- **Fields**:
  - `Hostname`: The host name of the endpoint, derived from the Pod hostname
    field or from a user-configured label.

## Producers

The following plugins produce this attribute:

- **`topology-extractor`** (Data Layer): Sets the `Topology` attribute on each
  endpoint when it is created, using the Pod hostname or a configured label.
