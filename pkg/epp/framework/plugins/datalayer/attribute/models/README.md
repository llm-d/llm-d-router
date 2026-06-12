# Models Attributes

This package defines the data structures for models served by an endpoint, as reported by the model server's `/v1/models` API.

## `ModelDataCollection`

A collection of `ModelData` entries describing the models exposed by an endpoint.

- **Key**: `ModelsAttributeKey` (`/v1/models`)
- **Fields** (per `ModelData`):
  - `ID`: Model identifier.
  - `Parent`: Parent model identifier (optional, e.g. for adapters).

## Producers

The following plugins produce this attribute:

- **`models-data-extractor`** (Data Layer): Extracts the list of served models from the endpoint's `/v1/models` API response.
