# Models Attributes

This package defines the data structures for models served by an endpoint, as reported by the model server's `/v1/models` API.

## `ModelDataCollection`

A collection of `ModelData` entries describing the models exposed by an endpoint.

- **Key**: `ModelsAttributeKey`; its string form is `/v1/models/models-data-extractor` (the data type `/v1/models` plus the producer name `models-data-extractor`).
- **Fields** (per `ModelData`):
  - `ID`: Model identifier.
  - `Object`: Object type as reported by the model server (i.e. `model`).
  - `Created`: Unix timestamp reported by the model server.
  - `OwnedBy`: Owner reported by the model server (e.g. `vllm`, `sglang`).
  - `Parent`: Parent model identifier (optional, e.g. for LoRA).

## Producers

The following plugins produce this attribute:

- **`models-data-extractor`** (Data Layer): Extracts the list of served models from the endpoint's `/v1/models` API response.
