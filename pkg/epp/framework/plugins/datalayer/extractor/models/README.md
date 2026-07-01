# Model Data Extractor

**Type:** `models-data-extractor`

The Models Data Extractor converts the response from a `models-data-source` into endpoint attributes consumed by filters and scorers.

## What it does

1. Receives the parsed API response forwarded by `models-data-source`.
2. Converts it into a `ModelDataCollection` — a slice of `ModelData` entries, each with:
   - `ID` (string): model identifier (e.g. `"llama-3-8b"`).
   - `Parent` (string, optional): base model the adapter derives from.
3. Stores the collection as an attribute on the corresponding endpoint.

## Attributes produced

- `ModelDataCollection` stored at attribute key `ModelsAttributeKey` (`"/v1/models"`) on each endpoint.

```go
attr, ok := endpoint.GetAttributes().Get(models.ModelsAttributeKey)
if !ok || attr == nil {
    return fmt.Errorf("no models found")
}
modelData, ok := attr.(models.ModelDataCollection)
```

## Configuration

No configuration parameters.

## Once-per-endpoint variant

**Type:** `models-endpoint-extractor`

`ModelEndpointExtractor` produces the same `ModelsAttributeKey` attribute but is
driven by endpoint lifecycle events instead of a poll loop. On endpoint add it
fetches `/v1/models` once and stores the parsed `ModelDataCollection`; the model
list is fixed for an endpoint's lifetime, so it is not re-fetched on a timer. A
failed fetch is logged and skipped, leaving the attribute unset so consumers fall
back to their default.

It is registered as the default producer of `ModelsAttributeKey` and self-wires
to an `endpoint-notification-source` (auto-created when absent), so any consumer
that declares the attribute as a dependency gets it without extra configuration.

### Configuration

| Field | Default | Description |
|---|---|---|
| `scheme` | `http` | Scheme used to reach the model server. |
| `path` | `/v1/models` | Path fetched on the model server. |
| `insecureSkipVerify` | `true` | Skip TLS verification when `scheme` is `https`. |
