# Model Data Extractor

**Type:** `models-data-extractor`

The Models Data Extractor converts the response from a `models-data-source` into endpoint attributes consumed by `models-responder` for model discovery and inference screening.

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
