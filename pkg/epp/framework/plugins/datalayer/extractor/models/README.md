# Model Data Extractor

**Type:** `models-data-extractor`

The Models Data Extractor converts the response from a `models-data-source` into endpoint attributes consumed by `models-responder` for model discovery and inference screening.

## What it does

1. Receives the parsed API response forwarded by `models-data-source`.
2. Converts it into a `ModelDataCollection`. See the
   [Models Attributes](../../attribute/models/README.md) documentation for its fields.
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
