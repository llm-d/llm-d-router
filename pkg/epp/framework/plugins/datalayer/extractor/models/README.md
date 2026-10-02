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

## Source binding

When `dataLayer.sources` does not list the extractor under a `models-data-source`, it is
bound to the configured `models-data-source`. When none is configured, or several are and
none lists the extractor, the EPP logs `datalayer: skipping unresolved dependency` at
start-up and the extractor stays unbound.

## Configuration

No configuration parameters.
