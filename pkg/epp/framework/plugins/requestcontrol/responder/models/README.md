# Models Responder

**Type:** `models-responder`
**Interfaces:** `requestcontrol.Responder`, `plugin.ConsumerPlugin`, `datalayer.Registrant`

Answers `GET /v1/models` from EPP, aggregating the model lists collected from every endpoint
in the pool. A model server only knows its own models, so routing the request to one of them
omits adapters loaded elsewhere.

Alpha: requires `--allow-experimental-plugins`.

## What it does

1. Declines anything that is not a `GET` for `/v1/models`, letting it route normally. A
   trailing slash matches; the query string is stripped first.
2. Reads the [`ModelDataCollection`](../../../datalayer/attribute/models/README.md) attribute
   from each endpoint.
3. Unions the entries, one per unique model `ID`. A duplicated ID resolves to the lowest
   endpoint ID, so the result is stable.
4. Returns HTTP 200, sorted by model `ID`:

```json
{
  "object": "list",
  "data": [
    { "id": "llama-3-8b" },
    { "id": "legal", "parent": "llama-3-8b" }
  ]
}
```

Endpoints not yet scraped are skipped. If none have been scraped, the plugin returns HTTP 503
rather than an empty list, so a client retries instead of caching it.

## Inputs consumed

- `ModelDataCollection` at `ModelsAttributeKey`, declared as an optional dependency.

## Configuration

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: models-responder
```

No parameters. Listing it will auto-create a
[`models-data-source`](../../../datalayer/source/models/README.md) and a
[`models-data-extractor`](../../../datalayer/extractor/models/README.md) to collect the
per-endpoint lists it reads.

Declare `models-data-source` explicitly to override the auto-created one:

```yaml
plugins:
- type: models-responder
- type: models-data-source
  parameters:
    scheme: "https"
    insecureSkipVerify: false
```

## Limitations

- The list reflects the last completed scrape, so a new adapter appears one interval later.
