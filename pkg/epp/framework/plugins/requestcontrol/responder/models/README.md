# Models Responder

**Type:** `models-responder`
**Interfaces:** `requestcontrol.Responder`, `requestcontrol.Screener`, `plugin.ConsumerPlugin`, `datalayer.Registrant`

Answers `GET /v1/models` from EPP, aggregating the model lists collected from every endpoint
in the pool. A model server only knows its own models, so routing the request to one of them
omits adapters loaded elsewhere.

Enable this Alpha plugin with the EPP command-line flag `--allow-experimental-plugins=true`.
See [Alpha Plugin CLI Flag](../../../README.md#alpha-plugin-cli-flag---allow-experimental-plugins).

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
    { "id": "legal" },
    { "id": "llama-3-8b" }
  ]
}
```

Each item returns only the model `id`, `object`, `created`, `owned_by`, and optional `shutdown_date`
fields reported by the model server.

Endpoints not yet scraped are skipped. If none have been scraped, the plugin returns HTTP 503
rather than an empty list, so a client retries instead of caching it. The 503 continues for as
long as scraping fails, for example when the source uses the wrong scheme.

For inference requests, the plugin checks each endpoint's collected `/v1/models` result:
- If the request does not contain any target model, all endpoints are candidates.
- If one or more endpoints list the requested base model, those endpoints and endpoints without
  collected model data remain candidates.
- If one or more endpoints list the requested LoRA adapter, only those endpoints remain
  candidates. An endpoint without collected model data may not have the adapter.
- If no endpoint lists the requested model, endpoints without collected model data remain
  candidates. If every endpoint has collected model data, all endpoints remain candidates
  because the model data may be incomplete.

## Inputs consumed

- `ModelDataCollection` at `ModelsAttributeKey`, declared as an optional dependency.

## Configuration

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: models-responder
```

No parameters. Listing it will auto-create a
[`models-data-source`](../../../datalayer/source/models/README.md) and a
[`models-data-extractor`](../../../datalayer/extractor/models/README.md) to collect the
per-endpoint lists it reads. The auto-created source scrapes over HTTP.

If the model servers serve `/v1/models` over HTTPS, declare `models-data-source` explicitly.
It overrides the auto-created one. Set the TLS parameters it needs, listed in its README:

```yaml
plugins:
- type: models-responder
- type: models-data-source
  parameters:
    scheme: "https"
    insecureSkipVerify: false
    caCertPath: "/etc/model-server-ca/ca.crt"
```

## Model data refresh

The model data source polls each endpoint at its configured interval. Requests that arrive after
the next collection use the updated model or LoRA adapter data. A hot-loaded adapter may remain
absent from inference candidates until the next successful collection from that endpoint.
