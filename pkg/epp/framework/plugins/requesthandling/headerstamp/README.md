# endpoint-header-stamper

A PreRequest plugin that carries attributes of the scheduling-selected endpoint onto outgoing request headers. It is the general mechanism for attaching a per-endpoint routing hint that a downstream gateway matches on — a path prefix, a model or lane name, a tenant tag.

The canonical producer pairing is the `attributes-producer` extractor, which surfaces discovery-declared `EndpointMetadata.Attributes` (e.g. a file-discovery entry's `attributes` section). Any producer works: set `attributeProducerName` to the instance whose attributes should be read.

## Configuration

```yaml
- type: endpoint-header-stamper
  name: header-stamper
  parameters:
    # Optional. Producer instance to read attributes from.
    # Defaults to "attributes-producer".
    attributeProducerName: attributes-producer
    stamps:
      - attributeKey: pathPrefix      # endpoint attribute to read
        headerName: ":path"           # header to write; pseudo-headers allowed
        mode: prepend                 # prepend | set (default: set)
      - attributeKey: model
        headerName: "x-llm-d-model"
        mode: set
```

| field | meaning |
|---|---|
| `attributeKey` | Endpoint attribute key, as declared by the producing plugin (e.g. `attributes-producer`'s `attributeKeys`). |
| `headerName` | Request header the value is written to, pseudo-headers such as `:path` included. EPP-owned names (stripped on the way out) and internal routing headers (`x-llm-d-prefiller-host-port` and siblings) are rejected at startup. |
| `mode` | `set` (default) overwrites the header — a client-supplied value can never pick the endpoint. `prepend` prefixes the existing value after normalizing the attribute to a single leading slash. |

Multiple stamps are applied in order. An endpoint missing a stamp's attribute is skipped for that stamp only.

## Stamping `:path`

`mode: prepend` on `:path` prefixes the request path with the attribute value (normalized to exactly one leading slash): an endpoint with `pathPrefix: /laneA` turns `/v1/chat/completions` into `/laneA/v1/chat/completions`, letting peers behind one shared gateway host be discriminated by path instead of by hostname.

Proxy support notes:

- **agentgateway** applies pseudo-header mutations from ext-proc responses, including `:path`, with no extra configuration.
- **Envoy** drops mutations of the routing pseudo-headers unless the ext-proc filter sets [`mutation_rules.allow_all_routing: true`](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/external_processor.proto). Without it, a `:path` stamp is silently ignored — set the filter option (or use a regular header with HeaderMatch routes) for the path-discrimination use case.

## Example

```yaml
# clusters.yaml (file discovery)
endpoints:
  - name: "in-sea-001-a"
    address: "shared-gw.example.com"
    port: "80"
    attributes:
      pathPrefix: "/laneA"
  - name: "in-sea-001-b"
    address: "shared-gw.example.com"
    port: "80"
    attributes:
      pathPrefix: "/laneB"
```

```yaml
# EndpointPickerConfig (excerpt)
plugins:
  - type: attributes-producer
    parameters:
      attributeKeys: [pathPrefix]
  - type: endpoint-header-stamper
    parameters:
      stamps:
        - attributeKey: pathPrefix
          headerName: ":path"
          mode: prepend
```

When the EPP selects `in-sea-001-b`, the forwarded request path is `/laneB/<original path>`, which the shared gateway matches with a `pathPrefix` route.
