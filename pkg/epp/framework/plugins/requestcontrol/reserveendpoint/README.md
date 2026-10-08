# Reserve Endpoint

**Type:** `reserve-endpoint`
**Interfaces:** `requestcontrol.PreRequest`

Lets a caller ask which endpoint EPP would pick for a request, without sending
the request there.

## What it does

SGLang disaggregation needs the decode request body to name the prefill pod,
so the coordinator must know the prefill pod before it sends either request.
The coordinator sends the prefill request body to the prefill profile with the
header `Prefer: reserve-endpoint`. EPP schedules that request as usual. The
caller gets `204 No Content` with the picked endpoint on the
`x-llm-d-reserved-host-port` response header, `Preference-Applied:
reserve-endpoint`, and the internal routing headers the scheduling plugins set
on the request. EPP forwards nothing to a model server.

## How It Works

1. Requests without the `reserve-endpoint` preference are not changed.
2. For a request with the preference, the plugin records the `<ip:port>` of the
   primary profile's first target endpoint as a request attribute.
3. After every PreRequest plugin ran, the director reads the attribute and
   builds the answer: `x-llm-d-reserved-host-port` with the recorded endpoint,
   `Preference-Applied: reserve-endpoint`, and each of these headers that a
   plugin set on the request: `x-prefiller-host-port`, `x-encoder-hosts-ports`,
   `x-data-parallel-host-port`, `x-kv-cache-source-host-port`. EPP drops those
   four headers on ingress, so a value present after scheduling was set by a
   plugin, and a forwarded request would have carried it to the model server
   or its sidecar. The caller gets it instead.
4. The ext_proc stream handler sends the answer to Envoy as an immediate
   response with status `204`, the answer headers and no body. Envoy replies to
   the caller and opens no connection to a model server.
5. If no plugin recorded an endpoint, the director rejects the request with
   `500`. A reservation without a body is rejected with `400` before
   scheduling. An EPP without this plugin never forwards a reservation to a
   model server.

The reservation runs every PreRequest plugin, so each plugin applies its
PreRequest effect to the picked endpoint. A plugin that undoes that effect at
the end of the stream (for example `inflight-load-producer`) releases it when
the answer is sent: the end-of-stream plugin call carries `TerminationCause:
answered`. A plugin with no end-of-stream release keeps the effect. For
example, `approx-prefix-cache-producer` records the prompt's block hashes for
the picked endpoint, and they stay recorded if the real request never reaches
it.

EPP treats the answer as a handled request: it is logged at info level, and it
is counted neither in `request_error_total` nor in `request_total`. The trace
span ends without an error status.

## Inputs consumed

- The `Prefer` request header (RFC 7240 token `reserve-endpoint`).
- The scheduling result of the primary profile.

## Output produced

- The request attribute `reserve-endpoint.endpoint`, the recorded `<ip:port>`.
- Through the director, a `204` answer with the headers
  `x-llm-d-reserved-host-port: <ip:port>`, `Preference-Applied: reserve-endpoint`
  and the internal routing headers set on the request, and no body.

## Configuration

The plugin takes no parameters. It is Alpha, so the EPP must run with
`--allow-experimental-plugins=true`. Put it in the EPP that serves the prefill
profile. With one scheduling profile no profile handler is needed, and the
default handler runs that profile for every request:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: reserve-endpoint
- type: prefill-filter
- type: queue-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
```

The plugin is discovered from the top-level `plugins` list. Do not add it to a
scheduling profile. Its position in the list does not matter: the director
builds the answer after every PreRequest plugin ran.

## Limitations

- The reservation holds no capacity. Between the answer and the arrival of the
  real request, another request can be scheduled to the same endpoint.
- The answer carries one endpoint, the primary profile's first target.

## Related Documentation

- [Coordinator architecture: KV and EC transfer protocols](../../../../../../docs/coordinator_architecture.md#kv-and-ec-transfer-protocols)
