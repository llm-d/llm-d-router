# Reserve Endpoint

**Type:** `reserve-endpoint`
**Interfaces:** `requestcontrol.PreRequest`

Lets a caller ask which endpoint EPP would pick for a request, without sending
the request there.

## What it does

SGLang disaggregation needs the decode request body to name the prefill pod,
so the coordinator must know the prefill pod before it sends either request.
The coordinator sends the prefill request body to the prefill profile with the
header `Prefer: reserve-endpoint`. EPP schedules that request as usual. This
plugin answers the caller with `204`, the picked endpoint on the
`x-llm-d-reserved-host-port` response header, and
`Preference-Applied: reserve-endpoint`. EPP forwards nothing to a model server.

## How It Works

1. Requests without the `reserve-endpoint` preference are not changed.
2. For a request with the preference, the plugin answers `204` with the
   `<ip:port>` of the primary profile's first target endpoint on
   `x-llm-d-reserved-host-port` and with `Preference-Applied: reserve-endpoint`.
   EPP sends the answer and does not forward the request.
3. If no plugin answered the request, the director rejects it with `500`. An
   EPP without this plugin never forwards a reservation to a model server.

The reservation runs every PreRequest plugin, so each plugin applies its
PreRequest effect to the picked endpoint. A plugin that undoes that effect at
the end of the stream (for example `inflight-load-producer`) releases it when
the stream ends: the end-of-stream plugin call carries `TerminationCause:
error`. A plugin with no end-of-stream release keeps the effect. For example,
`approx-prefix-cache-producer` records the prompt's block hashes for the picked
endpoint, and they stay recorded if the real request never reaches it.

EPP sends the answer as an immediate response and treats it as a handled
request: it is logged at info level, and it is counted neither in
`request_error_total` nor in `request_total`. The trace span ends without an
error status.

## Inputs consumed

- The `Prefer` request header (RFC 7240 token `reserve-endpoint`).
- The scheduling result of the primary profile.

## Output produced

- A `204` answer with the headers `x-llm-d-reserved-host-port: <ip:port>` and
  `Preference-Applied: reserve-endpoint`, and no body.

## Configuration

The plugin takes no parameters. It is Alpha, so the EPP must run with
`--allow-experimental-plugins=true`. Put it in the EPP that serves the prefill
profile:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: reserve-endpoint
- type: prefill-filter
- type: queue-scorer
- type: max-score-picker
- type: header-profile-handler
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
```

The plugin is discovered from the top-level `plugins` list. Do not add it to a
scheduling profile.

## Limitations

- The reservation holds no capacity. Between the answer and the arrival of the
  real request, another request can be scheduled to the same endpoint.
- The answer carries one endpoint, the primary profile's first target.

## Related Documentation

- [Coordinator architecture: KV and EC transfer protocols](../../../../../../docs/coordinator_architecture.md#kv-and-ec-transfer-protocols)
