# Reserve Endpoint

**Type:** `reserve-endpoint`
**Interfaces:** `requestcontrol.PreRequest`

Lets a caller ask which endpoint EPP would pick for a request, without sending
the request there.

## What it does

The coordinator's `prefill-decode` step sends the prefill request body to the
prefill profile with the header `Prefer: reserve-endpoint` before it sends any
real request. EPP schedules that request as usual. This plugin records the
picked endpoint, and EPP then answers the caller with `200` and the endpoint on
the `x-prefill-host-port` response header. EPP forwards nothing to a model
server.

The coordinator writes that endpoint into the SGLang prefill and decode bodies,
and pins the real prefill request to it with `x-prefill-pin` (see
[prefill-pin-screener](../screener/prefillpin/README.md)).

## How It Works

1. Requests without the `reserve-endpoint` preference are not changed.
2. For a request with the preference, the plugin stores the `<ip:port>` of the
   primary profile's first target endpoint as a request attribute.
3. After the PreRequest plugins, the director answers `200` with that endpoint
   on `x-prefill-host-port`, and does not forward the request.
4. If no plugin recorded an endpoint, the director rejects the request with
   `500`. An EPP without this plugin never forwards a reservation to a model
   server.

The reservation runs every PreRequest plugin, so a plugin that counts load in
PreRequest (for example `inflight-load-producer`) counts it on the endpoint. EPP
releases that state when the stream ends: the end-of-stream plugin call carries
`TerminationCause: answered`. The reservation is not counted in `request_total`.

## Inputs consumed

- The `Prefer` request header (RFC 7240 token `reserve-endpoint`).
- The scheduling result of the primary profile.

## Output produced

- A `200` answer with the header `x-prefill-host-port: <ip:port>` and no body.

## Configuration

The plugin takes no parameters. Put it in the EPP that serves the prefill
profile, together with `prefill-pin-screener`:

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: reserve-endpoint
- type: prefill-pin-screener
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

Both plugins are discovered from the top-level `plugins` list. Do not add them
to a scheduling profile.

## Limitations

- The reservation holds no capacity. Between the answer and the arrival of the
  pinned request, another request can be scheduled to the same endpoint.
- The answer carries one endpoint, the primary profile's first target.

## Related Documentation

- [Coordinator architecture: KV and EC transfer protocols](../../../../../../docs/coordinator_architecture.md#kv-and-ec-transfer-protocols)
