# OpenTelemetry JSON stdout logs

The EPP, coordinator, and routing sidecar emit one JSON object per line to
stdout using the OpenTelemetry Logs Data Model fields below. The output is not
OTLP/JSON. A log collector must apply the mapping before exporting OTLP.

## Record contract

| JSON field | Type | OpenTelemetry field |
| --- | --- | --- |
| `timestamp` | RFC 3339 UTC string | `Timestamp` |
| `severity_text` | string | `SeverityText` |
| `severity_number` | integer | `SeverityNumber` |
| `body` | string | `Body` |
| `service.name` | string | `Resource.Attributes["service.name"]` |
| `logger` | string | `InstrumentationScope.Name` |
| `trace_id` | optional 32-character lowercase hex string | `TraceId` |
| `span_id` | optional 16-character lowercase hex string | `SpanId` |
| all other fields | any JSON value | `Attributes` |

Collectors must parse `timestamp`, move `service.name` to the resource, move
`logger` to the instrumentation scope, decode valid trace and span identifiers,
and retain every other application field as a log attribute.

Severity numbers use the OpenTelemetry bands: `DEBUG=5`, `INFO=9`, `WARN=13`,
`ERROR=17`, and `FATAL=21`. Negative Zap levels used by logr verbosity map to
`DEBUG/5`. Zap `DPANIC` and `PANIC` retain their defined intermediate values,
18 and 19.

Example:

```json
{"timestamp":"2026-09-11T11:34:56.123Z","severity_text":"INFO","severity_number":9,"body":"request complete","logger":"epp","service.name":"llm-d-router","trace_id":"01234567890123456789012345678901","span_id":"0123456789012345"}
```

Trace fields are omitted when no valid span is active.

## Routing sidecar request line

Each request through the routing sidecar ends with one `INFO` record,
`request finished`, so the request can be found by `trace_id` at the default
verbosity. The per-stage records stay at debug (`--zap-log-level=4`).

| Field | Meaning |
| --- | --- |
| `path` | `decode-only`, `prefill-decode`, `encode-decode`, or `encode-prefill-decode` |
| `status` | HTTP status sent to the client, `0` when none was written |
| `duration` | Seconds the request spent in the sidecar |
| `decoder`, `prefiller` | Target `host:port`. `prefiller` only when prefill ran on another pod |
| `encoders` | Number of encoders, when any were used |
| `encodeDuration`, `prefillDuration`, `decodeDuration` | Seconds spent in each stage, when the connector measures it |
| `failedStage` | The first stage that failed |
| `clientWriteFailed` | `true` when the response could not be delivered to the client |
