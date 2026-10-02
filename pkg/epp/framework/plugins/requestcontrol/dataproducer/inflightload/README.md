# In-Flight Load Producer Plugin

**Type:** `inflight-load-producer`

Tracks real-time in-flight request and token counts per endpoint by hooking into the request lifecycle. Writes an `InFlightLoad` attribute onto each endpoint in the `Produce` phase, consumed by the following plugins:
- `token-load-scorer`: Scores endpoints based on in-flight tokens.
- `active-request-scorer`: Scores endpoints based on in-flight requests.
- `concurrency-detector`: Provides admission control based on in-flight requests/tokens.
- `prefix-cache-affinity-filter`: Uses in-flight tokens as a load gate to break stickiness.

## Default and estimated-output modes

- **Prefix Cache Discounting**: Automatically detects if an endpoint has a prefix cache hit (via `PrefixCacheMatchInfo`). Only the **uncached** portion of the prompt is added to the in-flight token counter, providing a more accurate estimate of the actual compute load.
- **Token Release Timing**: 
    - If `addEstimatedOutputTokens` is `false` (default): For streaming requests, all tokens are released as soon as the first chunk of the response is received (`StartOfStream`), as the prefill compute is complete. For non-streaming requests (or as a safety net), tokens are released when the response completes (`EndOfStream`).
    - If `addEstimatedOutputTokens` is `true`: The prompt portion is released at `StartOfStream` (for streaming) or `EndOfStream`, and the estimated output portion is released only when the response completes (`EndOfStream`).
- **Request Release**: In-flight request counters are always released when the response completes (`EndOfStream`).

The producer hooks three lifecycle phases:
- **Produce**: Writes current in-flight counts to each endpoint's attributes.
- **PreRequest**: Increments counters when a request is dispatched to an endpoint.
- **ResponseBody**: Decrements counters when a response completes or the request is aborted.

Endpoint departure events (pod removed from the pool) are handled via the `EndpointExtractor` interface to clean up stale counters.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addEstimatedOutputTokens` | `bool` | No | `false` | If true, adds an estimate of the generated output tokens to the in-flight counter. The estimate is read from the output-length bucket published by the `outlen-bucket` plugin; enable that plugin and order it before this producer so requests are classified. |
| `maxEstimatedOutputTokens` | `int` | No | _(none)_ | Optional upper bound on the estimated output tokens added per request when `addEstimatedOutputTokens` is true. Must be non-negative. Unset means no cap. |
| `prefixMatchInfoProducerName` | `string` | No | _(none)_ | Optional `prefix-cache producer` name to read to find cached prefix discount. Unset defaults to approximate-prefix producer. |
| `trackStreamingOutputTokens` | `bool` | No | `false` | Track full decode input plus cumulative generated tokens until completion. Requires a backend supporting continuous OpenAI usage reporting. |
| `outputTokenHeadroom` | `int` | With streaming tracking | _(none)_ | Positive growth allowance added to each decode request's token charge. Incompatible with estimated-output options. |

When `addEstimatedOutputTokens` is true, the estimated output per request is a flat
value determined by the output-length bucket published by the `outlen-bucket` plugin:

| Output-Length Bucket | Estimated output tokens |
|------------|------------------------|
| `LONG` (reasoning chains) | 4 096 |
| `SHORT` (tool-call JSON) | 100 |
| `UNKNOWN` (no reliable signal) | 1 000 |

The estimate is then bounded by the client-requested cap (`max_output_tokens` / `max_tokens`)
and `maxEstimatedOutputTokens`. Ranking invariant: SHORT (100) < UNKNOWN (1 000) < LONG (4 096).
When the `outlen-bucket` plugin is not enabled, every request reads as UNKNOWN and the producer
logs a one-time warning.

## Streaming decode context accounting

`trackStreamingOutputTokens` models resident context rather than remaining
generation work. A decode-capable or unlabeled endpoint is charged full prompt
tokens plus `outputTokenHeadroom` at dispatch. Each increase in engine-reported
cumulative completion tokens increases that charge. Cached prompt tokens remain
part of the decode context; shared physical KV pages are not deduplicated by this
accounting.

Prefill-only endpoints retain the uncached-input charge. The `prefill` profile
releases at the first response chunk; other profiles retain their charge until
completion or cancellation. Duplicate, decreasing, or absent usage does not
reduce a reservation. Default and estimated-output behavior are unchanged when
streaming tracking is disabled.

For streamed OpenAI chat/completions requests, the producer sets
`stream_options.include_usage` and `stream_options.continuous_usage_stats` to
`true`, preserving other stream options. Only enable this mode with compatible
backends. Clients receive the backend's additional usage metadata unchanged.
Anthropic and non-streaming requests are not modified. Backends that report usage
only at completion retain the initial prompt-plus-headroom charge during
generation; this is not continuous output accounting for those requests.

The headroom is a growth allowance, not a hard KV limit. Already admitted
requests can grow after admission closes. Operators must separately observe
physical KV and engine queues, and qualify admission limits for their workload.
The mode does not provide an interactivity or throughput guarantee.

```yaml
plugins:
- type: inflight-load-producer
  name: resident-decode-load
  parameters:
    trackStreamingOutputTokens: true
    outputTokenHeadroom: 2048
```

The example allowance is illustrative. `addEstimatedOutputTokens: true` and
`maxEstimatedOutputTokens` cannot be combined with this mode.

`llm_d_epp_streaming_accounting_requests_total` distinguishes requests that ask
for continuous usage from final-only requests. A request for continuous usage
does not prove the backend emitted it.
`llm_d_epp_streaming_output_observations_total` counts applied cumulative increases
before completion, not generated tokens. The existing in-flight gauges expose
the current token charge. Per-request token histograms record cumulative usage
once at completion rather than once per usage-bearing chunk.

---

## Related Documentation
- [Token Load Scorer](../../../scheduling/scorer/tokenload/README.md)
- [Active Request Scorer](../../../scheduling/scorer/activerequest/README.md)
- [Concurrency Detector](../../../flowcontrol/saturationdetector/concurrency/README.md)
- [Prefix Cache Affinity Filter](../../../scheduling/filter/prefixcacheaffinity/README.md)
- [Concurrency Attributes](../../../datalayer/attribute/concurrency/README.md)

**Configuration Example:**
```yaml
plugins:
  - type: inflight-load-producer
    name: inflight-load
    parameters:
      addEstimatedOutputTokens: true
```
