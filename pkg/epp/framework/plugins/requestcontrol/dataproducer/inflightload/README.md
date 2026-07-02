# In-Flight Load Producer Plugin

**Type:** `inflight-load-producer`

Tracks real-time in-flight request and token counts per endpoint by hooking into the request lifecycle. Writes an `InFlightLoad` attribute onto each endpoint in the `Produce` phase, consumed by the following plugins:
- `token-load-scorer`: Scores endpoints based on in-flight tokens.
- `active-request-scorer`: Scores endpoints based on in-flight requests.
- `concurrency-detector`: Provides admission control based on in-flight requests/tokens.
- `prefix-cache-affinity-filter`: Uses in-flight tokens as a load gate to break stickiness.

## Behavior

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
| `addEstimatedOutputTokens` | `bool` | No | `false` | If true, adds an estimate of the generated output tokens to the in-flight counter. |
| `outputRatio` | `float` | No | `1.5` | Estimated output-to-input token ratio applied when `addEstimatedOutputTokens` is true: estimated output = round(inputTokens * `outputRatio`). Computed against the full prompt, not the uncached portion. Must be non-negative. |
| `maxEstimatedOutputTokens` | `int` | No | _(none)_ | Optional upper bound on the estimated output tokens added per request when `addEstimatedOutputTokens` is true. Must be non-negative. Unset means no cap. |

When `addEstimatedOutputTokens` is true, the estimated output added per request is
`min(round(inputTokens * outputRatio), clientMaxOutputTokens?, maxEstimatedOutputTokens?)`.
The client cap is the request's own output limit (`max_tokens` /
`max_completion_tokens` / `max_output_tokens` depending on the API), applied only
when the client specified one. This keeps estimates realistic for high-input /
low-output workloads, where a fixed ratio would otherwise overstate output.

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
