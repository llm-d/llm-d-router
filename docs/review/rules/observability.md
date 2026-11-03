# Observability

## Register metrics under canonical component subsystems with bounded label cardinality
Register metrics under canonical subsystem constants (`LLMDRouterEndpointPickerSubsystem` / `"llm_d_epp"`, `LLMDRouterCoordinatorSubsystem` / `"llm_d_coordinator"`, or `llm_d_disagg_sidecar_`), label plugin metrics with `plugin_name` and `plugin_type`, and emit series only for configured plugins. Bound caller-influenced or pod-churn labels via closed enums, allowlists, or overflow buckets, expose live plugin gauges via scrape-time collectors, and delete series when endpoints are removed.
- Accept: Per-request ratio histograms alongside raw query and hit counters when the precomputed distribution simplifies alerting.
- Open: Whether per-tenant, per-workload, and per-user cost attribution belongs on opt-in Prometheus labels or only in structured logs and traces.

## Record at most one terminal outcome per request and keep latency histograms unbiased
Record at most one terminal outcome per request across early-return, buffered, and error paths, matching histogram bucket units to emitted values. Separate router overhead from backend inference wait, and skip stage duration observations when a stage is bypassed.

## Acquire package-scoped OTel tracers and propagate spans end to end
Acquire OpenTelemetry tracers via `tracing.Tracer(TracerScope)` with a package-scoped `TracerScope` constant (`llm-d.ai/...`), standard semantic-convention attribute keys, and route templates rather than raw URLs. Extract inbound W3C trace context at ingress and propagate spans across all outbound HTTP and gRPC calls.

## Summarize collections in String() and cap stateful plugin dumps
Summarize collections by count in `String()` on hot paths, and implement `DumpState` only on stateful plugins with sanitized entries capped to fixed top-K bounds.
- Accept: Omitting debug logs on routine plugin no-op branches when sibling plugins stay silent and proxy trace headers already record the decision.
