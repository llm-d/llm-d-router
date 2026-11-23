# Model inference health

This opt-in alpha producer observes valid business text output and probes inactive aggregated vLLM endpoints. It publishes reusable `InferenceHealth` state without changing routing or Pod readiness.

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: model-inference-health
  parameters: {inactivityInterval: 10s, requestTimeout: 3s, maxObservationAge: 1m, maxConcurrentProbes: 4}
schedulingProfiles: [{name: default, plugins: [{pluginRef: max-score-picker}]}]
```

The example shows defaults. Optional `scheme` is `http` or `https` with system certificates. Durations must be positive and at most 24h; evidence age must exceed inactivity plus 10 percent jitter and timeout. Concurrency is 1 to 64 per instance. Configure one instance per EPP. Notifications, in-flight load and model discovery are resolved automatically; no model parameter is needed. Discovery requires one base model from `/v1/models`, excluding entries with parents. Missing or ambiguous models leave health unknown; model/endpoint replacement invalidates evidence and late results. Explicit model-source configuration takes precedence. The automatic data-layer source path ignores polling intervals; explicitly configure `models-data-source` under `dataLayer.sources` to control discovery frequency.

Business observations require one profile, endpoint and text choice, matching models and Envoy `envoy.lb` metadata `x-gateway-destination-endpoint-served` matching the selected address and port. JSON requires complete text with a `stop` or `length` finish reason; SSE requires a complete event with a nonempty text delta. Errors, empty, role-only and usage-only output do not establish health. Bodies/events are bounded to 1 MiB. Probes call `/v1/chat/completions` directly with fixed text, `max_tokens: 1`, `ignore_eos: true`, temperature zero and streaming off, requiring complete valid output within the timeout. Redirects, backend authentication and custom TLS credentials are unsupported. Probes consume capacity outside EPP request accounting; cancellation may not stop backend execution.

In-flight or unknown load raises the base interval to at least one minute. The scheduler checks at intervals of at most one second, with up to 10 percent positive jitter and one outstanding probe per endpoint within the concurrency budget. Failure doubles the retry interval, capped at the greater of 60 seconds and the base interval; success resets backoff. Typed state retains model, known/up, reason, observation time and last success time.

| Metric | Meaning | Labels |
| --- | --- | --- |
| `llm_d_epp_model_inference_up` | 1 after success, 0 after a failed probe; absent for unknown/expired evidence or unavailable models. | `endpoint_name`, `namespace`, `port`, `producer_name`, `target_model_name` |
| `llm_d_epp_model_inference_probes_total` | Completed probes, including superseded results. | `producer_name`, `outcome`, `reason` |

Both metrics use EPP conventions and alpha stability. Outcomes are `success` or `error`; reasons are `success`, `http-error`, `transport-error`, `timeout`, `canceled`, `invalid-request`, `invalid-response`, or `response-too-large`. Model labels use the EPP limiter; endpoint/model removal deletes health series. Account for scrape timing. Unknown evidence is not failure or verified recovery.

Only unlabeled aggregated endpoints with rank index zero are supported. The backend must accept the fixed text chat request. PD, other protocols, aliases, reasoning-only output and multi-EPP coordination require separate support. Failures can reflect queueing, long prefill or connection issues; they do not identify GPU faults or measure throughput/quality. Real vLLM fault coverage and overhead remain unverified. TODO: custom JSON/file probe content, backend progress awareness, coordinated ownership and routing consumers with confirmation, freshness, recovery and all-endpoints-down policies.
