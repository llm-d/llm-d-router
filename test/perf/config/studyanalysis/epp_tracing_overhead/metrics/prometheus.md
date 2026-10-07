# Prometheus measurements

We query at a 10-second step over the GuideLLM measurement window and replace
`NAMESPACE` and `EPP_POD_REGEX` with the deployed namespace and EPP pod regular
expression.

```promql
sum by (pod) (
  rate(process_cpu_seconds_total{
    namespace="NAMESPACE", pod=~"EPP_POD_REGEX"
  }[1m])
)
```

```promql
container_memory_working_set_bytes{
  namespace="NAMESPACE", pod=~"EPP_POD_REGEX", container="epp"
}
```

```promql
sum by (pod) (
  rate(container_cpu_cfs_throttled_periods_total{
    namespace="NAMESPACE", pod=~"EPP_POD_REGEX", container="epp"
  }[5m])
)
/
clamp_min(
  sum by (pod) (
    rate(container_cpu_cfs_periods_total{
      namespace="NAMESPACE", pod=~"EPP_POD_REGEX", container="epp"
    }[5m])
  ),
  1
)
```

This study collects EPP scheduler, request-processing, and response-processing p50, p95,
and p99 by substituting the metric and quantile below:

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(llm_d_epp_scheduler_e2e_duration_seconds_bucket{
      namespace="NAMESPACE", pod=~"EPP_POD_REGEX"
    }[1m])
  )
)
```

The other histogram names are
`llm_d_epp_request_processing_duration_seconds_bucket` and
`llm_d_epp_response_processing_duration_seconds_bucket`. Use quantiles `0.5`,
`0.95`, and `0.99`. Also collect request rate:

```promql
rate(llm_d_epp_request_total{
  namespace="NAMESPACE", pod=~"EPP_POD_REGEX"
}[1m])
```

We verify the effective sampling ratio from counter increases over the same
measurement window:

```promql
sum(increase({
  __name__=~"otelcol_receiver_accepted_spans(_total)?",
  namespace="NAMESPACE",
  pod=~"OTEL_COLLECTOR_POD_REGEX"
}[BENCHMARK_WINDOW]))
```

```promql
sum(increase(llm_d_epp_request_total{
  namespace="NAMESPACE", pod=~"EPP_POD_REGEX"
}[BENCHMARK_WINDOW]))
```

This study calculates accepted spans per request for each ratio and divides it by the 100%
result. The normalized values is close to 0.01, 0.10, and 1.00. This
check detects incoming sampled trace contexts that the parent-based sampler
would follow instead of applying the configured root ratio. Finally, we retain both raw
counter results, average finite CPU and working-set samples across the steady measurement
window and report throttling separately and retain the raw time series.
