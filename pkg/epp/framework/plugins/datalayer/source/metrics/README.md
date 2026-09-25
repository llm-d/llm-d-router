# Metrics Data Source

**Type:** `metrics-data-source`

> [!NOTE]
> This plugin is enabled by default together with `core-metrics-extractor`. You do not need to explicitly declare it in your configuration, but it can be disabled if metrics collection is unnecessary.

The Metrics Data Source is a data layer plugin that polls a Prometheus-compatible metrics endpoint of a model server and parses the response into a structured format for extraction.

## What it does

1.  Periodically (or when triggered) performs an HTTP GET request to a configured metrics endpoint (e.g., `http://<endpoint-ip>:8080/metrics`).
2.  Parses the Prometheus text format response into a `PrometheusMetricMap`.
3.  Provides the parsed metrics to any registered extractors (like the `core-metrics-extractor`).

## Outputs produced

-   `PrometheusMetricType`: A map where keys are metric names and values are Prometheus `MetricFamily` objects.

## Configuration

The plugin config supports:

-   `scheme` (default "http"): The protocol scheme to use for metrics retrieval.
-   `path` (default "/metrics"): The URL path to use for metrics retrieval.
-   `insecureSkipVerify` (default true): Whether to skip TLS certificate verification when using the "https" scheme.
-   `caCertPath`: PEM CA bundle to verify the target's server cert.
-   `clientCertPath` / `clientKeyPath`: client certificate for mTLS. Set both together.
-   `interval` (string, optional): Scrape period (e.g. `"1s"`). Rounded to the nearest
    multiple of `--refresh-metrics-interval` (default 50ms). Omit to scrape every
    base tick.
-   `families` (list of strings, optional): Metric families to keep. Every other line of the
    response is dropped before parsing, which cuts the scrape's CPU and allocations when the
    model server exposes far more families than the extractors read. A family's samples may
    carry the `_bucket`, `_sum`, `_count`, `_total` or `_created` suffix. The list must cover
    every family the source's extractors read; `core-metrics-extractor` reads the engine's
    queued requests, running requests, KV cache usage, LoRA info and cache info metrics.
    Omit to parse the whole response.

### Example Configuration

```yaml
type: metrics-data-source
parameters:
  scheme: "http"
  path: "/metrics"
  insecureSkipVerify: true
  interval: "1s"
```

Parsing only what the core metrics extractor reads from vLLM:

```yaml
type: metrics-data-source
parameters:
  families:
  - vllm:num_requests_waiting
  - vllm:num_requests_running
  - vllm:kv_cache_usage_perc
  - vllm:lora_requests_info
  - vllm:cache_config_info
```

## Multi-cluster support

`multicluster-metrics-data-source` is the cluster-scoped variant. It scrapes a peer cluster's metrics like the pod source, and a distinct type lets a config pair it with `multicluster-metrics-extractor` independently of the pod pipeline. Because it crosses a cluster trust boundary it verifies the peer certificate by default, unlike the pod source. Set a `caCertPath` for a private cluster CA, or `insecureSkipVerify` to opt out. The response payload is size-capped. See the [wiring example](../../../README.md#example).
