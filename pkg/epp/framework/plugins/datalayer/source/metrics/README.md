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
-   `path` (default "/metrics"): The URL path to use for metrics retrieval. The path can have a
    query string, for example `/v1/loads?format=prometheus` (see
    [SGLang load for each rank](#sglang-load-for-each-rank)).
-   `insecureSkipVerify` (default true): Whether to skip TLS certificate verification when using the "https" scheme.
-   `caCertPath`: PEM CA bundle to verify the target's server cert.
-   `clientCertPath` / `clientKeyPath`: client certificate for mTLS. Set both together.
-   `interval` (string, optional): Scrape period (e.g. `"1s"`). Rounded to the nearest
    multiple of `--refresh-metrics-interval` (default 50ms). Omit to scrape every
    base tick.
-   `families` (list of strings, optional): Metric families to keep in addition to the ones
    the source's extractors declare. Every other line of the response is dropped before
    parsing, which cuts the scrape's CPU and allocations when the model server exposes far
    more families than the extractors read. `core-metrics-extractor` declares the metrics of
    every configured engine, `multicluster-metrics-extractor` its two pool metrics. The list
    must cover the families of any other extractor on the source; when it is omitted, such an
    extractor makes the source parse the whole response. A family is named as it appears in
    the `# TYPE` line: a counter by its `_total` name, a histogram or summary by its base name,
    which keeps its `_bucket`, `_sum` and `_count` samples.

### Example Configuration

```yaml
type: metrics-data-source
parameters:
  scheme: "http"
  path: "/metrics"
  insecureSkipVerify: true
  interval: "1s"
```

Keeping a vLLM counter for an extractor that does not declare its families, along with the
families the declaring extractors read. All other metrics are dropped before parsing:

```yaml
type: metrics-data-source
parameters:
  families:
  - vllm:prompt_tokens_total
```

### SGLang load for each rank

SGLang reports the load of each data-parallel rank as Prometheus text at
`/v1/loads?format=prometheus`. Use this path when `/metrics` of the scraped port does not have
every rank:

-   One SGLang HTTP server for a data-parallel group on several nodes. Its `/metrics` has only
    the ranks of its own node.
-   The SGLang Rust server (`SGLANG_RUST_SERVER`). Each rank has its own port, and only the port
    of the first rank serves `/metrics`.

`/v1/loads` uses other metric names than `/metrics`, so the built-in `sglang` engine config of
`core-metrics-extractor` finds no metric in the response. Set the names in `engineConfigs`:

```yaml
- type: metrics-data-source
  parameters:
    path: "/v1/loads?format=prometheus&include=core"
- type: core-metrics-extractor
  parameters:
    engineConfigs:
    - name: "sglang"
      queuedRequestsSpec: "sglang_num_waiting_reqs"
      runningRequestsSpec: "sglang_num_running_reqs"
      kvUsageSpec: "sglang_token_usage"
```

`/v1/loads` has no page size and no page count. With this config, the extractor does not set
`CacheBlockSize` and `CacheNumBlocks`.

## Multi-cluster support

`multicluster-metrics-data-source` is the cluster-scoped variant. It scrapes a peer cluster's metrics like the pod source, and a distinct type lets a config pair it with `multicluster-metrics-extractor` independently of the pod pipeline. Because it crosses a cluster trust boundary it verifies the peer certificate by default, unlike the pod source. Set a `caCertPath` for a private cluster CA, or `insecureSkipVerify` to opt out. The response payload is size-capped. See the [wiring example](../../../README.md#example).
