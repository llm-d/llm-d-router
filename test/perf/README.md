# EPP Performance Benchmarking Suite

This directory contains the performance testing pipeline for the Endpoint Picker (EPP) of the `llm-d-router`. The pipeline is used to deploy the router in standalone mode, run stress tests using `inference-perf`, and capture metrics such as CPU/Memory utilization and routing latency.

## Directory Layout

- **`config/`**: Configuration manifests for test runs.
  - **`router-configs/`**: Router Helm configuration values recipes (e.g. `optimized-baseline.yaml`).
  - **`studyanalysis/`**: Detailed analysis reports covering prefix-caching scaling performance, ITL streaming duration impact, input/output token scaling, and high QPS comparisons.
  - `llm-d-sim-deployment.yaml` / `llm-d-sim-service.yaml`: Kubernetes manifests for deploying the vLLM simulator.
  - `shared_prefix_job1.yaml`: Performance test workload specification defining load stages, API target, and request distributions.
- **`run_nightly_perf.py`**: The Python orchestrator script responsible for test namespace setup, EPP and simulator deployments, pprof profiling collection, metrics scraping, and markdown generation. It assumes that `kubectl` is already configured to target an active Kubernetes cluster.

---

## Running Locally

To execute the performance suite locally, you must target an active GKE cluster. Ensure your `kubectl` context is configured correctly.

### Prerequisites
- Python 3.10+
- `pyyaml`
- `helm`
- `go` (required to compile/parse pprof profiles)
- `graphviz` (required to generate SVG diagrams from profiles)
- A GKE cluster with Gateway API and Inference Extension CRDs installed.

### Execution Command

Run the orchestrator script from the repository root:

```bash
python3 test/perf/run_nightly_perf.py \
    --router-config test/perf/config/router-configs/optimized-baseline.yaml \
    --test-name optimized-baseline-job1 \
    --perf-job test/perf/config/shared_prefix_job1.yaml \
    --sim-replicas 10 \
    --gcp-project <your-gcp-project-id> \
    --results-dir test/perf/results/optimized-baseline \
    --router-machine-family e2
```

### Parameters
- `--router-config`: Path to the consolidated Helm values file for the EPP.
- `--test-name`: Unique name for this test run (defines the markdown filename).
- `--perf-job`: Path to the `inference-perf` job file config.
- `--sim-replicas`: Number of simulator pods to scale.
- `--gcp-project`: GCP Project ID hosting your GKE cluster.
- `--results-dir`: Output path for appending markdown result metrics.
- `--router-machine-family`: Optional node affinity mapping (e.g. `e2`, `c3`).
- `--collect-pprof-profiles`: (Optional) Collect EPP CPU & memory pprof profiles and render them as PNG and SVG visual artifacts.
- `--gcs-bucket`: (Optional) Target GCS bucket name or URI (e.g. `gs://llm-d-perf-results`) to sync and preserve test results and profiling artifacts without committing them to the Git repository.
- `--no-cleanup`: (Optional) Skip namespace deletion on test completion (useful for debugging).

### Tracing and latency metrics

Set the boolean `router.tracing.enabled` in the router config to control
benchmark tracing. If it is omitted or null, the runner uses
`router.epp.flags.tracing`, accepting Go boolean flag spellings. If both are
omitted or null, tracing is disabled. The runner sets the Chart switch and
EPP flag to the same value and rejects invalid tracing values before invoking
Helm. Sampling and exporter settings come from `router.tracing` in the same
config. The optimized baseline disables tracing.

P50, P95, and P99 are interpolated from the change in the EPP scheduler's
duration histogram and reported in milliseconds. They measure scheduler
latency, not client request latency. Missing metrics or no new events produce
zero values. A quantile in the unbounded bucket uses the highest finite
bucket boundary.

When appending to a report with a different table header, the writer starts
a new table and preserves the historical rows.

### Offline regression tests

With Python and `pyyaml` installed, run from the repository root:

```bash
python3 -m unittest discover -s test/perf -p test_run_nightly_perf.py -v
```

These tests exercise configuration generation, histogram calculations, and
report writing with cluster commands replaced by test doubles. They do not
deploy workloads or measure tracing overhead.

---

## Nightly GitHub Actions Run

The benchmarking pipeline runs daily via GHA:
- **Workflow Path**: `.github/workflows/nightly-router-perf-test-optimized-baseline-10k-1k.yaml`
- **Schedule**: Daily at 09:00 UTC.
- **Actions**:
  1. Authenticates to GCP and configures `kubectl` to point to the GKE development cluster.
  2. Runs `run_nightly_perf.py` (specifying `--router-machine-family e2` and `--gcs-bucket`).
  3. Appends the metrics results and uploads profiling artifacts directly to the specified Google Cloud Storage bucket.
