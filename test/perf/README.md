# EPP Performance Benchmarking Suite

This directory contains the performance testing pipeline for the Endpoint Picker (EPP) of the `llm-d-router`. The pipeline is used to deploy the router in standalone mode, run stress tests using `inference-perf`, and capture metrics such as CPU/Memory utilization and routing latency.

## Directory Layout

- **`config/`**: Configuration manifests for test runs.
  - **`router-configs/`**: Router Helm configuration values recipes (e.g. `optimized-baseline.yaml`).
  - **`targets/`**: Benchmark target manifests. Each defines the benchmark setup and the trend check settings for one test.
  - **`studyanalysis/`**: Detailed analysis reports covering prefix-caching scaling performance, ITL streaming duration impact, input/output token scaling, and high QPS comparisons.
  - `llm-d-sim-deployment.yaml` / `llm-d-sim-service.yaml`: Kubernetes manifests for deploying the vLLM simulator.
  - `shared_prefix_job1.yaml`: Performance test workload specification defining load stages, API target, and request distributions.
- **`run_nightly_perf.py`**: The Python orchestrator script responsible for test namespace setup, EPP and simulator deployments, pprof profiling collection, metrics scraping, and markdown generation. It assumes that `kubectl` is already configured to target an active Kubernetes cluster.
- **`trend.py`**: Compares the newest run record of a test with the median of its recent runs and writes the trend summary and issue body.

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
    --target test/perf/config/targets/optimized-baseline-job1.yaml \
    --gcp-project <your-gcp-project-id> \
    --results-dir test/perf/results/optimized-baseline
```

### Parameters
- `--target`: Benchmark target manifest under `config/targets/`. Its `setup` section sets the router config, perf job, simulator manifests and replicas, EPP replicas, CPU, memory and log verbosity, and machine family, overriding the matching flags below. The manifest's file name is the test name.
- `--router-config`: Path to the consolidated Helm values file for the EPP.
- `--test-name`: Unique name for this test run (defines the markdown filename).
- `--perf-job`: Path to the `inference-perf` job file config.
- `--sim-replicas`: Number of simulator pods to scale.
- `--gcp-project`: GCP Project ID hosting your GKE cluster.
- `--results-dir`: Output path for appending markdown result metrics.
- `--router-machine-family`: Optional node affinity mapping (e.g. `e2`, `c3`).
- `--epp-log-verbosity`: EPP log verbosity passed as `-v` (default 4).
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
bucket boundary. The mean is the change in the histogram's sum divided by the
change in its count, which does not depend on bucket boundaries.

When appending to a report with a different table header, the writer starts
a new table and preserves the historical rows.

### Tool-calling telemetry overhead

Compare a baseline build without tool-calling telemetry against the candidate
build using the same router configuration, simulator, resources, and load.
Scheduler histogram quantiles exclude request parsing and response handling;
they cannot establish the client-latency requirement for this instrumentation.

- Use non-tool requests at representative body sizes, including prompt text that
  mentions tool field names and text serialized with Unicode escapes.
- Run JSON and SSE responses with tracing disabled, then enabled with identical
  sampler and exporter settings on both builds.
- After warm-up, collect client request p50/p99 latency, completed requests per
  second, and errors from the load generator at identical offered load.
- Alternate repeated baseline and candidate runs to distinguish overhead from
  baseline variability. Missing measurements do not establish a passing result.

The local field-capture benchmarks and OTLP tests validate individual code paths;
they do not replace this gateway-to-model comparison.

### Run records

Next to each markdown row, the runner writes a JSON record to
`<results-dir>/runs/<test-name>/<UTC start time>.json`. The record holds the
run status; the target: its ID, its setup with file paths relative to the
repository root, a digest of that setup and of the contents of the files it
references other than the router config, and the SHA-256 of the router config;
the EPP commit from the `llm_d_epp_info` metric; the container
image digests; every container's idle and peak CPU and memory; and the
scheduler latency percentiles and mean. Records are written under the same
condition as the markdown row and sync to GCS with the rest of the results
folder.

### Trend check

`trend.py` compares the newest run record of a target, and the previous
successful run of the same target, with the successful runs from the window
before that previous run. Runs are comparable only when their target ID and
digest match, so editing the target or the workload or simulator files starts a
new baseline. Editing the router config keeps the baseline, so a config change
that slows the EPP is compared against the runs before it. A metric regresses
when both runs are worse than the baseline median by more than its threshold
and worse than every baseline run. With fewer than the minimum number of
baseline runs, nothing is compared.

The target's `trend` section sets the window (7 days), the minimum baseline
runs (4) and the compared metrics. Each metric has a name, a dotted path into
the run record, a direction (`lower` or `higher` is better), a threshold, and
whether it opens issues (`alerts`) or is only reported. Unknown keys in either
section are rejected. The `trend` section is not part of the digest, so tuning
it keeps the baseline.

```bash
python3 test/perf/trend.py \
    --results-dir test/perf/results/optimized-baseline \
    --target test/perf/config/targets/optimized-baseline-job1.yaml \
    --output-dir trend-output
```

It writes `trend-summary.md` and `trend-verdict.json` to the output folder, and
`trend-issue.md` when a metric regressed. When `GITHUB_STEP_SUMMARY` is set, the
summary is appended to it. A regression exits 0. A run record from another
schema version is skipped; an unreadable or malformed record of the current
version exits non-zero.

What to expect from the rule:

- A regression opens or updates the issue on its second run. The first run is
  shown as over threshold in the summary.
- A single bad run does not open an issue.
- A lasting regression is usually reported once. It is reported again only
  when two runs in a row are worse than every run in the week before them.
- Run records start with the first run that writes them, so the first
  comparison happens after 5 comparable successful runs: 4 for the baseline
  and the previous run.

### Offline regression tests

With Python and `pyyaml` installed, run from the repository root:

```bash
python3 -m unittest discover -s test/perf -p "test_*.py" -v
```

These tests exercise configuration generation, histogram calculations, report
and run record writing, and the trend comparison, with cluster commands
replaced by test doubles. They do not deploy workloads or measure tracing
overhead.

---

## Nightly GitHub Actions Run

The benchmarking pipeline runs daily via GHA:
- **Workflow Path**: `.github/workflows/nightly-router-perf-test-optimized-baseline-10k-1k.yaml`
- **Schedule**: Daily at 09:00 UTC.
- **Actions**:
  1. Authenticates to GCP and configures `kubectl` to point to the GKE development cluster.
  2. Checks out `inference-perf` at the commit pinned in the workflow.
  3. Runs `run_nightly_perf.py` with the `optimized-baseline-job1` target and `--gcs-bucket`.
  4. Appends the metrics results and uploads profiling artifacts directly to the specified Google Cloud Storage bucket.
