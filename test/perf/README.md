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
that slows the EPP is compared against the runs before it. When the router
config hash of either run differs from any baseline run, the summary and issue
show the old and new hashes. A metric regresses
when both runs are worse than the baseline median by more than its threshold
and worse than every baseline run. With fewer than the minimum number of
baseline runs, nothing is compared.

The target's `trend` section sets the window (7 days), the minimum baseline
runs (4) and the compared metrics. Each metric has a name, a dotted path into
the run record, a direction (`lower` or `higher` is better), a threshold, and
whether it opens issues (`alerts`) or is only reported. Unknown keys in either
section are rejected, as are a window or minimum baseline runs below 1, a
negative threshold, and duplicate metric names. The `trend` section is not part
of the digest, so tuning it keeps the baseline.

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

### Predicted latency A-C acceptance

[Issue 913](https://github.com/llm-d/llm-d-router/issues/913) requires separate
before/after measurements for the schema and registry phases. Compare the schema
phase with its prerequisite revision, then compare the registry phase with the
schema revision. For each phase, measure four configurations against the same
model backends: Kubernetes baseline, load-plus-prefix with weights `(1, 1, 1)`,
and predicted latency before and after the change. The Kubernetes baseline needs
an actual Kubernetes endpoint; a local round-robin proxy cannot supply that result.

The [load-plus-prefix values](config/router-configs/latency-load-prefix-111.yaml)
configure queue, KV-cache utilization and prefix scorers with weight 1 each.
The `optimized-baseline.yaml` recipe uses weights `(2, 2, 3)` and an additional
scorer, so it represents a different comparison. Pin the image and deployment
settings in the benchmark environment, provide the model-server label selector,
and save the effective EPP configuration. Use consistent prefix producers,
tokenization and cache settings across the comparison.

Use the same predicted-latency scheduling configuration before and after each
phase. Record its affinity filters, picker, prefix producer, prediction endpoints,
model state and `streamingMode`; an omitted `streamingMode` defaults to false.
The chart's default predicted-latency profile includes affinity and SLO filters,
so a smoke setup containing only a latency scorer and picker is not the same
profile. Check effective configuration and prediction/fallback telemetry during
the run. Confirm any differences from the published setup before comparing its
absolute results. The standalone proxy defaults to fail-open, which can bypass
the EPP when it is unavailable. Use the same fail-open setting across strategies
and verify that measured requests were handled by the intended path.

`generate_latency_workloads.py` prepares twelve JSON configurations per phase
for [inference-perf v0.6.0](https://github.com/kubernetes-sigs/inference-perf/tree/v0.6.0).
JSON configurations are accepted by its YAML reader. The generator starts no
services, sends no requests, and downloads no tokenizer or model. Supply an
existing local tokenizer directory and the four comparison endpoints:

```bash
python3 test/perf/generate_latency_workloads.py \
    --phase schema \
    --target k8s=http://k8s-baseline.invalid \
    --target load-prefix=http://load-prefix.invalid \
    --target pred-before=http://pred-before.invalid \
    --target pred-after=http://pred-after.invalid \
    --model MODEL_NAME --tokenizer-path /path/to/local/tokenizer \
    --qps 1 2 4 8 --workers 2 --max-concurrency-per-worker 64 \
    --clip-sigma 4 --output-dir /path/to/results/schema-configs
```

The endpoints, QPS values, worker count, concurrency and length clipping in this
example are placeholders. Select them for the approved benchmark environment.
Repeat with `--phase registry` and the corresponding before/after endpoints.
Keep every workload and load setting identical across the four strategies in
each comparison. The manifest records `prepared-not-run` and the tool version;
retain it alongside the generated configurations and actual run results.

The [published workloads](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms)
use multi-turn requests and the following token distributions:

| Scenario | Groups | Users/group | System | Question mean | Question std |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 6 | 1000 | 1000 | 30 | 9 |
| B | 6 | 1000 | 1000 | 3000 | 900 |
| C | 150 | 5 | 6000 | 1200 | 360 |

Output length has mean 1000 and standard deviation 300 in all scenarios. The
generator uses normal distributions with minimum 1 and maximum
`ceil(mean + clip_sigma * standard_deviation)`. The publication does not specify
clipping bounds, so record the selected bounds and establish their suitability
with the maintainers before claiming equivalent workload reproduction. Context
length and KV capacity must accommodate the long prompts and growing multi-turn
history. A 2048-token functional smoke test cannot cover these workloads.

Run each prepared configuration with the pinned tool only when the endpoints
and resource allocation are ready:

```bash
inference-perf --config_file /path/to/results/schema-configs/schema-A-pred-after.json
```

The generated stages offer load for 100 seconds each. The tool waits for stage
requests to finish before advancing. Keep per-request output and reject runs
with failed stages, incomplete requests or a load generator that cannot maintain
the offered QPS. Choose worker capacity above the expected outstanding load and
verify actual send times; a client concurrency cap can hide backend saturation.
Use the same QPS grid for all strategies and include points below, near and
above saturation. Shared GPU smoke measurements do not establish these gates.

The pinned tool starts its shared-prefix sequence again and clears user sessions
at each stage boundary. A 100-second stage schedules `int(QPS * 100)` requests;
the first 6000 requests in A and B, or 750 in C, use distinct sessions. Only
later requests can exercise another turn in that stage. The manifest records
the nominal number of repeated-session requests for every stage and scenario;
it does not prove that those requests completed successfully. Inspect the
per-request `chat_round` values in the actual results. The per-request export
has no stage ID; for completed, non-overlapping stages, sort by start time and
use each stage summary's request count to group the records. A stage with zero
repeated requests can still be part of the required QPS comparison, but it does
not test multi-turn history. If the sweep does not exercise history, include a
separate sustained multi-turn run and record any deviation from the published
workload.

Record router and predictor revisions, image digests, model revision, GPU and
replica topology, serving arguments, workload seed, and initial predictor
artifacts. Keep training schedules, cache state, warm-up and reset procedures
consistent. Alternate repeated before/after runs to assess measurement noise.
The small-model alternative requires agreement on its acceptance scope.

Calculate per-request NTPOT as end-to-end request latency divided by completed
output tokens. Decode-only TPOT and scheduler latency measure different work.
Retain timestamps and token counts so stage NTPOT and output tokens per second
can be recomputed with the same aggregation and measurement window. Report
errors, actual offered QPS and drain time with the latency and throughput data.

In inference-perf v0.6.0, the [lifecycle summary implementation](https://github.com/kubernetes-sigs/inference-perf/blob/v0.6.0/inference_perf/reportgen/base.py)
uses request start and end timestamps for NTPOT and sets NTPOT to zero for a
successful request with no recorded output tokens. Inspect per-request records
and reject a run containing such requests; otherwise the mean can look
artificially low. Compare scheduled and achieved load using the stage summary's
schedule-delay distribution. The per-request export does not include the
scheduled timestamp, so it cannot reconstruct arrival-to-completion latency.
The tool's output-token throughput divides completed tokens by the interval
from the first request start to the last completion, including drain time.
Use the same definition for every strategy and retain that interval.

With the tool's optional `analysis` dependencies installed, it can plot four
completed report directories using its existing analysis command. Keep the
generated legend and check the input paths before publication:

```bash
mkdir -p /path/to/analysis
inference-perf --analyze \
    /path/to/k8s /path/to/load-prefix \
    /path/to/pred-before /path/to/pred-after \
    --unified_analysis_dir /path/to/analysis
```

The [analysis implementation](https://github.com/kubernetes-sigs/inference-perf/blob/v0.6.0/inference_perf/analysis/analyze.py)
plots against achieved rate. Compare the requested QPS stages as well, and do
not use a delayed or under-driven load stage to establish a saturation gate.

For each phase and scenario, publish NTPOT-versus-QPS and output-tokens-per-second
curves with all four configurations. Include a table at approximately half
saturation, saturation and post-saturation with exact QPS, NTPOT and throughput.
Apply the issue's gates:

- NTPOT regresses by at most 5% at every QPS through saturation.
- Saturation output tokens per second regress by at most 5%.
- The saturation knee moves left by at most one QPS step.
- Predicted latency after the change beats Kubernetes at every QPS and matches
  or beats load-plus-prefix in the regime where the published results do so.
- Scenario A's baseline predicted-latency curve beats both baselines throughout
  the sweep, and the after curve beats both at saturation.
- Scenario B's after curve beats both baselines past eviction onset, and its
  crossover with load-plus-prefix moves right by at most one QPS step.
- Scenario C remains within 5% of load-plus-prefix and strictly beats Kubernetes.

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
