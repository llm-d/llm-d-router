# latencypredictorclient

Go client for the [llm-d-latency-predictor](https://github.com/llm-d/llm-d-latency-predictor)
Python service, which predicts per-request latency (TTFT / TPOT) for LLM
inference and is trained online from observed request metrics.

This client is used by the `predictedlatency` data producer plugin to fetch
predictions during scheduling and to stream training samples back to the
Python server.

## What it does

- **Prediction.** Issues bulk HTTP prediction calls to one or more
  prediction server URLs (`PredictionURLs`), with a coalescing layer that
  merges concurrent callers of `PredictBulkStrict` into a single batched
  HTTP request (`CoalesceWindow`, `MaxBulkSize`).
- **Training.** Buffers `TrainingEntry` samples locally and periodically
  flushes them to the training server (`TrainingURL`) on `FlushInterval`,
  downsampling to `MaxSampleSize` when the buffer grows large.
- **Model snapshot caching.** Periodically pulls the trained model
  itself from the Python server — coefficients (Bayesian Ridge),
  serialized trees (XGBoost), and bucket counts — on
  `MetricsRefreshInterval`, and caches it locally. (The server exposes
  this via a `/metrics`-style endpoint; the artifacts are not sent
  along with prediction requests.)
- **Local vs. remote inference.** With the cached snapshot, Bayesian
  Ridge predictions run in-process from the cached coefficients, no
  HTTP hop per request. XGBoost and LightGBM predictions go over HTTP
  via bulk prediction calls.

Configuration can be supplied programmatically via `Config` or loaded from
environment variables via `ConfigFromEnv()`. See [types.go](types.go) for
the full set of knobs.

## Entry points

- `New(cfg, logger) *Predictor` — construct a client. Spawns two
  background goroutines (flush loop + coalescing dispatcher).
- `Start(ctx) error` — prime the predictor with initial server status and
  model info.
- `PredictBulkStrict`, `AddTrainingEntry`, and related methods — see
  [prediction.go](prediction.go), [training.go](training.go).

## Tests

`tests/` contains a standalone load-test harness (`main.go`) that drives
the client against a running predictor server. It is a `package main`
binary, not part of the scheduler build.
