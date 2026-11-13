# Predicted Latency Producer (`predicted-latency-producer`)

**Type:** `predicted-latency-producer`

Trains XGBoost models via a sidecar and generates per-endpoint TTFT/TPOT predictions.

## Interfaces

DataProducer, PreRequest, ResponseHeader, ResponseBody, ProducerPlugin, ConsumerPlugin

## Responsibilities

- Bulk predictions during `Produce` (writes `LatencyPredictionInfo` to endpoint attributes)
- SLO headroom calculation per endpoint: `headroom = SLO - predicted_latency` (used by downstream scorer and admission plugins)
- TTFT training data collection on first token / EOS
- TPOT training data collection at EOS (streaming mode)
- Per-endpoint running request queue tracking (TPOT SLO priority queue)
- Prefix cache score forwarding from `PrefixCacheMatchInfo` attributes
- Multimodal encoder-cache size forwarding from `EncoderCacheMatchInfo` attributes
  (opt-in via `useEncoderCacheFeatures`)
- TPOT neutralization for prefill endpoints in disaggregated serving
- E2E latency metrics when `streamingMode=false`

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samplingMean` | `1000` | Mean interval for decode token sampling |
| `maxDecodeTokenSamplesForPrediction` | `0` | Max tokens to sample for TPOT prediction (0 = disabled) |
| `sloBufferFactor` | `1.0` | Multiplier for SLO headroom calculation |
| `contextTTL` | `5m` | TTL for per-request context in the cache, as a duration string such as `5m` |
| `streamingMode` | `false` | Record TTFT on first chunk (true) vs EOS (false) |
| `endpointRoleLabel` | `""` | Label key for disaggregated serving roles |
| `predictInProduce` | `true` | Enable/disable bulk predictions. Set false for training-only mode |
| `useEncoderCacheFeatures` | `false` | Feed multimodal encoder-cache sizes (`encoder_input_size`, `encoder_matched_size`) to the predictor. Requires (and auto-creates) a multimodal encoder-cache producer |
| `encoderCacheMatchInfoProducerName` | `""` | Multimodal encoder-cache producer to read match data from. Empty defaults to the auto-created producer |

## Default Behavior (`streamingMode: false`)

By default, the system assumes no SLO headers and trains for end-to-end request latency.
TTFT is recorded at EOS and represents the full e2e latency (reported as
`request_e2e_latency_seconds` in metrics). TPOT is not trained because there is no
per-token streaming. The scorer routes based on e2e latency predictions only, with TPOT
automatically neutralized.

## Streaming Mode (`streamingMode: true`)

Set this when clients send `"stream": true` and you want to train separate TTFT (time to
first token) and TPOT (time per output token) models. TTFT is recorded on the first
streaming chunk, and TPOT is sampled across subsequent tokens.

## Disaggregated Serving

Set `endpointRoleLabel` to the label distinguishing prefill from decode pods. TPOT is
automatically neutralized for prefill endpoints (`TPOTValid=true`, `TPOTHeadroom=0`),
ensuring TPOT doesn't affect scoring, admission, or tier classification for prefill pods.

## Live feature pipeline validation

`TestLiveFeaturePipeline` sends three sequential streaming requests with at most
16 output tokens each. It uses the production hooks and Go HTTP client with real
Python training and prediction services, checks the captured feature values,
waits for both model artifacts to change, and predicts after a model reload.
Token IDs and timing come from vLLM. Endpoint metrics are fixed nonzero fixtures.

Use disposable Python services with a shared model directory, an empty training
data store, and these training settings:

```bash
export LATENCY_MIN_SAMPLES_FOR_RETRAIN=2
export LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH=2
export LATENCY_RETRAINING_INTERVAL_SEC=2
export LATENCY_TEST_TRAIN_RATIO=0
export LATENCY_ENSEMBLE_MODE=false
```

With a vLLM server exposing `/tokenize` and `/v1/completions`, run:

```bash
LATENCY_TEST_VLLM_URL=http://127.0.0.1:8000 \
TRAINING_SERVER_URL=http://127.0.0.1:8001 \
PREDICTION_SERVER_URL=http://127.0.0.1:8002 \
go test -race -count=1 -run '^TestLiveFeaturePipeline$' \
  ./pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency
```

The test skips when `LATENCY_TEST_VLLM_URL` is unset. It requires exclusive access
to the disposable Python services. It does not start servers or download models.
The issue's A-C routing performance scenarios require separate validation.

## Files

| File | Purpose |
|------|---------|
| `plugin.go` | Struct, factory, config, per-request context, queue helpers |
| `requestcontrol_hooks.go` | PreRequest, ResponseHeader, ResponseBody hooks |
| `dataproducer_hooks.go` | Produce, Produces, Consumes |
| `features.go` | Schema providers, dependencies and capture scopes |
| `training.go` | Training entries, metric snapshots, bulk transport |
| `prediction.go` | generatePredictions, validatePrediction, TPOT neutralization |
| `decode_token_sampler.go` | Poisson-distributed token sampling for TPOT |
| `running_request_queue.go` | Per-pod request priority queue |
