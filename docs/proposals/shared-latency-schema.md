# Shared schema for latency predictor features

## Summary

This proposal defines the protocol, feature collection contract, rollout sequence and acceptance tests for [llm-d-router issue 913](https://github.com/llm-d/llm-d-router/issues/913). It follows the maintainer's [two-phase scope](https://github.com/llm-d/llm-d-router/issues/913#issuecomment-6749293360): first share the schema, then remove per-feature plumbing. The design requires review by Router and Predictor maintainers; implementation changes have separate pull requests and acceptance gates.

## Motivation

An optional numeric input currently needs matching declarations, validation, population and model-column edits across Router and Predictor. Missing one edit can silently produce a zero-valued input or a coefficient that is never used.

### Goals

- Define each raw field's name, type, default, validation and target participation once.
- Add an optional numeric input with one schema declaration and one Router provider, followed by binding generation.
- Preserve the existing HTTP shape, capture timing, model preprocessing and predictions during the refactor.
- Give reviewers concrete tests for old/new services, historical rows, stored models and end-to-end feature propagation.

### Non-goals

- Changing JSON/HTTP to gRPC, HTTP routes, model algorithms or scheduling policies.
- Shipping a production `image_size` feature, adding arbitrary derived transforms, or replacing joblib and LightGBM artifacts.
- Correcting the existing Go Ridge formula's differences from Python within the behavior-preserving refactor.

## Proposal

The Router repository owns this RFC, the canonical `proto/common/protocol/latency.proto` and `proto/testdata/wire.json`. Router generates Go bindings under `pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol/`. Predictor generates Python bindings under `common/protocol/` from an explicitly pinned Router schema revision. The compiler input name remains `common/protocol/latency.proto` in both repositories so ownership does not change descriptor names or Python import paths. Below, Router paths beginning with `predictedlatency/` are relative to `pkg/epp/framework/plugins/requesthandling/dataproducer/`.

Phase 1 introduces the schema, codecs and HTTP adapters while retaining the existing feature preparation and Go calculation. Phase 2 makes the schema descriptors drive feature selection and the Go registry drive capture and composition. The following definitions specify the proposed final contract, including metadata that Phase 1 declares but Phase 2 consumes.

### Review map and source baseline

This is a proposed contract. Prototype behavior comparisons used Predictor [`1962491`](https://github.com/llm-d/llm-d-latency-predictor/tree/1962491f9245e54db351b3a10ec76535989069b6) and Router [`6e6b3ee`](https://github.com/llm-d/llm-d-router/tree/6e6b3ee4f62e9405fed86c9b8eee97cbb44b0149); a subsequent Router presubmit covered `d66ac0e` plus the prototype changes. The RFC migration is based on Router [`647ca7c`](https://github.com/llm-d/llm-d-router/tree/647ca7cacd74de4fcc94b4717b72161de0f683b9). Its `contextTTL` configuration uses the duration wrapper introduced by `1072c54`; implementation patches must preserve that API and repeat checks on their actual target revisions. `common/protocol/`, `common/feature_plan.py` and Router `features.go` below are proposed additions. Existing integration points are the two Python server modules and Router's `predictedlatency` hooks/client. Schema ownership and the generation commands below describe the Router-owned design; the existing prototype's Predictor-owned generators require adaptation before their results can validate it.

| Priority / review question | Contract to review | Evidence required before implementation merge |
| --- | --- | --- |
| P0: What is shared? | Sections 1-2: complete messages, field numbers, defaults and target orders. | Regeneration and shared wire fixtures. |
| P0: Is JSON unchanged? | Section 3: adapter ownership and per-case conversion rules. | Old/new HTTP matrix and boundary validation cases. |
| P0: When is each signal sampled? | Sections 4-5: scopes, lifecycle table and selected endpoint handling. | Hook tests with changing metrics/load and distinct endpoints. |
| P0: What reaches each estimator? | Section 2: Feature Plan API, transforms and artifact-column authority. | Exact ordered frames across targets, gates, regimes and serving paths. |
| P1: Which models can activate or roll back? | Section 6: candidate checks, activation and rollback limits. | Historical-artifact and failed-reload tests, including mismatched model/scaler dimensions. |
| P1: What is independently mergeable? | Sections 8-10: file map, PR dependencies and phase gates. | Each phase's compatibility results and A-C report. |

## Design details

### 1. Protobuf messages and field numbers

The complete proposed v1 schema is below. No RPC service is declared. Every raw scalar uses `optional`; `required` in `FieldRules` is application validation, not protobuf wire-level requiredness.

```proto
syntax = "proto3";

package llmd.latency.v1;

import "google/protobuf/struct.proto";
import "google/protobuf/timestamp.proto";
import "google/protobuf/descriptor.proto";

option go_package = "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol";

// Model orders apply to raw inputs; engineered columns have their own transforms.
message FieldRules {
  bool required = 1;
  optional double minimum = 2;
  optional double maximum = 3;
  optional string default_json = 4;
  bool nullable = 5;
  int32 ttft_order = 6;
  int32 tpot_order = 7;
  string feature_group = 8;
  string model_column = 9;
  int32 ridge_ttft_order = 10;
  int32 ridge_tpot_order = 11;
  bool go_validate = 12;
  bool api_hidden = 13;
  bool default_now = 14;
}

extend google.protobuf.FieldOptions {
  FieldRules field_rules = 51000;
}

// Field presence distinguishes an unavailable signal from an observed zero.
message Features {
  optional double kv_cache_percentage = 1 [(field_rules) = {
    required: true minimum: 0 maximum: 1 go_validate: true
    ttft_order: 10 tpot_order: 10 ridge_ttft_order: 1 ridge_tpot_order: 1
  }];
  optional int64 input_token_length = 2 [(field_rules) = {
    required: true minimum: 0 go_validate: true
    ttft_order: 20 tpot_order: 20 ridge_ttft_order: 2 ridge_tpot_order: 2
  }];
  optional int64 num_request_waiting = 3 [(field_rules) = {
    required: true minimum: 0 go_validate: true
    ttft_order: 30 tpot_order: 30 ridge_ttft_order: 3 ridge_tpot_order: 3
  }];
  optional int64 num_request_running = 4 [(field_rules) = {
    required: true minimum: 0 go_validate: true
    ttft_order: 40 tpot_order: 40 ridge_ttft_order: 4 ridge_tpot_order: 4
  }];
  optional int64 num_tokens_generated = 5 [(field_rules) = {
    required: true minimum: 0 go_validate: true tpot_order: 70 ridge_tpot_order: 5
  }];
  optional double prefix_cache_score = 6 [(field_rules) = {
    required: true minimum: 0 maximum: 1 go_validate: true ttft_order: 90 ridge_ttft_order: 5
  }];
  optional int64 encoder_input_size = 7 [(field_rules) = {
    minimum: 0 go_validate: true ttft_order: 80 feature_group: "encoder" ridge_ttft_order: 6
  }];
  optional int64 encoder_matched_size = 8 [(field_rules) = {
    minimum: 0 go_validate: true ttft_order: 70 feature_group: "encoder" ridge_ttft_order: 7
  }];
  optional string pod_type = 9 [(field_rules) = {
    nullable: true ttft_order: 120 tpot_order: 80 model_column: "pod_type_cat"
  }];
  optional int64 prefill_tokens_in_flight = 10 [(field_rules) = {
    minimum: 0 ttft_order: 50 tpot_order: 50 feature_group: "tokens_in_flight"
  }];
  optional int64 decode_tokens_in_flight = 11 [(field_rules) = {
    minimum: 0 ttft_order: 60 tpot_order: 60 feature_group: "tokens_in_flight"
  }];
}

// The HTTP codec flattens features into the existing request JSON object.
message PredictionRequest {
  Features features = 1;
}

message TrainingEntry {
  Features features = 1;
  optional double actual_ttft_ms = 2 [(field_rules) = {required: true minimum: 0}];
  optional double actual_tpot_ms = 3 [(field_rules) = {required: true minimum: 0}];
  optional double predicted_ttft_ms = 4 [(field_rules) = {minimum: 0 api_hidden: true}];
  optional double predicted_tpot_ms = 5 [(field_rules) = {minimum: 0 api_hidden: true}];
  google.protobuf.Timestamp timestamp = 6 [(field_rules) = {default_now: true}];
}

message BulkPredictionRequest {
  repeated PredictionRequest requests = 1;
}

message BulkTrainingRequest {
  repeated TrainingEntry entries = 1;
}

// Coefficients are expressed in the original feature units, after descaling.
message ModelCoefficients {
  optional double ttft_intercept = 1;
  map<string, double> ttft_coefficients = 2;
  optional double tpot_intercept = 3;
  map<string, double> tpot_coefficients = 4;
}

// Tree payloads retain the model library's JSON representation.
message XGBoostTrees {
  google.protobuf.ListValue ttft_trees = 1;
  google.protobuf.ListValue tpot_trees = 2;
}

message BucketCount {
  optional string model = 1;
  optional int32 queue_bucket = 2;
  optional int32 cache_bucket = 3;
  optional int32 prefix_bucket = 4;
  optional int64 count = 5;
}

message BucketCounts {
  repeated BucketCount buckets = 1;
}

message ModelInfo {
  optional string model_type = 1;
  optional string objective_type = 2;
  map<string, bool> model_status = 3;
  optional double quantile = 4;
}

// This is the parsed representation of the Prometheus metrics response.
message MetricsResponse {
  optional string model_type = 1;
  ModelCoefficients coefficients = 2;
  XGBoostTrees xgboost_trees = 3;
  BucketCounts bucket_counts = 4;
  optional string raw_metrics = 5;
}

message ModelExportMetadata {
  optional string model_type = 1;
  optional double quantile_alpha = 2;
  google.protobuf.Timestamp exported_at = 3;
  optional int64 ttft_samples = 4;
  optional int64 tpot_samples = 5;
}
```

Labels and timestamps belong to `TrainingEntry`, not `Features`. An entry with `actual_ttft_ms > 0` contributes to the TTFT dataset; an entry with `actual_tpot_ms > 0` contributes to TPOT. Zero retains its existing meaning of no observation for that target. The two `predicted_*` fields describe selected-endpoint predictions, but `api_hidden` keeps them outside Python request models until drift retraining is implemented separately. Accepting them must not change training rows or their split assignment.

The artifact messages describe coefficients, library-owned tree JSON, model information and seed-export metadata. `/metrics` remains Prometheus text. `MetricsResponse` describes a parsed representation; it does not introduce a JSON metrics endpoint. `BucketCount` represents all trainer bucket labels. Router's legacy one-dimensional bucket parser is not made correct by declaring this message and remains separate work. Partial-error prediction response envelopes retain their existing nullable slots and indices; they are not replaced by protobuf response messages in this change.

### 2. Feature metadata and model order

| Metadata | Meaning and consumer |
| --- | --- |
| `required` | Python request field is mandatory; required historical training inputs cannot be defaulted. |
| `minimum`, `maximum` | Python validation bounds; Go applies them where `go_validate` is true. |
| `default_json` | Typed default for a missing optional raw input; absent metadata uses the protobuf scalar default. |
| `nullable` | Preserve the existing Python API's acceptance of explicit `null`, currently for `pod_type`. |
| `ttft_order`, `tpot_order` | Positive values select and order raw columns; zero excludes the raw field from that target. |
| `feature_group` | Existing configuration gate: `encoder` or `tokens_in_flight`; empty means always selected. |
| `model_column` | Name of an explicitly transformed column, currently `pod_type_cat`; it does not generate a new transform. |
| `ridge_ttft_order`, `ridge_tpot_order` | Terms and accumulation order of Go's existing local Ridge calculation. |
| `go_validate` | Preserve the Go client's current validation surface rather than introducing new checks implicitly. |
| `api_hidden`, `default_now` | Request-model controls for non-feature metadata: exclusion and timestamp factory. |

Feature collection sources and lifecycle scopes are Go registry data, because they depend on Router APIs. Bounds, defaults and model orders are schema data and are not copied into registration entries. Unknown feature groups, duplicate model orders, unsupported defaults or descriptor changes require review and tests; the metadata is not a generic transform language.

The raw-column orders are fixed as follows. `-` means excluded. Integers default to zero, floating-point inputs to zero, and `pod_type` to the empty string; these defaults apply only where missing fields are allowed. The first six feature fields and both actual labels are required by the Python API.

| Raw field | TTFT order | TPOT order | Go Ridge TTFT/TPOT | Group |
| --- | ---: | ---: | --- | --- |
| `kv_cache_percentage` | 10 | 10 | 1 / 1 | Always |
| `input_token_length` | 20 | 20 | 2 / 2 | Always |
| `num_request_waiting` | 30 | 30 | 3 / 3 | Always |
| `num_request_running` | 40 | 40 | 4 / 4 | Always |
| `num_tokens_generated` | - | 70 | - / 5 | Always |
| `prefix_cache_score` | 90 | - | 5 / - | Always |
| `encoder_input_size` | 80 | - | 6 / - | Encoder |
| `encoder_matched_size` | 70 | - | 7 / - | Encoder |
| `pod_type` | 120 | 80 | - / - | Always |
| `prefill_tokens_in_flight` | 50 | 50 | - / - | Tokens in flight |
| `decode_tokens_in_flight` | 60 | 60 | - / - | Tokens in flight |

Python inserts `is_queued = int(num_request_waiting > 0)` at order 0. TTFT additionally inserts `effective_input_tokens = (1 - prefix_cache_score) * input_token_length` at 100 and a categorical prefix bucket at 110. The prefix bucket is `min(int(clip(score, 0, 1) * 4), 3)`. `pod_type_cat` uses the existing categories `['', 'prefill', 'decode']`. The noqueue ensemble drops `is_queued` and `num_request_waiting`. Ridge drops the prefix bucket, one-hot encodes pod type and applies its stored `StandardScaler`; tree models retain categorical inputs. Schema order must reproduce these existing lists exactly before any new production input is enabled.

#### Feature Plan API and model association

The plan is a set of functions in `common/feature_plan.py`, not a new model class. Both existing server classes delegate their preparation helpers to it:

| Function | Input, output and obligation |
| --- | --- |
| `raw_fields(target, settings)` | Ordered descriptor fields for `ttft` or `tpot`; resolve encoder/in-flight gates. Cache descriptor selection by target and both gate values. |
| `feature_columns(target, settings, noqueue=False)` | Final ordered names after explicit engineered-column insertion and optional queue-column removal; reject duplicate positions. |
| `training_frame(rows, target, settings)` | Historical dictionaries to target raw frame plus label; fill optional defaults before dropping incomplete required rows. Do not mutate retained samples. |
| `raw_frames(rows, settings, plain_prediction=False)` / `raw_frames_fast(requests, settings)` | TTFT/TPOT raw frames from dictionaries or validated requests. `plain_prediction=True` preserves the plain API's defaults for the `tokens_in_flight` group; the strict fast path reads supplied values. Other fields share target selection and defaults. |
| `prepare_features(df, target, settings, prefix_buckets=4)` | Add the specified transforms/categories and return exactly `feature_columns(...)`. Callers supply a fresh frame or copy. |
| `align_features(frame, model, scaler=None)` | Select fitted names in fitted order; fill only known optional missing inputs; reject unknown columns or incompatible unnamed dimensions. |
| `validate_model_features(model, target, settings, scaler=None, noqueue=False)` | Exercise the target/regime preparation and artifact alignment before publishing a candidate; additionally enforce the model/scaler consistency requirements in section 6. |

For example, with encoder and in-flight gates off, the TTFT frame is `[is_queued, kv_cache_percentage, input_token_length, num_request_waiting, num_request_running, prefix_cache_score, effective_input_tokens, prefill_score_bucket, pod_type_cat]`; TPOT uses the same first five columns followed by `[num_tokens_generated, pod_type_cat]`. Enable in-flight by inserting its two fields after `num_request_running`; enable encoder by inserting `[encoder_matched_size, encoder_input_size]` immediately before TTFT's `prefix_cache_score`. Encoder does not add TPOT columns. Ridge then removes `prefill_score_bucket` and replaces `pod_type_cat` with the fixed three dummy columns produced by the existing pandas transform. Preserve those resulting fitted names and dtypes, not just the raw-field count.

There are three different identities: the schema digest identifies declarations, the fitted names/categories/scaler identify a model's input interpretation, and file checksums identify the downloaded artifact bytes. `ModelInfo` and `ModelExportMetadata` above do **not** contain a feature-plan version; neither `exported_at` nor a matching schema digest proves artifact compatibility. Phase 2 uses the existing estimator/scaler metadata rather than adding a required manifest that would break old joblib files. Training records names through the existing fit/save operations, and serving aligns to those saved names, never to a newly sorted coefficient map. Library versions, schema revision, flags, target/regime and artifact checksums must accompany test/rollout evidence. A transform or categorical-meaning change under an existing name requires a separate versioned artifact rollout; names alone cannot detect it.

### 3. JSON/HTTP boundary

Composition is internal. The wire representation of `PredictionRequest{features: ...}` remains a flat JSON object, for example:

```json
{
  "kv_cache_percentage": 0.4,
  "input_token_length": 128,
  "num_request_waiting": 2,
  "num_request_running": 3,
  "num_tokens_generated": 1,
  "prefix_cache_score": 0.5,
  "encoder_input_size": 0,
  "encoder_matched_size": 0,
  "prefill_tokens_in_flight": 400,
  "decode_tokens_in_flight": 0
}
```

Bulk payloads remain `{"requests": [flat_row, ...]}` and `{"entries": [flat_row, ...]}`. A nested HTTP `features` key is ignored as an unknown field; it is not a second supported request shape.

Router `protocol/json.go` uses `protojson` with proto field names, then flattens feature messages and converts quoted 64-bit integer values directly to `json.Number`, without a float conversion. Python `common/protocol/json_codec.py` uses `json_format` with matching transformations. Exact values above 2^53 are a wire guarantee, not a promise that estimators operating on float arrays retain integer precision. Non-finite scalar numbers cannot be emitted as ProtoJSON string substitutes.

Codecs retain optional zero presence, omit absent optional fields, ignore unknown HTTP fields and parse into a fresh message before replacing a destination. Unknown fields are discarded, not preserved for retransmission. Nullable `pod_type` maps to the existing empty-category behavior. Maps and arrays retain the endpoint-specific `{}`, `[]` and `null` behavior through compatibility facades; protobuf map presence alone cannot express all these distinctions.

Python `common/protocol/models.py:request_model()` constructs Pydantic classes once from descriptors. FastAPI keeps its normal validation and the strict bulk path can construct NumPy arrays directly from validated objects. A second protobuf parse per inference row is unnecessary. `ProtocolModel.to_proto()` remains available for explicit message conversion and contract tests. The training service's diagnostic `/predict` model continues to exclude both in-flight token fields; the independent prediction service accepts them. This asymmetry must be tested, not silently removed.

The prediction service has a further existing asymmetry: `/predict` and `/predict/bulk` accept in-flight fields but prepare them as defaults, while `/predict/bulk/strict` uses the supplied values. Preserve this by passing `plain_prediction=True` to `raw_frames()` for the two plain paths and defaulting the descriptor's `tokens_in_flight` group centrally. This is a group compatibility rule, not a new per-feature service list. New inputs outside that group retain automatic propagation. Correcting the in-flight omission requires a separate behavior-changing proposal; making all three APIs agree inside this refactor would fail fixed-artifact before/after comparison.

The boundary responsibilities are deliberately separate:

| Boundary / case | Conversion and validation rule | Implementation owner |
| --- | --- | --- |
| Go HTTP encode/decode | Public request facades call the shared codec; flatten `features`, retain snake_case output and emit int64 as JSON numbers without float conversion. | Client `types.go` marshal/unmarshal methods and `protocol/json.go`. |
| Python HTTP ingress | Build flat Pydantic models once from descriptors, then retain the existing FastAPI route/response logic. Do not replace request validation with a protobuf parse. | `common/protocol/models.py:request_model()` and declarations in both server modules. |
| Explicit Python message conversion | Flatten/unflatten using `to_dict()` / `from_dict()`; parse into a fresh message and replace the destination only on success. | `common/protocol/json_codec.py`; `ProtocolModel.to_proto()` is opt-in. |
| Missing field | Codec retains absence. API rejects missing required inputs; optional API fields and historical rows receive typed schema defaults at their respective application boundaries. | Codecs, `request_model()`, `training_frame()`. |
| Explicit numeric zero | Preserve presence; do not replace zero with the default. Bounds may still reject it, as in the disposable extension's minimum of 5. | Codecs and existing application validation. |
| Explicit `null` | Protobuf conversion treats scalar null as unset; HTTP request validation rejects numeric null, including optional numeric fields. Only nullable `pod_type` accepts null and preparation maps it to the empty category. | Codec semantics and Pydantic rules are tested separately. |
| Unknown or nested `features` key | Ignore and discard, including when a nested value conflicts with a valid flat field. A nested-only payload still fails required-field validation. | Both codecs and Pydantic's existing extra-field policy. |
| Naming/coercion/error response | Emit snake_case. ProtoJSON name aliases do not expand the FastAPI API's accepted aliases. Preserve existing Pydantic coercion, 422 validation bodies, strict-batch failure and ordinary-batch partial results. | Endpoint compatibility tests against the unmodified services. |

Router's `PredictionRequest` and `TrainingEntry` retain their existing public accessors and gain a `Features` message for additional fields. `PredictionFromFeatures` and `TrainingFromFeatures` clone a snapshot and project existing fields into those accessors. During encoding, accessor values override overlapping schema values, including zero, so existing callers that mutate fields keep working. New callers initialize through these helpers or constructor options:

```go
type PredictionRequestOption func(*PredictionRequest) error

func WithFeatures(*protocol.Features) PredictionRequestOption
func WithFeature(name string, value any) PredictionRequestOption

func NewPredictionRequest(
    kvCachePercentage float64, inputTokenLength int,
    numRequestWaiting int, numRequestRunning int,
    numTokensGenerated int, prefixCacheScore float64,
    options ...PredictionRequestOption,
) (PredictionRequest, error)
```

`WithFeature` rejects unknown schema names and invalid scalar values; final request validation applies bounds. Existing six-argument calls compile. Requiredness is enforced at the Python request boundary; legacy Go accessors cannot distinguish missing fields from zero. Go also retains the cross-field rule `encoder_matched_size <= encoder_input_size`.

### 4. Router Provider contract

The provider is a package-local registration struct with function fields, not a new framework plugin interface. The existing request-handling plugin remains the extension owner. The proposed types are:

```go
type featureScope uint8

const (
    requestFeature featureScope = iota
    endpointFeature
    metricFeature
    contextFeature
)

type featureUse uint8

const (
    predictionFeatures featureUse = 1 << iota
    ttftFeatures
    tpotFeatures
    allFeatureUses = predictionFeatures | ttftFeatures | tpotFeatures
)

type featureInput struct {
    producer *PredictedLatency
    state    *predictedLatencyCtx
    request  *fwksched.InferenceRequest
    endpoint fwksched.Endpoint
    metrics  *fwkdl.Metrics
    load     inFlightLoadSnapshot
    use      featureUse
}

type featureProvider struct {
    name         protoreflect.Name
    field        protoreflect.FieldDescriptor
    defaultValue protoreflect.Value
    scope        featureScope
    uses         featureUse
    enabled      func(*PredictedLatency) bool
    read         func(featureInput) protoreflect.Value
    dependency   func(*PredictedLatency) (plugin.DataKey, any)
}
```

Here `fwksched`, `fwkdl` and `plugin` refer to Router's existing scheduling, data-layer and plugin packages; `protocol` is the generated schema package. `inFlightLoadSnapshot` stores scalar token/request counts read from the existing concurrency attribute. `field` and `defaultValue` are resolved once from the descriptor; a registration declares `name`, `scope`, `uses`, the reader and any gate/dependency.

A concrete request feature registration is shown below. `tokenproducer` refers to the existing tokenizer producer package and `fwkrequest` to `pkg/epp/framework/common/request`.

```go
{
    name: "input_token_length",
    scope: requestFeature,
    uses: allFeatureUses,
    read: func(in featureInput) protoreflect.Value {
        if in.request != nil && in.request.Body != nil && in.request.Body.TokenizedRequest != nil {
            return protoreflect.ValueOfInt64(int64(in.request.Body.TokenizedRequest.TokenCount()))
        }
        return protoreflect.ValueOfInt64(0)
    },
    dependency: func(_ *PredictedLatency) (plugin.DataKey, any) {
        return tokenproducer.TokenizedPromptDataKey, fwkrequest.TokenizedRequest{}
    },
},
```

Readers return the descriptor's scalar kind and read already available data; they do not fetch metrics or call remote services. A reader may use only the inputs supplied at its capture scope:

| Scope | Invocation and available inputs | Existing signals |
| --- | --- | --- |
| `requestFeature` | Request-context creation after tokenization; `request` is available. | Input token length. |
| `endpointFeature` | DAG-ordered `Produce`; `producer`, `endpoint`, and scalar `load` snapshot are available. | Running count, prefix, encoder, pod role, tokens in flight. |
| `metricFeature` | Composition for prediction or a training observation; refreshed `metrics`, request `state`, and `use` are available. | KV usage and waiting queue size. |
| `contextFeature` | Same composition point; `state` and `use` are available. | Generated-token convention: 1 for scheduling prediction, 0 for training rows. |

The current decode-in-flight reader returns zero, preserving the existing signal. Encoder readers are gated by `UseEncoderCacheFeatures`, use encoder-cache attributes and clamp matched size to input size. Prefix defaults to zero when unavailable or NaN. `pod_type` comes from the configured endpoint-role label.

Initialization rejects unknown or duplicate provider names, missing schema providers, absent readers, empty uses and invalid scopes. `Consumes()` collects enabled dependencies into `DataDependencies.Required`, which creates the framework's ordering edges. Encoder dependencies remain required when enabled. Disabled captures use the schema default. Registry coverage tests and typed reader tests guard the correspondence with the schema.

`captureFeatures()` creates a message for one scope. `composeFeatures()` merges request and endpoint snapshots into a fresh message, applies metric/context readers and replaces fields excluded from a `featureUse` with their schema defaults. Captured request/endpoint messages are not mutated by later hooks. New registrations do not require per-feature fields in `predictedLatencyCtx` or new hook branches.

For a disposable extension test, add this field inside `Features`:

```proto
optional int64 extension_test_size = 99 [(field_rules) = {
  minimum: 5 default_json: "7" go_validate: true
  ttft_order: 85 tpot_order: 75
  ridge_ttft_order: 8 ridge_tpot_order: 6
}];
```

Add exactly one test-only provider entry, then regenerate both bindings:

```go
{
    name: "extension_test_size",
    scope: requestFeature,
    uses: allFeatureUses,
    read: func(featureInput) protoreflect.Value {
        return protoreflect.ValueOfInt64(17)
    },
},
```

The test reader deliberately returns a known scalar. It must appear in prediction, TTFT and TPOT rows as 17. A historical missing value defaults to 7, an explicit zero is rejected by the minimum, and a local Ridge model with intercept 1 and this coefficient 2 yields 15 for a missing field. Python discovers the field and its target order without service-specific edits. An actual production reader uses the appropriate existing request/endpoint signal instead; arbitrary derived or categorical features still require an explicit transform design. Neither test declaration is shipped in production.

### 5. End-to-end request and model lifecycle

```mermaid
sequenceDiagram
    participant R as Router hooks
    participant C as Go predictor client
    participant P as Prediction service
    participant T as Training service
    R->>R: Tokenize; capture request and endpoint features in Produce
    R->>C: Compose one scheduling row per candidate endpoint
    C->>P: POST /predict/bulk/strict with flat JSON rows
    P->>P: Validate; build target frames; align to loaded artifact columns
    P-->>C: Ordered TTFT/TPOT predictions
    C-->>R: Predictions used by the existing scheduler
    R->>R: PreRequest records selected primary/prefill endpoints
    R->>C: First token TTFT row; completion average-TPOT row
    C->>T: Buffered POST /add_training_data_bulk
    T->>T: Existing split/buckets; feature plan; fit and export artifacts
    P->>T: Existing model info/download endpoints
    T-->>P: joblib model/scaler files
    P->>P: Validate candidate columns; swap ready model references
```

`newPredictedLatencyContext()` creates immutable request-feature snapshots and the endpoint snapshot map. `generatePredictions()` converts composed rows through `PredictionFromFeatures`; `PredictBulkStrict()` preserves validation, coalescing, cancellation and output-count checks. Strict and ordinary batch calls retain their HTTP paths. Single `Predict()` retains local Ridge calculation and HTTP prediction for tree models. Endpoint selection and prediction fallback retain their existing contracts.

`AddTrainingDataBulk()` and the existing flush loop send measured rows to the trainer. Pydantic validates them. `add_training_sample()` retains positive-label filtering, queue/cache/prefix buckets, capacity, train/test split and retraining schedule. The Feature Plan functions in section 2 supply training, held-out evaluation, default-model initialization and all serving paths. Models retain fitted metadata; Ridge exports descaled coefficients by scaler names, while tree artifacts retain library JSON and LightGBM importances. Candidate activation follows section 6.

The hook-level sampling and label contract is:

| Event / existing hook | Snapshot and endpoint | Observation sent |
| --- | --- | --- |
| Candidate evaluation / `Produce` | Request token count plus an independent endpoint scalar snapshot per full endpoint ID; metrics at row composition. Later endpoint-attribute mutations must not rewrite these snapshots. | One scheduling row per candidate, `num_tokens_generated=1`; no measured labels. |
| Selection / `PreRequest` | Record primary and optional prefill endpoint metadata; clone selected metrics into profile-specific `lastSeenMetrics`. Reuse captured endpoint scalars. | Retain selected predictions; no training row yet. |
| First streaming data callback / `processFirstTokenForLatencyPrediction` | Use prefill snapshot/metrics when disaggregated, otherwise primary. Read the currently retained metrics before this helper's subsequent refresh. A response header alone is not a token event. | `actual_ttft_ms = now - requestReceivedTimestamp` in milliseconds, `actual_tpot_ms=0`, generated-token feature 0. |
| Later token callbacks / `processTokenForLatencyPrediction` | Advance observed-token count/last-token time and refresh retained metrics at the existing hook point. Do not recapture endpoint scalars. | No new per-token training row; retain the current request-average TPOT behavior. |
| Completion / `ResponseBody` with `EndOfStream` | Use primary snapshot and latest retained primary metrics; clear prefix/encoder features for TPOT. With more than one observed token, average TPOT is `(lastTokenTime - requestReceivedTime - TTFT) / (observedTokens - 1)` in milliseconds. | Send TPOT row only for a positive average, with actual TTFT 0. Non-streaming completion records end-to-end latency in the existing TTFT slot. |
| Truncation / cancellation / cleanup | Preserve the terminal-event contract of the target Router base. A synthesized completion is not an additional token. Remove request context on terminal cleanup. | Never fabricate a missing first-token label or duplicate a completed observation; a stale `Produce` must not recreate cancelled context. |

These are Router's current callback-derived token observations, not a new tokenizer measurement of output tokens. Tests must pin the base's callback semantics and check zero/one-token, non-streaming, missing-metrics and disaggregated cases explicitly.

Rereading the live in-flight tracker in `PreRequest` would depend on unspecified hook order and could include the current request itself. Request/endpoint snapshots therefore retain their `Produce` values, while KV/queue metrics follow the existing refresh points. A training row can legitimately differ from its scheduling row. Labels, selected predictions and timestamps remain outside `Features`.

The schema-driven local Ridge function remains `intercept + sum(coefficient[field.name] * feature_value)` in the specified ridge order. An absent coefficient contributes zero and an absent optional input uses its schema default. This preserves the existing subset of raw terms and its floating-point accumulation order. Python also uses engineered, categorical and in-flight terms and quantile adjustments; equality between local Go and Python is therefore not a general baseline invariant. New-feature tests isolate the new contribution; separate before/after tests prove the existing Go formula is unchanged.

### 6. Compatibility mechanisms and failure behavior

| Case | Mechanism | Implementation location and limit |
| --- | --- | --- |
| Old Router, new Predictor | Accept the same flat keys; default newly optional fields. | `common/protocol/models.py`; `common/feature_plan.py`. Required fields stay required. |
| New Router, old Predictor | Existing API models ignore extra keys. | Test against unmodified Python services. Ignoring a feature does not make its learned contribution available. |
| Historical row missing optional field | Add typed schema default before target-specific `dropna`. | `training_frame()`; request defaults. Missing required inputs remain invalid. |
| Explicit zero versus absence | Preserve `optional` presence in codec; apply defaults at the application boundary. | Both JSON codecs, `DefaultFor()` and `default_for()`. Legacy Go accessors still emit their zero values. |
| New plan, old named model | Select the artifact's recorded columns in recorded order; ignore extra current-plan columns. | `align_features()` at prediction/evaluation and `validate_model_features()` at load. |
| Artifact expects missing known optional column | Supply its schema default, including when a feature group is disabled. | `align_features()`; prediction then uses the fitted scaler/model order. Defaults are not a claim of unchanged quality after deliberately disabling a trained signal. |
| Unknown required artifact column | Reject candidate; do not guess column meaning. | Prediction `load_models()` keeps existing references/checksums. Trainer startup reports incompatible seed artifacts as an initialization error. |
| Unnamed artifact | Check `n_features_in_` where exposed; reject a dimension mismatch. | `align_features()`. Matching dimensions cannot establish unknown historical ordering; use a baseline-prediction fixture or migrate/retrain the artifact. |
| Incompatible gated artifact | Validate both regimes and targets before activation; use valid single models if gated loading fails. | Prediction `load_models()`. No partially loaded gated ensemble is published. |
| Timestamp/extra metadata | Preserve Go timestamp offset text and the existing Python request field set. | `TrainingEntry.MarshalJSON()` and `request_model()`. Avoid changing the sample dictionary used by the hash split. |

Field numbers and JSON names have distinct compatibility roles. Protobuf numbers are never reused; removed numbers and names must be reserved. A JSON rename is a wire break even with the same number, so it requires a separate migration. Unknown fields are discarded by these HTTP adapters; binary-protobuf unknown-field retention is not a rollout guarantee.

Phase 1 and Phase 2 add no production model columns and must retain the existing column lists, so they can reuse compatible existing artifacts with the same library versions. A future feature-adding release is a separate rollout: upgrade readers before enabling the writer/column, retain the prior model and settings for rollback, and prevent old workers from consuming new-column artifacts, for example through explicitly versioned deployment artifact directories. This RFC does not add an artifact-distribution versioning service or promise that an old binary can load every future model.

For repeatable before/after training tests, fix timestamps, `PYTHONHASHSEED`, RNG seeds, model settings and the submitted rows. The existing Python hash-based train/test split is process-seed dependent; retaining its code alone does not prove equal splits. Compare retained rows and split membership as well as final predictions.

#### Candidate activation and rollback

Prediction `LightweightPredictor.load_models()` is the activation boundary. Compute file checksums and deserialize into local candidate variables outside the serving lock. Require valid TTFT and TPOT single models, plus both scalers for Ridge. Validate named-column order and available dimensions for every candidate. For Ridge, the transformed/scaler column count must equal the fitted estimator's input count and coefficient length; checking scaler names alone is insufficient. Validate each target and each queue regime before setting `ensemble_active`. Corrupt, unknown-column and dimension-mismatch fixtures must fail before publication.

Admission tests pair a ten-column scaler with a two-input Ridge estimator and separately truncate fitted coefficients. Both must fail before activation, for TTFT/TPOT and queued/noqueue models. Tests must also verify that failed reload retains usable references, checksums, load timestamp and predictions, that a failed cold start remains unready with 503 inference, and that invalid gated candidates use valid single models. Even matching dimensions cannot identify every semantically mismatched multi-file model/scaler pair, so retain matched artifact sets and their baseline prediction fixtures.

After successful validation, swap both target references, their scalers, gated references/active flag, checksums and load timestamp under the existing lock. An in-flight prediction retains its previously captured references. On single-model candidate failure, return failure without replacing a previously usable in-memory set or advancing its loaded checksums. If only a gated candidate fails, preserve the existing policy of activating the validated single models with the ensemble inactive. This is an intentional fallback, not preservation of the old gated predictions.

At cold start there is no prior usable set: prediction readiness remains false and inference follows the existing 503 behavior until compatible files load. Trainer seed loading is a different startup path; incompatible seeds fail initialization rather than receiving this prediction-worker fallback. Retaining in-memory references is not a durable rollback: after a process restart only files remain. A deployment rollback restores the previous binary/configuration and its complete matching artifact set, then checks readiness and reference predictions. Stop incompatible candidate synchronization before restoring files. Restore model and scaler files together; never claim that reverting a feature flag alone restores the prior model.

The existing download path atomically replaces individual destination files, but trainer writes and multi-file generation publication are not transactional. This change must not claim otherwise or silently add a distribution subsystem. If tests expose a cross-file race, report it separately and keep using a matched, quiescent artifact snapshot for compatibility acceptance.

### 7. Generation and cross-repository review

Use protoc 25.3 and Router's existing protoc-gen-go version. Check generated files into both repositories. Predictor declares its protobuf runtime dependency and updates dependency files without upgrading unrelated packages. Router owns the schema and wire fixtures; its Go generation reads the local `proto/` directory without requiring a Predictor checkout.

Predictor records the upstream repository URL, full Router commit SHA, schema-relative path, schema SHA-256 and fixture SHA-256 in `common/protocol/schema-source.json`. Its generator accepts `ROUTER_PROTO_DIR` pointing at that pinned checkout's `proto/`, verifies source bytes against the recorded hashes, generates Python bindings and copies fixtures to `tests/testdata/latency-wire.json`. Updating the pin and expected hashes is an explicit reviewed change. Generated code and fixtures are checked in, so ordinary build, import and inference need no network access or Router checkout. There is one editable `.proto`; generated bindings and copied fixtures carry provenance and are not independently edited.

The lock keys are `repository`, `commit`, `schema_path`, `schema_sha256`, `fixture_path` and `fixture_sha256`. The paths are `proto/common/protocol/latency.proto` and `proto/testdata/wire.json`. Verify both hashes and each file's bytes in the recorded Git commit before invoking the compiler; updating a hash without committing the source must fail. Verification does not fetch code or prove remote availability. CI must obtain the recorded commit from the recorded upstream URL before checking generation; builds and inference use the checked-in generated files. A local POC may record a `file://` source only with the explicit `--allow-local-source` generator flag. Such a lock is unpublished development evidence and cannot satisfy the production upstream-pin gate.

```bash
# Router checkout: canonical source and Go bindings
make generate-proto-latencypredictor PROTOC=/path/to/protoc-25.3
make verify-latencypredictor-proto PROTOC=/path/to/protoc-25.3

# Predictor checkout: ../llm-d-router is at the recorded schema commit
make generate-proto ROUTER_PROTO_DIR=../llm-d-router/proto \
  PROTOC=/path/to/protoc-25.3
make verify-proto ROUTER_PROTO_DIR=../llm-d-router/proto \
  PROTOC=/path/to/protoc-25.3
```

These commands are proposed implementation targets. Verification generates into a temporary directory and compares bindings, provenance and fixtures without rewriting them. Predictor CI checks out the recorded Router commit for this generation check; runtime installation does not. A dependent Predictor PR must link an accessible published Router schema commit and demonstrate generation from it. A private local directory plus a digest is insufficient for independent reproduction. Paired development may use an explicit source override for experiments, but published provenance must name a fetchable full commit; uncommitted schema edits cannot satisfy the merge gate.

Generation tests reject a changed schema or fixture against an unchanged pin, reject output drift and missing source files, and pass with the pinned checkout in a clean directory. Compare bindings generated from the relocated schema against those generated from the identical schema bytes with the same compiler input name. The ownership migration changes source discovery and publication, not messages, JSON or descriptors. The disposable extension harness must mutate Router's canonical schema and add one Router provider; it may update the temporary Predictor pin/hash as test setup, without hand-editing Python feature lists.

### 8. Implementation files and independently reviewable phases

| Repository / phase | Files | Review focus |
| --- | --- | --- |
| Router / 1 | `proto/common/protocol/latency.proto`, `proto/testdata/wire.json` | Single editable schema and shared wire fixtures; protocol ownership reviewed by both components. |
| Predictor / 1 | `common/protocol/{latency_pb2.py,json_codec.py,schema.py,models.py,__init__.py,schema-source.json}`; `tests/testdata/latency-wire.json` | Python bindings, pinned Router source/provenance, copied fixtures, numeric JSON and descriptor-derived request models. |
| Predictor / 1 | `hack/generate-protocol.py`, `Makefile`, `pyproject.toml`, `requirements.txt`, `tests/requirements.txt` | Reproducible generation and the required runtime dependency only. |
| Predictor / 1 | `training/training_server.py`, `prediction/prediction_server.py`, `tests/test_protocol.py`, `tests/test_protocol_generation.py` | Replace duplicated request declarations, use protocol metadata export, retain existing preprocessing; reject source-pin drift without rewriting outputs. |
| Router / 1 | `Makefile.gen.mk`, `hack/generate-latency-proto.py`, `predictedlatency/latencypredictorclient/protocol/` | Bindings and copied fixtures from local canonical schema, schema digest, codecs and generation instructions. |
| Router / 1 | `predictedlatency/latencypredictorclient/{types.go,schema.go,artifacts.go,training.go}` | Compatibility facades, flat serialization and artifact decoding. Registry and Ridge formula remain unchanged in this phase. |
| Predictor / 2 | `common/feature_plan.py`, both service modules; `tests/test_feature_pipeline.py`, `tests/test_training_server.py` | Replace repeated raw-field lists across learning, evaluation, serving and reload; preserve transforms. |
| Predictor / 2 | `hack/test-feature-extension.py`, `hack/feature-extension-roundtrip.py`, `hack/testdata/feature-extension_test.go` | Disposable cross-repository schema/provider extension with real model engines. |
| Router / 2 | `predictedlatency/{features.go,plugin.go,dataproducer_hooks.go,requestcontrol_hooks.go,prediction.go,training.go}` | Registry coverage, required dependencies, request/endpoint snapshots and lifecycle projections. |
| Router / 2 | `predictedlatency/latencypredictorclient/{schema.go,prediction.go,training.go}` and corresponding tests | Options, descriptor validation and ordered local Ridge terms. |
| Router / 2 | `predictedlatency/live_pipeline_test.go`, hook/terminal-event tests, plugin README; `test/perf/` workload generator, tests, configuration and runbook | Actual row capture, reload, lifecycle regressions and reproducible performance preparation. |

Ridge coefficient-name alignment with the fitted scaler and conversion of LightGBM importance NumPy integers are independent artifact corrections with red/green actual-model tests. Review them separately before using their outputs to validate the schema. Unrelated build/review-routing corrections also remain separate. Record an unmodified-main baseline and report any prerequisite's behavioral effect; do not hide a correction inside a claim of behavior-neutral refactoring.

Publication order is: approve this design; review independent prerequisites; publish Router schema/adapters; review Predictor adapters against that exact Router schema commit; then review each repository's registry/feature-plan changes. Paired changes may be reviewed together, but both phase boundaries must build and have their own results. The schema phase must not depend on the registry being present. Publishing schema ownership is independent of runtime rollout order: deploy readers that tolerate a future field before enabling writers or model columns that rely on it.

The file map above is split into these review units; performance gates apply to the paired phase results rather than requiring unrelated algorithm changes:

| Review unit / dependency | Allowed scope | Merge gate |
| --- | --- | --- |
| Prerequisites / independent | Ridge export naming and LightGBM export scalar corrections, each with its own reproducer; unrelated Router fixes separately. | Actual-model red/green tests and explicit behavior impact, without a schema dependency. |
| 1A: Router schema / approved RFC | Canonical messages/fixtures, Go bindings, flat compatibility facades, artifact adapters and local generation support. Keep current hooks, capture timing and Ridge terms. | Existing callers compile; generation clean; old JSON fixtures and old-server compatibility pass; local fixed-coefficient behavior unchanged; paired Phase 1 A-C gate with 1B before runtime changes merge. |
| 1B: Predictor adapters / published Router 1A schema revision | Pinned source/provenance, Python bindings, codec and descriptor-derived HTTP models. Keep existing preparation/model logic. | Generation clean; old validation/error shapes preserved; full Go/Python mixed-version matrix and fixed-artifact comparisons pass; paired Phase 1 A-C gate. |
| 2A: Predictor Feature Plan / Phase 1 | Shared selection/preparation/alignment and candidate validation, including the Ridge pair dimension check. | Ordered frames/dtypes/transforms match the accepted baseline; historical artifacts, reload rejection and controlled retraining pass for all engines/regimes; paired Phase 2 A-C gate with 2B before runtime changes merge. |
| 2B: Router registry / Phase 1 and paired 2A | Providers, required DAG dependencies, lifecycle snapshots, descriptor validation and ordered Ridge terms. | Lifecycle/race tests, disposable schema-plus-provider roundtrip and paired Phase 2 A-C gate. |

Each unit's PR must name its exact baseline/dependency commits, link the corresponding tests and label any failed or pending gate. Phase 1 can be deployed with the old feature plumbing; Phase 2 is not required to make it functional. Neither a design approval nor a small/shared-GPU smoke test substitutes for a pending implementation merge gate.

The canonical schema and fixtures can be published in a smaller schema-only prerequisite, allowing both runtime PRs to pin an upstream schema commit while they are reviewed and tested together. This prerequisite changes no runtime path. Before accepting Phase 1 runtime changes as complete, both repositories' changes and the paired acceptance results must be available; merging a schema declaration alone establishes no runtime or performance result.

### 9. Tests that establish behavior preservation

| Test layer | Input and oracle | Required evidence |
| --- | --- | --- |
| Protocol | Shared old/current flat fixtures, bulk rows, optional zero/missing, integer limits, invalid/non-finite numbers, timestamps, unknown fields, maps and trees. | Go/Python codecs agree semantically; old accessors can decode output; failed decode does not replace the destination. |
| Mixed versions | Old/new Go clients against old/new Python services in both directions. | Existing valid rows and error shapes remain usable; unknown additions are ignored; strict batch ordering and ordinary partial failures retain their contracts. |
| Lifecycle | Same request with distinct prefill/primary metadata, changing live load after `Produce`, missing optional attributes, encoder gates and both streaming modes. | Scheduling and selected training rows use the intended snapshots; TTFT/TPOT projections and required dependency edges match the baseline. Cancellation and terminal events do not publish late context or duplicate training. |
| Fixed artifacts | Identical saved models/scalers and fixed rows through original and refactored paths. | Exact feature names/order, category encoding and transformed arrays; equal target predictions within a declared numeric tolerance. Exercise single, ordinary batch, strict batch, queue regimes and flags. |
| Controlled retraining | Identical retained datasets, timestamps, seeds and model settings for all three engines and mean/quantile objectives. | Equal preparation/split inputs; compare before/after predictions separately from stochastic quality tests. Verify exported coefficient names and feature contributions. |
| Historical artifacts/reload | Save artifacts with optional groups disabled, serve with them enabled and test the reverse with explicit defaults; inject corrupt, unknown-column and mismatched model/scaler candidates. | Recorded columns are honored; failed reload preserves usable references/checksums and predictions; cold start stays unready; gated and single fallback behavior is tested. |
| Extension | In disposable copies, add `extension_test_size` to the schema and one provider; regenerate; use missing/default/nonzero/invalid values. | Actual Go HTTP ingestion reaches training and all serving paths; all engines learn/export the field and reload it; Go local Ridge evaluates the new coefficient. No production test feature remains. |
| Deployment chain | Real HTTP training/prediction services, then Router hooks/Go client and an inference backend. | Captured rows match request/snapshot values, model hashes change, synchronized artifacts match and prediction succeeds after reload. Health checks alone do not satisfy this. |

The extension uses a nonzero default such as 7 to distinguish default propagation from accidental zeroing. Compare local Ridge before/after using the same coefficients and accumulation order. For Go/Python extension comparisons, isolate inputs for which the baseline formulas agree; do not generalize that result to all existing engineered terms or Python quantile outputs.

Fixed-artifact comparisons run in the same pinned runtime and require identical input names, category sets, order, dtypes and transformed values before comparing predictions. Compare each endpoint against its own baseline; single/plain-bulk and strict-bulk equality is not an existing invariant for in-flight inputs. Deterministic paths should produce identical predictions; any nonzero numeric tolerance must be stated and justified in the implementation PR before evaluating results. The 5% A-C performance allowance is not a prediction-value tolerance. Keep a separate report for retraining randomness and do not use it to waive an unexplained fixed-artifact difference.

The implementation PRs should provide reproducible commands such as:

```bash
# Predictor: protocol, feature-plan and artifact regressions
python -m pytest tests/test_protocol.py tests/test_training_server.py \
  tests/test_feature_pipeline.py tests/test_model_artifact_exports.py
python hack/test-feature-extension.py --router-dir ../llm-d-router \
  --protoc /path/to/protoc-25.3 --protoc-gen-go /path/to/protoc-gen-go

# Router: package race checks and the repository gate
go test -race ./pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/...
make presubmit
```

Run the full Predictor suite against disposable services as well. Default-capacity runs produced 41 passed/1 failed/1 skipped on unmodified Predictor, 74/1/1 on Phase 1 and 121/1/1 on final Phase 2. All fail `test_tif_features_quantile_learns_distribution_stress`; the observed low-TIF TTFT coverage is approximately 0.983-0.987 against the required 0.90 +/- 0.08. Increasing bucket capacity changed diagnostic results, but a non-default passing run cannot be reported as a passing default suite. Retain the default configuration and record the failure, baseline comparison and maintainer disposition without weakening thresholds. Tests proving compatibility and tests assessing model quality have different oracles.

### 10. A-C performance acceptance for both phases

Use identical hardware, serving model/revision, topology, cache settings, predictor settings, initial artifacts, workloads and QPS steps. Compare Kubernetes baseline, load-plus-prefix `(1,1,1)`, predicted-latency-before and predicted-latency-after. Record the original main revision and any prerequisite revisions; compare schema against its accepted prerequisite base, registry against schema, and retain the overall comparison against the original main baseline.

| Scenario | Groups x users/group | System tokens | Question tokens | Output tokens |
| --- | --- | ---: | --- | --- |
| A | 6 x 1000 | 1000 | 30 +/- 9 | 1000 +/- 300 |
| B | 6 x 1000 | 1000 | 3000 +/- 900 | 1000 +/- 300 |
| C | 150 x 5 | 6000 | 1200 +/- 360 | 1000 +/- 300 |

Enable multi-turn history and sweep below, through and above saturation. Pin the load generator and record distribution clipping. With the prepared inference-perf v0.6.0 workflow, each stage resets sessions; more than 6000 A/B requests or 750 C requests are needed for a second round in a stage. Inspect actual `chat_round` data. Placeholder QPS or a stage with no repeated sessions cannot by itself establish multi-turn workload equivalence. Agree any deviation from the issue's workload with the maintainers.

Acceptance requires:

- NTPOT regression at most 5% at every measured QPS through saturation and saturation output-token throughput regression at most 5%.
- A saturation knee moving left by at most one QPS step.
- Predicted latency still beating Kubernetes and matching/beating load-plus-prefix in the issue's specified regimes: A wins at saturation, B wins past eviction onset with crossover moving right by at most one step, and C stays within 5% of load-plus-prefix while strictly beating Kubernetes. Reproduce A's baseline advantage across the sweep before evaluating the refactor.
- Four-strategy NTPOT and output-token-throughput curves, tables at half-saturation/saturation/post-saturation, and exact configurations, revisions, image digests and raw results for each phase and scenario.

Compute NTPOT from end-to-end request time divided by completed output tokens. Retain achieved load, schedule delay, errors, zero-token requests and drain intervals so an under-driven client cannot masquerade as backend saturation. Use the same aggregation/window across comparisons and repeated before/after runs to characterize noise. Serialization microbenchmarks at 1, 20 and 100 rows help identify codec overhead but do not replace this gate. Shared-GPU smoke tests and small-model substitutions require an explicitly agreed acceptance scope before being counted as formal evidence.

### 11. Risks and review decisions

- ProtoJSON plus flattening adds work on the client path. Prototype CPU microbenchmarks measured approximately 42.5 microseconds per schema row versus 2.1 for the legacy encoder, and 3.48 milliseconds versus 0.123 for 100 rows, even with descriptor caching. These measurements are diagnostic evidence from the prototype, not A-C results for the Router-owned design. Repeat them on the implementation revision and measure the full routing path; the refactor cannot assume the overhead fits the 5% end-to-end gate. Avoid introducing a second unverified JSON implementation to optimize it.
- Preserving the Go local Ridge subset is a deliberate behavior constraint. A formula correction needs its own tests and performance comparison.
- Historical unnamed artifacts cannot prove their column semantics from dimensions alone. Their compatibility requires a reference fixture or an explicit migration.
- In-place trainer artifact writes can race downloads in the existing service. This proposal does not provide transactional multi-file publication; failures must remain visible and retryable rather than being presented as a successful rollout.
- Maintainers must approve schema ownership/publication, artifact handling for future feature additions, and any reduced performance workload before implementation acceptance.

### 12. Acceptance status and remaining work

This RFC specifies a reviewable design; its approval is a maintainer decision. Prototype test reports are supporting evidence for their recorded trees, not proof that the Router-owned implementation or the latest main revisions pass. The implementation requires these remaining gates:

- Publish the reviewed Router schema revision and replace any local POC source lock with that accessible upstream pin. Ownership/source-pin checks in a local checkout do not establish publication.
- Preserve the dimension-rejection and failure/fallback tests when splitting the Feature Plan changes into a reviewable implementation PR.
- Run generation, targeted compatibility tests and the repository gates on each exact phase tree, preserving duration-wrapper configuration and other upstream contracts. A combined POC package test does not replace independent phase gates or repository presubmit.
- Resolve or obtain maintainer disposition for the default-capacity quantile stress failure reproduced on both baseline and prototype; a capacity override is not a default-suite pass.
- Supply both phases' formal A-C before/after curves, tables, configurations and raw results. Configuration preparation, CPU comparisons and shared-GPU smoke tests do not satisfy those performance gates.

Model/scaler file publication races and the existing Go/Python Ridge formula gap require separate issue disposition if they affect the agreed acceptance baseline; this refactor does not claim to correct them.

#### Local POC evidence

The Router-owned POC uses Router base `647ca7c` and Predictor base `1962491`, with independent artifact-export prerequisites included in the Predictor test trees. It has separate Phase 1 schema/adapter and Phase 2 registry/Feature Plan worktrees. Source pins refer to unpublished local schema snapshots with explicit local-source opt-in; no upstream schema publication or implementation merge is implied.

| Check | Local result and scope |
| --- | --- |
| Phase 1 | 35 targeted Python tests and Go package race checks pass without the Phase 2 registry/Feature Plan. |
| Phase 2 | 82 targeted Python tests and Go package race checks pass, including source generation, ordered features, artifacts, invalid Ridge pairs and failure/fallback state. |
| Source ownership | Both phases verify generation from Router; Python has no editable `.proto`. Go/Python generated bytes match the identical Predictor-owned prototype schema. |
| Source pin failures | Changed schema/fixture bytes, missing files, unknown commits and hash-only updates fail before replacing outputs; `--check` detects output drift without rewriting it. |
| Disposable extension | One Router schema field and one provider reach actual Go HTTP ingestion, training, export, reload and prediction for Ridge, XGBoost and LightGBM. The test field and provider exist only in disposable copies. |
| Mixed versions | 12 old/new Go-client and Python-service combinations pass across both phases and all three engines. Prepare real trained models before positive-prediction assertions; ready default priors do not imply exported local Ridge coefficients are populated. |
| Fixed artifacts and retraining | 12 engine/objective/ensemble configurations pass, with 31 queries spanning queued/noqueue inputs. Each HTTP path's predictions, ordered frames, categories, retained training rows and validation responses match its unmodified baseline exactly; maximum prediction delta is 0 ms. |

The full Router `make presubmit` passes for each phase on the separate review-routing prerequisite in [PR 4213](https://github.com/llm-d/llm-d-router/pull/4213), with unchanged trees during checks. That one-line documentation repair covers an existing unmapped `forwardtokens` production path; it is not part of either issue implementation patch. Implementation-commit signatures require their own verification, separate from staged-tree code checks.

The extension harness bounds its own process and child CPU affinity to two available CPUs on Linux, because training engines can request all CPUs through `n_jobs=-1`. All POC model data and servers are disposable and use loopback ports. No passing default Predictor suite or formal A-C performance acceptance is claimed. Exact test trees, commands and local logs must accompany the implementation review artifacts.

## Alternatives

- Map-valued features avoid generated structs but lose the typed contract, presence and descriptor-driven validation.
- Duplicated flat protobuf messages retain separate feature definitions; composition plus a small compatibility adapter keeps one raw-field definition.
- Direct ProtoJSON would change flat nesting and 64-bit integer representation for existing consumers.
- gRPC changes deployment/transport without being required for schema sharing. A separate schema repository adds another ownership and release path; Router hosts the issue, RFC and canonical schema, with cross-component review for protocol changes.

## Release notes

NONE. The RFC defines a protocol and implementation plan without changing runtime behavior. User-facing implementation changes, if any, require release notes in their own PRs.
