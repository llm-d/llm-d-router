# Latency predictor protocol

The canonical schema and wire fixtures are owned by
[llm-d-router](https://github.com/llm-d/llm-d-router)
under `proto/`. The schema digest is recorded in `schema.sha256`.
Generated Go messages and `testdata/wire.json` are updated together.

The JSON codec preserves the flat snake_case HTTP fields and integer JSON
numbers. Optional zero values retain presence. Unknown HTTP fields are ignored
and discarded. Application validation, defaults and model feature selection
are separate responsibilities.

The Go client serializes prediction and training requests with these messages.
Legacy public struct fields remain as compatibility accessors; additional raw
features travel in the generated `Features` message. Model-info
and coefficient JSON decoding also use the generated types.

Use `PredictionFromFeatures` or `TrainingFromFeatures` to initialize a facade
from a schema message. For fields that also have legacy accessors, encoding uses
the accessor value, including zero, rather than the value in `Features`. Mutate
the accessor after initialization when changing one of those fields.

## Generate and verify

Use protoc 25.3 and the repository's protoc-gen-go target. Generation reads
the canonical schema in this Router checkout:

```bash
make generate-proto-latencypredictor

make verify-latencypredictor-proto
```

Verification generates temporary files and compares them with the checked-in
code, digest and fixtures. It does not rewrite them. Predictor pins a Router
schema commit and its schema/fixture hashes when generating Python bindings.

## Tests

```bash
go test -race ./pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol
```

The tests cover flat single and bulk rows, presence, exact 64-bit integers,
training metadata, model artifacts and decoding through the existing Go types.
The plugin tests exercise capture timing, dependency declarations and training
projections. End-to-end routing performance requires the issue's GPU scenarios.
