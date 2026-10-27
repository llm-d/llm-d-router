# Collecting EPP profiles

Enable the [`pprof values`](../deployments/tracing/pprof.values.yaml) for the
diagnostic run described in the [study instructions](../README.md). Run the
collection script while GuideLLM is generating load:

```sh
NAMESPACE=epp-tracing-overhead OUTPUT_DIR=/tmp/epp-pprof ./collect.sh
```

The script writes a 30-second CPU profile and a heap profile to one directory
per EPP pod under `OUTPUT_DIR`.

The profiles can be inspected with the pprof version pinned by llm-d-router:

```sh
PROFILE=/tmp/epp-pprof/EPP_POD_NAME/cpu.pprof
go run github.com/google/pprof@v0.0.0-20260402051712-545e8a4df936 -top "$PROFILE"
```

To select samples with at least one router tracing or OpenTelemetry frame:

```sh
go tool pprof -top \
  -focus='go.opentelemetry.io/otel|pkg/common/observability/tracing' \
  "$PROFILE"
```

The union counts each matching sample once. The individual function values are
cumulative call-stack attributions and must not be added. These measurements do
not assign all runtime and garbage-collection cost caused by trace allocations
to OpenTelemetry.
