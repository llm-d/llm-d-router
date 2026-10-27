# Deployments

The study used the llm-d optimized-baseline inference-scheduling topology with
an Istio Gateway and one EPP. The model pool contained eight replicas. We use a
separate namespace or remove the preceding deployment before switching between
the simulator and live targets.

## Prerequisites

The study installs the Gateway API, Gateway API Inference Extension, Istio, and an Istio
Gateway named `llm-d-inference-gateway`. The llm-d optimized-baseline guide
documents those platform prerequisites.

Clone and pin llm-d:

```sh
git clone https://github.com/llm-d/llm-d.git
cd llm-d
git checkout v0.9.0
export LLMD_ROOT=$(git rev-parse --show-toplevel)
export NAMESPACE=epp-tracing-overhead
export GUIDE_NAME=optimized-baseline
export PROVIDER_NAME=istio
export ROUTER_CHART_VERSION=v0.10.0
source "$LLMD_ROOT/guides/env.sh"
kubectl create namespace "$NAMESPACE"
```

Set `STUDY_DIR` to this directory and deploy the trace receiver:

```sh
export STUDY_DIR=/path/to/epp_tracing_overhead
kubectl apply -n "$NAMESPACE" -f "$STUDY_DIR/deployments/telemetry.yaml"
kubectl rollout status -n "$NAMESPACE" deployment/epp-tracing-otel-collector
kubectl rollout status -n "$NAMESPACE" deployment/epp-tracing-jaeger
```

## Model pool

For the isolated EPP experiment:

```sh
kubectl apply -n "$NAMESPACE" \
  -f "$STUDY_DIR/deployments/modelserver-simulator.yaml"
kubectl rollout status -n "$NAMESPACE" deployment/epp-tracing-modelserver
```

For live inference, create a `models-storage` PVC containing
`models/Qwen-Qwen3-0.6B` and a `huggingface-token` Secret with key `HF_TOKEN`,
then run:

```sh
kubectl apply -n "$NAMESPACE" \
  -f "$STUDY_DIR/deployments/modelserver-live.yaml"
kubectl rollout status -n "$NAMESPACE" deployment/epp-tracing-modelserver \
  --timeout=30m
```

## EPP variants

The router command layers the llm-d v0.9.0 recipe values with this study's
selector, image pin, and tracing values:

```sh
helm upgrade --install epp-tracing \
  "$ROUTER_GATEWAY_CHART" \
  -f "$LLMD_ROOT/guides/recipes/router/base.values.yaml" \
  -f "$LLMD_ROOT/guides/optimized-baseline/router/optimized-baseline.values.yaml" \
  -f "$STUDY_DIR/deployments/router-base.values.yaml" \
  --set provider.name=istio \
  --set httpRoute.create=true \
  --set httpRoute.inferenceGatewayName=llm-d-inference-gateway \
  -n "$NAMESPACE" \
  --version "$ROUTER_CHART_VERSION" \
  --wait
```

That command is the tracing-off case. For an enabled ratio, we append these values
arguments before `--set`:

```sh
-f "$STUDY_DIR/deployments/tracing/tracing-standard.values.yaml" \
-f "$STUDY_DIR/deployments/tracing/ratio-10.values.yaml"
```

The study replaces `ratio-10.values.yaml` with the required ratio, adds
`pprof.values.yaml` only for the diagnostic 100% run and uses `helm uninstall` and
installs again between measured variants so no process state carries across
runs.

We resolve the Gateway address for GuideLLM as:

```sh
export TARGET="http://$(kubectl get gateway llm-d-inference-gateway \
  -n "$NAMESPACE" -o jsonpath='{.status.addresses[0].value}')"
```

The captured EPP image identity was
`ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.10.0`, with digest
`sha256:2e516fa1310da7be59b82beb1445362139597d6d553ef04d546716abe3aaaa70`.

