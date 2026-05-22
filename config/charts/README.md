# LLM-D Router Helm Charts

This directory contains Helm charts for deploying the **LLM-D Router** components: the **Endpoint Picker (EPP)** and the **InferencePool** resource.

## Charts Overview

We provide two charts depending on your deployment mode, both leveraging a shared core library chart (`routerlib`):

*   **`llm-d-router-gateway`**: Used for **Gateway Mode**. It deploys EPP and creates an `InferencePool` resource. It integrates with the Kubernetes Gateway API (typically via `HTTPRoute` pointing to the `InferencePool`) for multi-pool, dynamic routing.
*   **`llm-d-router-standalone`**: Used for **Standalone Mode** (Service-backed or direct pod routing). EPP can be deployed without creating an `InferencePool` resource (by setting `router.inferencePool.create=false`). It supports running EPP with a sidecar proxy (Envoy or Agentgateway) to intercept and route traffic.
*   **`routerlib` (Library Chart)**: Encapsulates the core templates and default configurations for EPP and `InferencePool`. It is not deployable on its own.

---

## Prerequisites

Before installing the charts, ensure that the **Gateway API Inference Extension CRDs** are installed in your cluster. Refer to the [getting started guide](https://github.com/llm-d/llm-d-router/tree/main/deploy) for installation instructions.

---

## Installation & Usage

### 1. Standalone Mode (`llm-d-router-standalone`)

Standalone mode is useful when you want to run EPP as a local router/proxy for a specific model service, without integrating with a cluster-wide Gateway.

#### Standalone with Envoy Sidecar (Default Standalone)
Deploys EPP with an Envoy sidecar proxy that intercepts incoming HTTP/gRPC traffic and routes it using EPP:

```bash
helm install my-standalone-router ./config/charts/llm-d-router-standalone \
  --set router.modelServers.matchLabels.app=my-vllm-service
```

#### Standalone with Agentgateway Sidecar (Service-Backed)
Deploys EPP with an Agentgateway sidecar. This mode requires disabling the `InferencePool` resource creation (`create=false`) and routes traffic to an existing Kubernetes Service:

```bash
helm install my-standalone-router ./config/charts/llm-d-router-standalone \
  --set router.inferencePool.create=false \
  --set router.sidecar.proxyType=agentgateway \
  --set router.sidecar.agentgateway.service.name=my-model-service \
  --set router.sidecar.agentgateway.service.ports="8000"
```
---

### 2. Gateway Mode (`llm-d-router-gateway`)

To deploy an InferencePool named `vllm-qwen3-32b` that selects model servers with the label `app=vllm-qwen3-32b` and routes to port `8000`:

```bash
helm install vllm-qwen3-32b ./config/charts/llm-d-router-gateway \
  --set router.modelServers.matchLabels.app=vllm-qwen3-32b
```

#### Install with a Specific Provider (GKE or Istio)
To deploy provider-specific resources (like health check policies or destination rules), specify the provider name:

```bash
helm install vllm-qwen3-32b ./config/charts/llm-d-router-gateway \
  --set router.modelServers.matchLabels.app=vllm-qwen3-32b \
  --set provider.name=gke # Options: [none, gke, istio]
```
---

## Common Customizations

Since both charts use `routerlib` under the hood, most EPP customizations are shared and configured under the `router` values block.

### Custom Command-Line Flags for EPP
Pass additional flags to the EPP container using `router.flags`:

```bash
helm install vllm-pool ./config/charts/llm-d-router-gateway \
  --set router.modelServers.matchLabels.app=vllm-pool \
  --set router.flags.v=3 # Enable debug logging (verbosity 3)
```

### Custom Environment Variables
Define custom environment variables for EPP in your `values.yaml`:

```yaml
router:
  env:
    - name: FEATURE_FLAG_ENABLED
      value: "true"
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
```

### Custom EPP Plugins Configuration
EPP routing behavior is controlled by plugins. You can pass custom inline plugin configurations:

```yaml
router:
  pluginsCustomConfig:
    custom-plugins.yaml: |
      apiVersion: inference.networking.x-k8s.io/v1alpha1
      kind: EndpointPickerConfig
      plugins:
      - type: queue-scorer
      - type: custom-scorer
        parameters:
          threshold: 64
      schedulingProfiles:
      - name: default
        plugins:
        - pluginRef: queue-scorer
        - pluginRef: custom-scorer
```

### High Availability (HA)
To deploy EPP in an active-passive HA configuration, set `replicas` to a value greater than 1. Only one "leader" replica will process traffic, with others acting as warm standbys:

```bash
helm install vllm-pool ./config/charts/llm-d-router-gateway \
  --set router.modelServers.matchLabels.app=vllm-pool \
  --set router.replicas=3
```

### Monitoring
EPP exposes Prometheus metrics on port `9090`. You can configure metrics collection:

```yaml
router:
  monitoring:
    interval: "10s"
    provider:
      name: "gmp" # Options: [gmp (Google Managed Prometheus), prometheusoperator]
    prometheus:
      enabled: true
      auth:
        enabled: true # Set to false for unauthenticated /metrics access
```

### Tracing
EPP supports OpenTelemetry tracing:

```yaml
router:
  tracing:
    enabled: true
    otelExporterEndpoint: "http://otel-collector.monitoring.svc:4317"
    sampling:
      sampler: "parentbased_traceidratio"
      samplerArg: "0.1" # Sample 10% of traces
```

---

## Configuration Reference

The following table lists all configurable parameters for the LLM-D Router charts.

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| **InferencePool Config (`router.inferencePool.*`)** | | |
| `router.inferencePool.create` | Whether to create the `InferencePool` resource. Set to `false` in standalone mode for Service-backed routing. | `true` |
| `router.inferencePool.apiVersion` | The API version of the `InferencePool` resource. | `inference.networking.k8s.io/v1` |
| `router.inferencePool.group` | The API group of the `InferencePool` resource. | `inference.networking.k8s.io` |
| `router.inferencePool.failureMode` | The failure mode for the pool (e.g., `FailOpen`, `FailClosed`). | `FailOpen` |
| **Model Server Config (`router.modelServers.*`)** | | |
| `router.modelServers.matchLabels` | **REQUIRED** (when `create=true`). Label selector to match model server pods. | `{}` |
| `router.modelServers.type` | Type of model servers in the pool. Options: `[vllm, sglang, triton-tensorrt-llm, trtllm-serve, triton]`. | `vllm` |
| `router.modelServers.protocol` | Protocol used by model servers. Options: `[http, grpc]`. | `http` |
| `router.modelServers.targetPorts` | Port(s) EPP routes traffic to on the model servers. | `[{number: 8000}]` |
| `router.modelServers.targetPortNumber` | Legacy fallback port number for GKE health check policies. | `8000` |
| **EPP Core Config (`router.*`)** | | |
| `router.parser` | Request parser type for EPP. Options: `[openai-parser, vllmgrpc-parser, passthrough-parser]`. Empty for auto-selection. | `""` |
| `router.replicas` | Number of EPP replicas. Set > 1 to enable active-passive HA. | `1` |
| `router.extProcPort` | Port EPP uses for external processing gRPC communication. | `9002` |
| `router.failureMode` | EPP failure mode when external processing fails. | `FailOpen` |
| `router.image.registry` | EPP container image registry. | `ghcr.io/llm-d` |
| `router.image.repository` | EPP container image repository. | `llm-d-router-endpoint-picker-dev` |
| `router.image.tag` | EPP container image tag. | `main` |
| `router.image.pullPolicy` | EPP container image pull policy. | `Always` |
| `router.env` | Extra environment variables for EPP container. | `[]` |
| `router.extraContainerPorts` | Extra ports to expose on the EPP container. | `[]` |
| `router.extraServicePorts` | Extra ports to expose on the EPP Service. | `[]` |
| `router.flags` | Map of command-line flags passed directly to the EPP binary. | `{}` |
| `router.affinity` | Affinity rules for EPP pods. | `{}` |
| `router.tolerations` | Tolerations for EPP pods. | `[]` |
| `router.resources` | EPP container resource requests and limits. | `requests.cpu: "4"`, `requests.memory: 8Gi`, `limits.memory: 16Gi` |
| `router.pluginsConfigFile` | EPP plugins configuration file name. | `default-plugins.yaml` |
| `router.pluginsCustomConfig` | Inline custom YAML configuration for EPP plugins. | `{}` |
| `router.volumes` | Extra volumes for EPP pod. | `[]` |
| `router.volumeMounts` | Extra volume mounts for EPP container. | `[]` |
| **EPP Sidecar Config (`router.sidecar.*`)** | | |
| `router.sidecar.enabled` | Enable a sidecar proxy container in the EPP deployment. | `false` |
| `router.sidecar.proxyType` | **Standalone only**. Type of sidecar proxy. Options: `[envoy, agentgateway]`. | `envoy` |
| `router.sidecar.name` | Name of the sidecar container. | `""` |
| `router.sidecar.image` | Sidecar container image. | `""` |
| `router.sidecar.imagePullPolicy` | Sidecar container image pull policy. | `IfNotPresent` |
| `router.sidecar.command` | Sidecar container command. | `""` |
| `router.sidecar.args` | Sidecar container arguments. | `[]` |
| `router.sidecar.env` | Sidecar container environment variables. | `[]` |
| `router.sidecar.ports` | Sidecar container ports. | `[]` |
| `router.sidecar.livenessProbe` | Sidecar container liveness probe. | `{}` |
| `router.sidecar.readinessProbe` | Sidecar container readiness probe. | `{}` |
| `router.sidecar.resources` | Sidecar container resource requests and limits. | `{}` |
| `router.sidecar.volumeMounts` | Sidecar container volume mounts. | `[]` |
| `router.sidecar.volumes` | Sidecar container volumes. | `[]` |
| `router.sidecar.configMapData` | Key-value pairs to include in a ConfigMap created for the sidecar. | `{}` |
| **Standalone Sidecar Overrides (`router.sidecar.agentgateway.*`)** | | |
| `router.sidecar.agentgateway.service.create` | Create a dedicated model Service for the Agentgateway sidecar. | `true` |
| `router.sidecar.agentgateway.service.name` | Name of the model Service to route to. | `""` |
| `router.sidecar.agentgateway.service.namespace` | Namespace of the model Service. Defaults to release namespace. | `""` |
| `router.sidecar.agentgateway.service.ports` | Port list for the model Service (must match `modelServers.targetPorts`). | `[]` |
| **Monitoring & Tracing Config** | | |
| `router.monitoring.provider.name` | Metrics provider. Options: `[gmp, prometheusoperator]`. | `prometheusoperator` |
| `router.monitoring.provider.gmp.autopilot` | Set to `true` if deploying GMP on GKE Autopilot. | `false` |
| `router.tracing.enabled` | Enable OpenTelemetry tracing for EPP. | `false` |
| `router.tracing.otelExporterEndpoint` | OTLP gRPC collector endpoint. | `http://localhost:4317` |
| `router.tracing.sampling.sampler` | Trace sampler type. | `parentbased_traceidratio` |
| `router.tracing.sampling.samplerArg` | Sampler argument (e.g., sampling ratio `"0.1"`). | `"0.1"` |
| **EPP Latency Predictor Config (`router.latencyPredictor.*`)** | | |
| `router.latencyPredictor.enabled` | Enable latency-based routing (requires extra Borg/training setup). | `false` |
| `router.latencyPredictor.trainingServer.image` | Latency training server image configuration. | |
| `router.latencyPredictor.predictionServers.image` | Latency prediction server image configuration. | |
| `router.latencyPredictor.eppEnv` | EPP tuning variables for Latency Predictor. | |
| **Gateway-Specific Config (`llm-d-router-gateway` only)** | | |
| `inferenceObjectives` | List of names and priorities to create optional `InferenceObjective` resources. | `[]` |
| `provider.name` | Name of Gateway implementation. Options: `[none, gke, istio]`. | `none` |
| `provider.istio.destinationRule.host` | Custom host value for Istio DestinationRule. | `""` |
| `provider.istio.destinationRule.trafficPolicy.connectionPool` | Connection pool settings for Istio DestinationRule. | `{}` |
| `experimentalHttpRoute.enabled` | Deploy an `HTTPRoute` resource as part of the gateway chart. | `false` |
| `experimentalHttpRoute.inferenceGatewayName` | Target Gateway name for the `HTTPRoute`. | `inference-gateway` |
| `experimentalHttpRoute.inferenceGatewayNamespace` | Target Gateway namespace for the `HTTPRoute`. | `""` |
| `experimentalHttpRoute.requestTimeout` | Request timeout for the `HTTPRoute` (Istio/non-GKE only). | `300s` |
