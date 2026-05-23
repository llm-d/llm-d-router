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

#### Standalone with Envoy Proxy (Default Standalone)
Deploys EPP with an Envoy sidecar proxy that intercepts incoming HTTP/gRPC traffic and routes it using EPP:

```bash
helm install my-standalone-router ./config/charts/llm-d-router-standalone \
  --set router.modelServers.matchLabels.app=my-vllm-service
```

#### Standalone with Agentgateway Proxy (Service-Backed)
Deploys EPP with an Agentgateway proxy. This mode requires disabling the `InferencePool` resource creation (`create=false`) and routes traffic to an existing Kubernetes Service:

```bash
helm install my-standalone-router ./config/charts/llm-d-router-standalone \
  --set router.inferencePool.create=false \
  --set router.proxy.proxyType=agentgateway \
  --set router.proxy.agentgateway.service.name=my-model-service \
  --set router.proxy.agentgateway.service.ports="8000"
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

## Configuration & Customization

Since both charts use `routerlib` under the hood, all configurations and customizations are shared under the `router` values block. EPP and the LLM-D Router can be customized by grouping configuration blocks in your `values.yaml` file. Below is the complete documentation and reference for each component.

---

### 1. EPP Core Configuration

Core settings for the Endpoint Picker Proxy (EPP) container and pod, including scaling, images, command-line flags, custom environment variables, resources, and plugins config.

#### EPP Core Configuration Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.epp.parser` | Request parser type for EPP. Options: `[openai-parser, vllmgrpc-parser, passthrough-parser]`. Empty for auto-selection. | `""` |
| `router.epp.replicas` | Number of EPP replicas. Set > 1 to enable active-passive HA. | `1` |
| `router.epp.extProcPort` | Port EPP uses for external processing gRPC communication. | `9002` |
| `router.epp.image.registry` | EPP container image registry. | `ghcr.io/llm-d` |
| `router.epp.image.repository` | EPP container image repository. | `llm-d-router-endpoint-picker-dev` |
| `router.epp.image.tag` | EPP container image tag. | `main` |
| `router.epp.image.pullPolicy` | EPP container image pull policy. | `Always` |
| `router.epp.env` | Extra environment variables for EPP container. | `[]` |
| `router.epp.extraContainerPorts` | Extra ports to expose on the EPP container. | `[]` |
| `router.extraServicePorts` | Extra ports to expose on the EPP Service. | `[]` |
| `router.epp.flags` | Map of command-line flags passed directly to the EPP binary. | `{}` |
| `router.epp.affinity` | Affinity rules for EPP pods. | `{}` |
| `router.epp.tolerations` | Tolerations for EPP pods. | `[]` |
| `router.epp.resources` | EPP container resource requests and limits. | `requests.cpu: "4"`, `requests.memory: 8Gi`, `limits.memory: 16Gi` |
| `router.epp.pluginsConfigFile` | EPP plugins configuration file name. | `default-plugins.yaml` |
| `router.epp.pluginsCustomConfig` | Inline custom YAML configuration for EPP plugins. | `{}` |
| `router.epp.volumes` | Extra volumes for EPP pod. | `[]` |
| `router.epp.volumeMounts` | Extra volume mounts for EPP container. | `[]` |

#### Complete Custom EPP Core Example

To fully customize the EPP core container and pod (e.g., HA scaling, custom image, debugging flags, custom environment variables, custom plugin scheduling, and resource allocations), define the `router` block in your `values.yaml` as follows:

```yaml
router:
  epp:
    # Run EPP in active-passive HA mode
    replicas: 3
    image:
      registry: my-registry.io
      repository: my-epp-dev
      tag: v1.2.3
      pullPolicy: IfNotPresent
    extProcPort: 9002
    parser: vllmgrpc-parser
    flags:
      # Enable debug logging (verbosity 3)
      v: 3
      tracing: true
    env:
      - name: FEATURE_FLAG_ENABLED
        value: "true"
      - name: POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
    resources:
      requests:
        cpu: "8"
        memory: 16Gi
      limits:
        memory: 32Gi
    pluginsConfigFile: "custom-plugins.yaml"
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
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
          - matchExpressions:
            - key: topology.kubernetes.io/zone
              operator: In
              values:
              - us-central1-a
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    volumeMounts:
      - mountPath: /models
        name: model-volume
  volumes:
    - name: model-volume
      emptyDir: {}
```

---

### 2. Model Server Configuration

Configuration for the backend model servers that EPP routes traffic to. These settings are used by EPP to parse requests and route traffic correctly.

#### Model Server Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.modelServers.matchLabels` | **REQUIRED** (when `create=true`). Label selector to match model server pods. | `{}` |
| `router.modelServers.type` | Type of model servers in the pool. Options: `[vllm, sglang, triton-tensorrt-llm, trtllm-serve, triton]`. | `vllm` |
| `router.modelServers.protocol` | Protocol used by model servers. Options: `[http, grpc]`. | `http` |
| `router.modelServers.targetPorts` | Port(s) EPP routes traffic to on the model servers. | `[{number: 8000}]` |
| `router.modelServers.targetPortNumber` | Legacy fallback port number for GKE health check policies. | `8000` |

#### Complete Model Server Example

```yaml
router:
  modelServers:
    # Label selector to match your model server pods
    matchLabels:
      app: my-sglang-deployment
    type: sglang
    protocol: grpc
    targetPorts:
      - number: 50051
```

---

### 3. InferencePool Configuration

Settings for managing the `InferencePool` Kubernetes resource which logically groups the backend model servers.

#### InferencePool Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.inferencePool.create` | Whether to create the `InferencePool` resource. Set to `false` in standalone mode for Service-backed routing. | `true` |
| `router.inferencePool.apiVersion` | The API version of the `InferencePool` resource. | `inference.networking.k8s.io/v1` |
| `router.inferencePool.group` | The API group of the `InferencePool` resource. | `inference.networking.k8s.io` |
| `router.inferencePool.failureMode` | EPP failure mode when external processing fails (configured on the pool). Options: `[FailOpen, FailClosed]`. | `FailOpen` |

#### Complete InferencePool Example

```yaml
router:
  inferencePool:
    # Enable or disable InferencePool resource creation (false in standalone service-backed mode)
    create: true
    apiVersion: inference.networking.k8s.io/v1
    group: inference.networking.k8s.io
    # If EPP fails to process the request, route anyway (FailOpen) or fail the request (FailClosed)
    failureMode: FailClosed
```

---

### 4. Monitoring & Tracing Configuration

Configures metrics scraping via Prometheus (compatible with Google Managed Prometheus or standard Prometheus Operator) and OpenTelemetry distributed tracing endpoints.

#### Monitoring & Tracing Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.monitoring.provider.name` | Metrics provider. Options: `[gmp, prometheusoperator]`. | `prometheusoperator` |
| `router.monitoring.provider.gmp.autopilot` | Set to `true` if deploying GMP on GKE Autopilot. | `false` |
| `router.tracing.enabled` | Enable OpenTelemetry tracing for EPP. | `false` |
| `router.tracing.otelExporterEndpoint` | OTLP gRPC collector endpoint. | `http://localhost:4317` |
| `router.tracing.sampling.sampler` | Trace sampler type. | `parentbased_traceidratio` |
| `router.tracing.sampling.samplerArg` | Sampler argument (e.g., sampling ratio `"0.1"`). | `"0.1"` |

#### Complete Monitoring & Tracing Example

```yaml
router:
  monitoring:
    interval: "10s"
    provider:
      name: "gmp" # Use Google Managed Prometheus (GMP)
      gmp:
        autopilot: true # Enable if running on GKE Autopilot
    prometheus:
      enabled: true
      auth:
        enabled: true # Requires token authorization to scrape metrics on port 9090
  tracing:
    enabled: true
    otelExporterEndpoint: "http://otel-collector.monitoring.svc.cluster.local:4317"
    sampling:
      sampler: "parentbased_traceidratio"
      samplerArg: "0.1" # Sample 10% of requests
```

---

### 5. Gateway-Specific Configuration (`llm-d-router-gateway` only)

Configures routing policies and optional Gateway API integration features exclusive to the Gateway implementation chart.

#### Gateway-Specific Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `inferenceObjectives` | List of names and priorities to create optional `InferenceObjective` resources. | `[]` |
| `provider.name` | Name of Gateway implementation. Options: `[none, gke, istio]`. | `none` |
| `provider.istio.destinationRule.host` | Custom host value for Istio DestinationRule. | `""` |
| `provider.istio.destinationRule.trafficPolicy.connectionPool` | Connection pool settings for Istio DestinationRule. | `{}` |
| `experimentalHttpRoute.enabled` | Deploy an `HTTPRoute` resource as part of the gateway chart. | `false` |
| `experimentalHttpRoute.inferenceGatewayName` | Target Gateway name for the `HTTPRoute`. | `inference-gateway` |
| `experimentalHttpRoute.inferenceGatewayNamespace` | Target Gateway namespace for the `HTTPRoute`. | `""` |
| `experimentalHttpRoute.requestTimeout` | Request timeout for the `HTTPRoute` (Istio/non-GKE only). | `300s` |

#### Complete Gateway Example

```yaml
inferenceObjectives:
  - name: high-priority
    priority: 100
  - name: background-batch
    priority: 10

provider:
  name: gke # Use GKE gateway implementation
  
experimentalHttpRoute:
  enabled: true
  inferenceGatewayName: "my-company-gateway"
  inferenceGatewayNamespace: "gateway-infra"
  requestTimeout: "120s"
```

---

### 6. Sidecar Proxy Configuration (`router.proxy.*`)

Used primarily in **Standalone Mode** to spin up a proxy container (Envoy sidecar or Agentgateway sidecar) alongside EPP to intercept and route incoming traffic.

#### Proxy Sidecar Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.proxy.enabled` | Enable a sidecar proxy container in the EPP deployment. | `false` |
| `router.proxy.proxyType` | **Standalone only**. Type of sidecar proxy. Options: `[envoy, agentgateway]`. | `envoy` |
| `router.proxy.name` | Name of the sidecar container. | `""` |
| `router.proxy.image` | Sidecar container image. | `""` |
| `router.proxy.imagePullPolicy` | Sidecar container image pull policy. | `IfNotPresent` |
| `router.proxy.command` | Sidecar container command. | `""` |
| `router.proxy.args` | Sidecar container arguments. | `[]` |
| `router.proxy.env` | Sidecar container environment variables. | `[]` |
| `router.proxy.ports` | Sidecar container ports. | `[]` |
| `router.proxy.livenessProbe` | Sidecar container liveness probe. | `{}` |
| `router.proxy.readinessProbe` | Sidecar container readiness probe. | `{}` |
| `router.proxy.resources` | Sidecar container resource requests and limits. | `{}` |
| `router.proxy.volumeMounts` | Sidecar container volume mounts. | `[]` |
| `router.proxy.volumes` | Sidecar container volumes. | `[]` |
| `router.proxy.configMapData` | Key-value pairs to include in a ConfigMap created for the sidecar. | `{}` |
| `router.proxy.agentgateway.service.create` | **Agentgateway only**. Create a dedicated model Service for the Agentgateway proxy. | `true` |
| `router.proxy.agentgateway.service.name` | **Agentgateway only**. Name of the model Service to route to. | `""` |
| `router.proxy.agentgateway.service.namespace` | **Agentgateway only**. Namespace of the model Service. Defaults to release namespace. | `""` |
| `router.proxy.agentgateway.service.ports` | **Agentgateway only**. Port list for the model Service (must match `modelServers.targetPorts`). | `[]` |

#### Complete Proxy Sidecar Example (Agentgateway Service-Backed)

To deploy EPP in standalone mode with an Agentgateway sidecar routing traffic directly to an existing model Service `my-model-service` (bypassing `InferencePool` creation):

```yaml
router:
  inferencePool:
    create: false # Disable InferencePool creation

  proxy:
    enabled: true
    proxyType: agentgateway
    resources:
      requests:
        cpu: "2"
        memory: 4Gi
      limits:
        memory: 8Gi
    agentgateway:
      service:
        create: true # Create a Service to route client traffic to EPP
        name: "my-model-service"
        ports:
          - 8000 # Intercept traffic on port 8000
```

---

### 7. Sidecar Tokenizer Configuration (`router.tokenizer.*`)

Runs a tokenizer sidecar container to expose a Unix Domain Socket (UDS) to EPP, allowing EPP to tokenize incoming requests and enable precise, token-count-aware routing policies (e.g., precise prefix-cache matching).

#### Tokenizer Sidecar Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.tokenizer.enabled` | Enable a tokenizer UDS sidecar container in the EPP deployment. | `false` |
| `router.tokenizer.image.registry` | Tokenizer container image registry. | `ghcr.io/llm-d` |
| `router.tokenizer.image.repository` | Tokenizer container image repository. | `llm-d-uds-tokenizer` |
| `router.tokenizer.image.tag` | Tokenizer container image tag. | `vllm-v0.19.1` |
| `router.tokenizer.image.pullPolicy` | Tokenizer container image pull policy. | `IfNotPresent` |
| `router.tokenizer.env` | Extra environment variables for the tokenizer container (e.g., HuggingFace token). `TOKENIZERS_DIR` and `HF_HOME` are templated automatically. | `[HF_TOKEN]` |
| `router.tokenizer.resources` | Tokenizer container resource requests and limits. | `requests.cpu: "8"`, `requests.memory: 8Gi` |
| `router.tokenizer.volumeMounts` | Extra volume mounts for the tokenizer container (e.g., for model files). | `[]` |

#### Complete Tokenizer Sidecar Example

```yaml
router:
  tokenizer:
    enabled: true
    image:
      registry: ghcr.io/llm-d
      repository: llm-d-uds-tokenizer
      tag: vllm-v0.19.1
    env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: my-hf-token-secret
            key: token
    resources:
      requests:
        cpu: "8"
        memory: 8Gi
      limits:
        memory: 16Gi
    volumeMounts:
      # If pre-loading models from an external volume:
      - mountPath: /models
        name: model-cache-volume
  volumes:
    - name: model-cache-volume
      persistentVolumeClaim:
        claimName: pvc-model-cache
```

---

### 8. Sidecar Latency Predictor Configuration (`router.latencyPredictor.*`)

Enables latency predictor containers inside the EPP deployment to feed metrics to a latency scorer plugin, allowing EPP to route traffic based on real-time predicted latencies.

#### Latency Predictor Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.latencyPredictor.enabled` | Enable latency-based routing (requires extra Borg/training setup). | `false` |
| `router.latencyPredictor.trainingServer.image` | Latency training server image configuration. | |
| `router.latencyPredictor.predictionServers.image` | Latency prediction server image configuration. | |
| `router.latencyPredictor.eppEnv` | EPP tuning variables for Latency Predictor. | |

#### Complete Latency Predictor Example

```yaml
router:
  latencyPredictor:
    enabled: true
    trainingServer:
      image:
        registry: my-company-docker.pkg.dev/k8s-staging-images
        repository: latency-training-server
        tag: latest
    predictionServers:
      image:
        registry: my-company-docker.pkg.dev/k8s-staging-images
        repository: latency-prediction-server
        tag: latest
    eppEnv:
      LATENCY_MAX_SAMPLE_SIZE: "20000"
      LATENCY_MAX_CONCURRENT_DISPATCHES: "48"
      LATENCY_COALESCE_WINDOW_MS: "2"
```
