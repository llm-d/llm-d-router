# llm-d Router Helm Charts

This directory contains Helm charts for deploying the **llm-d Router** components: the **Endpoint Picker (EPP)** and the **InferencePool** resource.

## Charts Overview

We provide two charts depending on your deployment mode, both leveraging a shared core library chart (`routerlib`):

*   **`llm-d-router-gateway`**: Used for **Gateway Mode**. It deploys EPP and creates an `InferencePool` resource. It integrates with the Kubernetes Gateway API (typically via `HTTPRoute` pointing to the `InferencePool`) for multi-pool, dynamic routing.
*   **`llm-d-router-standalone`**: Used for **Standalone Mode** (Service-backed or direct pod routing). EPP can be deployed without creating an `InferencePool` resource (by setting `router.inferencePool.create=false`). It supports running a proxy (Envoy or Agentgateway) to intercept and route traffic, either as a sidecar in the EPP pod or as a separate horizontally scalable service.
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

#### Standalone with a Separate Proxy Service
Deploys the proxy as its own horizontally scalable Deployment and Service instead of as a sidecar in the EPP pod. The proxy reaches EPP over the EPP Service and fails open if EPP is unreachable, so the data path stays available across EPP active/passive failover:

```bash
helm install my-standalone-router ./config/charts/llm-d-router-standalone \
  --set router.modelServers.matchLabels.app=my-vllm-service \
  --set router.inferencePool.create=false \
  --set router.proxy.mode=service \
  --set router.proxy.replicas=3
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

## Migrating from gateway-api-inference-extension

If you previously used the `gateway-api-inference-extension` Helm charts (either `inferencepool` or `standalone`), please note that the `llm-d-router` charts have been restructured.

To prevent accidental misconfiguration, the new charts include strict validation checks that will **fail the installation** if they detect legacy top-level keys in your values file or command-line flags.

You must migrate your values according to the following mapping:

### Value Mapping Guide

| Legacy Key (Top-level) | New Key (Nested/Renamed) | Description |
| :--- | :--- | :--- |
| `inferenceExtension.*` | `router.epp.*` <br> `router.monitoring.*` <br> `router.tracing.*` <br> `router.latencyPredictor.*` | EPP configuration has been restructured and nested under the `router` block. |
| `inferencePool.*` | `router.modelServers.*` <br> `router.inferencePool.*` | Model server configuration and pool control flags have moved under `router`. |
| `inferenceObjectives` | `router.inferenceObjectives` | Objectives list has moved under `router`. |
| `experimentalHttpRoute` | `httpRoute` | Gateway HTTPRoute configuration has been renamed (Gateway chart only). |

For a detailed list of all new configuration options, refer to the [Configuration & Customization](#configuration--customization) section below.

---

## Configuration & Customization

Since both charts use `routerlib` under the hood, all configurations and customizations are shared under the `router` values block. EPP and the llm-d Router can be customized by grouping configuration blocks in your `values.yaml` file. Below is the complete documentation and reference for each component.

---

### 1. EPP Core Configuration

Core settings for the Endpoint Picker Proxy (EPP) container and pod, including scaling, images, command-line flags, custom environment variables, resources, and custom plugins configuration (`pluginsCustomConfig`).

> [!NOTE]
> **High Availability (HA) Modes**:
> When EPP is scaled (`router.epp.replicas > 1`):
> *   **Active-Passive Mode (Default)**: The chart automatically enables the `--ha-enable-leader-election` flag. Only one leader replica active-routes traffic, coordinates lease status, and maintains absolute routing state, while other replicas act as warm standbys.
> *   **Active-Active Mode**: You can explicitly disable leader-election by passing `ha-enable-leader-election: false` under `router.epp.flags`. In this mode, all replicas process traffic concurrently.
>     *   *Warning*: In active-active mode, you **must only use active-active compatible plugins**—specifically plugins that pull real-time metrics/state dynamically from the backend model servers (such as the precise prefix cache, queue, and KV-cache utilization scorers). Avoid plugins that rely on local in-memory routing state, as this state is not synchronized across replicas.

#### EPP Core Configuration Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.epp.parsers` | List of request parser types for EPP. Options: `[openai-parser, anthropic-parser, vllmgrpc-parser, vllmhttp-parser, passthrough-parser]`. Empty for auto-selection. | `[]` |
| `router.epp.replicas` | Number of EPP replicas. Set > 1 to enable multi-replica EPP. | `1` |
| `router.epp.extProcPort` | Port EPP uses for external processing gRPC communication. | `9002` |
| `router.epp.image.registry` | EPP container image registry. | `ghcr.io/llm-d` |
| `router.epp.image.repository` | EPP container image repository. | `llm-d-router-endpoint-picker-dev` |
| `router.epp.image.tag` | EPP container image tag. | `main` |
| `router.epp.image.pullPolicy` | EPP container image pull policy. | `Always` |
| `router.epp.env` | Extra environment variables for EPP container. | `[]` |
| `router.epp.extraContainerPorts` | Extra ports to expose on the EPP container. | `[]` |
| `router.extraServicePorts` | Extra ports to expose on the EPP Service. | `[]` |
| `router.clusterDomain` | Kubernetes cluster DNS domain used to build in-cluster Service FQDNs. | `cluster.local` |
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
    # Run EPP in multi-replica mode
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
      # Enable active-passive mode (one active leader, the other two are standby).
      ha-enable-leader-election: true
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

### 3. InferencePool & InferenceObjectives Configuration

Settings for managing the `InferencePool` Kubernetes resource and optional `InferenceObjective` priority routing rules.

> [!IMPORTANT]
> `inferenceObjectives` can only be configured when `inferencePool.create` is `true`. Creating `InferenceObjectives` without creating an `InferencePool` is not supported and will trigger a chart validation error.

#### InferencePool & InferenceObjectives Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.inferencePool.create` | Whether to create the `InferencePool` resource. Set to `false` in standalone mode for Service-backed routing. | `true` |
| `router.inferencePool.failureMode` | EPP failure mode when external processing fails (configured on the pool). Options: `[FailOpen, FailClosed]`. | `FailOpen` |
| `router.inferenceObjectives` | List of names and priorities to create optional `InferenceObjective` resources. | `[]` |

#### Complete InferencePool & InferenceObjectives Example

```yaml
router:
  inferencePool:
    # Enable or disable InferencePool resource creation (false in standalone service-backed mode)
    create: true
    # If EPP fails to process the request, route anyway (FailOpen) or fail the request (FailClosed)
    failureMode: FailClosed
  
  # Optional: Define InferenceObjective(s) for this InferencePool
  inferenceObjectives:
    - name: high-priority
      priority: 100
    - name: background-batch
      priority: 10
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

### 5. Sidecar Tokenizer Configuration (`router.tokenizer.*`)

Runs a tokenizer sidecar that EPP queries to tokenize incoming requests, enabling precise, token-count-aware routing policies (e.g., precise prefix-cache matching).

The sidecar runs vLLM's `vllm launch render <modelName>` and exposes `/v1/completions/render` and `/v1/chat/completions/render` over loopback HTTP. Wire EPP to it via `router.epp.pluginsCustomConfig` with `type: token-producer` and `vllm:`.

#### Tokenizer Sidecar Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.tokenizer.enabled` | Enable the vLLM `/render` tokenizer sidecar in the EPP deployment. | `false` |
| `router.tokenizer.modelName` | **REQUIRED** when enabled. Model name passed as the first positional arg to the sidecar's `vllm launch render` command. | `""` |
| `router.tokenizer.image.registry` | Tokenizer container image registry. | `docker.io` |
| `router.tokenizer.image.repository` | Tokenizer container image repository. | `vllm/vllm-openai-cpu` |
| `router.tokenizer.image.tag` | Tokenizer container image tag. | `v0.19.1` |
| `router.tokenizer.image.pullPolicy` | Tokenizer container image pull policy. | `IfNotPresent` |
| `router.tokenizer.port` | Container port the sidecar listens on. | `8000` |
| `router.tokenizer.command` | Override container command. Empty renders `["vllm", "launch", "render"]`. | `[]` |
| `router.tokenizer.args` | Override container args. Empty renders `["<modelName>", "--port=<port>"]`. | `[]` |
| `router.tokenizer.env` | Extra environment variables (e.g., HuggingFace token). | `[HF_TOKEN]` |
| `router.tokenizer.resources` | Tokenizer container resource requests and limits. | `requests.cpu: "4"`, `requests.memory: 8Gi` |
| `router.tokenizer.volumeMounts` | Extra volume mounts for the tokenizer container. | `[]` |
| `router.tokenizer.readinessProbe` | Readiness probe spec. | `httpGet /health on the render-http named port` |

#### Complete Tokenizer Render Sidecar Example

```yaml
router:
  epp:
    volumes:
      - name: model-cache-volume
        persistentVolumeClaim:
          claimName: pvc-model-cache
  tokenizer:
    enabled: true
    modelName: "Qwen/Qwen3-32B"
    image:
      registry: docker.io
      repository: vllm/vllm-openai-cpu
      tag: v0.19.1
    env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: my-hf-token-secret
            key: token
      - name: HF_HOME
          value: /models
    resources:
      requests:
        cpu: "4"
        memory: 8Gi
      limits:
        memory: 16Gi
    volumeMounts:
      # If pre-loading models from an external volume:
      - mountPath: /models
        name: model-cache-volume
```

#### UDS Tokenizer Backend (deprecated)

The `llm-d-uds-tokenizer` sidecar (gRPC over a Unix Domain Socket) is no longer
templated by this chart; the render backend above supersedes it. If you still
need it during migration, set it up in two steps.

**Step 1 — inject the sidecar into the EPP deployment.** The chart does not
render it, so patch it in yourself (e.g. via kustomize). The EPP container must
mount the shared socket volume so it can reach the tokenizer:

```yaml
spec:
  template:
    spec:
      containers:
        # Existing EPP container — add the shared socket mount.
        - name: epp
          volumeMounts:
            - name: tokenizer-uds
              mountPath: /tmp/tokenizer
        # The deprecated sidecar.
        - name: tokenizer-uds
          image: ghcr.io/llm-d/llm-d-uds-tokenizer:vllm-v0.19.1
          env:
            - name: TOKENIZERS_DIR
              value: /tokenizers
            - name: HF_HOME
              value: /tokenizers
            # Required for gated/private tokenizers on HuggingFace.
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: llm-d-hf-token
                  key: HF_TOKEN
          volumeMounts:
            - name: tokenizers
              mountPath: /tokenizers
            - name: tokenizer-uds
              mountPath: /tmp/tokenizer
      volumes:
        - name: tokenizers
          emptyDir: {}
        - name: tokenizer-uds
          emptyDir: {}
```

**Step 2 — point EPP at the socket.** Configure the tokenizer plugin via
`router.epp.pluginsCustomConfig` (the chart writes this into the EPP config
mounted at `/config`). Select the deprecated UDS backend with
`udsTokenizerConfig`:

```yaml
router:
  epp:
    pluginsCustomConfig:
      custom-plugins.yaml: |
        plugins:
          - type: token-producer
            parameters:
              modelName: "Qwen/Qwen3-32B"
              udsTokenizerConfig:
                socketFile: /tmp/tokenizer/tokenizer-uds.socket
```
---

### 6. Sidecar Latency Predictor Configuration (`router.latencyPredictor.*`)

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

---

## Deployment Mode Specific Configurations

Depending on your target deployment architecture (Gateway Mode vs Standalone Mode), utilize the following specific configurations.

---

### Gateway Mode Configuration

Configures routing policies, target Gateway interfaces, and priority objectives exclusive to the Gateway implementation chart (`llm-d-router-gateway`).

#### Gateway-Specific Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `provider.name` | Name of Gateway implementation. Options: `[none, gke, istio]`. | `none` |
| `provider.istio.destinationRule.host` | Custom host value for Istio DestinationRule. | `""` |
| `provider.istio.destinationRule.trafficPolicy.connectionPool` | Connection pool settings for Istio DestinationRule. | `{}` |
| `httpRoute.create` | Deploy an `HTTPRoute` resource as part of the gateway chart. | `false` |
| `httpRoute.inferenceGatewayName` | Target Gateway name for the `HTTPRoute`. | `inference-gateway` |
| `httpRoute.inferenceGatewayNamespace` | Target Gateway namespace for the `HTTPRoute`. | `""` |
| `httpRoute.requestTimeout` | Request timeout for the `HTTPRoute` (Istio/non-GKE only). | `300s` |

#### Complete Gateway Example

```yaml
provider:
  name: gke # Use GKE gateway implementation
  
httpRoute:
  create: true
  inferenceGatewayName: "my-company-gateway"
  inferenceGatewayNamespace: "gateway-infra"
  requestTimeout: "120s"
```

---

### Standalone Mode Configuration

Configures EPP to run with a proxy (Envoy proxy or Agentgateway proxy) that intercepts and routes client traffic directly to model servers (exclusive to `llm-d-router-standalone`). The proxy runs as a sidecar in the EPP pod by default, or as a separate horizontally scalable service when `router.proxy.mode=service` (Envoy only).

#### Proxy Sidecar Parameters

| **Parameter Name** | **Description** | **Default** |
| :--- | :--- | :--- |
| `router.proxy.enabled` | Enable the proxy (Envoy or Agentgateway) in front of EPP. | `false` |
| `router.proxy.proxyType` | Type of proxy. Options: `[envoy, agentgateway]`. | `envoy` |
| `router.proxy.mode` | Proxy deployment mode. `sidecar` runs the proxy in the EPP pod; `service` runs it as its own horizontally scalable Deployment and Service reaching EPP over the EPP Service (Envoy only). | `sidecar` |
| `router.proxy.replicas` | Replica count for the proxy Deployment when `mode=service`. | `2` |
| `router.proxy.failOpen` | Whether the proxy passes traffic through (fail-open) when EPP is unreachable. | `true` |
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
