[![Go Report Card](https://goreportcard.com/badge/github.com/llm-d/llm-d-inference-scheduler)](https://goreportcard.com/report/github.com/llm-d/llm-d-inference-scheduler)
[![Go Reference](https://pkg.go.dev/badge/github.com/llm-d/llm-d-inference-scheduler.svg)](https://pkg.go.dev/github.com/llm-d/llm-d-inference-scheduler)
[![License](https://img.shields.io/github/license/llm-d/llm-d-inference-scheduler)](/LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://llm-d.slack.com/archives/C08SBNRRSBD)

# llm-d Router

> [!IMPORTANT]
> **Terminology Change**: The *Inference Scheduler* has been renamed to **llm-d Router**; see [Terminology](README.md#terminology).

> [!IMPORTANT]
> **API & Code Consolidation**: Core Endpoint Picker (EPP) code and the `InferenceObjective` and `InferenceModelRewrite` APIs have been merged into this repository from [Gateway API Inference Extension (GIE)]. The GIE repository now exclusively hosts the `InferencePool` API—an extension of the [Kubernetes Gateway API]—and defines the Endpoint Picker Protocol.

The **llm-d Router** is the intelligent entry point for inference traffic, delivering LLM load and prefix-cache aware routing, request prioritization, and advanced flow control across diverse request formats to fulfill complex serving objectives. It supports a flexible deployment model: it can run in **Standalone Mode** (where a self-managed Envoy proxy runs alongside the EPP in the same pod) or integrate with L7 load balancers—including self-managed instances (e.g., Istio, AgentGateway) and cloud-managed services (e.g., Google Cloud's Application Load Balancer)—via the Kubernetes Gateway API. 

The router achieves its intelligence through an **Endpoint Picker (EPP)** that integrates with production-grade proxies (such as [Envoy]) via the [ext-proc] protocol, injecting real-time signals into the data plane to optimize request placement.

<p align="center">
  <img src="docs/images/llm-d-router.svg" width="800" alt="llm-d Router Architecture">
</p>

## Core Components and APIs

This repository hosts the following core components:

- **Endpoint Picker (EPP)**: The intelligent routing engine that serves as the "brain" of the router. It evaluates incoming requests against the current state of the [InferencePool], considering factors like KV-cache locality, current load, and priority to make optimal placement decisions. It integrates with L7 proxies via the `ext-proc` protocol.
- **Request Management APIs**: These resources directly influence the EPP's request handling behavior:
    - **InferenceObjective**: Configures the EPP's scheduling goals for specific requests, including priority levels and performance targets.
    - **InferenceModelRewrite**: Directs the EPP to perform model name rewriting, enabling flexible traffic management for A/B testing and canary rollouts.
- **Disaggregation Sidecar**: A coordination component deployed alongside model servers (typically as a sidecar to the decode worker). It orchestrates complex multi-stage inference lifecycles, such as **P/D (Prefill/Decode)** and **E/P/D (Encode/Prefill/Decode)**, by communicating with specialized encode and prefill workers to manage KV-cache and embedding transfers. For more details, see the [Disaggregation Documentation].

## Modes of Operation

The llm-d Router supports two primary deployment modes as specified in the [Kubernetes Gateway API Inference Extensions]:

### 1. Standalone Mode
A lightweight deployment where a self-managed Envoy proxy runs alongside the EPP in the same pod. This mode is ideal for clusters without Gateway API infrastructure or for basic testing and local evaluations.

### 2. Gateway Mode (Inference Gateway)
The recommended mode for production environments, leveraging the official [Gateway API]. In this mode, the EPP acts as a backend for an `InferencePool`, which is referenced by an `HTTPRoute` on a shared `Gateway`. This enables advanced traffic management, multi-cluster load balancing, and shared infrastructure for both inference and traditional workloads.

For more details on the router architecture, routing logic, and different plugins (filters and scorers), see the [Architecture Documentation].

---

> [!NOTE]
> The project provides tools for automatic Envoy installation. However, if you install or
> configure it yourself, please note that the only supported [request_body_mode and response_body_mode](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto)
> is `FULL_DUPLEX_STREAMED`

## Terminology

To ensure clarity across the project, we use the following standard terminology:

- **llm-d Router**: The complete intelligent entry point, comprising both the **Proxy** (e.g., Envoy) and the **Endpoint Picker (EPP)**. This term replaces "Inference Scheduler" in all contexts.
- **llm-d Endpoint Picker (EPP)**: The specific component that implements the routing intelligence and scoring logic. Use this term when referring to capabilities or configurations specific to the EPP itself, rather than the request routing system as a whole.
- **Inference Gateway**: A synonym for the **llm-d Router** when operating in **Gateway Mode**.
- **Request Scheduler**: A sub-component within the EPP responsible for the queuing and dispatching of requests.

[Kubernetes]:https://kubernetes.io
[Kubernetes Gateway API]:https://gateway-api.sigs.k8s.io/
[Architecture Documentation]:docs/architecture.md
[Disaggregation Documentation]:docs/disaggregation.md
[InferencePool]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Gateway API Inference Extension (GIE)]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Kubernetes Gateway API Inference Extensions]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Gateway API]:https://github.com/kubernetes-sigs/gateway-api
[Envoy]:https://github.com/envoyproxy/envoy
[ext-proc]:https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter

## Contributing

Our community meeting is bi-weekly at Wednesday 10AM PDT ([Google Meet], [Meeting Notes]).

We currently utilize the [#sig-router] channel in llm-d Slack workspace for communications.

For large changes please [create an issue] first describing the change so the
maintainers can do an assessment, and work on the details with you. See
[DEVELOPMENT.md](DEVELOPMENT.md) for details on how to work with the codebase.

Contributions are welcome!

[create an issue]:https://github.com/llm-d/llm-d-inference-scheduler/issues/new
[discussion]:https://github.com/llm-d/llm-d-inference-scheduler/discussions/new?category=q-a
[Slack]:https://llm-d.slack.com/
[Google Meet]:https://meet.google.com/zij-zekm-jvt
[Meeting Notes]:https://docs.google.com/document/d/1Pf3x7ZM8nNpU56nt6CzePAOmFZ24NXDeXyaYb565Wq4
[#sig-router]:https://llm-d.slack.com/?redir=%2Fmessages%2Fsig-router
