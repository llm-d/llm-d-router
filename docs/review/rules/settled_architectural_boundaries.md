# Settled architectural boundaries

## Capacity shedding in SaturationDetector or AdmissionPlugin
When implementing overload shedding or scheduling filters: place capacity shedding in `SaturationDetector` or `AdmissionPlugin` (`Admitter`) rather than `Filter` plugins.

## Homogeneous base model and role per InferencePool
When routing across models or disaggregated roles in an `InferencePool` or EPP: scope each pool and EPP to one homogeneous base model and role, routing distinct base models to separate pools via Gateway `HTTPRoute`.
- Accept: Unioning endpoint model lists in `models-responder` on `GET /v1/models` for subset-loaded LoRA adapters.
- Open: Whether non-Kubernetes deployments load `InferenceObjective` files from Kubernetes manifests with ignored fields or a strict schema.

## External conversation history vs bounded routing state
When adding multi-turn session handling or stateful routing plugins: never store client conversation history in the router, while allowing bounded in-memory routing and session state with capacity caps and endpoint-deletion cleanup.

## Cluster orchestration in pkg/coordinator and pod proxy in pkg/sidecar
When placing disaggregated multi-stage logic across `pkg/coordinator` and `pkg/sidecar`: keep cluster multi-stage orchestration in `pkg/coordinator` and keep `pkg/sidecar` scoped to the pod-local worker proxy.
- Open: Whether decode-pod sidecars should watch `InferencePool` and `Pod` objects for SSRF allowlists or avoid workload-pod API watches.

## Ingress-only header stripping
When stripping headers or forwarding `kv_transfer_params`: strip spoofing headers and client `kv_transfer_params` at every external client entry point, passthrough routes included, while treating internal EPP, Coordinator, and Sidecar `kv_transfer_params` and routing metadata hops as trusted.
- Open: Whether session affinity uses standard HTTP `Set-Cookie` headers or an explicit header such as `x-session-token`.
