# Open Questions

Unresolved maintainer design tensions carried at `status: draft` until settled into `decisions/`.

* [Design proposal storage in repository vs issues](design_proposals_location.md) - Whether accepted design proposals belong in an in-repo proposals directory or in GitHub issues and linked design documents.
* [Non-Kubernetes InferenceObjective file schema](non_kubernetes_objectives_schema.md) - Whether standalone deployments load InferenceObjective definitions from Kubernetes CRD manifests with ignored fields or from a dedicated strict file schema.
* [Disaggregation sidecar Kubernetes watches for SSRF allowlists](sidecar_ssrf_allowlist_watches.md) - Whether every decode pod disaggregation sidecar should watch InferencePool and Pod resources directly or avoid workload-pod API-server watches.
* [Remote fetch vs repository cache for upstream test CRDs](upstream_test_crd_caching.md) - Whether unmodified upstream Gateway API and GIE CRDs used by verification and test scripts are fetched remotely on every run or cached in the repository.
* [Plugin panic isolation at the invocation boundary](plugin_panic_isolation.md) - Whether the EPP plugin framework wraps extension-point calls in recover guards per request or lets panicking plugins terminate the process.
* [Multi-tenant cost attribution on Prometheus labels vs logs and traces](cost_attribution_prometheus_labels.md) - Whether tenant, workload, and user cost-attribution identifiers belong on opt-in Prometheus metric labels or exclusively in structured logs and traces.
* [Session affinity transport via Set-Cookie vs request header](session_affinity_cookie_vs_header.md) - Whether session affinity state is carried by standard HTTP Set-Cookie headers or by explicit session-token headers.
* [Identifier hashing vs cleartext in DumpState debug endpoints](dump_state_identifier_hashing.md) - Whether stateful plugin DumpState endpoints hash tenant and program identifiers by default or emit cleartext values behind network access controls.
* [Prefix-cache affinity decode hotspot release formula](prefix_cache_affinity_hotspot_release.md) - Whether prefix-cache-affinity-filter releases decode hotspots through a unified ttftEstimate formula by default or behind opt-in configuration.
* [Sub-agent wire identity for concurrent KV session accounting](sub_agent_wire_identity.md) - How parallel sub-agents supply distinct wire-level identifiers so stateful session ledgers do not overwrite parent-session KV token counts.
* [Single multi-protocol Vertex AI parser vs separate REST and gRPC parsers](vertexai_multi_protocol_parser_split.md) - Whether platforms serving both REST and gRPC inference routes use one parser with segment wildcards or split into protocol-specific parser plugins.
* [Coordinator SGLang P/D route scoping, bootstrap port, and stream abort handling](coordinator_sglang_pd_streaming_abort.md) - How the Coordinator scopes SGLang disaggregated /generate routing, Python vs Rust bootstrap_port behavior, and mid-stream prefill error termination.
