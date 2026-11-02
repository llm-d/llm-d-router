# Review Rules

Compact Stage-1 subsystem and architectural review rules loaded on demand via [review_a_pull_request.md](../playbooks/review_a_pull_request.md). Citation provenance, recurrence counts, and rationale live in [`provenance.json`](../provenance.json) and compile into [`docs/review_playbook.md`](../../review_playbook.md) for human readers. See [AGENTS.md](../../../AGENTS.md) for repository-wide coding, prose, logging, constant-reuse, and pull request rules.

* [Scope and process](scope_and_process.md) - Public design issue alignment, auto-close keyword scope, and release-note hygiene.
* [Plugin framework contracts](plugin_framework_contracts.md) - Core-vs-plugin import boundaries, extension-point separation, typed data-layer slots, factory validation, naming, and dead-code removal.
* [Project direction](project_direction.md) - Architectural boundaries across built-in runtimes, upstream signal producers, engine adapters, orchestrator-neutral endpoints, pool homogeneity, coordinator vs sidecar, Helm packaging, and upstream CRDs.
* [Configuration, Helm, and compatibility](configuration_helm_and_compatibility.md) - Two-release deprecation windows, strict unknown-field decoding, safe defaults, shared Helm library templates, and lossless CRD conversion.
* [Concurrency and shared state](concurrency_and_shared_state.md) - Mutex critical sections, bounded state with endpoint deletion cleanup, edge-triggered priority bands, drain-scoped contexts, per-replica state sync, and KV-event sequence recovery.
* [Error handling and boundaries](error_handling_and_boundaries.md) - Boundary input validation, explicit error propagation without fabricated fallbacks, and unwrapped status-code mapping.
* [Protocol parsing and wire fidelity](protocol_parsing_and_wire_fidelity.md) - Passthrough JSON fidelity, declarative route claims, complete SSE frame buffering, multi-API multimodal parity, host-port-rank endpoint keys, and disaggregated stage ordering.
* [Performance on the request path](performance_on_the_request_path.md) - Zero avoidable hot-path allocations, pooled transports and multiplexed event sockets, orthogonal load signals, contiguous prefix-cache scoring, and reproducible benchmarks.
* [Observability](observability.md) - Canonical metric subsystems and bounded label cardinality, single terminal request outcomes, package-scoped OpenTelemetry spans, and guarded debug logs.
* [Testing and CI](testing_and_ci.md) - Production factory tests with exact output assertions, colocated sleep-free unit tests, and namespace-isolated E2E teardown.
* [Security and trust boundaries](security_and_trust_boundaries.md) - Header namespacing and ingress/egress scrubbing, InferencePool SSRF target allowlists, TLS minimums, namespace-scoped watches, and sensitive payload redaction.
* [Documentation and comments](documentation_and_comments.md) - Colocated plugin READMEs, upstream guide linking, and review-round documentation synchronization.
