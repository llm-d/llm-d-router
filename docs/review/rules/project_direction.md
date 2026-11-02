# Project direction

## Keep core runtimes built in and policy plugins optional
Keep core execution runtimes and cross-replica state-sync orchestration in built-in runtime packages, and keep optional operator-configured policies in `pkg/epp/framework/plugins`. Route scalar telemetry through generic attribute-driven filters and scorers, share lightweight math via plain Go helpers, and require multiple concrete use cases before adding a framework extension point.
- Accept: A new extension point with one initial caller when the alternative forces core runtime code to reference a concrete plugin.

## Extract scheduling signals in upstream producers and pass native payloads through
Run tokenization, multimodal extraction, and token estimation in upstream pre-scheduling data producers (`token-producer`) so downstream filters and scorers stay free of wire-format parsing. Forward client payloads to native engine render endpoints (`/v1/responses/render`, `/v1/messages/render`) instead of translating API schemas in the router (deprecated `legacy` render mode retains chat conversion only for engines lacking native render), and order pre-admission queues by immutable admission-time request attributes.
- Accept: A contained interim translation layer for an API surface that the backend engine cannot render natively.

## Configure backend engine identity explicitly and isolate engine schemas
Configure backend engine identity explicitly at deployment time rather than inferring engine type from unrelated connector flags. Isolate engine-specific wire schemas, CLI flags, and metric names behind declarative adapters, and close router pull requests blocked on stalled unmerged upstream engine changes.

## Model endpoints with orchestrator-neutral domain types
Model discovery, server registration, metric extraction, and `datalayer` interfaces around orchestrator-neutral `Endpoint` types rather than Kubernetes `Pod` structs or controller managers. Do not widen the Kubernetes-specific seam beyond the datastore registration path and `Pod` controllers.
- Accept: Core identity fields on `EndpointMetadata` with optional per-endpoint topology such as rank stored in `AttributeMap`.
- Open: Whether non-Kubernetes deployments load `InferenceObjective` definitions from Kubernetes manifests with ignored fields or from a separate strict file schema.

## Scope each pool to a single homogeneous base model and role
Scope each `InferencePool` and EPP process to homogeneous replicas serving one base model and disaggregated role so every endpoint contributes to shared pool capacity. Route distinct base models to separate pools via Gateway `HTTPRoute` rules.
- Accept: Serving `GET /v1/models` from EPP via `models-responder` by unioning endpoint model lists so subset-loaded LoRA adapters remain visible.

## Keep client conversation state out of the router and orchestrate multi-stage pipelines in the coordinator
Keep conversation persistence, intra-backend rank balancing, and offline calibration outside the router, and reject `/v1/responses` stateful fields (`previous_response_id`, `conversation`) with an explicit error instead of dropping them. Place cluster-level multi-stage disaggregated orchestration in `pkg/coordinator` above endpoint selection, and keep `pkg/sidecar` scoped to the pod-local proxy whose only Kubernetes API access is the optional namespace-scoped `InferencePool` and `Pod` watch behind `--enable-ssrf-protection`.
- Accept: Bounded in-memory session registries and cross-replica synced routing state when client payloads and conversation history remain external.
- Open: Whether the disaggregation sidecar should watch `InferencePool` and `Pod` objects from every decode pod for SSRF allowlists or avoid workload-pod API-server watches.

## Deploy through Helm charts and configure components via structured files
Ship end-user deployments through Helm charts under `config/charts` and configure components via hierarchical structured config files rather than adding binary CLI flags. Publish multi-component operations guides and umbrella dashboards in `llm-d/llm-d`, keeping Kustomize manifests under `deploy` scoped to local development and E2E suites.
- Accept: Updates to Kustomize manifests under `deploy` needed to keep local kind environments and E2E suites running.

## Consume versioned upstream CRDs and protobufs without vendoring
Consume upstream Gateway API Inference Extension CRDs and protobuf definitions from versioned releases or the pinned Go module cache instead of vendoring local copies (`deploy/components/crds-istio` is the sole local-dev exception).
- Open: Whether unmodified upstream test CRDs should be fetched from GitHub on each run or cached in-repo to avoid rate-limit failures.
