# Review Playbook

<!-- Generated from docs/review/ by hack/render_playbook.py. Do not edit directly; regenerate with: python3 hack/render_playbook.py -->

> **Two-stage workflow:** This file is the compiled human-readable review guide for contributors and maintainers, combining subsystem rules with rationale, accepted exceptions, open questions, and representative pull request threads. Coding and review agents must not load `docs/review_playbook.md` or `docs/review/provenance.json`; agents route changed paths through [review_a_pull_request.md](review/playbooks/review_a_pull_request.md) and load only matching rule files under `docs/review/rules/`. See [AGENTS.md](../AGENTS.md) for repository-wide coding, prose, logging, constant-reuse, and pull request rules.

## Contents

- [Scope and process](#scope-and-process) (2 rules)
- [Plugin framework contracts](#plugin-framework-contracts) (6 rules)
- [Project direction](#project-direction) (8 rules)
- [Configuration, Helm, and compatibility](#configuration-helm-and-compatibility) (5 rules)
- [Concurrency and shared state](#concurrency-and-shared-state) (6 rules)
- [Error handling and boundaries](#error-handling-and-boundaries) (3 rules)
- [Protocol parsing and wire fidelity](#protocol-parsing-and-wire-fidelity) (6 rules)
- [Performance on the request path](#performance-on-the-request-path) (5 rules)
- [Observability](#observability) (4 rules)
- [Testing and CI](#testing-and-ci) (3 rules)
- [Security and trust boundaries](#security-and-trust-boundaries) (3 rules)
- [Documentation and comments](#documentation-and-comments) (2 rules)

## Scope and process

Public design issue alignment, auto-close keyword scope, and release-note hygiene.

### Settle design tradeoffs on public issues and scope auto-close keywords

Discuss architectural changes and behavior-changing requirements on public GitHub issues rather than private trackers before iterating on pull requests. Use auto-closing keywords (`Fixes #`, `Closes #`) only when a pull request resolves the entire issue, and use `Part of #` on multi-slice work.

- **Why it matters:** Unaligned design changes stall in late review, and premature auto-close keywords close multi-slice tracking issues while work remains.
- **Reviewers accept:** Narrow wire-formatting adjustments motivated by an external conformance verifier when the issue and pull request state that compatibility scope ([#2590](https://github.com/llm-d/llm-d-router/issues/2590#issuecomment-5499142962)).
- **Settled decision:** [Pull request template distinguishes Fixes from Part of](review/decisions/pr_template_fixes_vs_part_of.md).
- **Open question:** Whether accepted design proposals live in an in-repo proposals directory ([#2376](https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3832021146)) or only in issues and linked design documents ([#2376](https://github.com/llm-d/llm-d-router/pull/2376#discussion_r3837894575)) ([details](review/open_questions/design_proposals_location.md)).
- **Examples:** Observed in 74 PRs: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4541930806), [#857](https://github.com/llm-d/llm-d-router/issues/857), [#1090](https://github.com/llm-d/llm-d-router/issues/1090#issuecomment-5347088232)

### Use `NONE` release notes for internal changes and keep PR metadata synchronized

Set the `release-note` block to `NONE` for internal refactors, tests, CI edits, and fixes to unshipped features on `main`, and write operator-focused notes when defaults, flags, metrics, headers, or upgrade steps change. Update the pull request title, description, and release note whenever review iterations rename symbols or narrow scope.

- **Why it matters:** Internal churn in `release-note` blocks pollutes tag release notes, and stale PR descriptions misstate what merged.
- **Examples:** Observed in 49 PRs: [#1206](https://github.com/llm-d/llm-d-router/pull/1206#issuecomment-4647621711), [#1657](https://github.com/llm-d/llm-d-router/pull/1657#issuecomment-4708030072), [#1939](https://github.com/llm-d/llm-d-router/pull/1939#issuecomment-4925378102)

## Plugin framework contracts

Core-vs-plugin import boundaries, extension-point separation, typed data-layer slots, factory validation, naming, and dead-code removal.

### Keep core runtime packages free of concrete plugin logic

Do not import concrete plugin packages or switch on concrete plugin types inside `cmd/epp/runner`, `pkg/epp/datastore`, or `pkg/epp/framework/interface`. Expose shared capabilities through `fwkplugin.Handle` and small optional interfaces verified with compile-time guards (`var _ Interface = (*Type)(nil)`), resolved once at startup.

- **Why it matters:** Importing concrete plugins into core runtime packages creates import cycles and couples core dispatch to optional policies.
- **Examples:** Observed in 60 PRs: [#836](https://github.com/llm-d/llm-d-router/issues/836), [#877](https://github.com/llm-d/llm-d-router/issues/877), [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3261995399)

### Separate data producers from scheduling filters and scorers

Keep stateful indexing and raw endpoint extraction in data producers and place policy evaluation in scheduling plugins. Restrict `Filter` plugins to binary eligibility with fail-open bypass when optional telemetry is absent, express soft preferences as weighted `Scorer` plugins, reserve `Screener` plugins for mandatory profile-wide constraints, and leave load shedding to saturation detectors.

- **Why it matters:** Mixing raw telemetry extraction with scoring or encoding soft preferences as hard filters causes premature endpoint elimination and duplicate state.
- **Reviewers accept:** Strict single-endpoint request pinning as a pre-scheduling `Screener` once weighed against `Filter`, or a shared stateless helper package for header parsing without I/O or caching ([Session affinity as a shared stateless helper package](review/decisions/session_affinity_shared_helper.md), [#3145](https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6088007878), [#3145](https://github.com/llm-d/llm-d-router/pull/3145#pullrequestreview-5462400380), [#815](https://github.com/llm-d/llm-d-router/issues/815#issuecomment-4603503350)).
- **Examples:** Observed in 37 PRs: [#545](https://github.com/llm-d/llm-d-router/issues/545#issuecomment-5194281537), [#837](https://github.com/llm-d/llm-d-router/issues/837), [#977](https://github.com/llm-d/llm-d-router/issues/977)

### Exchange cross-plugin state through typed data-layer slots and keys

Pass cross-plugin request and endpoint metadata through typed `datalayer.Slot[T]` handles or `Clone()`-safe values in `datalayer.AttributeMap` instead of global singletons or direct peer calls. Declare dependencies via `Produces()` and `Consumes()`, key instance-scoped attributes by non-empty plugin names, and validate producer-consumer DAG wiring at startup after default producers resolve.

- **Why it matters:** Sharing state via globals or undeclared attribute keys bypasses startup DAG validation and causes silent nil reads or aliasing across plugin instances.
- **Settled decisions:** [Typed data-layer Slot handles over generic Extractor interfaces](review/decisions/typed_datalayer_slots_over_generic_extractors.md), [Per-request endpoint attribute isolation with dynamic metric read-through](review/decisions/per_request_endpoint_attribute_isolation_with_dynamic_readthrough.md).
- **Examples:** Observed in 58 PRs: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4396004237), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252785994), [#1096](https://github.com/llm-d/llm-d-router/pull/1096#discussion_r3255496885)

### Validate plugin parameters once in the factory constructor

Validate parameter bounds, required handles, and forbidden system headers inside the factory constructor without exposing mutable post-construction setters or repeating checks per request. Register built-in plugins in the central catalog with an explicit stability tier and enforce experimental feature gates across explicit and default instances.

- **Why it matters:** Deferring parameter checks past factory construction delays invalid-config failures until live traffic or allows post-init mutation races.
- **Reviewers accept:** Documenting redundant plugin combinations in plugin READMEs instead of rejecting sibling plugins inside factory constructors ([#1396](https://github.com/llm-d/llm-d-router/pull/1396#issuecomment-4642995683)).
- **Examples:** Observed in 49 PRs: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3268601757), [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3196010531), [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628902)

### Name plugin packages with concise domain nouns and qualify config units

Name plugin packages with short lowercase domain nouns (`bylabel`, `queuedepth`, `tokenload`, `sessionaffinity`), suffix newly registered plugin type strings with their role (`-filter`, `-scorer`, `-picker`, `-producer`, `-screener`, `-decider`), and omit package-name stutter on exported identifiers (shipped unsuffixed type strings keep their names). Suffix quantitative config fields with explicit units or portable token counts rather than backend block counts, and use `resource.Quantity` for byte sizes.

- **Why it matters:** Ambiguous plugin names and unitless block-count fields cause misconfigured thresholds when operators switch backend engines.
- **Reviewers accept:** Standard data-structure names on internal types and preserving shipped public constant names across internal variable renames ([#2036](https://github.com/llm-d/llm-d-router/pull/2036#issuecomment-4999184223), [#2342](https://github.com/llm-d/llm-d-router/pull/2342#discussion_r3752875804)).
- **Settled decision:** [Prefix-cache block size parameter includes Tokens unit suffix](review/decisions/prefix_cache_block_size_tokens_unit_suffix.md).
- **Examples:** Observed in 72 PRs: [#747](https://github.com/llm-d/llm-d-router/issues/747), [#813](https://github.com/llm-d/llm-d-router/issues/813), [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143427635)

### Share common helpers and delete dead code in the stranding change

Decouple pure payload parsing from HTTP response writing and reuse shared utilities in `pkg/common` rather than duplicating logic across packages. Delete dead branches, write-only fields, unreferenced methods, and single-caller pass-through wrappers in the change that strands them.

- **Why it matters:** Duplicated helpers drift on edge cases, and stranded shims leave dead execution paths that mislead maintainers.
- **Settled decision:** [Delete stranded deprecation helpers in the removal pull request](review/decisions/delete_stranded_deprecation_helpers_immediately.md).
- **Examples:** Observed in 97 PRs: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143173577), [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236349435), [#1444](https://github.com/llm-d/llm-d-router/pull/1444#pullrequestreview-4425939275)

## Project direction

Architectural boundaries across built-in runtimes, upstream signal producers, engine adapters, orchestrator-neutral endpoints, pool homogeneity, coordinator vs sidecar, Helm packaging, and upstream CRDs.

### Keep core runtimes built in and policy plugins optional

Keep core execution runtimes and cross-replica state-sync orchestration in built-in runtime packages, and keep optional operator-configured policies in `pkg/epp/framework/plugins`. Route scalar telemetry through generic attribute-driven filters and scorers, share lightweight math via plain Go helpers, and require multiple concrete use cases before adding a framework extension point.

- **Why it matters:** Adding single-purpose extension points bloats the framework interface, while hardcoding policies in core runtimes prevents operator opt-out.
- **Reviewers accept:** A new extension point with one initial caller when the alternative forces core runtime code to reference a concrete plugin ([Pre-admission header attribute extraction via dedicated plugin extension point](review/decisions/request_header_extractor_before_admission.md), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#discussion_r3740526294), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061)).
- **Settled decision:** [Scorer-internal fallback over scheduler fallback profiles](review/decisions/scorer_internal_fallback_over_fallback_profiles.md).
- **Examples:** Observed in 23 PRs: [#818](https://github.com/llm-d/llm-d-router/issues/818), [#1139](https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080), [#1450](https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814)

### Extract scheduling signals in upstream producers and pass native payloads through

Run tokenization, multimodal extraction, and token estimation in upstream pre-scheduling data producers (`token-producer`) so downstream filters and scorers stay free of wire-format parsing. Forward client payloads to native engine render endpoints (`/v1/responses/render`, `/v1/messages/render`) instead of translating API schemas in the router (deprecated `legacy` render mode retains chat conversion only for engines lacking native render), and order pre-admission queues by immutable admission-time request attributes.

- **Why it matters:** Router-side schema translation drops model-specific fields, and mutable queue keys corrupt pre-admission priority ordering.
- **Reviewers accept:** A contained interim translation layer for an API surface that the backend engine cannot render natively ([#1896](https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643), [#1896](https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118)).
- **Examples:** Observed in 23 PRs: [#850](https://github.com/llm-d/llm-d-router/issues/850), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3279332668), [#1364](https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492)

### Configure backend engine identity explicitly and isolate engine schemas

Configure backend engine identity explicitly at deployment time rather than inferring engine type from unrelated connector flags. Isolate engine-specific wire schemas, CLI flags, and metric names behind declarative adapters, and close router pull requests blocked on stalled unmerged upstream engine changes.

- **Why it matters:** Inferring engine types from tangential flags or scattering engine conditionals across routing packages breaks multi-engine deployments.
- **Examples:** Observed in 11 PRs: [#382](https://github.com/llm-d/llm-d-router/issues/382#issuecomment-4585925394), [#3030](https://github.com/llm-d/llm-d-router/issues/3030), [#3359](https://github.com/llm-d/llm-d-router/pull/3359#discussion_r4361817276)

### Model endpoints with orchestrator-neutral domain types

Model discovery, server registration, metric extraction, and `datalayer` interfaces around orchestrator-neutral `Endpoint` types rather than Kubernetes `Pod` structs or controller managers. Do not widen the Kubernetes-specific seam beyond the datastore registration path and `Pod` controllers.

- **Why it matters:** Leaking Kubernetes `Pod` or controller types into discovery and `datalayer` interfaces blocks standalone non-Kubernetes deployments.
- **Reviewers accept:** Core identity fields on `EndpointMetadata` with optional per-endpoint topology such as rank stored in `AttributeMap` ([Core endpoint identity on EndpointMetadata and optional topology in AttributeMap](review/decisions/endpoint_identity_metadata_vs_optional_attributes.md), [#1106](https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3273288488), [#2147](https://github.com/llm-d/llm-d-router/pull/2147#pullrequestreview-4766963622)).
- **Open question:** Whether non-Kubernetes deployments load `InferenceObjective` definitions from Kubernetes manifests with unused fields ignored ([#3502](https://github.com/llm-d/llm-d-router/pull/3502)) or from a separate strict file schema ([#3502](https://github.com/llm-d/llm-d-router/pull/3502#pullrequestreview-5649007338)) ([details](review/open_questions/non_kubernetes_objectives_schema.md)).
- **Examples:** Observed in 7 PRs: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628879), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3271206173), [#1148](https://github.com/llm-d/llm-d-router/pull/1148#discussion_r3294724682)

### Scope each pool to a single homogeneous base model and role

Scope each `InferencePool` and EPP process to homogeneous replicas serving one base model and disaggregated role so every endpoint contributes to shared pool capacity. Route distinct base models to separate pools via Gateway `HTTPRoute` rules.

- **Why it matters:** Mixing base models or disaggregated roles inside one `InferencePool` invalidates pool-wide load, saturation, and KV-cache capacity assumptions.
- **Reviewers accept:** Serving `GET /v1/models` from EPP via `models-responder` by unioning endpoint model lists so subset-loaded LoRA adapters remain visible ([Pool-wide /v1/models aggregation via generic Responder plugin interface](review/decisions/models_responder_generic_extension_point.md), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5483618104)).
- **Examples:** Observed in 7 PRs: [#1878](https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197), [#2333](https://github.com/llm-d/llm-d-router/pull/2333#issuecomment-5935642884), [#2692](https://github.com/llm-d/llm-d-router/pull/2692#issuecomment-5626620151)

### Keep client conversation state out of the router and orchestrate multi-stage pipelines in the coordinator

Keep conversation persistence, intra-backend rank balancing, and offline calibration outside the router, and reject `/v1/responses` stateful fields (`previous_response_id`, `conversation`) with an explicit error instead of dropping them. Place cluster-level multi-stage disaggregated orchestration in `pkg/coordinator` above endpoint selection, and keep `pkg/sidecar` scoped to the pod-local proxy whose only Kubernetes API access is the optional namespace-scoped `InferencePool` and `Pod` watch behind `--enable-ssrf-protection`.

- **Why it matters:** Dropping stateful `/v1/responses` fields silently breaks multi-turn context, and placing multi-stage orchestration in pod sidecars expands workload-pod privileges.
- **Reviewers accept:** Bounded in-memory session registries and cross-replica synced routing state when client payloads and conversation history remain external ([#1892](https://github.com/llm-d/llm-d-router/issues/1892), [#2531](https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755)).
- **Open question:** Whether the disaggregation sidecar should watch `InferencePool` and `Pod` objects from every decode pod to build its SSRF allowlist ([#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771)) or avoid API-server watches from workload pods ([#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953)) ([details](review/open_questions/sidecar_ssrf_allowlist_watches.md)).
- **Examples:** Observed in 14 PRs: [#896](https://github.com/llm-d/llm-d-router/issues/896#issuecomment-5923079423), [#1104](https://github.com/llm-d/llm-d-router/issues/1104#issuecomment-4449848790), [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953)

### Deploy through Helm charts and configure components via structured files

Ship end-user deployments through Helm charts under `config/charts` and configure components via hierarchical structured config files rather than adding binary CLI flags. Publish multi-component operations guides and umbrella dashboards in `llm-d/llm-d`, keeping Kustomize manifests under `deploy` scoped to local development and E2E suites.

- **Why it matters:** Adding ad-hoc CLI flags or diverging Kustomize production paths fragments operator configuration across deployment methods.
- **Reviewers accept:** Updates to Kustomize manifests under `deploy` needed to keep local kind environments and E2E suites running ([Helm charts for end-user releases and deploy Kustomize manifests for local kind and E2E](review/decisions/helm_release_charts_with_dev_kustomize_manifests.md), [#1177](https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514), [#2048](https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577)).
- **Examples:** Observed in 30 PRs: [#634](https://github.com/llm-d/llm-d-router/issues/634), [#930](https://github.com/llm-d/llm-d-router/pull/930#issuecomment-4366069128), [#1550](https://github.com/llm-d/llm-d-router/pull/1550#issuecomment-4656209920)

### Consume versioned upstream CRDs and protobufs without vendoring

Consume upstream Gateway API Inference Extension CRDs and protobuf definitions from versioned releases or the pinned Go module cache instead of vendoring local copies (`deploy/components/crds-istio` is the sole local-dev exception).

- **Why it matters:** Vendoring upstream CRDs or protobufs forks schemas from pinned Go module versions and masks upstream drift.
- **Open question:** Whether unmodified upstream CRDs used by tests are fetched from GitHub on every run ([#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490)) or cached in the repository to avoid rate-limited fetch failures ([#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4661288603)) ([details](review/open_questions/upstream_test_crd_caching.md)).
- **Examples:** Observed in 9 PRs: [#968](https://github.com/llm-d/llm-d-router/issues/968#issuecomment-4602495714), [#1035](https://github.com/llm-d/llm-d-router/pull/1035#discussion_r3244818645), [#1043](https://github.com/llm-d/llm-d-router/pull/1043#discussion_r3279489618)

## Configuration, Helm, and compatibility

Two-release deprecation windows, strict unknown-field decoding, safe defaults, shared Helm library templates, and lossless CRD conversion.

### Provide a two-release deprecation window for shipped user-facing surfaces

Support shipped CLI flags, configuration fields, plugin types, HTTP headers, and metrics across at least two releases (`N+2`) before removal. Log a startup warning when a deprecated surface is used, migrate all in-tree manifests and tests to the replacement immediately, and separate structural moves from renames.

- **Why it matters:** Removing shipped flags, config fields, headers, or metrics without an `N+2` window breaks existing Helm upgrades and operator dashboards.
- **Reviewers accept:** Renaming or removing unshipped config keys, unreleased Helm charts, and internal headers without deprecation shims, or deleting shims after `N+2` ([#894](https://github.com/llm-d/llm-d-router/pull/894#discussion_r3151838341), [#1254](https://github.com/llm-d/llm-d-router/pull/1254#issuecomment-4512813466), [#2365](https://github.com/llm-d/llm-d-router/pull/2365#discussion_r4241052101), [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379506333)).
- **Settled decisions:** [Two-release deprecation shims required even during pre-1.0 config refactors](review/decisions/two_release_deprecation_shims_for_pre_v1_config.md), [Plugin graduation path across Alpha, Beta, and Stable tiers](review/decisions/plugin_graduation_tiers.md).
- **Examples:** Observed in 68 PRs: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4395649694), [#842](https://github.com/llm-d/llm-d-router/pull/842#discussion_r3180806498), [#905](https://github.com/llm-d/llm-d-router/pull/905#issuecomment-4351689606)

### Decode configuration with strict unknown-field rejection and startup validation

Seed config structs from canonical default constructors, decode through `fwkplugin.StrictDecoder` (`DisallowUnknownFields()`), `yaml.UnmarshalStrict`, or `serializer.EnableStrict`, and validate cross-field and mode-scoped invariants before serving traffic. Prefer explicit mode enums over boolean flag combinations, and use pointer numeric fields with `omitempty` only when distinguishing unset defaults from explicit zero.

- **Why it matters:** Permissive YAML or JSON decoding silently ignores misspelled operator config fields and runs pods with unintended defaults.
- **Examples:** Observed in 63 PRs: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143157150), [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190715413), [#919](https://github.com/llm-d/llm-d-router/pull/919#discussion_r3161391196)

### Ship safe out-of-the-box defaults and gate workload-sensitive features behind opt-in config

Choose defaults that run within standard CPU and memory budgets without external sidecars or cross-component boot-order dependencies. Gate workload-sensitive heuristics, speculative estimators, and outbound payload rewrites behind opt-in config so disabled features leave the default request path untouched.

- **Why it matters:** Enabling workload-sensitive heuristics or sidecar dependencies by default regresses baseline deployments and causes boot-order deadlocks.
- **Settled decision:** [Flow control remains opt-in by default until standalone default-on graduation](review/decisions/flow_control_opt_in_feature_gate_posture.md).
- **Examples:** Observed in 35 PRs: [#1029](https://github.com/llm-d/llm-d-router/pull/1029#discussion_r3243916222), [#1156](https://github.com/llm-d/llm-d-router/pull/1156#discussion_r3454102412), [#1187](https://github.com/llm-d/llm-d-router/issues/1187#issuecomment-5139243431)

### Deduplicate Helm chart defaults through shared library templates

Share common Helm logic through library templates while declaring values only in charts that render them, fail template rendering with `fail` on removed keys, and keep lifecycle hooks compatible with shell-less distroless images.

- **Why it matters:** Copy-pasted Helm values drift across charts, and shell-based hooks fail at runtime inside distroless containers.
- **Reviewers accept:** Repeating stable Kubernetes API group literals directly in Helm templates and gating lifecycle features on supported Kubernetes versions with docs rather than shell fallbacks ([#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294998818), [#2961](https://github.com/llm-d/llm-d-router/pull/2961#discussion_r4069441791)).
- **Examples:** Observed in 52 PRs: [#924](https://github.com/llm-d/llm-d-router/issues/924), [#1040](https://github.com/llm-d/llm-d-router/pull/1040#issuecomment-4415659669), [#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294949972)

### Preserve lossless conversion across multi-version CRDs

Serve multi-version CRDs through lossless conversion under a single storage version with codegen validation markers, updating `config/charts`, `deploy`, and test manifests in the same change.

- **Why it matters:** Lossy CRD conversion drops fields during rolling upgrades across mixed API versions.
- **Examples:** Observed in 13 PRs: [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178415392), [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4144921130), [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4155389367)

## Concurrency and shared state

Mutex critical sections, bounded state with endpoint deletion cleanup, edge-triggered priority bands, drain-scoped contexts, per-replica state sync, and KV-event sequence recovery.

### Keep mutex critical sections tight and release locks before network or blocking calls

Encapsulate synchronization on the state-owning type, finish struct initialization before registering callbacks, and never hold a mutex across network RPCs, channel sends, or socket teardown. Combine read-modify-write steps and coupled counter reads under one lock acquisition, and copy caller-owned maps before asynchronous dispatch.

- **Why it matters:** Holding mutexes across network I/O or channel sends stalls concurrent request routing and risks lock-order deadlocks.
- **Reviewers accept:** Defensive slice or map copies at goroutine boundaries, snapshot coalescing in capacity-one channels, uncontended startup or shutdown locks, and locked handle-invalidation loops when unlocking early races with removal ([#942](https://github.com/llm-d/llm-d-router/pull/942#discussion_r3178202254), [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314150441), [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314202507), [#1690](https://github.com/llm-d/llm-d-router/pull/1690#discussion_r3455137825)).
- **Examples:** Observed in 60 PRs: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190593630), [#889](https://github.com/llm-d/llm-d-router/issues/889), [#1128](https://github.com/llm-d/llm-d-router/pull/1128#discussion_r3275924435)

### Bound in-memory state with capacity limits and endpoint deletion cleanup

Bound in-memory maps and queues keyed by requests, sessions, prompts, or endpoints with hard capacity caps and periodic TTL sweeps. Subscribe stateful plugins to endpoint lifecycle events so pod removals and empty controller reconciles purge per-endpoint state.

- **Why it matters:** Unbounded session or endpoint maps leak memory under pod churn and route traffic using stale deleted-pod state.
- **Reviewers accept:** A dedicated bounded registry when state requires no-eviction pinning of in-flight entries, or a periodic cleanup loop when missed lifecycle events could strand entries ([#2531](https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3269839644)).
- **Examples:** Observed in 27 PRs: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190233226), [#1004](https://github.com/llm-d/llm-d-router/pull/1004#discussion_r3202130559), [#1727](https://github.com/llm-d/llm-d-router/pull/1727#discussion_r3513605187)

### Reconcile dynamic priority bands on controller edges and drain before removal

Add and remove dynamic `InferenceObjective` priority bands on controller reconciliation edges rather than data-plane idleness sweeps, preserving static default bands and draining in-flight queues before band deletion.

- **Why it matters:** Idleness-based band sweeps race with arriving requests and drop queued work when active priority bands are deleted.
- **Settled decision:** [Channel-fed maintenance loop with lock protection for dynamic priority bands](review/decisions/channel_fed_priority_band_reconciliation.md).
- **Examples:** Observed in 29 PRs: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314143560), [#1606](https://github.com/llm-d/llm-d-router/issues/1606), [#1608](https://github.com/llm-d/llm-d-router/pull/1608#issuecomment-4688020707)

### Propagate caller contexts and bound goroutine lifetimes to server drain

Root background loops on a drain-scoped context, pass caller request contexts to all outbound RPCs, and run graceful readiness drain on leader lease loss. Check `ctx.Err()` across loop iterations and multi-channel `select` wakeups so client disconnects halt downstream work and release in-flight reservations.

- **Why it matters:** Ignoring `ctx.Err()` or server drain leaks background goroutines and strands capacity reservations after client disconnects.
- **Examples:** Observed in 17 PRs: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740756), [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3149928582), [#2713](https://github.com/llm-d/llm-d-router/pull/2713#discussion_r3996027352)

### Scope cross-replica state to per-replica keys with short remote-store TTLs

Publish cross-replica load and cache state under collision-free replica IDs, combining live local counters with cached peer levels at read time so replica restarts never overwrite peer state. Expire peer entries via short remote-store TTLs scaled to the publish interval instead of host wall-clock comparisons, and keep startup non-blocking when optional peer stores are unreachable.

- **Why it matters:** Shared cross-replica keys or host clock comparisons let one restarting replica clobber peer load counters or stall startup.
- **Examples:** Observed in 7 PRs: [#1892](https://github.com/llm-d/llm-d-router/issues/1892), [#1946](https://github.com/llm-d/llm-d-router/issues/1946), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955475084)

### Recover KV-event sequence gaps and epoch resets from base snapshots

Track per-endpoint KV-event sequence numbers to detect gaps or worker restarts (`seq < lastSeq`), resetting stale index entries and loading a base snapshot before applying subsequent events.

- **Why it matters:** Applying KV-cache deltas across sequence gaps or worker restarts leaves phantom cached blocks in the prefix index.
- **Examples:** Observed in 27 PRs: [#1737](https://github.com/llm-d/llm-d-router/issues/1737), [#1887](https://github.com/llm-d/llm-d-router/pull/1887#discussion_r3684757376), [#2946](https://github.com/llm-d/llm-d-router/pull/2946#pullrequestreview-5438730630)

## Error handling and boundaries

Boundary input validation, explicit error propagation without fabricated fallbacks, and unwrapped status-code mapping.

### Validate untrusted inputs at ingress and configuration boundaries

Validate nil pointers, type assertions, slice bounds, integer overflows, content types, and payload size limits at network, parser, and config boundaries before buffering bodies or constructing domain types. Include field and expected-type context in boundary errors, and close caller-supplied listeners when early setup fails.

- **Why it matters:** Unchecked untrusted payloads trigger panics or unbounded memory reads before domain logic runs.
- **Reviewers accept:** Omitting redundant defensive nil checks inside internal helpers whose callers already guarantee valid inputs ([#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3192331180), [#944](https://github.com/llm-d/llm-d-router/pull/944#discussion_r3435978187), [#3252](https://github.com/llm-d/llm-d-router/pull/3252#discussion_r4218084207)).
- **Examples:** Observed in 52 PRs: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143190014), [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3220172664), [#1147](https://github.com/llm-d/llm-d-router/pull/1147#issuecomment-4461397416)

### Propagate errors explicitly without fabricated fallbacks

Return producer, parser, and config failures as explicit errors instead of substituting placeholder counts, synthetic block sizes, or silent zero defaults, and never call `panic` on domain or invariant errors in request or flow-control paths. Fail startup when TLS certificates, metrics listeners, or required plugin bindings are invalid.

- **Why it matters:** Fabricated fallback values route traffic on bogus capacity numbers, and request-path panics crash the router replica.
- **Open question:** Whether the plugin framework recovers plugin panics at the invocation boundary and reports them per request ([#2105](https://github.com/llm-d/llm-d-router/issues/2105)) or lets a panicking plugin stop the process ([#2098](https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5032265086)) ([details](review/open_questions/plugin_panic_isolation.md)).
- **Examples:** Observed in 44 PRs: [#876](https://github.com/llm-d/llm-d-router/issues/876), [#1002](https://github.com/llm-d/llm-d-router/pull/1002#discussion_r3403097625), [#1061](https://github.com/llm-d/llm-d-router/pull/1061#discussion_r3246024767)

### Map unwrapped error chains to canonical status codes

Classify errors with `errors.Is` and `errors.As` across unwrapped chains rather than matching error strings, formatting error bodies for the active API surface. Map client validation faults to 4xx, upstream worker protocol faults to 502, empty pools to 503, and saturation drops to 429.

- **Why it matters:** Substring error matching breaks when messages change, and misclassified 4xx vs 5xx codes break client retry and gateway circuit-breaking.
- **Reviewers accept:** Returning 502 Bad Gateway rather than 400 Bad Request when an upstream worker returns an invalid content type or malformed payload for a valid client request ([#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379124714)).
- **Examples:** Observed in 36 PRs: [#1116](https://github.com/llm-d/llm-d-router/issues/1116#issuecomment-4450544045), [#1603](https://github.com/llm-d/llm-d-router/pull/1603#discussion_r3399020119), [#2052](https://github.com/llm-d/llm-d-router/pull/2052#discussion_r3597666475)

## Protocol parsing and wire fidelity

Passthrough JSON fidelity, declarative route claims, complete SSE frame buffering, multi-API multimodal parity, host-port-rank endpoint keys, and disaggregated stage ordering.

### Preserve unrecognized client request fields and wire fidelity

Decode only router-inspected fields into typed structs while preserving unrecognized client JSON fields, 64-bit integer precision, key ordering, and explicit booleans on forwarded bodies. Normalize headers and paths once at ingress through read-only accessors without mutating forwarded client headers.

- **Why it matters:** Full-struct JSON re-serialization strips vendor-specific sampling fields and rounds 64-bit integers via `float64`.
- **Settled decision:** [Opt-in backend request priority rewrite via PriorityRewriter interface](review/decisions/opt_in_backend_priority_rewrite.md).
- **Examples:** Observed in 51 PRs: [#1045](https://github.com/llm-d/llm-d-router/issues/1045), [#1072](https://github.com/llm-d/llm-d-router/issues/1072), [#1092](https://github.com/llm-d/llm-d-router/issues/1092)

### Claim inference routes declaratively with unmatched-route fallthrough

Register plugin and endpoint route claims declaratively by HTTP method and path so unmatched routes fall through to downstream handlers without hardcoded paths in core dispatchers.

- **Why it matters:** Hardcoded endpoint paths in core dispatchers block custom engine routes that should pass through untouched.
- **Reviewers accept:** Separate path and protocol slices when claimed paths apply uniformly, and forwarding native payloads directly across disaggregated stages running the same engine ([#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358239412), [#2987](https://github.com/llm-d/llm-d-router/pull/2987#discussion_r4090885940)).
- **Examples:** Observed in 14 PRs: [#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358215465), [#1516](https://github.com/llm-d/llm-d-router/issues/1516), [#1524](https://github.com/llm-d/llm-d-router/issues/1524)

### Buffer streaming responses to complete SSE event boundaries and count terminal usage once

Buffer chunked streaming responses to complete SSE event delimiters before inspecting frames so mid-event TCP splits never drop `data:` lines. Preserve `http.Flusher` via `Unwrap()` on response writer wrappers, record token usage only once from the terminal chunk, and surface premature EOF or missing terminal events as stream errors.

- **Why it matters:** Parsing raw TCP read chunks as whole SSE frames corrupts split JSON events, double-counts tokens, or hides truncated streams.
- **Examples:** Observed in 15 PRs: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740743), [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3161138182), [#2391](https://github.com/llm-d/llm-d-router/pull/2391#discussion_r3798712574)

### Maintain feature parity across inference APIs and validate multimodal items individually

Keep tokenization, tool-call rendering, usage extraction, and length classification consistent across chat completions, legacy completions, responses, messages, and generate endpoints. Group per-item fields into element structs rather than parallel slices, and validate multimodal metadata per item and modality rather than with request-wide booleans.

- **Why it matters:** API-skewed parsers or parallel multimodal slices misindex mixed-media prompts and bypass token accounting on newer endpoints.
- **Reviewers accept:** Hashing multimodal items atomically as whole units when the backend inference engine hashes media items without chunking ([#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4399431503), [#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4400172236)).
- **Examples:** Observed in 63 PRs: [#848](https://github.com/llm-d/llm-d-router/issues/848), [#932](https://github.com/llm-d/llm-d-router/pull/932#discussion_r3178050269), [#1019](https://github.com/llm-d/llm-d-router/issues/1019)

### Key endpoint state by host, port, and worker rank

Key per-endpoint state, cache indexes, eviction hooks, and event-socket ports by host, port, and worker rank rather than pod IP alone.

- **Why it matters:** Keying endpoint state by pod IP alone collides data-parallel worker ranks sharing one pod network namespace.
- **Examples:** Observed in 13 PRs: [#841](https://github.com/llm-d/llm-d-router/issues/841), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242332547), [#2831](https://github.com/llm-d/llm-d-router/pull/2831#discussion_r4033597903)

### Order disaggregated stages and isolate stage-specific payloads

Execute disaggregated encode and prefill stages before dependent decode stages, keep prefill-only payload mutations out of decode requests, and cancel downstream stages immediately when an upstream stage fails.

- **Why it matters:** Leaking prefill-only fields into decode payloads or continuing after upstream stage failure wastes GPU decode capacity on broken requests.
- **Examples:** Observed in 29 PRs: [#1444](https://github.com/llm-d/llm-d-router/pull/1444#discussion_r3346561765), [#1653](https://github.com/llm-d/llm-d-router/pull/1653#discussion_r3412845630), [#2094](https://github.com/llm-d/llm-d-router/pull/2094#discussion_r3730268179)

## Performance on the request path

Zero avoidable hot-path allocations, pooled transports and multiplexed event sockets, orthogonal load signals, contiguous prefix-cache scoring, and reproducible benchmarks.

### Avoid heap allocations and unhoisted work on the request path

Avoid unnecessary byte-to-string conversions, slice or map clones, untyped map unmarshaling, and unmutated payload re-serialization on per-request routing, scoring, and event-decoding paths. Hoist header normalization, lookup maps, and type checks to startup, pre-allocate known-length slices, and combine consecutive read-locked lookups into one pass.

- **Why it matters:** Per-request heap allocations and repeated map lookups inflate GC pause pressure and router p99 latency under high QPS.
- **Examples:** Observed in 85 PRs: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143174078), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3367176286), [#880](https://github.com/llm-d/llm-d-router/pull/880#discussion_r3149083076)

### Reuse pooled HTTP transports and multiplex per-endpoint event subscriptions

Reuse HTTP clients with keep-alive connection pooling across backend calls and metric scrapes, rebuilding transports only on TLS certificate rotation. Multiplex per-endpoint ZMQ or event streams across in-process consumers instead of opening duplicate sockets per plugin, and batch background sweeps off the request path.

- **Why it matters:** Per-scrape HTTP transports and duplicate ZMQ sockets exhaust ephemeral ports and add handshake overhead.
- **Examples:** Observed in 4 PRs: [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3224041877), [#2045](https://github.com/llm-d/llm-d-router/issues/2045), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r4095311906)

### Evaluate orthogonal load signals over filtered candidate subsets

Evaluate queue depth, in-flight requests, and KV-cache utilization as distinct signals over the filtered candidate subset, and decrement every in-flight counter on cancellation or error.

- **Why it matters:** Scoring unfiltered endpoints wastes CPU and skews normalization, while leaked in-flight increments permanently starve healthy pods.
- **Reviewers accept:** Constant-time byte-length token estimation and single-endpoint soft-pinning session affinity when backed by engine contracts or multi-pod benchmarks ([#1438](https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3338979029), [#2968](https://github.com/llm-d/llm-d-router/pull/2968#discussion_r4214852975)).
- **Examples:** Observed in 32 PRs: [#1534](https://github.com/llm-d/llm-d-router/pull/1534#discussion_r3401170547), [#1580](https://github.com/llm-d/llm-d-router/pull/1580#discussion_r3391563214), [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3643867571)

### Score prefix caches contiguously and break near-ties by load

Score prefix caches contiguously from the start of each sequence by stopping block credit at the first missing block, salt cache keys by tenant, and break near-ties with load-weighted sampling.

- **Why it matters:** Crediting non-contiguous blocks overestimates KV-cache reuse, and deterministic tie-breaking herds shared-prefix traffic onto one pod.
- **Examples:** Observed in 33 PRs: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3354232991), [#2031](https://github.com/llm-d/llm-d-router/issues/2031), [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3610603777)

### Gate routing policies and hot-path changes on reproducible before-and-after benchmarks

Back new scoring formulas, cache indexes, hot-path locks, and default threshold changes with reproducible hit-and-miss microbenchmarks and cluster comparisons against active routing baselines. Commit benchmark scripts and summary reports while excluding raw execution logs from git.

- **Why it matters:** Unbenchmarked scoring or lock changes regress throughput or tail latency on cache-miss workloads without detection.
- **Examples:** Observed in 43 PRs: [#655](https://github.com/llm-d/llm-d-router/issues/655), [#1006](https://github.com/llm-d/llm-d-router/issues/1006), [#3068](https://github.com/llm-d/llm-d-router/pull/3068#issuecomment-5930096809)

## Observability

Canonical metric subsystems and bounded label cardinality, single terminal request outcomes, package-scoped OpenTelemetry spans, and bounded debug state.

### Register metrics under canonical component subsystems with bounded label cardinality

Register metrics under canonical subsystem constants (`LLMDRouterEndpointPickerSubsystem` / `"llm_d_epp"`, `LLMDRouterCoordinatorSubsystem` / `"llm_d_coordinator"`, or `llm_d_disagg_sidecar_`), label plugin metrics with `plugin_name` and `plugin_type`, and emit series only for configured plugins. Bound caller-influenced or pod-churn labels via closed enums, allowlists, or overflow buckets, expose live plugin gauges via scrape-time collectors, and delete series when endpoints are removed.

- **Why it matters:** Unbounded label values or orphaned pod series explode Prometheus cardinality and break shared dashboard queries.
- **Reviewers accept:** Per-request ratio histograms alongside raw query and hit counters when the precomputed distribution simplifies alerting ([#1429](https://github.com/llm-d/llm-d-router/pull/1429#pullrequestreview-4493962366)).
- **Settled decision:** [Disaggregation sidecar serves /metrics on its existing HTTP listener mux](review/decisions/sidecar_metrics_on_shared_http_mux.md).
- **Open question:** Whether per-tenant, per-workload, and per-user cost attribution belongs on Prometheus labels that operators opt into ([#2834](https://github.com/llm-d/llm-d-router/pull/2834)) or only in structured logs and traces because header-derived values have no closed set ([#2834](https://github.com/llm-d/llm-d-router/pull/2834#discussion_r4032724626)) ([details](review/open_questions/cost_attribution_prometheus_labels.md)).
- **Examples:** Observed in 60 PRs: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190768619), [#793](https://github.com/llm-d/llm-d-router/pull/793#discussion_r3324344552), [#1089](https://github.com/llm-d/llm-d-router/issues/1089#issuecomment-4703238704)

### Record at most one terminal outcome per request and keep latency histograms unbiased

Record at most one terminal outcome per request across early-return, buffered, and error paths, matching histogram bucket units to emitted values. Separate router overhead from backend inference wait, and skip stage duration observations when a stage is bypassed.

- **Why it matters:** Double-recording error paths inflates failure rates, and recording bypassed zero-cost stages skews latency histograms downward.
- **Examples:** Observed in 37 PRs: [#1553](https://github.com/llm-d/llm-d-router/issues/1553), [#1658](https://github.com/llm-d/llm-d-router/pull/1658#discussion_r3618566988), [#1677](https://github.com/llm-d/llm-d-router/pull/1677#discussion_r3852749939)

### Acquire package-scoped OTel tracers and propagate spans end to end

Acquire OpenTelemetry tracers via `tracing.Tracer(TracerScope)` with a package-scoped `TracerScope` constant (`llm-d.ai/...`), standard semantic-convention attribute keys, and route templates rather than raw URLs. Extract inbound W3C trace context at ingress and propagate spans across all outbound HTTP and gRPC calls.

- **Why it matters:** Dropping W3C trace headers or placing raw URLs in span names breaks distributed traces across stages and explodes trace cardinality.
- **Settled decision:** [Ratio-sampled per-plugin OpenTelemetry spans with high-overhead KV-event spans gated off](review/decisions/sampled_per_plugin_otel_spans_and_opt_in_kv_event_spans.md).
- **Examples:** Observed in 37 PRs: [#708](https://github.com/llm-d/llm-d-router/issues/708), [#922](https://github.com/llm-d/llm-d-router/pull/922#discussion_r3178210848), [#1513](https://github.com/llm-d/llm-d-router/pull/1513#discussion_r3369205741)

### Summarize collections in String() and cap stateful plugin dumps

Summarize collections by count in `String()` on hot paths, and implement `DumpState` only on stateful plugins with sanitized entries capped to fixed top-K bounds.

- **Why it matters:** Formatting full collections on the request path allocates per item even at low log verbosity, and uncapped state dumps stall debug endpoints.
- **Reviewers accept:** Omitting debug logs on routine plugin no-op branches when sibling plugins stay silent and proxy trace headers already record the decision ([#2319](https://github.com/llm-d/llm-d-router/pull/2319#issuecomment-5265371475)).
- **Settled decision:** [Once-per-request operational logs use V(logging.DEFAULT)](review/decisions/once_per_request_logs_default_verbosity.md).
- **Examples:** Observed in 57 PRs: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3189062435), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252750496), [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178428969)

## Testing and CI

Production factory tests with exact output assertions, colocated sleep-free unit tests, and namespace-isolated E2E teardown.

### Exercise production factories and assert exact behavioral outputs

Instantiate plugins and runners in tests via production factories and `fwkplugin.StrictDecoder` rather than literal internal structs. Assert exact output fields, error types, metric values, and JSON wire payloads without broad `cmpopts.IgnoreFields`, covering both states of feature gates, skipped disaggregated stages, and boundary inputs.

- **Why it matters:** Constructing internal structs directly in tests bypasses factory defaults and strict config decoding, letting broken wiring pass CI.
- **Reviewers accept:** End-to-end flow tests over unreachable unit combinations, and calling pure parsers directly in table-driven unit tests separate from env-var wrappers ([#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524933896), [#1937](https://github.com/llm-d/llm-d-router/pull/1937#discussion_r3552704054)).
- **Examples:** Observed in 150 PRs: [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3215017290), [#972](https://github.com/llm-d/llm-d-router/pull/972#pullrequestreview-4216545185), [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3226190634)

### Colocate unit tests and synchronize without fixed sleeps

Keep unit tests in colocated `*_test.go` files moved alongside production types, and place shared mocks or builders in cycle-free leaf test packages. Synchronize async startup and reconciles with injectable clocks, barriers, or `require.Eventually` instead of `time.Sleep`, and bind ephemeral loopback listeners atomically without close-then-rebind races.

- **Why it matters:** Fixed `time.Sleep` calls and close-then-rebind port helpers cause flaky unit tests under CI CPU contention.
- **Reviewers accept:** Calibrated sleep windows when testing refresh-on-read cache TTL expiration where polling the read method would reset the TTL ([#1396](https://github.com/llm-d/llm-d-router/pull/1396#discussion_r3369095849)).
- **Examples:** Observed in 47 PRs: [#1066](https://github.com/llm-d/llm-d-router/issues/1066), [#1188](https://github.com/llm-d/llm-d-router/issues/1188), [#1528](https://github.com/llm-d/llm-d-router/pull/1528#discussion_r3369900484)

### Isolate parallel end-to-end suites with non-aborting teardown

Run E2E suites against freshly built PR images and shipping Helm charts, isolating parallel runs by namespace and port (per-process NodePorts on kind; supervised `kubectl port-forward` on shared clusters). Record created cluster resources before readiness waits, run non-aborting cleanup across all resources on teardown, and dump pod logs only on failure.

- **Why it matters:** Shared E2E namespaces or early-aborting teardown scripts collide parallel CI jobs and strand leaked cluster resources.
- **Reviewers accept:** Separate E2E runner scripts backed by shared shell or Go helpers when cluster ownership, image sets, and environment contracts differ ([#1698](https://github.com/llm-d/llm-d-router/pull/1698#issuecomment-4762216995)).
- **Examples:** Observed in 15 PRs: [#1048](https://github.com/llm-d/llm-d-router/pull/1048#pullrequestreview-4310234203), [#1481](https://github.com/llm-d/llm-d-router/pull/1481#discussion_r3372638867), [#1976](https://github.com/llm-d/llm-d-router/pull/1976#discussion_r3568707788)

## Security and trust boundaries

Header namespacing and ingress/egress scrubbing, InferencePool SSRF target allowlists, TLS minimums, namespace-scoped watches, and sensitive payload redaction.

### Namespace new router headers under `x-llm-d-` and scrub internal headers at ingress and egress

Prefix new router-owned HTTP headers with lowercase `x-llm-d-`, suffix host-and-port headers with `-host-port`, and normalize inbound header lookups to lowercase at ingress (`x-gateway-destination-endpoint`, `x-data-parallel-host-port`, and `x-session-token` keep their names). Strip client-supplied internal routing headers, deprecated aliases, and untrusted transfer fields across all routes at ingress, and scrub internal pod topology headers before returning responses.

- **Why it matters:** Unscrubbed client routing headers allow external callers to spoof internal endpoint selection or leak pod topology on responses.
- **Reviewers accept:** Passing allowlist-validated `x-request-id` correlation headers verbatim, documenting body-over-header precedence, and scrubbing internal headers on Envoy fallback routes via `request_headers_to_remove` when early stripping would hide `ScreeningHeaders` from EPP ([#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524966490), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955566592), [#3433](https://github.com/llm-d/llm-d-router/pull/3433#discussion_r4274645712)).
- **Open question:** Whether session affinity should be carried by a standard `Set-Cookie` response ([#28](https://github.com/llm-d/llm-d-router/issues/28)) or by a request header such as `x-session-token` ([#600](https://github.com/llm-d/llm-d-router/pull/600#discussion_r3156539950)) ([details](review/open_questions/session_affinity_cookie_vs_header.md)).
- **Examples:** Observed in 34 PRs: [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560), [#1027](https://github.com/llm-d/llm-d-router/pull/1027#discussion_r3219971477), [#2319](https://github.com/llm-d/llm-d-router/pull/2319#discussion_r3758140767)

### Allowlist dynamic endpoint targets and reject control characters

Validate peer endpoint URLs and KV-cache transfer targets against `InferencePool` host-and-port allowlists across all target ports and data-parallel ranks, blocking self-pulls and unverified cluster addresses. Reject whitespace and CR/LF control characters in header-derived targets.

- **Why it matters:** Dialing unverified header-derived targets or permitting CR/LF characters exposes decode and EPP pods to SSRF and HTTP request smuggling.
- **Examples:** Observed in 61 PRs: [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560), [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771), [#1857](https://github.com/llm-d/llm-d-router/pull/1857#discussion_r3508246239)

### Enforce TLS minimums, namespace-scoped watches, and payload redaction

Enforce minimum TLS versions and header-read timeouts on HTTP servers, trim trailing newlines from mounted secret files, and scope Kubernetes informers to the configured namespace rather than cluster-wide watches. Never log raw prompts, base64 media payloads, or end-user identifiers at default or debug verbosity.

- **Why it matters:** Missing read timeouts expose listeners to slow-header DoS, cluster-wide watches over-privilege service accounts, and unredacted logs leak user prompts.
- **Open question:** Whether `DumpState` output hashes tenant and program identifiers by default ([#2008](https://github.com/llm-d/llm-d-router/pull/2008#pullrequestreview-4759516825)) or emits them in cleartext because hashed identifiers carry little debugging value ([#2008](https://github.com/llm-d/llm-d-router/pull/2008#issuecomment-5073212799)) ([details](review/open_questions/dump_state_identifier_hashing.md)).
- **Examples:** Observed in 26 PRs: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628868), [#1350](https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310324692), [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473483)

## Documentation and comments

Colocated plugin READMEs, upstream guide linking, and review-round documentation synchronization.

### Colocate plugin READMEs and link to canonical upstream guides

Document each plugin's parameter schema, code-verified defaults, and minimal YAML example in a colocated `README.md` inside its package under `pkg/epp/framework/plugins`. Link to upstream `llm-d/llm-d` architecture guides rather than duplicating them, using file or symbol links without source line numbers.

- **Why it matters:** Scattered plugin docs or hardcoded source line links drift from code defaults and mislead operators.
- **Settled decision:** [Component metrics reference in llm-d-router paired with user guides in llm-d/llm-d](review/decisions/local_metrics_docs_paired_with_umbrella_user_guides.md).
- **Examples:** Observed in 167 PRs: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4398304029), [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143455213), [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3177858398)

### Synchronize docstrings, diagrams, and configuration examples across review rounds

Update affected package docstrings, architecture diagrams, and YAML configuration examples whenever code or types change during review or rebase, and omit hardcoded plugin or feature counts.

- **Why it matters:** Review-round renames that leave stale YAML examples or docstrings cause copy-paste configuration failures.
- **Examples:** Observed in 55 PRs: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236365197), [#933](https://github.com/llm-d/llm-d-router/pull/933#issuecomment-4864629122), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242601033)
