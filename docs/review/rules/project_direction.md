---
type: Review Rule Set
title: Project direction
description: "Architectural boundaries across built-in runtimes, upstream signal producers, engine adapters, orchestrator-neutral endpoints, pool homogeneity, coordinator vs sidecar, Helm packaging, and upstream CRDs."
tags: ["project-direction", direction, "mvm-07", "mvm-16", "unr-13", "mvm-06", "nvc-03", "unr-12", "mvm-22", "nvd-01", "mvm-08", "mvm-09", "nvc-14", "nvd-04", "nvc-10", "unr-03"]
status: stable
stale_after: "2027-04-26T00:00:00Z"
sources:
  - id: "issue-818"
    resource: "https://github.com/llm-d/llm-d-router/issues/818"
    author: "human:ahg-g"
    last_modified: "2026-04-13T15:51:07Z"
  - id: "issue-1139-ic-4601790080"
    resource: "https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080"
    author: "human:elevran"
    last_modified: "2026-06-02T11:11:13Z"
  - id: "issue-1450-ic-4604976814"
    resource: "https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814"
    author: "human:liu-cong"
    last_modified: "2026-06-02T17:14:08Z"
  - id: "pr-1904-ic-4898714192"
    resource: "https://github.com/llm-d/llm-d-router/pull/1904#issuecomment-4898714192"
    author: "human:liu-cong"
    last_modified: "2026-07-06T23:59:43Z"
  - id: "pr-1921-r3587753917"
    resource: "https://github.com/llm-d/llm-d-router/pull/1921#discussion_r3587753917"
    author: "human:ahg-g"
    last_modified: "2026-07-15T13:42:52Z"
  - id: "pr-2308-r3740526294"
    resource: "https://github.com/llm-d/llm-d-router/pull/2308#discussion_r3740526294"
    author: "human:ahg-g"
    last_modified: "2026-08-08T10:48:32Z"
  - id: "pr-2308-rev-5453802061"
    resource: "https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061"
    author: "human:elevran"
    last_modified: "2026-10-08T09:12:13Z"
  - id: "issue-850"
    resource: "https://github.com/llm-d/llm-d-router/issues/850"
    author: "human:vMaroon"
    last_modified: "2026-04-20T11:00:58Z"
  - id: "pr-860-r3279332668"
    resource: "https://github.com/llm-d/llm-d-router/pull/860#discussion_r3279332668"
    author: "human:ahg-g"
    last_modified: "2026-05-21T07:16:29Z"
  - id: "issue-970"
    resource: "https://github.com/llm-d/llm-d-router/issues/970"
    author: "human:ahg-g"
    last_modified: "2026-05-03T12:02:52Z"
  - id: "issue-1364-ic-4587275492"
    resource: "https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492"
    author: "human:elevran"
    last_modified: "2026-05-31T16:17:22Z"
  - id: "issue-1634-ic-5942141855"
    resource: "https://github.com/llm-d/llm-d-router/issues/1634#issuecomment-5942141855"
    author: "human:LukeAVanDrie"
    last_modified: "2026-10-01T22:41:53Z"
  - id: "pr-2963-ic-5811157439"
    resource: "https://github.com/llm-d/llm-d-router/pull/2963#issuecomment-5811157439"
    author: "human:vMaroon"
    last_modified: "2026-09-24T09:04:11Z"
  - id: "pr-1896-ic-4924616643"
    resource: "https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643"
    author: "human:vMaroon"
    last_modified: "2026-07-09T11:33:43Z"
  - id: "pr-1896-rev-4657668118"
    resource: "https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118"
    author: "human:sagearc"
    last_modified: "2026-07-08T20:25:23Z"
  - id: "issue-382-ic-4585925394"
    resource: "https://github.com/llm-d/llm-d-router/issues/382#issuecomment-4585925394"
    author: "human:elevran"
    last_modified: "2026-05-31T06:28:42Z"
  - id: "issue-2447-ic-5342388312"
    resource: "https://github.com/llm-d/llm-d-router/issues/2447#issuecomment-5342388312"
    author: "human:elevran"
    last_modified: "2026-08-19T12:54:13Z"
  - id: "issue-3030"
    resource: "https://github.com/llm-d/llm-d-router/issues/3030"
    author: "human:sagearc"
    last_modified: "2026-09-24T12:18:01Z"
  - id: "pr-3359-r4361817276"
    resource: "https://github.com/llm-d/llm-d-router/pull/3359#discussion_r4361817276"
    author: "human:liu-cong"
    last_modified: "2026-10-23T16:26:24Z"
  - id: "pr-1007-r3203628879"
    resource: "https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628879"
    author: "human:elevran"
    last_modified: "2026-05-07T17:55:06Z"
  - id: "pr-1112-r3271206173"
    resource: "https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3271206173"
    author: "human:ahg-g"
    last_modified: "2026-05-20T04:24:25Z"
  - id: "pr-1148-r3294724682"
    resource: "https://github.com/llm-d/llm-d-router/pull/1148#discussion_r3294724682"
    author: "human:elevran"
    last_modified: "2026-05-24T13:32:29Z"
  - id: "pr-1106-r3273288488"
    resource: "https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3273288488"
    author: "human:elevran"
    last_modified: "2026-05-20T10:50:25Z"
  - id: "pr-2147-rev-4766963622"
    resource: "https://github.com/llm-d/llm-d-router/pull/2147#pullrequestreview-4766963622"
    author: "human:elevran"
    last_modified: "2026-07-23T18:26:04Z"
  - id: "issue-1878-ic-4864624197"
    resource: "https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197"
    author: "human:shmuelk"
    last_modified: "2026-07-02T10:16:39Z"
  - id: "pr-2333-ic-5935642884"
    resource: "https://github.com/llm-d/llm-d-router/pull/2333#issuecomment-5935642884"
    author: "human:ahg-g"
    last_modified: "2026-10-01T16:20:46Z"
  - id: "pr-2692-ic-5626620151"
    resource: "https://github.com/llm-d/llm-d-router/pull/2692#issuecomment-5626620151"
    author: "human:zetxqx"
    last_modified: "2026-09-10T23:06:02Z"
  - id: "pr-2308-rev-5483618104"
    resource: "https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5483618104"
    author: "human:ahg-g"
    last_modified: "2026-10-11T14:20:56Z"
  - id: "issue-896-ic-5923079423"
    resource: "https://github.com/llm-d/llm-d-router/issues/896#issuecomment-5923079423"
    author: "human:liu-cong"
    last_modified: "2026-10-01T01:47:27Z"
  - id: "issue-1104-ic-4449848790"
    resource: "https://github.com/llm-d/llm-d-router/issues/1104#issuecomment-4449848790"
    author: "human:elevran"
    last_modified: "2026-05-14T10:29:32Z"
  - id: "pr-1145-ic-4489027953"
    resource: "https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953"
    author: "human:ahg-g"
    last_modified: "2026-05-19T14:55:39Z"
  - id: "issue-1979"
    resource: "https://github.com/llm-d/llm-d-router/issues/1979"
    author: "human:vMaroon"
    last_modified: "2026-07-12T21:36:30Z"
  - id: "pr-2950-r4070533796"
    resource: "https://github.com/llm-d/llm-d-router/pull/2950#discussion_r4070533796"
    author: "human:elevran"
    last_modified: "2026-09-22T10:13:45Z"
  - id: "issue-1892"
    resource: "https://github.com/llm-d/llm-d-router/issues/1892"
    author: "human:vMaroon"
    last_modified: "2026-07-04T21:06:11Z"
  - id: "pr-2531-r4086142755"
    resource: "https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755"
    author: "human:ahg-g"
    last_modified: "2026-09-23T18:57:36Z"
  - id: "issue-634"
    resource: "https://github.com/llm-d/llm-d-router/issues/634"
    author: "human:elevran"
    last_modified: "2026-02-19T17:30:46Z"
  - id: "pr-930-ic-4366069128"
    resource: "https://github.com/llm-d/llm-d-router/pull/930#issuecomment-4366069128"
    author: "human:ahg-g"
    last_modified: "2026-05-03T11:34:24Z"
  - id: "pr-1550-ic-4656209920"
    resource: "https://github.com/llm-d/llm-d-router/pull/1550#issuecomment-4656209920"
    author: "human:ahg-g"
    last_modified: "2026-06-09T04:43:18Z"
  - id: "pr-1601-ic-4770357895"
    resource: "https://github.com/llm-d/llm-d-router/pull/1601#issuecomment-4770357895"
    author: "human:ahg-g"
    last_modified: "2026-06-22T16:06:28Z"
  - id: "issue-1177-ic-4478503514"
    resource: "https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514"
    author: "human:shmuelk"
    last_modified: "2026-05-18T14:08:19Z"
  - id: "pr-2048-r3596786577"
    resource: "https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577"
    author: "human:shmuelk"
    last_modified: "2026-07-16T15:28:50Z"
  - id: "issue-968-ic-4602495714"
    resource: "https://github.com/llm-d/llm-d-router/issues/968#issuecomment-4602495714"
    author: "human:elevran"
    last_modified: "2026-06-02T12:44:42Z"
  - id: "pr-1035-r3244818645"
    resource: "https://github.com/llm-d/llm-d-router/pull/1035#discussion_r3244818645"
    author: "human:zetxqx"
    last_modified: "2026-05-14T23:08:33Z"
  - id: "pr-1043-r3279489618"
    resource: "https://github.com/llm-d/llm-d-router/pull/1043#discussion_r3279489618"
    author: "human:ahg-g"
    last_modified: "2026-05-21T07:46:34Z"
  - id: "pr-1482-ic-4633716490"
    resource: "https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490"
    author: "human:liu-cong"
    last_modified: "2026-06-05T16:59:52Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Project direction

Architectural boundaries across built-in runtimes, upstream signal producers, engine adapters, orchestrator-neutral endpoints, pool homogeneity, coordinator vs sidecar, Helm packaging, and upstream CRDs.

## Keep core runtimes built in and policy plugins optional

### Rule

Core execution runtimes and cross-replica state synchronization orchestration belong in built-in runtime packages, whereas `pkg/epp/framework/plugins` holds optional policy units that operators enable or disable via configuration (see [docs/plugins-vs-builtins.md](../../plugins-vs-builtins.md)). Route scalar telemetry through generic attribute-driven filters and scorers, share lightweight math via plain Go helpers, and require multiple concrete use cases before adding a new framework extension point.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod` (referenced symbols: `pkg/epp/framework/plugins`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Related decisions**: [Scorer-internal fallback over scheduler fallback profiles](../decisions/scorer_internal_fallback_over_fallback_profiles.md), [Pre-admission header attribute extraction via dedicated plugin extension point](../decisions/request_header_extractor_before_admission.md).

### Reviewers accept

A new extension point with one initial caller when the alternative is runtime code that references a concrete plugin.[^pr-2308-r3740526294][^pr-2308-rev-5453802061]

### Evidence

Distilled from 3 review clusters (21 pull requests / 8 maintainers, 1 pull requests / 3 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-818][^issue-1139-ic-4601790080][^issue-1450-ic-4604976814][^pr-1904-ic-4898714192][^pr-1921-r3587753917]

## Extract scheduling signals in upstream producers and pass native payloads through

### Rule

Run tokenization, multimodal extraction, and token estimation in canonical upstream pre-scheduling data producers (`token-producer`) that populate request attributes, keeping downstream filters and scorers free of wire-format parsing. Forward client payloads to the engine's native render endpoints (`/v1/responses/render`, `/v1/messages/render`) instead of translating one API schema into another inside the router; the deprecated `legacy` render mode keeps a chat conversion for engines without the native endpoint. Order pre-admission flow-control queues by immutable request attributes fixed at admission rather than mutable endpoint state.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod` (referenced symbols: `/v1/responses/render`, `/v1/messages/render`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.

### Reviewers accept

A contained interim translation layer for an API the engine cannot render natively.[^pr-1896-ic-4924616643][^pr-1896-rev-4657668118]

### Evidence

Distilled from 3 review clusters (10 pull requests / 6 maintainers, 12 pull requests / 10 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-850][^pr-860-r3279332668][^issue-970][^issue-1364-ic-4587275492][^issue-1634-ic-5942141855][^pr-2963-ic-5811157439]

## Configure backend engine identity explicitly and isolate engine schemas

### Rule

Configure backend inference-engine identity explicitly at deployment time rather than inferring engine type from unrelated connector flags. Isolate engine-specific wire schemas, CLI flags, and metric names behind declarative adapters and per-plugin documentation instead of scattering engine conditionals across shared routing packages, and close router pull requests blocked on stalled unmerged upstream engine changes.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Conflict audit references**: `UNR-13`.

### Evidence

Distilled from 2 review clusters (10 pull requests / 5 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-382-ic-4585925394][^issue-2447-ic-5342388312][^issue-3030][^pr-3359-r4361817276]

## Model endpoints with orchestrator-neutral domain types

### Rule

Model new discovery, server registration, metric-extraction, and `datalayer` interfaces around orchestrator-neutral `Endpoint` entities rather than Kubernetes `Pod` structs or controller managers so routing operates across both Kubernetes and standalone deployments. The datastore registration path and the controllers that consume `Pod` objects are the existing Kubernetes-specific seam; do not widen it.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Related decisions**: [Core endpoint identity on EndpointMetadata and optional topology in AttributeMap](../decisions/endpoint_identity_metadata_vs_optional_attributes.md).

### Reviewers accept

Identity fields on `EndpointMetadata` with optional per-endpoint data such as rank carried in the `AttributeMap`.[^pr-1106-r3273288488][^pr-2147-rev-4766963622]

### Open question

See [Non-Kubernetes InferenceObjective file schema](../open_questions/non_kubernetes_objectives_schema.md).

### Evidence

Observed across 7 pull requests and 2 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^pr-1007-r3203628879][^pr-1112-r3271206173][^pr-1148-r3294724682]

## Scope each pool to a single homogeneous base model and role

### Rule

Scope each `InferencePool` and EPP process to homogeneous replicas serving a single base model and disaggregated role so every endpoint contributes to shared pool capacity, and route distinct base models through Gateway `HTTPRoute` rules to separate pools.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod`; see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Related decisions**: [Pool-wide /v1/models aggregation via generic Responder plugin interface](../decisions/models_responder_generic_extension_point.md).

### Reviewers accept

Answering `GET /v1/models` from EPP through the `models-responder` plugin, which unions the model lists reported by the pool's endpoints so adapters loaded on a subset of endpoints stay visible.[^pr-2308-rev-5453802061][^pr-2308-rev-5483618104]

### Evidence

Distilled from 2 review clusters (6 pull requests / 3 maintainers, 1 pull requests / 1 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-1878-ic-4864624197][^pr-2333-ic-5935642884][^pr-2692-ic-5626620151]

## Keep client conversation state out of the router and orchestrate multi-stage pipelines in the coordinator

### Rule

Keep conversation persistence, intra-backend rank balancing, and offline calibration in external layers; the router rejects `/v1/responses` fields that reference state it does not hold (`previous_response_id`, `conversation`) with an explicit error instead of dropping them. Router-resident routing state (session registries, per-pod session ledgers, cross-replica synced load and cache levels) follows the capacity and eviction rules under Concurrency and shared state. Place cluster-level multi-stage disaggregated orchestration in `pkg/coordinator` above endpoint selection, and keep `pkg/sidecar` scoped to the pod-local proxy whose only Kubernetes API access is the optional namespace-scoped `InferencePool` and `Pod` watch behind `--enable-ssrf-protection` (see [docs/architecture.md](../../architecture.md), [docs/disaggregation.md](../../disaggregation.md), and [docs/coordinator_architecture.md](../../coordinator_architecture.md#coordinator-vs-the-llm-d-router-sidecar-model)).

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod` (referenced symbols: `/v1/responses`, `pkg/coordinator`, `pkg/sidecar`, `--enable-ssrf-protection`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Conflict audit references**: `MVM-08`.

### Reviewers accept

Bounded in-memory session registries and cross-replica synced routing state in the router when client payloads and conversation history stay external.[^issue-1892][^pr-2531-r4086142755]

### Open question

See [Disaggregation sidecar Kubernetes watches for SSRF allowlists](../open_questions/sidecar_ssrf_allowlist_watches.md).

### Evidence

Distilled from 3 review clusters (12 pull requests / 10 maintainers, 2 pull requests / 3 maintainers, 44 pull requests / 12 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-896-ic-5923079423][^issue-1104-ic-4449848790][^pr-1145-ic-4489027953][^issue-1979][^pr-2950-r4070533796]

## Deploy through Helm charts and configure components via structured files

### Rule

Ship end-user deployments through lightweight Helm charts under `config/charts` and align configuration schemas hierarchically with top-level components in structured configuration files instead of expanding binary CLI flags. Publish multi-component operations guides and umbrella dashboards in `llm-d/llm-d`, and keep the Kustomize manifests under `deploy` scoped to local development environments and E2E suites.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod` (referenced symbols: `config/charts`, `llm-d/llm-d`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Related decisions**: [Helm charts for end-user releases and deploy Kustomize manifests for local kind and E2E](../decisions/helm_release_charts_with_dev_kustomize_manifests.md).

### Reviewers accept

Updates to the manifests under `deploy` that keep local kind environments and E2E suites running.[^issue-1177-ic-4478503514][^pr-2048-r3596786577]

### Evidence

Distilled from 2 review clusters (16 pull requests / 9 maintainers, 14 pull requests / 7 maintainers) in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-634][^pr-930-ic-4366069128][^pr-1550-ic-4656209920][^pr-1601-ic-4770357895]

## Consume versioned upstream CRDs and protobufs without vendoring

### Rule

Consume upstream Gateway API Inference Extension CRDs and protobuf definitions from versioned releases or the pinned Go module cache instead of vendoring local copies in the repository. The Istio CRDs under `deploy/components/crds-istio` are the checked-in exception for the local development environment.

### Scope and consequence

- **Applies to**: `cmd/**`, `pkg/epp/**`, `pkg/coordinator/**`, `pkg/sidecar/**`, `config/charts/**`, `go.mod` (referenced symbols: `deploy/components/crds-istio`); see [Failure classes](../references/failure_classes.md).
- **Consequence if violated**: Placing state or orchestration in the wrong component boundary forces cross-component rewrites and fragments operator deployment surfaces.
- **Conflict audit references**: `NVC-10`, `UNR-03`.

### Open question

See [Remote fetch vs repository cache for upstream test CRDs](../open_questions/upstream_test_crd_caching.md).

### Evidence

Observed across 7 pull requests and 6 maintainers in the 2026-04-26 to 2026-10-26 review corpus.

Representative review threads:[^issue-968-ic-4602495714][^pr-1035-r3244818645][^pr-1043-r3279489618][^pr-1482-ic-4633716490]

[^issue-818]: [#818](https://github.com/llm-d/llm-d-router/issues/818) (human:ahg-g)
[^issue-1139-ic-4601790080]: [#1139](https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080) (human:elevran)
[^issue-1450-ic-4604976814]: [#1450](https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814) (human:liu-cong)
[^pr-1904-ic-4898714192]: [#1904](https://github.com/llm-d/llm-d-router/pull/1904#issuecomment-4898714192) (human:liu-cong)
[^pr-1921-r3587753917]: [#1921](https://github.com/llm-d/llm-d-router/pull/1921#discussion_r3587753917) (human:ahg-g)
[^pr-2308-r3740526294]: [#2308](https://github.com/llm-d/llm-d-router/pull/2308#discussion_r3740526294) (human:ahg-g)
[^pr-2308-rev-5453802061]: [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061) (human:elevran)
[^issue-850]: [#850](https://github.com/llm-d/llm-d-router/issues/850) (human:vMaroon)
[^pr-860-r3279332668]: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3279332668) (human:ahg-g)
[^issue-970]: [#970](https://github.com/llm-d/llm-d-router/issues/970) (human:ahg-g)
[^issue-1364-ic-4587275492]: [#1364](https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492) (human:elevran)
[^issue-1634-ic-5942141855]: [#1634](https://github.com/llm-d/llm-d-router/issues/1634#issuecomment-5942141855) (human:LukeAVanDrie)
[^pr-2963-ic-5811157439]: [#2963](https://github.com/llm-d/llm-d-router/pull/2963#issuecomment-5811157439) (human:vMaroon)
[^pr-1896-ic-4924616643]: [#1896](https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643) (human:vMaroon)
[^pr-1896-rev-4657668118]: [#1896](https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118) (human:sagearc)
[^issue-382-ic-4585925394]: [#382](https://github.com/llm-d/llm-d-router/issues/382#issuecomment-4585925394) (human:elevran)
[^issue-2447-ic-5342388312]: [#2447](https://github.com/llm-d/llm-d-router/issues/2447#issuecomment-5342388312) (human:elevran)
[^issue-3030]: [#3030](https://github.com/llm-d/llm-d-router/issues/3030) (human:sagearc)
[^pr-3359-r4361817276]: [#3359](https://github.com/llm-d/llm-d-router/pull/3359#discussion_r4361817276) (human:liu-cong)
[^pr-1007-r3203628879]: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628879) (human:elevran)
[^pr-1112-r3271206173]: [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3271206173) (human:ahg-g)
[^pr-1148-r3294724682]: [#1148](https://github.com/llm-d/llm-d-router/pull/1148#discussion_r3294724682) (human:elevran)
[^pr-1106-r3273288488]: [#1106](https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3273288488) (human:elevran)
[^pr-2147-rev-4766963622]: [#2147](https://github.com/llm-d/llm-d-router/pull/2147#pullrequestreview-4766963622) (human:elevran)
[^issue-1878-ic-4864624197]: [#1878](https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197) (human:shmuelk)
[^pr-2333-ic-5935642884]: [#2333](https://github.com/llm-d/llm-d-router/pull/2333#issuecomment-5935642884) (human:ahg-g)
[^pr-2692-ic-5626620151]: [#2692](https://github.com/llm-d/llm-d-router/pull/2692#issuecomment-5626620151) (human:zetxqx)
[^pr-2308-rev-5483618104]: [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5483618104) (human:ahg-g)
[^issue-896-ic-5923079423]: [#896](https://github.com/llm-d/llm-d-router/issues/896#issuecomment-5923079423) (human:liu-cong)
[^issue-1104-ic-4449848790]: [#1104](https://github.com/llm-d/llm-d-router/issues/1104#issuecomment-4449848790) (human:elevran)
[^pr-1145-ic-4489027953]: [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953) (human:ahg-g)
[^issue-1979]: [#1979](https://github.com/llm-d/llm-d-router/issues/1979) (human:vMaroon)
[^pr-2950-r4070533796]: [#2950](https://github.com/llm-d/llm-d-router/pull/2950#discussion_r4070533796) (human:elevran)
[^issue-1892]: [#1892](https://github.com/llm-d/llm-d-router/issues/1892) (human:vMaroon)
[^pr-2531-r4086142755]: [#2531](https://github.com/llm-d/llm-d-router/pull/2531#discussion_r4086142755) (human:ahg-g)
[^issue-634]: [#634](https://github.com/llm-d/llm-d-router/issues/634) (human:elevran)
[^pr-930-ic-4366069128]: [#930](https://github.com/llm-d/llm-d-router/pull/930#issuecomment-4366069128) (human:ahg-g)
[^pr-1550-ic-4656209920]: [#1550](https://github.com/llm-d/llm-d-router/pull/1550#issuecomment-4656209920) (human:ahg-g)
[^pr-1601-ic-4770357895]: [#1601](https://github.com/llm-d/llm-d-router/pull/1601#issuecomment-4770357895) (human:ahg-g)
[^issue-1177-ic-4478503514]: [#1177](https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514) (human:shmuelk)
[^pr-2048-r3596786577]: [#2048](https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577) (human:shmuelk)
[^issue-968-ic-4602495714]: [#968](https://github.com/llm-d/llm-d-router/issues/968#issuecomment-4602495714) (human:elevran)
[^pr-1035-r3244818645]: [#1035](https://github.com/llm-d/llm-d-router/pull/1035#discussion_r3244818645) (human:zetxqx)
[^pr-1043-r3279489618]: [#1043](https://github.com/llm-d/llm-d-router/pull/1043#discussion_r3279489618) (human:ahg-g)
[^pr-1482-ic-4633716490]: [#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490) (human:liu-cong)
