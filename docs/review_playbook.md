# Review Playbook

<!-- Generated from docs/review/ by hack/render_playbook.py. Do not edit directly; regenerate with: python3 hack/render_playbook.py -->

> **Two-stage workflow:** This file is the compiled human-readable review guide for contributors and maintainers, combining targeted non-code review rules with rationale, accepted exceptions, open questions, and representative pull request threads. Coding and review agents must not load `docs/review_playbook.md` or `docs/review/provenance.json`; agents inspect code and search the repository first, then route changed paths through [review_a_pull_request.md](review/playbooks/review_a_pull_request.md) to load at most `0` to `2` matching rule files under `docs/review/rules/`. See [AGENTS.md](../AGENTS.md) for repository-wide coding, prose, logging, constant-reuse, and pull request rules.

## Contents

- [External wire and engine contracts](#external-wire-and-engine-contracts) (5 rules)
- [Unencoded runtime hazards](#unencoded-runtime-hazards) (6 rules)
- [Governance and compatibility policy](#governance-and-compatibility-policy) (4 rules)
- [Settled architectural boundaries](#settled-architectural-boundaries) (5 rules)

## External wire and engine contracts

External API token semantics, engine streaming and render wire contracts, HTTP header invariants, router-built P/D transfer params, and ZMQ KV-event sequence recovery not visible in router Go structs.

### Anthropic split prompt tokens and vLLM streaming usage flags

When parsing or forwarding token usage and `stream_options`: sum Anthropic `/v1/messages` prompt tokens as `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (OpenAI nests `cached_tokens` instead), and treat vLLM `stream_options.continuous_usage_stats` (cumulative usage on every chunk) as separate from one-shot `include_usage` without injecting `include_usage` beside it.

- **Why it matters:** Ignoring Anthropic cache token fields undercounts prompt tokens, and injecting include_usage alongside continuous_usage_stats duplicates or corrupts streaming usage.
- **Examples:** Observed in 63 PRs: [#848](https://github.com/llm-d/llm-d-router/issues/848), [#932](https://github.com/llm-d/llm-d-router/pull/932#discussion_r3178050269), [#2391](https://github.com/llm-d/llm-d-router/pull/2391#discussion_r3798712574)

### Client JSON passthrough, Content-Length, and Connection upgrade

When decoding JSON bodies, rewriting payloads, or copying client headers upstream: unmarshal only router-inspected keys so other fields pass through, delete or recompute `Content-Length` on rewrite, and scan every `Connection` line and token case-insensitively (RFC 9110) for `upgrade`. Do not forward client `Accept-Encoding` to legs whose responses the router decodes; Go `http.Transport` only decompresses gzip it requested itself.

- **Why it matters:** Full-struct JSON re-serialization drops unknown client fields, stale Content-Length headers truncate or hang streams after body rewrites, single-line Connection checks miss RFC 9110 upgrades, and a forwarded client Accept-Encoding lets upstreams return gzip the router then fails to decode.
- **Settled decision:** [Opt-in backend request priority rewrite via PriorityRewriter interface](review/decisions/opt_in_backend_priority_rewrite.md).
- **Examples:** Observed in 52 PRs: [#1045](https://github.com/llm-d/llm-d-router/issues/1045), [#1092](https://github.com/llm-d/llm-d-router/issues/1092), [#3638](https://github.com/llm-d/llm-d-router/pull/3638#pullrequestreview-5665600572)

### Native engine render passthrough and stateful Responses rejection

When integrating inference APIs, prompt rendering, or `/v1/responses`: forward raw client payloads to native engine `/render` routes (`/v1/responses/render`, `/v1/messages/render`) instead of translating schemas in the router, and reject `/v1/responses` `previous_response_id` and `conversation` fields with an explicit error.

- **Why it matters:** Router-side prompt translation drifts from backend tokenizer templates, and silently ignoring previous_response_id or conversation fields drops prior multi-turn context.
- **Reviewers accept:** Interim translation while the engine lacks a native `/render` endpoint ([#1896](https://github.com/llm-d/llm-d-router/pull/1896#issuecomment-4924616643), [#1896](https://github.com/llm-d/llm-d-router/pull/1896#pullrequestreview-4657668118)).
- **Examples:** Observed in 23 PRs: [#850](https://github.com/llm-d/llm-d-router/issues/850), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3279332668), [#1364](https://github.com/llm-d/llm-d-router/issues/1364#issuecomment-4587275492)

### Router-built P/D transfer params

When a sidecar connector builds decode-leg `kv_transfer_params` itself instead of forwarding prefill's response: set every key that engine's connector reads unconditionally (vLLM NIXL: `remote_request_id`), never carry one engine's keys into another engine's connector (SGLang uses `bootstrap_host`/`port`/`room` and has no `remote_request_id`), and drop encoder `ec_transfer_params` from decode bodies.

- **Why it matters:** Router-built decode params that omit a key the engine connector reads unconditionally, or carry another engine's keys, crash the decode engine or break its KV transfer, and forwarded encoder ec_transfer_params trigger connector errors on decode workers.
- **Examples:** Observed in 2 PRs: [#3607](https://github.com/llm-d/llm-d-router/issues/3607), [#3608](https://github.com/llm-d/llm-d-router/pull/3608#pullrequestreview-5627567379), [#3732](https://github.com/llm-d/llm-d-router/pull/3732#pullrequestreview-5701261477)

### ZMQ KV-events sequence gaps and worker epoch resets

When consuming ZMQ KV-event streams: track per-endpoint sequence numbers to detect gaps (`seq > lastSeq + 1`) and worker-restart epoch resets (`seq < lastSeq`), purging stale index entries and loading a base snapshot before applying deltas.

- **Why it matters:** Applying KV-event deltas across dropped packets or worker restarts leaves stale block hashes in the prefix-cache index and misroutes requests.
- **Examples:** Observed in 27 PRs: [#1737](https://github.com/llm-d/llm-d-router/issues/1737), [#1887](https://github.com/llm-d/llm-d-router/pull/1887#discussion_r3684757376), [#2946](https://github.com/llm-d/llm-d-router/pull/2946#pullrequestreview-5438730630)

## Unencoded runtime hazards

Non-local concurrency, Envoy ext_proc skip-path, SSE chunk-boundary accumulation, ResponseWriter wrapper, multi-worker DP endpoint-key, metric cardinality, and lock-across-network hazards not visible in local function signatures.

### Unjoined timed-out DataProducer goroutines and buffer lifetimes

When writing `DataProducer` plugins or body handling around `dataProducerPluginsWithTimeout`: timeout returns without joining the producer goroutine, so check `ctx.Err()` before writes, and never mutate shared `*InferenceRequestBody` fields in place after timeout or recycle `sync.Pool` buffers while an unjoined producer or snapshot reader aliases their bytes.

- **Why it matters:** Unjoined producer goroutines that outlive timeout mutate request bodies or recycled sync.Pool buffers concurrently with downstream scheduling.
- **Open question:** Whether the plugin framework recovers plugin panics at the invocation boundary and reports them per request ([#2105](https://github.com/llm-d/llm-d-router/issues/2105)) or lets a panicking plugin stop the process ([#2098](https://github.com/llm-d/llm-d-router/issues/2098#issuecomment-5032265086)) ([details](review/open_questions/plugin_panic_isolation.md)).
- **Examples:** Observed in 44 PRs: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740756), [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3149928582), [#1002](https://github.com/llm-d/llm-d-router/pull/1002#discussion_r3403097625)

### Envoy ext_proc early header-skip metrics and decode-pod filtering

When changing header-only or skipped-body exits in Envoy `ext_proc` handlers: record request latency and one terminal outcome on early-skip branches, and restrict fallback endpoint selection in disaggregated pools to decode-capable pods.

- **Why it matters:** Skipping latency recording on header-only exit branches skews request histograms, and unfiltered fallback selection can route client traffic to prefill-only pods.
- **Examples:** Observed in 37 PRs: [#1553](https://github.com/llm-d/llm-d-router/issues/1553), [#1658](https://github.com/llm-d/llm-d-router/pull/1658#discussion_r3618566988), [#1444](https://github.com/llm-d/llm-d-router/pull/1444#discussion_r3346561765)

### SSE chunk boundaries, usage folding, and ResponseWriter Unwrap

When parsing streamed responses, folding usage, or wrapping `http.ResponseWriter`: `ext_proc` chunks split SSE events at any byte, so every SSE parser carries partial lines per request, capping each assembled line, keep the last cumulative usage event in a chunk, and do not let later zeros overwrite counts. Implement `Unwrap() http.ResponseWriter` so `http.Flusher` works.

- **Why it matters:** SSE events split across ext_proc chunks or several usage events in one chunk record zero or stale token usage, unbounded carry buffers grow on malformed streams, and missing Unwrap breaks SSE flushing.
- **Examples:** Observed in 17 PRs: [#3650](https://github.com/llm-d/llm-d-router/pull/3650#discussion_r4402061735), [#3650](https://github.com/llm-d/llm-d-router/pull/3650#discussion_r4402061747), [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740743)

### Multi-worker DP endpoint keys by host, port, and rank

When keying per-endpoint maps, prefix-cache indexes, ZMQ sockets, or SSRF allowlists: key by `(host, port, rank)` not IP alone: multi-worker DP ranks share one pod IP.

- **Why it matters:** Keying endpoint state by pod IP alone collides across multi-worker data-parallel ranks sharing one pod network namespace.
- **Examples:** Observed in 13 PRs: [#841](https://github.com/llm-d/llm-d-router/issues/841), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242332547), [#2831](https://github.com/llm-d/llm-d-router/pull/2831#discussion_r4033597903)

### Bounded metric labels and series cleanup

When adding Prometheus `*Vec` labels or configurable buckets: wrap caller-controlled values (`model_name`, `fairness_id`) in `BoundedLabel` at every record site, plugin metrics included, delete series when their endpoint or flow is removed, and cap configurable bucket counts.

- **Why it matters:** Caller-controlled label values, series never deleted after endpoint or flow removal, and unbounded configurable buckets grow Prometheus series without limit and freeze stale values in aggregates.
- **Examples:** Observed in 4 PRs: [#2106](https://github.com/llm-d/llm-d-router/issues/2106), [#2529](https://github.com/llm-d/llm-d-router/issues/2529), [#3539](https://github.com/llm-d/llm-d-router/pull/3539#discussion_r4347141065)

### No locks held across network calls

When a mutex guards scheduling-path state: never hold it (read lock included) across Redis, HTTP, or gRPC calls. A waiting writer blocks new `RWMutex` readers, so a slow call becomes request latency; copy state under the lock and call after release with a bounded context.

- **Why it matters:** Holding a scheduling-path mutex across a Redis or network round trip queues every reader behind a waiting writer and turns backend latency or outages into request latency.
- **Examples:** Observed in 1 PR: [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473471), [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473477)

## Governance and compatibility policy

Release deprecation policy, plugin graduation tiers and frozen Stable v1 fixtures, upstream engine and GIE CRD coupling rules, and shipped Helm vs local Kustomize boundaries.

### N+2 deprecation for shipped surfaces and zero shims for unshipped surfaces

When renaming or removing CLI flags, config fields, plugin types, HTTP headers, or metrics: keep `N+2` two-release deprecation shims with startup warnings for surfaces shipped in a prior release tag while migrating in-tree configs and tests immediately, and modify or delete unshipped post-tag surfaces with zero deprecation shim.

- **Why it matters:** Removing shipped flags or config fields without a two-release shim breaks upgrades, while keeping shims for unshipped surfaces leaves dead compatibility code.
- **Settled decisions:** [Two-release deprecation shims required even during pre-1.0 config refactors](review/decisions/two_release_deprecation_shims_for_pre_v1_config.md), [Delete stranded deprecation helpers in the removal pull request](review/decisions/delete_stranded_deprecation_helpers_immediately.md).
- **Examples:** Observed in 68 PRs: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4395649694), [#842](https://github.com/llm-d/llm-d-router/pull/842#discussion_r3180806498), [#894](https://github.com/llm-d/llm-d-router/pull/894#discussion_r3151838341)

### Plugin Alpha, Beta, and Stable graduation and frozen v1 fixtures

When registering plugins, changing plugin parameters, or editing `test/testdata/plugins/stable/v1/`: register new plugins at `Alpha` (`--allow-experimental-plugins`, no compat guarantee), preserve two minor versions at `Beta`, and freeze `Stable` schemas in `test/testdata/plugins/stable/v1/` without editing existing fixtures for breaking changes.

- **Why it matters:** Graduating plugins without tier gates or editing frozen Stable v1 fixtures silently breaks production EndpointPickerConfig manifests.
- **Settled decision:** [Plugin graduation path across Alpha, Beta, and Stable tiers](review/decisions/plugin_graduation_tiers.md).
- **Examples:** Observed in 12 PRs: [#905](https://github.com/llm-d/llm-d-router/pull/905#issuecomment-4351689606), [#2365](https://github.com/llm-d/llm-d-router/pull/2365#discussion_r4241052101), [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379506333)

### Unmerged upstream engine contracts and versioned GIE CRDs

When targeting backend engine flags or endpoints, or updating GIE CRDs and protos: hold or close router PRs coupled to unmerged upstream engine flags or endpoints, and consume versioned upstream GIE CRDs and protos without vendoring (`test/e2e/artifacts/components/crds-istio` is the sole local-dev exception).

- **Why it matters:** Merging router support against unmerged engine flags or vendored upstream CRDs strands incompatible wire shapes when upstream lands differently.
- **Open question:** Whether unmodified upstream CRDs used by tests are fetched from GitHub on every run ([#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490)) or cached in the repository to avoid rate-limited fetch failures ([#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4661288603)) ([details](review/open_questions/upstream_test_crd_caching.md)).
- **Examples:** Observed in 20 PRs: [#3359](https://github.com/llm-d/llm-d-router/pull/3359#discussion_r4361817276), [#968](https://github.com/llm-d/llm-d-router/issues/968#issuecomment-4602495714), [#1035](https://github.com/llm-d/llm-d-router/pull/1035#discussion_r3244818645)

### Helm charts in config/charts/ and local Kustomize in test/e2e/artifacts/

When changing deployment manifests or component flags under `config/charts/` or `test/e2e/artifacts/`: ship end-user deployment surfaces solely via `config/charts/` Helm charts and structured config, keeping `test/e2e/artifacts/` Kustomize for local kind and E2E only.

- **Why it matters:** Shipping production flags only in test/e2e/artifacts/ Kustomize overlays leaves Helm chart installations missing supported configuration.
- **Reviewers accept:** Updating `test/e2e/artifacts/` Kustomize manifests for local kind and E2E suites ([Helm charts for end-user releases and test/e2e/artifacts Kustomize manifests for local kind and E2E](review/decisions/helm_release_charts_with_dev_kustomize_manifests.md), [#1177](https://github.com/llm-d/llm-d-router/issues/1177#issuecomment-4478503514), [#2048](https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577)).
- **Examples:** Observed in 30 PRs: [#634](https://github.com/llm-d/llm-d-router/issues/634), [#930](https://github.com/llm-d/llm-d-router/pull/930#issuecomment-4366069128), [#1550](https://github.com/llm-d/llm-d-router/pull/1550#issuecomment-4656209920)

## Settled architectural boundaries

Maintainer-settled subsystem boundaries across overload shedding, pool homogeneity, session state, coordinator vs sidecar responsibilities, and ingress trust boundaries.

### Capacity shedding in SaturationDetector or AdmissionPlugin

When implementing overload shedding or scheduling filters: place capacity shedding in `SaturationDetector` or `AdmissionPlugin` (`Admitter`) rather than `Filter` plugins.

- **Why it matters:** Shedding load inside scheduling Filter plugins bypasses flow-control queuing, priority fairness, and admission metrics.
- **Settled decisions:** [Scorer-internal fallback over scheduler fallback profiles](review/decisions/scorer_internal_fallback_over_fallback_profiles.md), [Pre-admission header attribute extraction via dedicated plugin extension point](review/decisions/request_header_extractor_before_admission.md).
- **Examples:** Observed in 37 PRs: [#545](https://github.com/llm-d/llm-d-router/issues/545#issuecomment-5194281537), [#837](https://github.com/llm-d/llm-d-router/issues/837), [#977](https://github.com/llm-d/llm-d-router/issues/977)

### Homogeneous base model and role per InferencePool

When routing across models or disaggregated roles in an `InferencePool` or EPP: scope each pool and EPP to one homogeneous base model and role, routing distinct base models to separate pools via Gateway `HTTPRoute`.

- **Why it matters:** Mixing heterogeneous base models or roles in one InferencePool breaks capacity accounting, KV-cache compatibility, and scheduling assumptions.
- **Reviewers accept:** Unioning endpoint model lists in `models-responder` on `GET /v1/models` for subset-loaded LoRA adapters ([Pool-wide /v1/models aggregation via generic Responder plugin interface](review/decisions/models_responder_generic_extension_point.md), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5453802061), [#2308](https://github.com/llm-d/llm-d-router/pull/2308#pullrequestreview-5483618104)).
- **Open question:** Whether non-Kubernetes deployments load `InferenceObjective` definitions from Kubernetes manifests with unused fields ignored ([#3502](https://github.com/llm-d/llm-d-router/pull/3502)) or from a separate strict file schema ([#3502](https://github.com/llm-d/llm-d-router/pull/3502#pullrequestreview-5649007338)) ([details](review/open_questions/non_kubernetes_objectives_schema.md)).
- **Examples:** Observed in 7 PRs: [#1878](https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197), [#2333](https://github.com/llm-d/llm-d-router/pull/2333#issuecomment-5935642884), [#2692](https://github.com/llm-d/llm-d-router/pull/2692#issuecomment-5626620151)

### External conversation history vs bounded routing state

When adding multi-turn session handling or stateful routing plugins: never store client conversation history in the router, while allowing bounded in-memory routing and session state with capacity caps and endpoint-deletion cleanup.

- **Why it matters:** Storing client conversation history turns the stateless router into a stateful data store, while unbounded routing caches leak memory on pod churn.
- **Settled decision:** [Session affinity as a shared stateless helper package](review/decisions/session_affinity_shared_helper.md).
- **Examples:** Observed in 27 PRs: [#896](https://github.com/llm-d/llm-d-router/issues/896#issuecomment-5923079423), [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190233226), [#1004](https://github.com/llm-d/llm-d-router/pull/1004#discussion_r3202130559)

### Cluster orchestration in pkg/coordinator and pod proxy in pkg/sidecar

When placing disaggregated multi-stage logic across `pkg/coordinator` and `pkg/sidecar`: keep cluster multi-stage orchestration in `pkg/coordinator` and keep `pkg/sidecar` scoped to the pod-local worker proxy.

- **Why it matters:** Placing cluster-wide stage orchestration inside the pod-local sidecar couples worker proxies to cluster routing state and control-plane watches.
- **Open question:** Whether the disaggregation sidecar should watch `InferencePool` and `Pod` objects from every decode pod to build its SSRF allowlist ([#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771)) or avoid API-server watches from workload pods ([#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953)) ([details](review/open_questions/sidecar_ssrf_allowlist_watches.md)).
- **Examples:** Observed in 14 PRs: [#1104](https://github.com/llm-d/llm-d-router/issues/1104#issuecomment-4449848790), [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953), [#1653](https://github.com/llm-d/llm-d-router/pull/1653#discussion_r3412845630)

### Ingress-only header stripping

When stripping headers or forwarding `kv_transfer_params`: strip spoofing headers and client `kv_transfer_params` at every external client entry point, passthrough routes included, while treating internal EPP, Coordinator, and Sidecar `kv_transfer_params` and routing metadata hops as trusted.

- **Why it matters:** Failing to strip internal headers at client ingress enables routing spoofing, while stripping kv_transfer_params on internal hops breaks disaggregated KV transfer.
- **Open question:** Whether session affinity should be carried by a standard `Set-Cookie` response ([#28](https://github.com/llm-d/llm-d-router/issues/28)) or by a request header such as `x-session-token` ([#600](https://github.com/llm-d/llm-d-router/pull/600#discussion_r3156539950)) ([details](review/open_questions/session_affinity_cookie_vs_header.md)).
- **Examples:** Observed in 35 PRs: [#3523](https://github.com/llm-d/llm-d-router/pull/3523#pullrequestreview-5584533523), [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560), [#2319](https://github.com/llm-d/llm-d-router/pull/2319#discussion_r3758140767)
