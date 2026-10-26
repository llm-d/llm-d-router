# Review Playbook

This playbook records the engineering and review standards for `llm-d-router`. It is distilled from maintainer review feedback across pull requests and issues. Reviewers cite these rules when flagging defects, and authors check their changes against each section before opening a pull request. See [AGENTS.md](../AGENTS.md) for pull request mechanics, commit sign-off, logging conventions, and prose style rules.

## Scope and process

- **Separate mechanical moves from behavioral changes.**
  Place mechanical package moves, file renames, linter rollouts, and build-tool updates in standalone pull requests separate from behavioral changes so collateral diffs do not obscure logic defects, and split large test additions by tier across unit, integration, and benchmark suites. See [AGENTS.md](../AGENTS.md#pull-requests) for pull request scope and template mechanics.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4642288302), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3274050146), [#1018](https://github.com/llm-d/llm-d-router/pull/1018#discussion_r3270503284), [#1528](https://github.com/llm-d/llm-d-router/pull/1528#issuecomment-4650669675)
  Reviewers accept: Deferring cross-package migrations, sibling plugin gaps, RPC batching follow-ups, and broad documentation rewrites to linked issues to keep the active diff focused ([#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3178005111), [#1071](https://github.com/llm-d/llm-d-router/pull/1071#discussion_r3262248799), [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3374238301)).

- **Settle design tradeoffs on public issues and scope auto-close keywords.**
  Discuss architecture and vulnerability fixes on public GitHub issues rather than private trackers before iterating on pull requests. Use auto-closing issue keywords only when a pull request resolves the entire local issue, leaving multi-slice tracking issues open until the final slice lands.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4541930806), [#857](https://github.com/llm-d/llm-d-router/issues/857), [#1090](https://github.com/llm-d/llm-d-router/issues/1090#issuecomment-5347088232), [#2813](https://github.com/llm-d/llm-d-router/pull/2813#discussion_r4000003650)
  Reviewers accept: Narrow wire-formatting adjustments motivated by an external conformance verifier when the issue and pull request state that compatibility scope explicitly ([#2590](https://github.com/llm-d/llm-d-router/issues/2590#issuecomment-5499142962)).

- **Use `NONE` release notes for internal changes and keep PR metadata synchronized.**
  Mark the `release-note` block as `NONE` for internal refactors, tests, CI edits, and fixes to unshipped features on `main`, and write operator-focused notes when defaults, flags, metrics, headers, or upgrade steps change. Update the pull request title, body, and release note whenever review iterations rename symbols or narrow scope.
  Examples: [#1206](https://github.com/llm-d/llm-d-router/pull/1206#issuecomment-4647621711), [#1657](https://github.com/llm-d/llm-d-router/pull/1657#issuecomment-4708030072), [#1939](https://github.com/llm-d/llm-d-router/pull/1939#issuecomment-4925378102), [#2221](https://github.com/llm-d/llm-d-router/pull/2221#discussion_r3708529206)

- **Iterate in place from personal fork branches with incremental commits.**
  Open pull requests from a personal fork instead of pushing feature branches to the upstream repository, update active pull requests in place, and close superseded pull requests with a link to the replacement. Push new commits during review rounds rather than force-pushing rebased history so reviewers can inspect incremental diffs.
  Examples: [#1172](https://github.com/llm-d/llm-d-router/pull/1172#issuecomment-4477576689), [#1330](https://github.com/llm-d/llm-d-router/pull/1330#issuecomment-4614740432), [#2228](https://github.com/llm-d/llm-d-router/pull/2228#issuecomment-5409613155), [#2600](https://github.com/llm-d/llm-d-router/pull/2600#issuecomment-5539415543), [#2823](https://github.com/llm-d/llm-d-router/pull/2823#issuecomment-5668357219)

## Plugin framework contracts

- **Keep core runtime packages free of concrete plugin logic.**
  Do not import concrete plugin packages or add plugin-specific type switches inside `cmd/epp/runner`, `pkg/epp/datastore`, or `pkg/epp/framework/interface`. Expose shared runtime capabilities through `fwkplugin.Handle` and small optional behavioral interfaces verified with compile-time interface guards (`var _ Interface = (*Type)(nil)`), resolving capabilities once at startup registration.
  Examples: [#836](https://github.com/llm-d/llm-d-router/issues/836), [#877](https://github.com/llm-d/llm-d-router/issues/877), [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3261995399), [#1130](https://github.com/llm-d/llm-d-router/issues/1130)

- **Separate data producers from scheduling filters and scorers.**
  Keep stateful indexing and raw endpoint extraction in data producers while placing policy evaluation in downstream scheduling plugins (see [docs/architecture.md](architecture.md#routing-flow)). Restrict `Filter` plugins to binary endpoint eligibility with fail-open bypass when optional telemetry is absent, express soft preferences as weighted `Scorer` plugins rather than positional filters, use pre-scheduling `Screener` plugins only for single-endpoint request pinning, and delegate load shedding to saturation detectors.
  Examples: [#545](https://github.com/llm-d/llm-d-router/issues/545#issuecomment-5194281537), [#837](https://github.com/llm-d/llm-d-router/issues/837), [#977](https://github.com/llm-d/llm-d-router/issues/977), [#2003](https://github.com/llm-d/llm-d-router/issues/2003#issuecomment-5073147346)
  Reviewers accept: Implementing strict single-endpoint request pinning as a pre-scheduling `Screener` once the tradeoff against a scheduling `Filter` is evaluated ([#3145](https://github.com/llm-d/llm-d-router/pull/3145#issuecomment-6088007878), [#3145](https://github.com/llm-d/llm-d-router/pull/3145#pullrequestreview-5462400380)).

- **Exchange cross-plugin state through typed data-layer slots and keys.**
  Pass cross-plugin request and endpoint metadata through typed `datalayer.Slot[T]` handles (`pkg/epp/framework/interface/datalayer/slot.go`) or `Clone()`-safe values in `datalayer.AttributeMap` (`pkg/epp/framework/interface/datalayer/registry.go`) rather than global singletons or direct peer calls. Declare cross-plugin dependencies through `Produces()` and `Consumes()`, key instance-scoped attributes by non-empty plugin names, and validate producer-consumer DAG wiring at startup after default producers resolve.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#issuecomment-4396004237), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252785994), [#1096](https://github.com/llm-d/llm-d-router/pull/1096#discussion_r3255496885), [#1106](https://github.com/llm-d/llm-d-router/pull/1106#discussion_r3267965090), [#1249](https://github.com/llm-d/llm-d-router/pull/1249#issuecomment-4565953909)

- **Validate plugin parameters once in the factory constructor.**
  Validate parameter bounds, required handles, and forbidden system headers inside the factory constructor rather than exposing mutable post-construction setters or repeating checks per request. Register built-in plugins in the central catalog with an explicit stability tier and enforce experimental feature gates across explicit and default instances; see [pkg/epp/framework/plugins/README.md](../pkg/epp/framework/plugins/README.md) and [docs/create_new_filter.md](create_new_filter.md) for plugin registration and tier lifecycle rules.
  Examples: [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3268601757), [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3196010531), [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628902), [#1912](https://github.com/llm-d/llm-d-router/issues/1912)
  Reviewers accept: Documenting redundant plugin combinations in plugin READMEs rather than rejecting sibling plugins inside factory constructors ([#1396](https://github.com/llm-d/llm-d-router/pull/1396#issuecomment-4642995683)).

- **Name plugin packages with concise domain nouns and qualify config units.**
  Name plugin packages with short lowercase domain nouns (such as `bylabel`, `queuedepth`, `tokenload`, or `sessionaffinity`), suffix registered plugin type strings with their extension-point role (`-filter`, `-scorer`, `-picker`, `-producer`, or `-decider`), and avoid repeating the package name in exported identifiers. Suffix quantitative configuration fields with explicit units or portable token counts rather than backend-dependent block counts, and accept human-readable quantity strings (`resource.Quantity`) for byte sizes.
  Examples: [#747](https://github.com/llm-d/llm-d-router/issues/747), [#813](https://github.com/llm-d/llm-d-router/issues/813), [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143427635), [#1924](https://github.com/llm-d/llm-d-router/pull/1924#discussion_r3542577754)
  Reviewers accept: Standard computer-science data-structure names on internal types and preserving established public constant identifiers across internal variable renames ([#2036](https://github.com/llm-d/llm-d-router/pull/2036#issuecomment-4999184223), [#2342](https://github.com/llm-d/llm-d-router/pull/2342#discussion_r3752875804)).

- **Share common helpers and delete dead code in the stranding change.**
  Decouple pure payload parsing from HTTP response writing, reuse shared utilities in `pkg/common` rather than duplicating logic across packages, and follow [AGENTS.md](../AGENTS.md#working-in-the-codebase) for canonical constant ownership across packages. Delete dead branches, write-only fields, unreferenced methods, and single-caller pass-through wrappers in the change that strands them.
  Examples: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143173577), [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236349435), [#1200](https://github.com/llm-d/llm-d-router/pull/1200#discussion_r3274000360), [#1444](https://github.com/llm-d/llm-d-router/pull/1444#pullrequestreview-4425939275)

## Project direction

- **Keep core runtimes built in and policy plugins optional.**
  Core execution runtimes and cross-replica state synchronization orchestration belong in built-in runtime packages, whereas `pkg/epp/framework/plugins` holds optional policy units that operators enable or disable via configuration (see [docs/plugins-vs-builtins.md](plugins-vs-builtins.md)). Route scalar telemetry through generic attribute-driven filters and scorers, share lightweight math via plain Go helpers, and require multiple concrete use cases before adding a new framework extension point.
  Examples: [#818](https://github.com/llm-d/llm-d-router/issues/818), [#1139](https://github.com/llm-d/llm-d-router/issues/1139#issuecomment-4601790080), [#1450](https://github.com/llm-d/llm-d-router/issues/1450#issuecomment-4604976814), [#1904](https://github.com/llm-d/llm-d-router/pull/1904#issuecomment-4898714192), [#1921](https://github.com/llm-d/llm-d-router/pull/1921#discussion_r3587753917)

- **Normalize requests in upstream producers into a protocol-agnostic representation.**
  Run request payload normalization, tokenization, multimodal extraction, and token estimation in canonical upstream pre-scheduling data producers that populate a protocol-agnostic internal request representation, keeping downstream filters and scorers free of wire-format parsing. Order pre-admission flow-control queues by immutable request attributes fixed at admission rather than mutable endpoint state.
  Examples: [#850](https://github.com/llm-d/llm-d-router/issues/850), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3279332668), [#970](https://github.com/llm-d/llm-d-router/issues/970), [#1634](https://github.com/llm-d/llm-d-router/issues/1634#issuecomment-5942141855)

- **Configure backend engine identity explicitly and isolate engine schemas.**
  Configure backend inference-engine identity explicitly at deployment time rather than inferring engine type from unrelated connector flags. Isolate engine-specific wire schemas, CLI flags, and metric names behind declarative adapters and per-plugin documentation instead of scattering engine conditionals across shared routing packages, and close router pull requests blocked on stalled unmerged upstream engine changes.
  Examples: [#382](https://github.com/llm-d/llm-d-router/issues/382#issuecomment-4585925394), [#2447](https://github.com/llm-d/llm-d-router/issues/2447#issuecomment-5342388312), [#3030](https://github.com/llm-d/llm-d-router/issues/3030), [#3359](https://github.com/llm-d/llm-d-router/pull/3359#discussion_r4361817276)

- **Model endpoints with orchestrator-neutral domain types.**
  Model internal datastore, discovery, server registration, and metric-extraction interfaces around orchestrator-neutral `Endpoint` entities rather than Kubernetes `Pod` structs or controller managers so routing operates across both Kubernetes and standalone deployments.
  Examples: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628879), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3271206173), [#1148](https://github.com/llm-d/llm-d-router/pull/1148#discussion_r3294724682)

- **Scope each pool to a single homogeneous base model and role.**
  Scope each `InferencePool` and EPP process to homogeneous replicas serving a single base model and disaggregated role so every endpoint contributes to shared pool capacity. Route distinct base models through Gateway `HTTPRoute` rules to separate pools, and forward model-discovery requests directly to backend servers rather than synthesizing aggregated model lists in the router.
  Examples: [#1878](https://github.com/llm-d/llm-d-router/issues/1878#issuecomment-4864624197), [#2333](https://github.com/llm-d/llm-d-router/pull/2333#issuecomment-5935642884), [#2692](https://github.com/llm-d/llm-d-router/pull/2692#issuecomment-5626620151)

- **Keep the router stateless and orchestrate multi-stage pipelines in the coordinator.**
  Keep the router focused on stateless HTTP and SSE inference routing, narrow metadata extraction, and cluster-wide cache or session placement while delegating conversation persistence, intra-backend rank balancing, and offline calibration to external layers. Place cluster-level multi-stage disaggregated orchestration in `pkg/coordinator` above endpoint selection, while keeping `pkg/sidecar` scoped to the pod-local decode proxy without Kubernetes API watches or cluster privileges (see [docs/architecture.md](architecture.md) and [docs/disaggregation.md](disaggregation.md)).
  Examples: [#896](https://github.com/llm-d/llm-d-router/issues/896#issuecomment-5923079423), [#1104](https://github.com/llm-d/llm-d-router/issues/1104#issuecomment-4449848790), [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4489027953), [#1979](https://github.com/llm-d/llm-d-router/issues/1979)

- **Deploy through Helm charts and configure components via structured files.**
  Ship end-user deployments through lightweight Helm charts under `config/charts` and align configuration schemas hierarchically with top-level components in structured configuration files instead of expanding binary CLI flags. Publish multi-component operations guides and umbrella dashboards in `llm-d/llm-d`, keep `deploy` scoped to development and E2E overlays, and review design proposals in issues or external docs rather than committing static files under `docs/`.
  Examples: [#634](https://github.com/llm-d/llm-d-router/issues/634), [#930](https://github.com/llm-d/llm-d-router/pull/930#issuecomment-4366069128), [#1550](https://github.com/llm-d/llm-d-router/pull/1550#issuecomment-4656209920), [#1601](https://github.com/llm-d/llm-d-router/pull/1601#issuecomment-4770357895), [#1844](https://github.com/llm-d/llm-d-router/pull/1844#issuecomment-5101886369)

- **Consume versioned upstream CRDs and protobufs without vendoring.**
  Consume upstream Gateway API Inference Extension CRDs and protobuf definitions from versioned releases or the pinned Go module cache rather than vendoring local copies in the repository.
  Examples: [#968](https://github.com/llm-d/llm-d-router/issues/968#issuecomment-4602495714), [#1035](https://github.com/llm-d/llm-d-router/pull/1035#discussion_r3244818645), [#1043](https://github.com/llm-d/llm-d-router/pull/1043#discussion_r3279489618), [#1482](https://github.com/llm-d/llm-d-router/pull/1482#issuecomment-4633716490)

## Configuration, Helm, and compatibility

- **Provide a two-release deprecation window for shipped user-facing surfaces.**
  Support released CLI flags, configuration fields, plugin types, HTTP headers, and metrics across at least two releases (`N+2`) before removal. Emit a prominent startup warning when a deprecated surface is used, migrate all in-tree manifests and tests to the replacement immediately, and sequence structural moves separately from renames so operators migrate once.
  Examples: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4395649694), [#842](https://github.com/llm-d/llm-d-router/pull/842#discussion_r3180806498), [#905](https://github.com/llm-d/llm-d-router/pull/905#issuecomment-4351689606), [#922](https://github.com/llm-d/llm-d-router/pull/922#issuecomment-4352132975)
  Reviewers accept: Renaming or removing unshipped configuration keys, unreleased Helm charts, and internal headers without deprecation shims, and deleting shims once their deprecation window closes ([#894](https://github.com/llm-d/llm-d-router/pull/894#discussion_r3151838341), [#1254](https://github.com/llm-d/llm-d-router/pull/1254#issuecomment-4512813466), [#2365](https://github.com/llm-d/llm-d-router/pull/2365#discussion_r4241052101), [#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379506333)).

- **Decode configuration with strict unknown-field rejection and startup validation.**
  Seed configuration structs from canonical default constructors, decode plugin and component configurations through `fwkplugin.StrictDecoder` (`DisallowUnknownFields()` in `pkg/epp/framework/interface/plugin/registry.go`), `yaml.UnmarshalStrict`, or `serializer.EnableStrict` (`pkg/epp/config/loader/configloader.go`) so unknown fields fail at startup, and validate cross-field and mode-scoped invariants deterministically before serving traffic. Use explicit mode enums over boolean flag combinations, and use pointer numeric fields with `omitempty` only when distinguishing an unset default from an explicit zero.
  Examples: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143157150), [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190715413), [#919](https://github.com/llm-d/llm-d-router/pull/919#discussion_r3161391196), [#933](https://github.com/llm-d/llm-d-router/pull/933#discussion_r3195228231)

- **Ship safe out-of-the-box defaults and gate workload-sensitive features behind opt-in config.**
  Choose defaults that run within standard memory and CPU budgets without external sidecars or cross-component boot-order dependencies. Gate workload-sensitive heuristics, speculative estimators, and outbound payload rewrites behind explicit opt-in configuration so disabled features leave the default request path untouched.
  Examples: [#1029](https://github.com/llm-d/llm-d-router/pull/1029#discussion_r3243916222), [#1156](https://github.com/llm-d/llm-d-router/pull/1156#discussion_r3454102412), [#1187](https://github.com/llm-d/llm-d-router/issues/1187#issuecomment-5139243431), [#1220](https://github.com/llm-d/llm-d-router/issues/1220#issuecomment-4506123176)

- **Deduplicate Helm chart defaults through shared library templates.**
  Share common Helm logic in library templates while declaring values only in charts that render them, emit template `fail` errors when operators supply removed keys, and keep lifecycle hooks compatible with shell-less distroless images.
  Examples: [#924](https://github.com/llm-d/llm-d-router/issues/924), [#1040](https://github.com/llm-d/llm-d-router/pull/1040#issuecomment-4415659669), [#1042](https://github.com/llm-d/llm-d-router/issues/1042), [#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294949972)
  Reviewers accept: Repeating graduated stable Kubernetes API group literals directly in Helm templates and gating lifecycle features on supported Kubernetes versions with a documentation note rather than shell fallbacks ([#1275](https://github.com/llm-d/llm-d-router/pull/1275#discussion_r3294998818), [#2961](https://github.com/llm-d/llm-d-router/pull/2961#discussion_r4069441791)).

- **Preserve lossless conversion across multi-version CRDs.**
  Serve multi-version CRDs through lossless conversion under a single storage version with codegen validation markers, and update all in-tree `config/charts`, `deploy`, and test manifests in the same change.
  Examples: [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178415392), [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4144921130), [#2925](https://github.com/llm-d/llm-d-router/pull/2925#discussion_r4155389367), [#2925](https://github.com/llm-d/llm-d-router/pull/2925#issuecomment-5749511075)

## Concurrency and shared state

- **Keep mutex critical sections tight and release locks before network or blocking calls.**
  Encapsulate synchronization inside methods on the state-owning type, complete struct initialization before registering callbacks, and never hold a mutex across network RPCs, channel sends, or socket teardown. Combine read-modify-write transitions and coupled counter reads under a single lock acquisition, and copy caller-owned maps before asynchronous handoff.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190593630), [#889](https://github.com/llm-d/llm-d-router/issues/889), [#1128](https://github.com/llm-d/llm-d-router/pull/1128#discussion_r3275924435), [#1678](https://github.com/llm-d/llm-d-router/pull/1678#discussion_r3448540954)
  Reviewers accept: Defensively copying caller-supplied slices or maps at public goroutine boundaries, coalescing full state snapshots in capacity-one channels, and holding uncontended startup or shutdown locks when unlocking risks handle races ([#942](https://github.com/llm-d/llm-d-router/pull/942#discussion_r3178202254), [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314150441), [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314202507)).

- **Bound in-memory state with capacity limits and endpoint deletion cleanup.**
  Guard in-memory maps and queues keyed by requests, sessions, prompts, or endpoints with hard capacity caps and periodic TTL sweeps so long-running processes do not leak memory. Subscribe stateful plugins to endpoint lifecycle events to purge per-endpoint state on pod removal or empty controller reconciles.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190233226), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252764201), [#1004](https://github.com/llm-d/llm-d-router/pull/1004#discussion_r3202130559), [#1727](https://github.com/llm-d/llm-d-router/pull/1727#discussion_r3513605187), [#1736](https://github.com/llm-d/llm-d-router/issues/1736)

- **Reconcile dynamic priority bands on controller edges and drain before removal.**
  Provision and remove dynamic `InferenceObjective` priority bands through edge-triggered controller reconciliation rather than background data-plane idleness sweeps, preserving statically configured default bands and draining in-flight queues before deleting removed bands.
  Examples: [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3314143560), [#1354](https://github.com/llm-d/llm-d-router/pull/1354#discussion_r3391664697), [#1606](https://github.com/llm-d/llm-d-router/issues/1606), [#1608](https://github.com/llm-d/llm-d-router/pull/1608#issuecomment-4688020707)

- **Propagate caller contexts and bound goroutine lifetimes to server drain.**
  Root background collectors and serving loops on a drain-scoped context, and pass caller request contexts to all outbound RPCs. Check `ctx.Err()` on loop iterations and multi-channel `select` wakeups so client disconnects halt downstream work and release in-flight reservations, and run graceful readiness drain on leader lease loss.
  Examples: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740756), [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3149928582), [#2713](https://github.com/llm-d/llm-d-router/pull/2713#discussion_r3996027352), [#3140](https://github.com/llm-d/llm-d-router/issues/3140)

- **Scope cross-replica state to per-replica keys with short remote-store TTLs.**
  Publish shared load and cache state under collision-free replica IDs and combine live local counters with cached peer levels at read time so one replica reconnecting or resetting never wipes another replica's state. Expire transient peer state with short remote-store TTLs scaled to the publish interval rather than comparing wall clocks across hosts, and do not block startup when optional peer stores are unreachable.
  Examples: [#1892](https://github.com/llm-d/llm-d-router/issues/1892), [#1946](https://github.com/llm-d/llm-d-router/issues/1946), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955475084), [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4214243393)

- **Recover KV-event sequence gaps and epoch resets from base snapshots.**
  Track per-endpoint sequence numbers on KV-cache event streams to detect missed batches or container restarts (`seq < lastSeq`), resetting stale index state and loading a base snapshot before applying subsequent events.
  Examples: [#1737](https://github.com/llm-d/llm-d-router/issues/1737), [#1887](https://github.com/llm-d/llm-d-router/pull/1887#discussion_r3684757376), [#2946](https://github.com/llm-d/llm-d-router/pull/2946#pullrequestreview-5438730630), [#3399](https://github.com/llm-d/llm-d-router/issues/3399), [#3400](https://github.com/llm-d/llm-d-router/issues/3400)

## Error handling and boundaries

- **Validate untrusted inputs at ingress and configuration boundaries.**
  Concentrate nil, type-assertion, slice-bound, integer-overflow, content-type, and payload-size checks at network, parser, and configuration boundaries before buffering bodies or constructing domain types. Include field and expected-type context in boundary errors, and close caller-supplied listeners if early setup fails.
  Examples: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143190014), [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3220172664), [#1147](https://github.com/llm-d/llm-d-router/pull/1147#issuecomment-4461397416), [#1163](https://github.com/llm-d/llm-d-router/pull/1163#discussion_r3254240458)
  Reviewers accept: Omitting redundant defensive nil checks inside internal helpers whose callers guarantee valid inputs ([#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3192331180), [#944](https://github.com/llm-d/llm-d-router/pull/944#discussion_r3435978187), [#3252](https://github.com/llm-d/llm-d-router/pull/3252#discussion_r4218084207)).

- **Propagate errors explicitly without fabricated fallbacks.**
  Return producer, parser, and configuration failures as explicit errors instead of substituting fake placeholder counts, synthetic block sizes, or silent zero defaults that corrupt downstream routing decisions, and never call `panic` in request or flow-control packages (enforced by `forbidigo` in `.golangci.yml` for `pkg/epp/flowcontrol`). Fail startup when TLS certificates, metrics listeners, or required plugin bindings are invalid.
  Examples: [#876](https://github.com/llm-d/llm-d-router/issues/876), [#1002](https://github.com/llm-d/llm-d-router/pull/1002#discussion_r3403097625), [#1061](https://github.com/llm-d/llm-d-router/pull/1061#discussion_r3246024767), [#1895](https://github.com/llm-d/llm-d-router/pull/1895#discussion_r3615464093)

- **Map unwrapped error chains to canonical status codes.**
  Classify errors for retry, admission, and fail-open logic with `errors.Is` and `errors.As` across unwrapped chains rather than matching error message substrings, and format error responses to match the active API surface. Map client validation faults to 4xx, upstream worker protocol faults to 502, empty pools to 503, and saturation drops to 429.
  Examples: [#1116](https://github.com/llm-d/llm-d-router/issues/1116#issuecomment-4450544045), [#1603](https://github.com/llm-d/llm-d-router/pull/1603#discussion_r3399020119), [#2052](https://github.com/llm-d/llm-d-router/pull/2052#discussion_r3597666475), [#2215](https://github.com/llm-d/llm-d-router/pull/2215#discussion_r3698734025)
  Reviewers accept: Returning 502 Bad Gateway rather than 400 Bad Request when an upstream worker returns an invalid content type or malformed response for a valid client request ([#3511](https://github.com/llm-d/llm-d-router/pull/3511#discussion_r4379124714)).

- **Handle non-deferred close errors and scope linter suppressions to exact rules.**
  Handle or explicitly discard (`_ =`) non-deferred `Close()` errors on early-return paths and `t.Cleanup` callbacks (where `gosec` `G104` or `errcheck` applies outside `.golangci.yml` `std-error-handling` exclusions), and never use broad `//nolint:all` directives. When a verified invariant requires a static-analysis or security suppression, scope the directive to the specific rule identifier (`//nolint:<rule>`) at that call site and include a comment stating why the invariant holds.
  Examples: [#2995](https://github.com/llm-d/llm-d-router/issues/2995), [#2997](https://github.com/llm-d/llm-d-router/issues/2997), [#3048](https://github.com/llm-d/llm-d-router/pull/3048#pullrequestreview-5330221439), [#3055](https://github.com/llm-d/llm-d-router/pull/3055#discussion_r4177698311)

## Protocol parsing and wire fidelity

- **Preserve unrecognized client request fields and wire fidelity.**
  Decode only router-inspected fields into typed request structs while preserving unknown client fields, 64-bit integer precision, JSON key ordering, and explicit boolean flags on forwarded payloads. Normalize headers and paths once at ingress through read-only accessors without mutating forwarded headers.
  Examples: [#1045](https://github.com/llm-d/llm-d-router/issues/1045), [#1072](https://github.com/llm-d/llm-d-router/issues/1072), [#1092](https://github.com/llm-d/llm-d-router/issues/1092), [#1350](https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310261653)

- **Claim inference routes declaratively with unmatched-route fallthrough.**
  Register plugin and endpoint route claims declaratively by HTTP method and path so unmatched routes fall through cleanly to downstream handlers without hardcoding endpoint paths in core dispatchers.
  Examples: [#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358215465), [#1516](https://github.com/llm-d/llm-d-router/issues/1516), [#1524](https://github.com/llm-d/llm-d-router/issues/1524), [#3479](https://github.com/llm-d/llm-d-router/pull/3479#discussion_r4316182503)
  Reviewers accept: Representing plugin routing claims as separate path and protocol slices when claimed paths apply uniformly, and forwarding native client payloads directly to disaggregated workers when all stages run the same engine ([#1475](https://github.com/llm-d/llm-d-router/pull/1475#discussion_r3358239412), [#2987](https://github.com/llm-d/llm-d-router/pull/2987#discussion_r4090885940)).

- **Buffer streaming responses to complete SSE event boundaries and count terminal usage once.**
  Buffer chunked streaming responses to full SSE event delimiters before inspecting frames so network splits mid-event never drop `data:` prefixes. Preserve `http.Flusher` via `Unwrap()` on response writer wrappers, record token usage only from the terminal stream chunk, and treat premature EOF or missing terminal events as stream errors.
  Examples: [#822](https://github.com/llm-d/llm-d-router/pull/822#discussion_r3182740743), [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3161138182), [#2391](https://github.com/llm-d/llm-d-router/pull/2391#discussion_r3798712574), [#2432](https://github.com/llm-d/llm-d-router/pull/2432#pullrequestreview-4976441002)

- **Maintain feature parity across inference APIs and validate multimodal items individually.**
  Support tokenization, tool-call rendering, usage extraction, and length classification symmetrically across chat completions, legacy completions, responses, messages, and generate endpoints. Group per-item attributes into element structs instead of parallel slices, and validate multimodal metadata per item and modality rather than with request-wide booleans.
  Examples: [#848](https://github.com/llm-d/llm-d-router/issues/848), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3273772782), [#932](https://github.com/llm-d/llm-d-router/pull/932#discussion_r3178050269), [#1019](https://github.com/llm-d/llm-d-router/issues/1019), [#2723](https://github.com/llm-d/llm-d-router/pull/2723#discussion_r4165902195)
  Reviewers accept: Hashing multimodal items as whole units rather than chunking like text when the upstream inference engine hashes media items atomically ([#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4399431503), [#1009](https://github.com/llm-d/llm-d-router/issues/1009#issuecomment-4400172236)).

- **Key endpoint state by host, port, and worker rank.**
  Key per-endpoint state, cache indexes, eviction hooks, and event-socket ports by host, port, and worker rank rather than pod IP alone so multi-rank workers on one host never collide.
  Examples: [#841](https://github.com/llm-d/llm-d-router/issues/841), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242332547), [#1738](https://github.com/llm-d/llm-d-router/pull/1738#discussion_r3474223470), [#2831](https://github.com/llm-d/llm-d-router/pull/2831#discussion_r4033597903)

- **Order disaggregated stages and isolate stage-specific payloads.**
  In disaggregated pipelines, run upstream encode and prefill stages before dependent decode stages, isolate prefill-only payload mutations from decode requests, and cancel downstream stages immediately when an upstream stage fails.
  Examples: [#1444](https://github.com/llm-d/llm-d-router/pull/1444#discussion_r3346561765), [#1653](https://github.com/llm-d/llm-d-router/pull/1653#discussion_r3412845630), [#2094](https://github.com/llm-d/llm-d-router/pull/2094#discussion_r3730268179), [#2299](https://github.com/llm-d/llm-d-router/pull/2299#discussion_r3731458268)

## Performance on the request path

- **Avoid heap allocations and unhoisted work on the request path.**
  Keep per-request routing, scoring, and event-decoding paths free of avoidable byte-to-string conversions, slice or map clones, untyped map unmarshaling, and unmutated payload re-serialization. Hoist header normalization, extractor lookup maps, and type checks to startup, pre-allocate slices of known length, and fold consecutive read-locked lookups into one pass.
  Examples: [#683](https://github.com/llm-d/llm-d-router/pull/683#discussion_r3143174078), [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3367176286), [#880](https://github.com/llm-d/llm-d-router/pull/880#discussion_r3149083076), [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236327373)

- **Reuse pooled HTTP transports and multiplex per-endpoint event subscriptions.**
  Configure backend and metrics-scraper HTTP clients with keep-alive connection pooling, swapping the transport only when TLS certificates rotate rather than per scrape. Multiplex per-endpoint ZMQ or event subscriptions across in-process consumers instead of opening duplicate sockets per plugin, and batch periodic background network sweeps off the request path.
  Examples: [#890](https://github.com/llm-d/llm-d-router/pull/890#discussion_r3224041877), [#2045](https://github.com/llm-d/llm-d-router/issues/2045), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r4095311906), [#2837](https://github.com/llm-d/llm-d-router/pull/2837#issuecomment-6087844569)

- **Evaluate orthogonal load signals over filtered candidate subsets.**
  Keep queue depth, in-flight requests, and KV-cache utilization as separate signals evaluated over the filtered candidate subset, and release every in-flight increment on cancellation or error.
  Examples: [#1534](https://github.com/llm-d/llm-d-router/pull/1534#discussion_r3401170547), [#1580](https://github.com/llm-d/llm-d-router/pull/1580#discussion_r3391563214), [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3643867571), [#2066](https://github.com/llm-d/llm-d-router/pull/2066#discussion_r3605216570)
  Reviewers accept: Constant-time byte-length token estimation and single-endpoint soft-pinning session affinity when backed by engine contracts or multi-pod benchmarks ([#1438](https://github.com/llm-d/llm-d-router/pull/1438#discussion_r3338979029), [#2968](https://github.com/llm-d/llm-d-router/pull/2968#discussion_r4214852975)).

- **Score prefix caches contiguously and break near-ties by load.**
  Score prefix caches contiguously per sequence by halting block credit at the first missing block, isolate cache keys by tenant salt, and break near-ties with load-weighted sampling so hot prefixes do not overload a single worker.
  Examples: [#860](https://github.com/llm-d/llm-d-router/pull/860#discussion_r3354232991), [#1727](https://github.com/llm-d/llm-d-router/pull/1727#discussion_r3513605134), [#2031](https://github.com/llm-d/llm-d-router/issues/2031), [#2032](https://github.com/llm-d/llm-d-router/pull/2032#discussion_r3610603777)

- **Gate routing policies and hot-path changes on reproducible before-and-after benchmarks.**
  Validate new scoring formulas, cache indexes, hot-path locks, and default threshold changes with reproducible microbenchmarks covering both hit and miss paths alongside cluster evaluations against active routing baselines. Commit reproducible benchmark scripts and summary reports while keeping raw execution logs out of version control.
  Examples: [#655](https://github.com/llm-d/llm-d-router/issues/655), [#1006](https://github.com/llm-d/llm-d-router/issues/1006), [#1445](https://github.com/llm-d/llm-d-router/pull/1445#issuecomment-4605185125), [#3068](https://github.com/llm-d/llm-d-router/pull/3068#issuecomment-5930096809)

## Observability

- **Register metrics under canonical component subsystems with bounded label cardinality.**
  Register metrics under the component's canonical subsystem constant (`LLMDRouterEndpointPickerSubsystem` / `"llm_d_epp"`, `LLMDRouterCoordinatorSubsystem` / `"llm_d_coordinator"`, or the `llm_d_disagg_sidecar_` prefix; see [docs/metrics.md](metrics.md)), attach `plugin_name` and `plugin_type` labels to plugin-scoped metrics, and emit series only for configured plugins. Bound caller-influenced or pod-churn labels using closed enums, allowlists, or overflow buckets, expose live plugin state through scrape-time collectors, and delete series for removed endpoints so scaled-down pods do not leak stale metrics.
  Examples: [#707](https://github.com/llm-d/llm-d-router/pull/707#discussion_r3190768619), [#793](https://github.com/llm-d/llm-d-router/pull/793#discussion_r3324344552), [#1089](https://github.com/llm-d/llm-d-router/issues/1089#issuecomment-4703238704), [#1144](https://github.com/llm-d/llm-d-router/issues/1144#issuecomment-5943961626), [#1385](https://github.com/llm-d/llm-d-router/pull/1385#discussion_r3338737052)
  Reviewers accept: Emitting per-request ratio histograms alongside raw query and hit counters when the precomputed distribution simplifies operator alerting ([#1429](https://github.com/llm-d/llm-d-router/pull/1429#pullrequestreview-4493962366)).

- **Record at most one terminal outcome per request and keep latency histograms unbiased.**
  Emit at most one terminal outcome observation per request across early-return, buffered, and error paths, matching histogram bucket units to emitted values. Separate router processing overhead from backend inference wait, and skip stage duration observations when a stage is bypassed so sub-microsecond no-ops do not skew low buckets.
  Examples: [#1553](https://github.com/llm-d/llm-d-router/issues/1553), [#1658](https://github.com/llm-d/llm-d-router/pull/1658#discussion_r3618566988), [#1677](https://github.com/llm-d/llm-d-router/pull/1677#discussion_r3852749939)

- **Acquire package-scoped OTel tracers and propagate spans end to end.**
  Acquire OpenTelemetry tracers via `tracing.Tracer(TracerScope)` using a package-scoped `TracerScope` constant (`llm-d.ai/...`), standard semantic-convention attribute keys, and route templates rather than raw URLs. Extract inbound W3C trace context at ingress and propagate spans across all outbound HTTP and gRPC calls.
  Examples: [#708](https://github.com/llm-d/llm-d-router/issues/708), [#922](https://github.com/llm-d/llm-d-router/pull/922#discussion_r3178210848), [#1513](https://github.com/llm-d/llm-d-router/pull/1513#discussion_r3369205741), [#1531](https://github.com/llm-d/llm-d-router/pull/1531#discussion_r3376534183)

- **Guard per-item debug loops and cap stateful plugin dumps.**
  Follow the `go-logr` verbosity conventions in [AGENTS.md](../AGENTS.md#logging), guarding per-candidate and per-block debug loops behind `v.Enabled()`. Summarize collections by count in `String()`, and implement `DumpState` only on stateful plugins with sanitized entries capped to fixed top-K bounds.
  Examples: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3189062435), [#901](https://github.com/llm-d/llm-d-router/pull/901#discussion_r3252750496), [#972](https://github.com/llm-d/llm-d-router/pull/972#discussion_r3178428969), [#1074](https://github.com/llm-d/llm-d-router/issues/1074), [#1148](https://github.com/llm-d/llm-d-router/pull/1148#discussion_r3294713104)
  Reviewers accept: Omitting debug log statements on routine plugin no-op branches when sibling plugins are silent and proxy header traces already record the state ([#2319](https://github.com/llm-d/llm-d-router/pull/2319#issuecomment-5265371475)).

## Testing and CI

- **Exercise production factories and assert exact behavioral outputs.**
  Instantiate plugins and runners in tests through production factories and `fwkplugin.StrictDecoder` rather than constructing internal structs directly. Assert exact output fields, error types, Prometheus metric values, and JSON wire payloads without masking tested fields via broad `cmpopts.IgnoreFields`, covering both states of feature gates, skipped disaggregated stages, and boundary inputs.
  Examples: [#916](https://github.com/llm-d/llm-d-router/pull/916#discussion_r3215017290), [#972](https://github.com/llm-d/llm-d-router/pull/972#pullrequestreview-4216545185), [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3226190634), [#1030](https://github.com/llm-d/llm-d-router/pull/1030#discussion_r3289321053)
  Reviewers accept: Relying on end-to-end flow tests over unreachable unit combinations and keeping pure parsing functions separate from environment-variable wrappers so table-driven unit tests can call the parser directly ([#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524933896), [#1937](https://github.com/llm-d/llm-d-router/pull/1937#discussion_r3552704054)).

- **Colocate unit tests and synchronize without fixed sleeps.**
  Keep unit tests in colocated `*_test.go` files beside the source file they test, move paired test files whenever production types move, and place shared mocks or builders in leaf test packages that cannot cause import cycles. Synchronize asynchronous startup and controller reconciliation with injectable clocks, lifecycle barriers, or `require.Eventually` rather than fixed `time.Sleep` calls, and bind ephemeral loopback listeners atomically without close-then-rebind port races.
  Examples: [#1066](https://github.com/llm-d/llm-d-router/issues/1066), [#1188](https://github.com/llm-d/llm-d-router/issues/1188), [#1528](https://github.com/llm-d/llm-d-router/pull/1528#discussion_r3369900484), [#2865](https://github.com/llm-d/llm-d-router/pull/2865#discussion_r4016088694)
  Reviewers accept: Calibrated sleep windows when testing refresh-on-read cache TTL expiration where polling the read method would reset the entry TTL ([#1396](https://github.com/llm-d/llm-d-router/pull/1396#discussion_r3369095849)).

- **Isolate parallel end-to-end suites with non-aborting teardown.**
  Run end-to-end tests against freshly built pull request images and shipping Helm charts using per-process namespaces and namespace-scoped port forwarding rather than cluster-global NodePorts. Record created cluster resources before readiness waits, run non-aborting cleanup across all resources on teardown, and dump verbose pod logs only when a test fails.
  Examples: [#1048](https://github.com/llm-d/llm-d-router/pull/1048#pullrequestreview-4310234203), [#1481](https://github.com/llm-d/llm-d-router/pull/1481#discussion_r3372638867), [#1976](https://github.com/llm-d/llm-d-router/pull/1976#discussion_r3568707788), [#2048](https://github.com/llm-d/llm-d-router/pull/2048#discussion_r3596786577)
  Reviewers accept: Maintaining separate E2E runner scripts backed by shared helper libraries when cluster ownership, image sets, and environment contracts differ ([#1698](https://github.com/llm-d/llm-d-router/pull/1698#issuecomment-4762216995)).

- **Expose every CI gate in `make presubmit` and pin container image tags.**
  Wire every linter, manifest validator, protobuf generator, and security check into `make presubmit` inside the version-pinned builder container so local runs match CI across architectures. Pin container image tags in manifests and workflows rather than using `:latest`, keep local developer tool configs out of version control, and hold dependency updates that introduce unfixed upstream regressions.
  Examples: [#949](https://github.com/llm-d/llm-d-router/issues/949#issuecomment-4565428236), [#960](https://github.com/llm-d/llm-d-router/pull/960#discussion_r3182827088), [#1234](https://github.com/llm-d/llm-d-router/pull/1234#discussion_r3294875921), [#3159](https://github.com/llm-d/llm-d-router/pull/3159#issuecomment-6053625347)
  Reviewers accept: Pinning CI container images to immutable version release tags rather than opaque SHA256 digests so workflow diffs remain readable ([#3130](https://github.com/llm-d/llm-d-router/pull/3130#discussion_r4166716568)).

## Security and trust boundaries

- **Namespace router headers under `x-llm-d-` and scrub internal headers at ingress and egress.**
  Prefix router-managed HTTP headers with lowercase `x-llm-d-` (such as `x-llm-d-epp-profile` in `pkg/common/request` and routing headers in `pkg/epp/metadata`) and normalize inbound header lookups to lowercase at ingress. Strip client-supplied internal routing headers, deprecated header aliases, and untrusted transfer fields across all routes at ingress, and scrub internal pod topology headers before returning responses to clients.
  Examples: [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560), [#1027](https://github.com/llm-d/llm-d-router/pull/1027#discussion_r3219971477), [#1161](https://github.com/llm-d/llm-d-router/pull/1161#discussion_r3264933290), [#2319](https://github.com/llm-d/llm-d-router/pull/2319#discussion_r3758140767)
  Reviewers accept: Passing allowlist-validated `x-request-id` correlation headers through verbatim and documenting when body stamping overwrites client fields that take precedence over headers ([#1872](https://github.com/llm-d/llm-d-router/pull/1872#discussion_r3524966490), [#2716](https://github.com/llm-d/llm-d-router/pull/2716#discussion_r3955566592)).

- **Allowlist dynamic endpoint targets and reject control characters.**
  Validate peer endpoint URLs and KV-cache transfer targets against `InferencePool` host-and-port allowlists across all target ports and data-parallel ranks, blocking self-pulls and unverified internal cluster addresses. Reject whitespace and carriage-return or newline control characters in header-derived targets to prevent SSRF and request smuggling.
  Examples: [#979](https://github.com/llm-d/llm-d-router/issues/979#issuecomment-5944114560), [#1145](https://github.com/llm-d/llm-d-router/pull/1145#issuecomment-4479522771), [#1857](https://github.com/llm-d/llm-d-router/pull/1857#discussion_r3508246239), [#2234](https://github.com/llm-d/llm-d-router/pull/2234#discussion_r3682907408)

- **Enforce TLS minimums, namespace-scoped watches, and payload redaction.**
  Enforce minimum TLS versions and header-read timeouts on HTTP servers, trim trailing newlines when reading mounted secret files, and scope Kubernetes informers to the configured namespace rather than cluster-wide watches. Never log raw prompts, base64 media payloads, or end-user identifiers at debug or default verbosity.
  Examples: [#1007](https://github.com/llm-d/llm-d-router/pull/1007#discussion_r3203628868), [#1350](https://github.com/llm-d/llm-d-router/pull/1350#discussion_r3310324692), [#2837](https://github.com/llm-d/llm-d-router/pull/2837#discussion_r4225473483)

## Documentation and comments

- **Colocate plugin READMEs and link to canonical upstream guides.**
  Place each plugin's parameter schema, code-verified defaults, and minimal reference YAML example in a colocated `README.md` inside its package under `pkg/epp/framework/plugins` (see [AGENTS.md](../AGENTS.md#code-style) for single-source prose rules). Link to upstream `llm-d/llm-d` architecture guides or foundation policies rather than duplicating them, and use file or symbol links without fragile source line numbers.
  Examples: [#820](https://github.com/llm-d/llm-d-router/pull/820#issuecomment-4398304029), [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143455213), [#888](https://github.com/llm-d/llm-d-router/pull/888#discussion_r3143457636), [#959](https://github.com/llm-d/llm-d-router/pull/959#discussion_r3177858398)
  Reviewers accept: Folding small configuration examples directly into the plugin `README.md` rather than adding standalone deployment YAML files ([#1904](https://github.com/llm-d/llm-d-router/pull/1904#discussion_r3652301223)).

- **Synchronize docstrings, diagrams, and configuration examples across review rounds.**
  Re-verify affected package docstrings, architecture diagrams, and configuration examples whenever code or types change during review or rebase, and avoid hardcoded plugin or feature counts that drift as plugins are added.
  Examples: [#891](https://github.com/llm-d/llm-d-router/pull/891#discussion_r3236365197), [#933](https://github.com/llm-d/llm-d-router/pull/933#issuecomment-4864629122), [#1112](https://github.com/llm-d/llm-d-router/pull/1112#discussion_r3242601033), [#1128](https://github.com/llm-d/llm-d-router/pull/1128#discussion_r3280179807)

- **Document lock ordering, shallow-copy safety, and engine protocol quirks in code comments.**
  Follow [AGENTS.md](../AGENTS.md#code-style) for comment style, recording lock ordering preconditions, shallow-copy aliasing constraints, upstream engine protocol quirks, and deliberate interface contract overrides directly on the affected type or method.
  Examples: [#1115](https://github.com/llm-d/llm-d-router/pull/1115#discussion_r3306897196), [#1135](https://github.com/llm-d/llm-d-router/pull/1135#discussion_r3255416032), [#1207](https://github.com/llm-d/llm-d-router/pull/1207#discussion_r3267187365), [#1244](https://github.com/llm-d/llm-d-router/pull/1244#discussion_r3294891967)
