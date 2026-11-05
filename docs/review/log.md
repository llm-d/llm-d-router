# Update Log

## 2026-11-05

* **Update**: Mined maintainer review threads and issues for pull requests `#3512` to `#3744` with comments through `2026-10-30T14:00:00Z` (59 pull requests, 85 review threads, 35 issues). Added 3 Stage-1 rules (router-built P/D `kv_transfer_params`, bounded metric labels and series cleanup, no locks held across network calls), refined the SSE chunk-boundary usage rule, the client header passthrough rule (`Accept-Encoding`), and the ingress stripping rule (every client entry point), dropped the constant-location sentence already covered by [AGENTS.md](../../AGENTS.md), routed `pkg/*/metrics/**` and observability packages to [unencoded_runtime_hazards.md](rules/unencoded_runtime_hazards.md), and recorded `corpus.mined_through_pr` `3744` in [`provenance.json`](provenance.json).
* **Update**: Matched routing globs and rule paths to the current tree: `pkg/epp/framework/plugins/requesthandling/**` replaces the merged `requestcontrol` plugin and interface packages (`#3830`), `pkg/common/grpc/**` routes with the engine wire contracts, and `test/e2e/artifacts/` replaces the removed `deploy/` Kustomize tree (`#3034`). Raised the documented Stage-1 budget in [maintain_review_bundle.md](playbooks/maintain_review_bundle.md) to `9,500` bytes.

## 2026-11-03

* **Update**: Pruned Stage-1 `rules/*.md` from 12 files (53 rules) to 4 files (17 `When:`-triggered non-code rules) by removing contracts inferable from repository code search and `AGENTS.md`, removed broad `- Accept:` nit-suppression waivers, updated `AGENTS.md` and [review_a_pull_request.md](playbooks/review_a_pull_request.md) to the two-pass additive review protocol (`0` to `2` rule files per diff with explicit `None (AGENTS.md and repo search only)` routing rows), updated `hack/render_playbook.py` and [`provenance.json`](provenance.json), and regenerated [`docs/review_playbook.md`](../review_playbook.md).
* **Update**: Expanded diff-path routing in [review_a_pull_request.md](playbooks/review_a_pull_request.md) to cover all non-test Go packages under `cmd/`, `pkg/`, and `internal/`, added production Go routing coverage, unified `rules/index.md` section discovery, code-span-safe link validation, and `provenance.json` schema validation to `hack/render_playbook.py`, and removed duplicate verbosity-guard prose from `rules/observability.md`.

## 2026-11-02

* **Update**: Refactored the review bundle into a two-stage workflow: stripped YAML frontmatter, `sources:`, and `[^id]` footnotes from Stage-1 `rules/*.md`, removed 5 rules duplicating `AGENTS.md` and CI linters (58 to 53 rules), moved all citation provenance, `stale_after` timestamps, recurrence counts, and per-rule consequences into [`provenance.json`](provenance.json), compiled a human-readable [`docs/review_playbook.md`](../review_playbook.md) via `hack/render_playbook.py`, restricted path existence validation to routing-table globs in [review_a_pull_request.md](playbooks/review_a_pull_request.md), and moved `stale_after` expiration enforcement into opt-in `--check-staleness`.
* **Update**: Added `stale_after` freshness, `Applies to` glob, diff-path routing table glob, and backticked repository path existence checks to `hack/render_playbook.py`, and expanded [Maintain the review knowledge bundle](playbooks/maintain_review_bundle.md) with the same-PR, rule-authoring, and periodic audit loops.

## 2026-11-01

* **Update**: Capped per-rule representative citations at three threads across `rules/`, compacted `sources:` entries to single-line YAML flow mappings, added unused-source validation to `hack/render_playbook.py`, and recorded context-bounded loading guidance in [Review a pull request](playbooks/review_a_pull_request.md).

## 2026-10-29

* **Creation**: Imported 12 `Review Rule Set` concepts (58 rules) under `rules/`, 28 `Decision` concepts under `decisions/` (including 5 human-verified maintainer decisions by `human:liu-cong`), 12 `Open Question` concepts under `open_questions/`, 3 `Reference` concepts under `references/`, and 3 `Playbook` concepts under `playbooks/` from commit `c50f49e7` and the `2026-04-26` to `2026-10-26` GitHub review dump.
