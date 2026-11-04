---
okf_version: "0.2"
---

# llm-d-router Review Knowledge Bundle

Two-stage review bundle for `llm-d-router` designed for two-pass additive code review. Pass 1 inspects the diff, reads modified files in full, and searches the repository directly (`AGENTS.md`, `docs/architecture.md`, callers, sibling implementations, `pkg/common/`, `config/charts/`, and tests). Pass 2 routes touched paths via [review_a_pull_request.md](playbooks/review_a_pull_request.md) to `0` to `2` compact Stage-1 rule files under `rules/` that record non-code external wire contracts, non-local runtime hazards, release governance, and settled architectural boundaries. Stage 2 combines `rules/*.md`, [`provenance.json`](provenance.json), `decisions/`, and `open_questions/` into the human-readable guide at [`docs/review_playbook.md`](../review_playbook.md) via `python3 hack/render_playbook.py`. Agents must not load `docs/review_playbook.md` or `docs/review/provenance.json`.

## Start here

* [Review a pull request](playbooks/review_a_pull_request.md) - Two-pass additive review procedure and Stage-1 diff-path routing table for agents.
* [Review Playbook (human guide)](../review_playbook.md) - Stage-2 compiled single-page guide with rationale, exceptions, open questions, and PR examples.
* [Failure classes and section taxonomy](references/failure_classes.md) - Overview of the 4 Stage-1 rule sections and 17 non-code review rules.

## Contents

* [Review Rules](rules/index.md) - 4 compact Stage-1 rule files housing 17 `When:`-triggered non-code review invariants.
* [Provenance Sidecar](provenance.json) - Per-section `stale_after` timestamps and per-rule consequences, recurrence counts, and PR thread citations.
* [Decisions](decisions/index.md) - 28 settled maintainer tradeoffs (`MVM-*`) and human-verified policy decisions.
* [Open Questions](open_questions/index.md) - 12 unresolved architectural and operational questions (`status: draft`).
* [References](references/index.md) - Failure-class taxonomy, maintainer roster, and conflict-class glossary.
* [Playbooks](playbooks/index.md) - Procedures for reviewing, authoring, and maintaining this bundle.
