# Update Log

## 2026-11-02

* **Update**: Refactored the review bundle into a two-stage workflow: stripped YAML frontmatter, `sources:`, and `[^id]` footnotes from Stage-1 `rules/*.md`, removed 5 rules duplicating `AGENTS.md` and CI linters (58 to 53 rules), moved all citation provenance, `stale_after` timestamps, recurrence counts, and per-rule consequences into [`provenance.json`](provenance.json), compiled a human-readable [`docs/review_playbook.md`](../review_playbook.md) via `hack/render_playbook.py`, restricted path existence validation to routing-table globs in [review_a_pull_request.md](playbooks/review_a_pull_request.md), and moved `stale_after` expiration enforcement into opt-in `--check-staleness`.
* **Update**: Added `stale_after` freshness, `Applies to` glob, diff-path routing table glob, and backticked repository path existence checks to `hack/render_playbook.py`, and expanded [Maintain the review knowledge bundle](playbooks/maintain_review_bundle.md) with the same-PR, rule-authoring, and periodic audit loops.

## 2026-11-01

* **Update**: Capped per-rule representative citations at three threads across `rules/`, compacted `sources:` entries to single-line YAML flow mappings, added unused-source validation to `hack/render_playbook.py`, and recorded context-bounded loading guidance in [Review a pull request](playbooks/review_a_pull_request.md).

## 2026-10-29

* **Creation**: Imported 12 `Review Rule Set` concepts (58 rules) under `rules/`, 28 `Decision` concepts under `decisions/` (including 5 human-verified maintainer decisions by `human:liu-cong`), 12 `Open Question` concepts under `open_questions/`, 3 `Reference` concepts under `references/`, and 3 `Playbook` concepts under `playbooks/` from commit `c50f49e7` and the `2026-04-26` to `2026-10-26` GitHub review dump.
