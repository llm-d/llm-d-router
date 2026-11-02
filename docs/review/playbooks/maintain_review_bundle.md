---
type: Playbook
title: Maintain the review knowledge bundle
description: "Workflow for keeping the two-stage docs/review/ bundle current: same-PR contract and routing updates, rule authoring in rules/*.md + provenance.json, open-question graduation, opt-in stale_after audits, and compiling docs/review_playbook.md."
tags: [playbook, maintenance, okf]
status: stable
---

# Maintenance loops

`make verify-review-playbook` runs `python3 hack/render_playbook.py --check` in `make presubmit`. It fails when `docs/review_playbook.md` differs from the compiled output, when rules in `docs/review/rules/*.md` and keys in `docs/review/provenance.json` are out of sync, when a Markdown link or anchor is broken, or when a first-column glob in [review_a_pull_request.md](review_a_pull_request.md)'s routing table matches nothing in the repository. It does not validate inline backticked code paths in rule prose, and an expired `stale_after` timestamp emits a stderr `WARNING:` without failing `make presubmit` (`--check-staleness` enforces expiration during audits). Each loop ends with `python3 hack/render_playbook.py` followed by `python3 hack/render_playbook.py --check`.

# Updating the bundle in the same pull request

A pull request that changes an architectural contract, deprecates or removes a surface, moves or deletes a top-level path matched by the routing table in [review_a_pull_request.md](review_a_pull_request.md), or settles a question under `open_questions/` updates `docs/review/` in that pull request.

1. Edit the affected Stage-1 rule in `rules/<section>.md`, metadata in `provenance.json`, or concept file under `decisions/` or `references/`, and update routing-table globs if top-level directory layouts moved.
2. Append a dated entry under `## YYYY-MM-DD` at the top of `log.md` naming the pull request and the files changed.
3. Run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Adding or refining a rule

Add or update a rule when maintainers repeat a subsystem or architectural correction across at least three review threads (and it is not already covered by `AGENTS.md` or automated CI linters), or agree on a recurring exception under `- Accept:`.

1. Add `## <Rule Title>` with 1-3 imperative sentences (plus optional `- Accept:` and `- Open:` one-sentence bounds) to `rules/<section>.md`, with zero YAML frontmatter or `[^id]` footnotes.
2. Add the matching `<section_stem>#<rule_anchor>` entry to `provenance.json` with `applies_to` globs, a specific one-sentence `consequence`, `observed_prs`, and 1-3 representative `examples` (plus optional `accept_decisions`, `accept_examples`, `decisions`, or `open_question`).
3. Update counts in [failure_classes.md](../references/failure_classes.md) and [index.md](../index.md) if the rule count changed, append a dated entry to `log.md`, and run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Settling an open question

1. Move the concept file with `git mv docs/review/open_questions/<slug>.md docs/review/decisions/<slug>.md`.
2. Change frontmatter `type` from `Open Question` to `Decision`, change `status` from `draft` to `stable`, add `verified: { by: human:<github-login>, at: <ISO-8601> }`, and replace `# Resolution criteria` with `# Outcome` citing the deciding comment in `sources`.
3. Remove the `- Open:` line from the rule in `rules/<section>.md`, move the `open_question` reference to `decisions` (or `accept_decisions`) in `provenance.json`, and update `open_questions/index.md` and `decisions/index.md`.
4. Append a dated entry to `log.md`, then run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Periodic stale_after audit

Every section in `provenance.json` carries a `stale_after` timestamp six months after its last audit. Run `python3 hack/render_playbook.py --check --check-staleness` to enforce freshness checks during scheduled maintainer audits.

1. Compare each rule against pull requests merged and review threads opened since the previous audit. Remove rules that duplicate CI or no longer match maintainer reviews, refine `- Accept:` bounds, and graduate settled open questions into `decisions/`.
2. Advance `stale_after` on every audited section in `provenance.json` by six months.
3. Append a dated `Audit` entry to `log.md`, then run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check --check-staleness`.
