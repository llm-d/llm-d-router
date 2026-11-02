---
type: Playbook
title: Maintain the review knowledge bundle
description: "Workflow for keeping docs/review/ current: same-PR updates when contracts or paths change, adding or refining rules from repeated review corrections, periodic stale_after audits, and regenerating docs/review_playbook.md."
tags: [playbook, maintenance, okf]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Maintenance loops

`make verify-review-playbook` runs `python3 hack/render_playbook.py --check` in `make presubmit`. It fails when `docs/review_playbook.md` differs from the rendered bundle, when a Markdown link or anchor is broken, when a backticked repository path, an `Applies to` glob, or a diff-path routing table glob in `docs/review/` matches nothing in the repository, or when a `stale_after` timestamp in `rules/*.md` is missing, malformed, or in the past. Each loop ends with `python3 hack/render_playbook.py` followed by `python3 hack/render_playbook.py --check`.

# Updating the bundle in the same pull request

A pull request that changes an architectural contract, deprecates or removes a surface, moves or deletes a path referenced by `docs/review/`, or settles a question under `open_questions/` updates the matching `docs/review/` file in that pull request.

1. Edit the rule, decision, or reference that states the contract. Update `Applies to` globs and backticked paths to the new layout.
2. Append a dated entry under `## YYYY-MM-DD` at the top of `log.md` naming the pull request and the files changed.
3. Regenerate and verify.

# Adding or refining a rule

Add or update a rule when maintainers repeat the same correction across at least three review threads, or agree on a recurring exception that belongs under `Reviewers accept`.

1. Add the rule block to `rules/<section>.md` with `### Rule`, `### Evidence`, and optional `### Reviewers accept` and `### Open question` subsections. Cite at most three threads under `Representative review threads:` and at most eight distinct citations per rule, each registered in frontmatter `sources`.
2. Set `Applies to` to the globs the rule governs.
3. Append a dated entry to `log.md`, then regenerate and verify.

# Settling an open question

1. Move the concept file with `git mv docs/review/open_questions/<slug>.md docs/review/decisions/<slug>.md`.
2. Change frontmatter `type` from `Open Question` to `Decision`, change `status` from `draft` to `stable`, add `verified: { by: human:<github-login>, at: <ISO-8601> }`, and replace `# Resolution criteria` with `# Outcome` citing the deciding comment in `sources`.
3. Update `open_questions/index.md`, `decisions/index.md`, and any rule block in `rules/<section>.md` linking to the former open question, then append a dated entry to `log.md`.
4. Regenerate and verify.

# Periodic stale_after audit

Every `Review Rule Set` under `rules/` carries a `stale_after` timestamp six months after its last audit. `make verify-review-playbook` fails once that date passes, so the audit happens before expiry.

1. Compare each rule against pull requests merged and review threads opened since the previous audit. Remove rules that no longer match maintainer behavior, tighten `Reviewers accept` bounds that drifted, and settle resolved open questions into `decisions/`.
2. Advance `stale_after` on every audited rule set by six months.
3. Append a dated `Audit` entry to `log.md`, then regenerate and verify.
