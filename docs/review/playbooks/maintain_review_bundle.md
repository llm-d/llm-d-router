---
type: Playbook
title: Maintain the review knowledge bundle
description: "Workflow for keeping the two-stage docs/review/ bundle current: same-PR contract and routing updates, When:-framed non-code rule authoring in rules/*.md + provenance.json, open-question graduation, opt-in stale_after audits, and compiling docs/review_playbook.md."
tags: [playbook, maintenance, okf]
status: stable
---

# Maintenance loops

`make verify-review-playbook` runs `python3 hack/render_playbook.py --check` in `make presubmit`. It fails when `docs/review_playbook.md` differs from the compiled output, when rules in `docs/review/rules/*.md` and keys in `docs/review/provenance.json` are out of sync, when a Stage-1 rule omits its `When <condition>:` prefix, when a Markdown link or anchor is broken, when a routing row in [review_a_pull_request.md](review_a_pull_request.md) lacks either `None (AGENTS.md and repo search only)` or `1` to `2` rule file links, when a first-column glob matches nothing in the repository, or when a non-test Go file under `cmd/`, `pkg/`, or `internal/` matches no routing row. It does not validate inline backticked code paths in rule prose, and an expired `stale_after` timestamp emits a stderr `WARNING:` without failing `make presubmit` (`--check-staleness` enforces expiration during audits). Each loop ends with `python3 hack/render_playbook.py` followed by `python3 hack/render_playbook.py --check`.

# Updating the bundle in the same pull request

A pull request that changes an architectural contract, deprecates or removes a surface, moves or deletes a top-level path matched by the routing table in [review_a_pull_request.md](review_a_pull_request.md), or settles a question under `open_questions/` updates `docs/review/` in that pull request.

1. Edit the affected Stage-1 rule in `rules/<section>.md`, metadata in `provenance.json`, or concept file under `decisions/` or `references/`, and update routing-table globs if directory layouts moved (routing self-contained packages with no non-code rule dependencies to `None (AGENTS.md and repo search only)`).
2. Append a dated entry under `## YYYY-MM-DD` at the top of `log.md` naming the pull request and the files changed.
3. Run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Adding or refining a rule

Add or update a rule only when maintainers establish a recurring external contract, non-local runtime hazard, release policy, or architectural boundary that cannot be inferred by local code search, `AGENTS.md`, `docs/architecture.md`, or CI linters.

1. Add `## <Rule Title>` starting with `When <specific diff condition>:` in 1-3 crisp ASCII sentences (plus optional `- Accept:` settled architectural exceptions and `- Open:` unsettled debates) to `rules/<section>.md`, with zero YAML frontmatter or `[^id]` footnotes. Do not add broad `- Accept:` nit-suppression waivers or rules inferable from sibling `.go` files.
2. Add the matching `<section_stem>#<rule_anchor>` entry to `provenance.json` with `applies_to` globs, a unique one-sentence `consequence`, `observed_prs`, and 1-3 representative `examples` (plus optional `accept_decisions`, `accept_examples`, `decisions`, or `open_question`).
3. Keep the combined Stage-1 `rules/*.md` corpus under `8,000` bytes and `2,000` tokens across the 4 files, update counts in [failure_classes.md](../references/failure_classes.md) and [index.md](../index.md) if the rule count changed, append a dated entry to `log.md`, and run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Settling an open question

1. Move the concept file with `git mv docs/review/open_questions/<slug>.md docs/review/decisions/<slug>.md`.
2. Change frontmatter `type` from `Open Question` to `Decision`, change `status` from `draft` to `stable`, add `verified: { by: human:<github-login>, at: <ISO-8601> }`, and replace `# Resolution criteria` with `# Outcome` citing the deciding comment in `sources`.
3. Remove the `- Open:` line from the rule in `rules/<section>.md`, move the `open_question` reference to `decisions` (or `accept_decisions`) in `provenance.json`, and update `open_questions/index.md` and `decisions/index.md`.
4. Append a dated entry to `log.md`, then run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check`.

# Periodic stale_after audit

Every section in `provenance.json` carries a `stale_after` timestamp six months after its last audit. Run `python3 hack/render_playbook.py --check --check-staleness` to enforce freshness checks during scheduled maintainer audits.

1. Compare each rule against pull requests merged and review threads opened since the previous audit. Prune rules that have become directly visible in code, tests, or CI linters, verify `- Accept:` architectural exceptions, and graduate settled open questions into `decisions/`.
2. Advance `stale_after` on every audited section in `provenance.json` by six months.
3. Append a dated `Audit` entry to `log.md`, then run `python3 hack/render_playbook.py` and `python3 hack/render_playbook.py --check --check-staleness`.
