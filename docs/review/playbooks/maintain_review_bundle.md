---
type: Playbook
title: Maintain the review knowledge bundle
description: "Workflow for adding review rules, settling open questions into decisions, updating log.md, and regenerating docs/review_playbook.md."
tags: [playbook, maintenance, okf]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Settling an open question

1. Move the concept file with `git mv docs/review/open_questions/<slug>.md docs/review/decisions/<slug>.md`.
2. Change frontmatter `type` from `Open Question` to `Decision`, change `status` from `draft` to `stable`, add `verified: { by: human:<github-login>, at: <ISO-8601> }`, and replace `# Resolution criteria` with `# Outcome` citing the deciding comment in `sources`.
3. Update `open_questions/index.md`, `decisions/index.md`, and any rule block in `rules/<section>.md` linking to the former open question, then append a dated entry under `## YYYY-MM-DD` at the top of `log.md`.
4. Run `python3 hack/render_playbook.py` to regenerate `docs/review_playbook.md` and verify `python3 hack/render_playbook.py --check` exits 0.
