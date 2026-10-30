---
okf_version: "0.2"
---

# llm-d-router Review Knowledge Bundle

Canonical Open Knowledge Format (OKF) v0.2 source for `llm-d-router` engineering standards, architectural direction, settled review decisions, and open maintainer questions. The single-page human checklist at `docs/review_playbook.md` is generated deterministically from this directory via `python3 hack/render_playbook.py`.

## Start here

* [Review a pull request](playbooks/review_a_pull_request.md) - Diff-path routing table and review procedure for agents and human reviewers.
* [Failure classes and section taxonomy](references/failure_classes.md) - Overview of the 12 rule sections and defect categories.

## Directories

* [Review Rules](rules/index.md) - 12 `Review Rule Set` concept files housing 58 normative review rules and project-direction statements.
* [Decisions](decisions/index.md) - 28 settled maintainer tradeoffs (`MVM-*`) and human-verified policy decisions.
* [Open Questions](open_questions/index.md) - 12 unresolved architectural and operational questions (`status: draft`).
* [References](references/index.md) - Failure-class taxonomy, maintainer roster, and conflict-class glossary.
* [Playbooks](playbooks/index.md) - Agent and maintainer procedures for reviewing, authoring, and updating this bundle.
