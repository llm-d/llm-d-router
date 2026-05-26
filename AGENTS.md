# AGENTS.md

llm-d Router. Go service that routes inference requests to model-serving pods via an Endpoint Picker (EPP), with a sidecar that coordinates disaggregated inference. Multi-vendor open source under the llm-d project — review bandwidth is a shared community resource, so scope work tightly and discuss substantive changes in the open before code lands.

`make help` lists targets. `make presubmit` is the pre-merge gate. All targets run inside a builder container; host Go is not required.

## Working in the codebase

- State your interpretation before coding. When the task has multiple valid reads, ask — don't pick one silently. For clear failure signals (logs, failing tests, reproducer), act — the ask rule is about unclear requirements, not unclear bugs.
- Define success as a checkable outcome — "add validation" becomes "write failing tests for invalid inputs, then make them pass".
- Before changing or extending a component, read an analogous one in the repository. The closest existing implementation is the canonical pattern — follow its structure, naming, and tests rather than introducing new conventions.
- The plugin model is the main extension surface. Start at [docs/architecture.md](docs/architecture.md); existing filters, scorers, and profile handlers are the canonical references.
- Tests in the same package describe the contract. Read them before changing behavior.
- Verify behavior against the code, not from filenames or familiarity. Run the build or read the test when uncertain.
- If execution goes sideways — unexpected state, cascading failures, a fix that breaks adjacent code — stop and re-plan. Restate what you know, identify where the plan broke, propose a revised path before continuing.

## Pull requests

- Minimalism: smallest correct change inside the smallest scope.
- Non-trivial work must be tracked in an issue. If there isn't one, ask the user to file or link it.
- The PR addresses that issue and nothing else — no renames, reformatting, refactors, new abstractions, or pattern changes beyond what the issue requires.
- Unrelated improvements belong in their own issue, not folded into this PR. If you spot dead code or unrelated bugs in passing, mention them — don't fix them.
- Self-check on the way out: if the change grew larger than expected or the fix feels hacky, rewrite the clean version before opening the PR.

## Code style

- Standard Go. `make format` and `make lint` are authoritative.
- Comments are terse and only present when the WHY is non-obvious. Never paraphrase the code.
- Docs and comments describe the current state on its own terms. No "previously", "now", "recently", "renamed from", "added to fix", or other temporal or conversational framing. A reader with no context for the change must still understand the text.
- State each fact once, in its canonical location. Do not duplicate across struct docs, prose, tables, inline comments, and examples.
- No emojis unless explicitly requested.

## Git workflow

- DCO sign-off is required. Use `git commit -s`.
- Commit subject: imperative, ~72 characters. Body short and focused on the WHY; long narrative belongs in the PR description.
- Do not add machine-generated co-author trailers. Sign-off is the only required trailer.
- Do not bypass hooks (`--no-verify`) or signing checks.

## Agent operating rules

**Allowed.** Edit code, run `make` targets, read the codebase and GitHub state, push commits to the current feature branch when the task implies it.

**Ask first.** Rewriting pushed history, edits under `.github/` or to `OWNERS`, dependency upgrades.

**Never, without explicit per-turn authorization.** Public actions under the user's identity — GitHub comments, reviews, reactions, PR state changes, label or reviewer assignment, posts to Slack or any external surface. Draft such replies as quoted text for the user to send. Authorization is per-action and does not carry between actions or to sub-agents.
