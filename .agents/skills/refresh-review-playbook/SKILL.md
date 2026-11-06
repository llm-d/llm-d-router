---
name: refresh-review-playbook
description: Refresh the llm-d-router review playbook (`docs/review/` and `docs/review_playbook.md`) from maintainer PR reviews and issues mined since the `provenance.json` watermark. Use when asked to refresh, update, or holdout-benchmark the review playbook; produces validated non-code rule and decision updates ready for a PR.
---

# Refresh the llm-d-router Review Playbook

Refresh the two-stage `docs/review/` bundle and compiled `docs/review_playbook.md` guide from newly merged pull requests, maintainer review threads, and closed or updated issues since the watermark in `docs/review/provenance.json`.

Commands below run from the repository root. `<skill>` is this skill directory (`.agents/skills/refresh-review-playbook` in `llm-d-router`). The scripts are stdlib-only Python 3.10+; mining also needs an authenticated `gh` CLI.

## Hard Rules

1. **Non-code knowledge only**: Never add rules that repeat `AGENTS.md`, `.golangci.yml`, `Makefile`, `docs/architecture.md`, or contracts a reviewer can discover directly via `rg` across the touched package, interface definition, or sibling implementations.
2. **Preserve the Two-Pass Additive Review Protocol**: Stage-1 rules in `docs/review/rules/*.md` supplement Pass-1 code inspection. Never weaken the Pass-1 Retention Guard in `docs/review/playbooks/review_a_pull_request.md` or `AGENTS.md`, and never add broad `- Accept:` nit-suppression waivers.
3. **Respect the Temporal Holdout Firewall**: When refreshing up to a cutoff (`--until-pr <MAX_PR>` or `--until-date <ISO_DATE>`) for holdout benchmarking, zero PR citations, examples, or rules from PRs `> <MAX_PR>` or after `<ISO_DATE>` may enter `docs/review/` or `docs/review_playbook.md`.
4. **Pure ASCII and zero em-dashes**: `docs/review/**`, `docs/review_playbook.md`, and `AGENTS.md` must contain strictly 7-bit ASCII characters and zero `U+2014` em-dashes (`hack/render_playbook.py --check` rejects non-ASCII bytes).
5. **No unauthorized GitHub writes**: Mining is strictly read-only (`gh api graphql`). Follow the [AGENTS.md](../../../AGENTS.md) operating rules before pushing branches, opening PRs, or posting comments.

---

## Step 1: Read Current Watermark and Mine New Maintainer Reviews (`scripts/mine_delta.py`)

`docs/review/provenance.json` has exactly three top-level keys (`corpus`, `sections`, `rules`); `hack/render_playbook.py --check` rejects any other top-level key, so watermarks live under `corpus`. `scripts/mine_delta.py` derives the watermark in this order: explicit `--since-pr` / `--since-issue` / `--since-date`, then `corpus.mined_through_pr` / `corpus.mined_through_issue` / `corpus.mined_through_date`, then the highest `llm-d/llm-d-router` PR and issue numbers cited anywhere in `docs/review/` (`rules.*.examples`, decision and open-question `sources`) with `corpus.window_end` as the date. The cited-number fallback can re-mine PRs that were reviewed but not cited; that is safe, and writing `corpus.mined_through_pr` in Step 4 makes the next run exact.

```bash
python3 <skill>/scripts/mine_delta.py \
  --repo-root . \
  --output <scratch>/delta_candidates.json \
  [--until-pr <MAX_PR> --until-date <YYYY-MM-DD>]
```

- **Routine refresh (default to `HEAD`)**: Omit `--until-pr` and `--until-date` to mine every PR and updated issue from the watermark through `HEAD`.
- **Benchmarking with a temporal holdout firewall**: Pass both `--until-pr <MAX_PR>` and `--until-date <YYYY-MM-DD>`. PRs and issues share one GitHub number counter, so `--until-pr` also caps issue numbers unless `--until-issue` overrides it. `--until-date` drops review and issue comments posted after the cutoff; the script warns when `--until-pr` is passed alone.
- **What `mine_delta.py` extracts**:
  - Queries `gh api graphql` with pagination for PRs in `(since_pr, until_pr]` and issues updated since the watermark. It fails hard on GraphQL errors, on malformed dates, when `--max-pages` runs out before the watermark boundary, and when any nested `reviews`, `reviewThreads`, or `comments` connection reports `hasNextPage`.
  - Includes a PR when it was created by `--until-date`, then keeps only reviews and comments posted by the cutoff. `watermark_after.mined_through_pr` never passes the lowest PR still open, so the next run re-mines comments posted on it later.
  - Drops bot accounts (`github-actions`, `codecov`, `dependabot`, `gemini-code-assist`, `renovate`, `k8s-ci-robot`, `*[bot]`, `*-bot`) and self-only threads where no human other than the PR or issue author commented.
  - Tags each candidate with exactly one `suggested_category`: one of the 4 Stage-1 sections, `decisions` (settled or debated design direction for Stage 2), or `code_local_reject`. Tags are triage hints; Step 2 makes the call.
- Offline runs and tests use `--mock-graphql-file <json>` with `{"pullRequests": [...], "issues": [...]}`.

---

## Step 2: Non-Code Triage and Deduplication Rubric

Evaluate every candidate finding in `delta_candidates.json` against the **4-question Non-Code Admission Gate**. A candidate must pass all four checks before touching `docs/review/`:

1. **Not in `AGENTS.md`**: Is the finding already covered by `AGENTS.md` (constant reuse across packages, `go-logr` verbosity levels `DEFAULT`/`VERBOSE`/`DEBUG`/`TRACE`, DCO sign-off, PR minimalism, or prose rules)?
   - *If yes -> **REJECT**.*
2. **Not discoverable by code search (`rg`)**: Can a reviewer with full-repo `rg` discover this contract simply by reading the touched package, interface definition, callers, or adjacent sibling implementation?
   - *If yes -> **REJECT** (Pass 1 already catches code-discoverable bugs).*
3. **Fits one of the 4 Stage-1 Non-Code Rule Categories or Stage-2 Architectural History**:
   - `docs/review/rules/external_wire_and_engine_contracts.md`: External vLLM, SGLang, Triton, Anthropic, OpenAI, OTel, ZMQ, or Gateway API wire formats, msgpack field orderings, SSE `[DONE]` or chunk framing, token-accounting quirks, and streaming vs diffusion endpoints.
   - `docs/review/rules/unencoded_runtime_hazards.md`: Cross-cutting runtime, concurrency, memory, lock, or metric-cardinality traps that look idiomatic locally (for example, `bytes.Buffer.Bytes()` backing-array retention across streaming lifetime, unjoined timed-out goroutines mutating shared request state or `sync.Pool` buffers, holding a mutex across external network or Redis calls, unbounded per-pod or per-request metric labels in hot loops, or mutating caller-owned slices/maps across goroutines).
   - `docs/review/rules/governance_and_compatibility_policy.md`: Multi-release `N+2` deprecation windows for shipped surfaces vs zero shims for unshipped surfaces, `Alpha`/`Beta`/`Stable` plugin graduation and frozen `test/testdata/plugins/stable/v1/` fixtures, GIE CRD and unmerged upstream engine coupling gates, env var proliferation bans, Helm (`config/charts/`) vs local Kustomize (`test/e2e/artifacts/`) parity, and `VerifyClusterState` e2e requirements.
   - `docs/review/rules/settled_architectural_boundaries.md`: Settled subsystem ownership boundaries and anti-patterns rejected in prior PRs/issues (`SaturationDetector`/`AdmissionPlugin` vs `Filter` shedding, `PrepareRequestData` vs `Score`, `EndpointCandidates` discovery vs routing, single base model and role per `InferencePool`, external conversation history vs bounded routing state, `pkg/coordinator` vs `pkg/sidecar`, status-code translation ownership, and ingress-only header stripping).
   - **Stage-2 `docs/review/decisions/*.md` or `docs/review/open_questions/*.md`**: Settled design rationale (including graduating a resolved `open_questions/*.md` file to `decisions/*.md` via `git mv`) or newly surfaced unresolved maintainer tradeoffs (`status: draft`).
4. **General Domain Invariant (Zero Overfitting)**:
   - Formulate every Stage-1 rule as a general architectural, runtime, governance, or wire invariant with a concrete `When <diff condition>:` trigger, never overfitting to a single variable name or one-off typo. `hack/render_playbook.py` parses exactly this grammar (no frontmatter, no `###` headings, no `[^id]` footnotes, no inline PR links):
     ```markdown
     ## <Rule Title>
     When <specific diff condition>: <actionable check in 1 to 3 plain ASCII sentences>.
     - Accept: <optional settled exception>
     - Open: <optional unresolved question>
     ```
   - Citations live only in `docs/review/provenance.json` under `rules["<section_stem>#<rule_anchor>"].examples` (`1` to `3` entries with `id`, `url`, `author`, `last_modified`), never in the rule file.
   - Check existing rules in the target file first: if the new PR or issue reinforces an existing rule, increment `observed_prs` and swap it into `examples` (keep at most `3` representative citations) instead of creating a duplicate rule.

---

## Step 3: Token Budget, Routing Cap, and Pass-1 Retention Invariants

Enforce these structural bounds whenever editing `docs/review/`:

1. **Exactly 4 Stage-1 Rule Files**: Keep only the 4 canonical rule files in `docs/review/rules/` (`external_wire_and_engine_contracts.md`, `unencoded_runtime_hazards.md`, `governance_and_compatibility_policy.md`, `settled_architectural_boundaries.md`, plus `index.md`).
2. **Routing Row Cap (`<= 2` Rule Files and `<= 4,400` Bytes per Row)**: Every row in `## Diff-path to rule-section routing` inside `docs/review/playbooks/review_a_pull_request.md` must route either to `None (AGENTS.md and repo search only)` or to `1` to `2` rule files whose combined size is `<= 4,400` bytes (about `1,100` tokens at `4` bytes per token). Every non-test `.go` file under `cmd/`, `pkg/`, and `internal/` must match at least one production routing row.
3. **Total Stage-1 Budget (`<= 9,500` Bytes / about `2,375` Tokens)**: Keep the combined size of the 4 Stage-1 rule files (excluding `rules/index.md`, which Pass 2 never loads) `<= 9,500` bytes. `validate_refresh.py` prints the current rule count and byte total on its first line (`Stage-1: <rules> rules, <bytes> bytes ...`). When a new rule approaches the cap, merge overlapping rules or prune rules that are now encoded in Go types, linters, or `AGENTS.md`.
4. **Pass-1 Retention Guard**: Keep the Two-Pass Additive Review Protocol and Pass-1 Retention Guard intact in both `docs/review/playbooks/review_a_pull_request.md` and `AGENTS.md`. `validate_refresh.py` requires both files to keep these two sentences verbatim (whitespace-normalized):
   - "Record all verified Pass-1 code findings before opening any Stage-1 rule file"
   - "never drop, suppress, or downgrade verified Pass-1 code"
5. **Procedure Stability**: A refresh updates Stage-1 rules and routing rows. Leave the review procedure steps and the `AGENTS.md` review section unchanged unless a holdout comparison shows the edit does not lower real bugs caught (maintainer comments plus verified extra bugs). Steps that prune output to cut noise also prune real findings.

---

## Step 4: Update `provenance.json`, Compile, and Validate (`scripts/validate_refresh.py`)

1. **Update `docs/review/provenance.json`**:
   - Update `corpus` (`window_end`, `pull_requests`, `maintainer_pr_items`, `issues`, `maintainer_issue_items`) and write the new `corpus.mined_through_pr`, `corpus.mined_through_issue`, and `corpus.mined_through_date` from `watermark_after` in `delta_candidates.json`. Never add top-level keys.
   - Advance `sections.*.stale_after` timestamps (6 months ahead in `YYYY-MM-DDTHH:MM:SSZ` format) when performing a periodic audit.
   - Ensure every `<section_stem>#<rule_anchor>` in `rules/*.md` has a matching entry in `provenance.json` (`rules`) with `applies_to`, a unique one-sentence `consequence`, `observed_prs >= 1`, and `1` to `3` `examples` whose `url` is an `llm-d/llm-d-router` PR or issue (plus synced `accept_decisions`/`accept_examples` for `- Accept:` and `open_question` for `- Open:`).
2. **Update index counts and changelog**:
   - If the rule, decision, or open-question count changed, update `docs/review/index.md`, `docs/review/references/failure_classes.md`, `docs/review/decisions/index.md`, and `docs/review/open_questions/index.md`.
   - Prepend a dated entry under `## YYYY-MM-DD` in `docs/review/log.md`.
3. **Compile and run deterministic validation**:
   ```bash
   python3 <skill>/scripts/validate_refresh.py \
     --repo-root . \
     --render \
     [--max-pr <MAX_PR>] \
     [--sync-prdraft]
   ```
   `validate_refresh.py` imports the parsers from `hack/render_playbook.py` under `--repo-root`, so it checks the same grammar CI enforces. It runs `hack/render_playbook.py` (with `--render`) and `hack/render_playbook.py --check --check-staleness`, then checks: the required bundle files, exactly the 4 canonical rule files plus `index.md`, the `provenance.json` schema, a `When <condition>: <check>` body and at least one PR or issue citation per rule, the `9,500` byte Stage-1 budget (`--max-stage1-bytes`), the `2` file and `4,400` byte routing row cap (`--max-row-bytes`), the Pass-1 Retention Guard, `0` `U+2014` characters in `docs/review/**`, `docs/review_playbook.md`, and `AGENTS.md`, and (with `--max-pr <N>`) zero PR or issue citations above `N` in `docs/review/**`, `docs/review_playbook.md`, and `AGENTS.md`. Only when every check passes does the optional `--sync-prdraft` copy `docs/review_playbook.md` to a local review-tool knowledge file (`--prdraft-path`, default `~/.config/prdraft/knowledge/llm-d-router/review-playbook.md`).
4. **Script tests**: `python3 <skill>/scripts/test_refresh_skill.py -v` validates the enclosing checkout read-only and runs each failure case on a scratch copy. Set `REFRESH_TEST_WORKTREE=<checkout>` to validate another checkout. A copy of the skill outside the repository validates a fixture built from `scripts/testdata/` when that directory exists (see the test module docstring).

---

## Step 5: Holdout A/B Verification and Automation Integration

### Holdout A/B Verification Before Opening a PR

When testing a refreshed playbook against the baseline playbook on recent PRs:

1. Pick a split PR watermark `<SPLIT_PR>` (for example, the prior `provenance.json` watermark or an intermediate cutoff) and run Step 1 with `--until-pr <SPLIT_PR>` and Step 4 with `--max-pr <SPLIT_PR>` so the refreshed playbook contains zero citations above `<SPLIT_PR>`.
2. Select post-watermark holdout PRs (`PR > <SPLIT_PR>`) that have ground-truth maintainer review comments or post-merge bug fixes.
3. Run blind A/B reviews comparing the previous playbook against the refreshed playbook on the same holdout PR head SHAs, with the same reviewer prompt and model.
4. Verify two acceptance criteria before opening a PR:
   - **Zero Pass-1 regression**: The refreshed playbook retains 100% of Pass-1 code defects caught by the baseline.
   - **Non-negative net recall on non-code hazards**: Newly added or refined rules catch recurring non-code defects on the holdout PRs without introducing false-positive nit noise.

### Scheduled Refresh

To run `refresh-review-playbook` on a recurring schedule:

1. Create a branch or worktree from the latest upstream `main`.
2. Run `<skill>/scripts/mine_delta.py --repo-root <worktree> --output <scratch>/delta_candidates.json`.
3. If `delta_candidates.json` yields `0` qualifying non-code rules, decision graduations, or open-question updates after Step 2 triage, exit cleanly with a no-op summary (do not open a churn PR that only bumps dates).
4. When qualifying updates exist, apply Steps 2 through 4, verify `validate_refresh.py --repo-root <worktree> --render` exits `0`, and propose the refresh PR with `git commit -s` after `make verify-review-playbook` passes.
