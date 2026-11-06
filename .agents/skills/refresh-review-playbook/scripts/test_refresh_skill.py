#!/usr/bin/env python3
"""Unit and integration tests for the refresh-review-playbook scripts.

Validation tests run read-only against WORKTREE: REFRESH_TEST_WORKTREE when set,
otherwise a fixture repo built from testdata/ when it exists (a valid refreshed
bundle plus an empty file for every path in repo_paths.txt, so routing coverage
checks see the real repository layout), otherwise the llm-d-router checkout that
contains this skill under .agents/skills/. Every mutation runs on a copy under
the scratch root (REFRESH_TEST_SCRATCH, default a fresh temp dir).

To refresh the fixture from a checkout with a valid bundle:
  cp -r <repo>/{AGENTS.md,docs,hack} testdata/bundle/  (keep only the bundle files)
  git -C <repo> ls-tree -r --name-only HEAD | grep -vE \\
    '^(AGENTS.md|docs/review/|docs/review_playbook.md|hack/render_playbook.py)' \\
    > testdata/repo_paths.txt

Run: python3 test_refresh_skill.py -v
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

sys.dont_write_bytecode = True
SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import mine_delta  # noqa: E402
import validate_refresh  # noqa: E402

TESTDATA = SCRIPTS / "testdata"
SCRATCH_ROOT = pathlib.Path(
    os.environ.get("REFRESH_TEST_SCRATCH") or tempfile.mkdtemp(prefix="refresh-skill-test-")
)
# <repo>/.agents/skills/refresh-review-playbook/scripts -> <repo>
ENCLOSING_REPO = SCRIPTS.parents[3]
BUILD_FIXTURE = not os.environ.get("REFRESH_TEST_WORKTREE") and TESTDATA.is_dir()
if os.environ.get("REFRESH_TEST_WORKTREE"):
    WORKTREE = pathlib.Path(os.environ["REFRESH_TEST_WORKTREE"])
elif BUILD_FIXTURE:
    WORKTREE = SCRATCH_ROOT / "fixture-repo"
else:
    WORKTREE = ENCLOSING_REPO
COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "bin", ".worktrees")


def _build_fixture_repo(root: pathlib.Path) -> None:
    root.mkdir(parents=True)
    for rel in (TESTDATA / "repo_paths.txt").read_text(encoding="utf-8").splitlines():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    shutil.copytree(TESTDATA / "bundle", root, dirs_exist_ok=True)
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "-A"], cwd=root, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "fixture"], cwd=root, check=True)


def setUpModule() -> None:
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    if BUILD_FIXTURE:
        shutil.rmtree(WORKTREE, ignore_errors=True)
        _build_fixture_repo(WORKTREE)
    elif not (WORKTREE / "docs" / "review" / "provenance.json").is_file():
        raise RuntimeError(f"{WORKTREE} has no docs/review bundle; set REFRESH_TEST_WORKTREE")


def tearDownModule() -> None:
    if not os.environ.get("REFRESH_TEST_SCRATCH"):
        shutil.rmtree(SCRATCH_ROOT)


def _git_status(repo: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _comment(author: str, body: str, ts: str = "2026-11-01T10:00:00Z", cid: int = 1) -> Dict[str, Any]:
    return {
        "databaseId": cid,
        "body": body,
        "createdAt": ts,
        "url": f"https://github.com/llm-d/llm-d-router/pull/0#discussion_r{cid}",
        "author": {"login": author},
    }


def _pr(number: int, author: str, threads: List[List[Dict[str, Any]]], reviews: Any = None,
        merged: str = "2026-11-02T00:00:00Z", state: str = "MERGED",
        created: str = "2026-10-30T00:00:00Z") -> Dict[str, Any]:
    return {
        "number": number,
        "title": f"PR {number}",
        "url": f"https://github.com/llm-d/llm-d-router/pull/{number}",
        "state": state,
        "createdAt": created,
        "mergedAt": merged,
        "updatedAt": merged,
        "author": {"login": author},
        "reviews": {"nodes": reviews or []},
        "reviewThreads": {
            "nodes": [
                {"isResolved": True, "path": "pkg/kvevents/pool.go", "line": 3, "comments": {"nodes": t}}
                for t in threads
            ]
        },
    }


def _issue(number: int, author: str, comments: List[Dict[str, Any]],
           updated: str = "2026-11-02T00:00:00Z") -> Dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://github.com/llm-d/llm-d-router/issues/{number}",
        "createdAt": "2026-10-28T00:00:00Z",
        "updatedAt": updated,
        "author": {"login": author},
        "comments": {"nodes": comments},
    }


def _write_synthetic_repo(root: pathlib.Path, prov: Dict[str, Any]) -> pathlib.Path:
    bundle = root / "docs" / "review"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return root


class ScoringTest(unittest.TestCase):
    def test_bot_logins(self) -> None:
        for login in ("github-actions[bot]", "gemini-code-assist", "renovate", "foo-bot", "", None):
            self.assertTrue(mine_delta.is_bot_login(login), login)
        self.assertFalse(mine_delta.is_bot_login("ahg-g"))

    def test_categories(self) -> None:
        cases = {
            "vLLM continuous_usage_stats already reports usage per chunk": "external_wire_and_engine_contracts",
            "This goroutine keeps writing to the sync.Pool buffer after timeout": "unencoded_runtime_hazards",
            "Shipped flags need an N+2 deprecation window": "governance_and_compatibility_policy",
            "Shedding belongs in the SaturationDetector, not a filter": "settled_architectural_boundaries",
            "We agreed this is out of scope; please open a follow-up issue": "decisions",
            "nit: typo in variable name": "code_local_reject",
        }
        for text, want in cases.items():
            got = mine_delta.score_candidate_text(text)
            self.assertEqual(got["suggested_category"], want, text)
            self.assertEqual(got["non_code_candidate"], want != "code_local_reject", text)

    def test_path_hint_alone_is_rejected(self) -> None:
        got = mine_delta.score_candidate_text("please rename", path="pkg/kvevents/pool.go")
        self.assertEqual(got["suggested_category"], "code_local_reject")

    def test_non_word_edge_tokens_match_wire_category(self) -> None:
        cases = {
            "the stream must end with data: [DONE]": "external_wire_and_engine_contracts:sse_framing",
            "the final chunk is [DONE] here": "external_wire_and_engine_contracts:sse_framing",
            "POST /v1/responses needs previous_response_id": "external_wire_and_engine_contracts:http_wire",
            "proxy http://localhost:8000/v1/responses as-is": "external_wire_and_engine_contracts:http_wire",
            "call /render before tokenizing": "external_wire_and_engine_contracts:http_wire",
            "the /v1/messages route returns input_tokens": "external_wire_and_engine_contracts:http_wire",
        }
        for text, signal in cases.items():
            got = mine_delta.score_candidate_text(text)
            self.assertIn(signal, got["matched_signals"], text)
            self.assertEqual(got["suggested_category"], "external_wire_and_engine_contracts", text)
        self.assertNotIn(
            "external_wire_and_engine_contracts:http_wire",
            mine_delta.score_candidate_text("see /rendering docs")["matched_signals"],
        )


class DateAndWindowTest(unittest.TestCase):
    def test_missing_timestamp_rejected_under_upper_cutoff(self) -> None:
        cutoff = "2026-11-03T23:59:59Z"
        for ts in (None, ""):
            self.assertFalse(mine_delta.timestamp_in_window(ts, until_iso=cutoff), ts)
            self.assertTrue(mine_delta.timestamp_in_window(ts), ts)
        self.assertTrue(mine_delta.timestamp_in_window("2026-11-01T00:00:00Z", until_iso=cutoff))
        self.assertFalse(mine_delta.timestamp_in_window("2026-11-04T00:00:00Z", until_iso=cutoff))

    def test_comment_without_timestamp_dropped_by_until_date(self) -> None:
        undated = _comment("bob", "msgpack undated", cid=1)
        undated["createdAt"] = None
        nodes = [_pr(5, "alice", [[undated, _comment("carol", "msgpack dated", cid=2)]])]
        cands, _ = mine_delta.filter_and_extract_pr_candidates(nodes, since_pr=0, until_date="2026-11-03")
        self.assertEqual([hc["author"] for hc in cands[0]["thread_comments"]], ["carol"])

    def test_normalizes_valid_dates_to_utc(self) -> None:
        self.assertEqual(mine_delta.normalize_iso_cutoff("2026-11-03"), "2026-11-03T23:59:59Z")
        self.assertEqual(mine_delta.normalize_iso_cutoff("2026-11-03", is_end=False), "2026-11-03T00:00:00Z")
        self.assertEqual(mine_delta.normalize_iso_cutoff("2026-11-03T10:00:00Z"), "2026-11-03T10:00:00Z")
        self.assertEqual(mine_delta.normalize_iso_cutoff("2026-11-03T10:00:00-02:00"), "2026-11-03T12:00:00Z")

    def test_rejects_malformed_dates(self) -> None:
        for bad in ("2026-13-01", "2026-02-30", "yesterday", "11/03/2026", "2026-11-03T10:00:00"):
            with self.assertRaises(ValueError, msg=bad):
                mine_delta.normalize_iso_cutoff(bad)

    def test_cli_rejects_malformed_until_date(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            root = _write_synthetic_repo(pathlib.Path(tmp), {"corpus": {"window_end": "2026-10-26"}})
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS / "mine_delta.py"), "--repo-root", str(root),
                 "--until-date", "not-a-date", "--output", str(root / "out.json")],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid date", proc.stderr)

    def test_malformed_provenance_date_fails_hard(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            root = _write_synthetic_repo(pathlib.Path(tmp), {"corpus": {"mined_through_date": "26-10-2026"}})
            with self.assertRaisesRegex(ValueError, "mined_through_date"):
                mine_delta.extract_watermarks(root)


class WatermarkTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = pathlib.Path(tempfile.mkdtemp(dir=SCRATCH_ROOT, prefix="wm-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_real_worktree_watermark(self) -> None:
        wm = mine_delta.extract_watermarks(WORKTREE)
        prov = json.loads((WORKTREE / "docs/review/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(wm["mined_through_date"], prov["corpus"]["window_end"])
        cited = [
            int(ex["url"].split("/pull/")[1].split("#")[0])
            for r in prov["rules"].values()
            for ex in r["examples"]
            if "/pull/" in ex["url"]
        ]
        self.assertGreaterEqual(wm["mined_through_pr"], max(cited))

    def test_foreign_repo_urls_ignored_and_corpus_override_wins(self) -> None:
        prov = {
            "corpus": {"window_end": "2026-10-26"},
            "rules": {"x#y": {"examples": [
                {"url": "https://github.com/llm-d/llm-d-router/pull/120"},
                {"url": "https://github.com/vllm-project/vllm/pull/99999"},
                {"url": "https://github.com/llm-d/llm-d-router/issues/130"},
            ]}},
        }
        root = _write_synthetic_repo(self.tmp, prov)
        wm = mine_delta.extract_watermarks(root)
        self.assertEqual((wm["mined_through_pr"], wm["mined_through_issue"]), (120, 130))
        prov["corpus"]["mined_through_pr"] = 150
        _write_synthetic_repo(self.tmp, prov)
        self.assertEqual(mine_delta.extract_watermarks(root)["mined_through_pr"], 150)


class FilterTest(unittest.TestCase):
    def test_pr_window_bots_and_self_talk(self) -> None:
        nodes = [
            _pr(100, "alice", [[_comment("bob", "vLLM msgpack order")]]),
            _pr(101, "alice", [
                [_comment("alice", "self note on msgpack")],
                [_comment("github-actions[bot]", "msgpack lint")],
                [_comment("alice", "q"), _comment("bob", "msgpack field order matters", cid=7)],
            ], reviews=[
                {"databaseId": 9, "state": "COMMENTED", "body": "lgtm", "author": {"login": "bob"},
                 "submittedAt": "2026-11-01T00:00:00Z"},
                {"databaseId": 10, "state": "COMMENTED", "body": "Needs N+2 deprecation",
                 "author": {"login": "carol"}, "submittedAt": "2026-11-01T00:00:00Z"},
            ]),
            _pr(102, "alice", [[_comment("bob", "SaturationDetector owns shedding")]]),
        ]
        cands, stats = mine_delta.filter_and_extract_pr_candidates(nodes, since_pr=100, until_pr=101)
        self.assertEqual(stats["prs_in_window"], 1)
        self.assertEqual({c["pr_number"] for c in cands}, {101})
        kinds = sorted((c["kind"], c["reviewer"]) for c in cands)
        self.assertEqual(kinds, [("pull_request_review", "human:carol"), ("review_thread", "human:bob")])

    def test_until_date_drops_later_comments(self) -> None:
        nodes = [_pr(5, "alice", [[
            _comment("bob", "msgpack early", ts="2026-11-01T00:00:00Z", cid=1),
            _comment("carol", "msgpack late", ts="2026-11-09T00:00:00Z", cid=2),
        ]], merged="2026-11-02T00:00:00Z")]
        cands, _ = mine_delta.filter_and_extract_pr_candidates(nodes, since_pr=0, until_date="2026-11-03")
        self.assertEqual([hc["author"] for hc in cands[0]["thread_comments"]], ["bob"])

    def test_pr_created_before_cutoff_kept_when_merged_after(self) -> None:
        nodes = [
            _pr(5, "alice", [[_comment("bob", "msgpack early", ts="2026-11-01T00:00:00Z")]],
                merged="2026-11-20T00:00:00Z"),
            _pr(6, "alice", [[_comment("bob", "msgpack", ts="2026-11-05T00:00:00Z")]],
                created="2026-11-04T00:00:00Z"),
        ]
        cands, stats = mine_delta.filter_and_extract_pr_candidates(nodes, since_pr=0, until_date="2026-11-03")
        self.assertEqual(stats["prs_in_window"], 1)
        self.assertEqual([c["pr_number"] for c in cands], [5])

    def test_open_pr_holds_back_watermark(self) -> None:
        nodes = [
            _pr(n, "alice", [[_comment("bob", "msgpack")]], state=s)
            for n, s in ((103, "MERGED"), (104, "OPEN"), (106, "CLOSED"), (108, "MERGED"))
        ]
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            root = _write_synthetic_repo(pathlib.Path(tmp), {"corpus": {"window_end": "2026-10-26"}})
            report = mine_delta.build_delta_report(root, nodes, [], since_pr=100, since_issue=100)
            self.assertEqual(report["watermark_after"]["mined_through_pr"], 103)
            capped = mine_delta.build_delta_report(root, nodes, [], since_pr=100, since_issue=100, until_pr=107)
            self.assertEqual(capped["watermark_after"]["mined_through_pr"], 103)
            settled = [n for n in nodes if n["state"] != "OPEN"]
            done = mine_delta.build_delta_report(root, settled, [], since_pr=100, since_issue=100)
            self.assertEqual(done["watermark_after"]["mined_through_pr"], 108)
        self.assertEqual(sorted({c["pr_number"] for c in report["candidate_threads"]}), [103, 104, 106, 108])

    def test_nested_truncation_fails_hard(self) -> None:
        pr = _pr(7, "alice", [[_comment("bob", "msgpack")]])
        pr["reviewThreads"]["pageInfo"] = {"hasNextPage": True}
        with self.assertRaisesRegex(RuntimeError, r"PR #7: nested reviewThreads connection is truncated"):
            mine_delta.filter_and_extract_pr_candidates([pr], since_pr=0)

        pr = _pr(8, "alice", [[_comment("bob", "msgpack")]])
        pr["reviewThreads"]["nodes"][0]["comments"]["pageInfo"] = {"hasNextPage": True}
        with self.assertRaisesRegex(RuntimeError, r"PR #8: nested reviewThreads.comments"):
            mine_delta.filter_and_extract_pr_candidates([pr], since_pr=0)

        pr = _pr(9, "alice", [])
        pr["reviews"]["pageInfo"] = {"hasNextPage": True}
        with self.assertRaisesRegex(RuntimeError, r"PR #9: nested reviews"):
            mine_delta.filter_and_extract_pr_candidates([pr], since_pr=0)

        iss = _issue(210, "alice", [_comment("bob", "consensus")])
        iss["comments"]["pageInfo"] = {"hasNextPage": True}
        with self.assertRaisesRegex(RuntimeError, r"Issue #210: nested comments"):
            mine_delta.filter_and_extract_issue_candidates([iss], since_issue=0)

        complete = _pr(10, "alice", [[_comment("bob", "msgpack")]])
        complete["reviewThreads"]["pageInfo"] = {"hasNextPage": False}
        cands, _ = mine_delta.filter_and_extract_pr_candidates([complete], since_pr=0)
        self.assertEqual(len(cands), 1)

    def test_issue_self_only_and_shared_counter_cutoff(self) -> None:
        issues = [
            _issue(200, "alice", [_comment("alice", "bump")]),
            _issue(201, "alice", [_comment("bob", "we decided going forward to keep this out of scope")]),
            _issue(205, "alice", [_comment("bob", "consensus reached")]),
        ]
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            root = _write_synthetic_repo(pathlib.Path(tmp), {"corpus": {"window_end": "2026-10-26"}})
            report = mine_delta.build_delta_report(
                root, [], issues, since_pr=150, since_issue=150, until_pr=203
            )
        self.assertEqual([c["issue_number"] for c in report["candidate_issues"]], [201])
        self.assertEqual(report["candidate_issues"][0]["heuristics"]["suggested_category"], "decisions")
        self.assertEqual(report["watermark_after"]["mined_through_issue"], 203)


class PaginationTest(unittest.TestCase):
    def _page(self, field: str, nums: List[int], more: bool) -> Dict[str, Any]:
        return {"data": {"repository": {field: {
            "pageInfo": {"hasNextPage": more, "endCursor": "c"},
            "nodes": [{"number": n, "updatedAt": "2026-11-01T00:00:00Z"} for n in nums],
        }}}}

    def test_fails_hard_when_pages_run_out(self) -> None:
        with mock.patch.object(mine_delta, "_run_gh_graphql", return_value=self._page("pullRequests", [500, 499], True)):
            with self.assertRaisesRegex(RuntimeError, "max-pages"):
                mine_delta.fetch_github_nodes("llm-d/llm-d-router", 100, 100, None, max_pages=2)

    def test_fails_hard_on_graphql_errors(self) -> None:
        with mock.patch.object(mine_delta, "_run_gh_graphql", return_value={"errors": [{"message": "rate"}]}):
            with self.assertRaisesRegex(RuntimeError, "errors"):
                mine_delta.fetch_github_nodes("llm-d/llm-d-router", 100, 100, None)

    def test_stops_at_watermark(self) -> None:
        pages = [self._page("pullRequests", [300, 101], True), self._page("issues", [99], True)]
        with mock.patch.object(mine_delta, "_run_gh_graphql", side_effect=pages) as m:
            prs, issues = mine_delta.fetch_github_nodes("llm-d/llm-d-router", 101, 100, None)
        self.assertEqual((len(prs), len(issues), m.call_count), (2, 1, 2))


class MineCliTest(unittest.TestCase):
    def test_mock_graphql_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            root = _write_synthetic_repo(pathlib.Path(tmp), {
                "corpus": {"window_end": "2026-10-26", "mined_through_pr": 10, "mined_through_issue": 10},
            })
            mock_file = root / "mock.json"
            mock_file.write_text(json.dumps({
                "pullRequests": [_pr(11, "a", [[_comment("b", "vLLM SSE [DONE] framing")]]),
                                 _pr(12, "a", [[_comment("b", "msgpack")]])],
                "issues": [],
            }), encoding="utf-8")
            out = root / "out" / "delta.json"
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS / "mine_delta.py"), "--repo-root", str(root),
                 "--mock-graphql-file", str(mock_file), "--until-pr", "11", "--output", str(out)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--until-date", proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["watermark_before"]["mined_through_pr"], 10)
        self.assertEqual([c["pr_number"] for c in report["candidate_threads"]], [11])
        self.assertEqual(report["summary"]["category_breakdown"], {"external_wire_and_engine_contracts": 1})


class ValidateRealWorktreeTest(unittest.TestCase):
    def test_real_worktree_passes_read_only(self) -> None:
        before = _git_status(WORKTREE)
        errs, stats = validate_refresh.validate(WORKTREE, max_pr=10**6)
        self.assertEqual(errs, [])
        self.assertLessEqual(stats["stage1_bytes"], validate_refresh.DEFAULT_MAX_STAGE1_BYTES)
        self.assertEqual(_git_status(WORKTREE), before)


class ValidateMutationTest(unittest.TestCase):
    """Each test breaks one invariant on a scratch copy and expects a named error."""

    base: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        cls.base = pathlib.Path(tempfile.mkdtemp(dir=SCRATCH_ROOT, prefix="base-")) / "repo"
        shutil.copytree(WORKTREE, cls.base, ignore=COPY_IGNORE)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.base.parent)

    def setUp(self) -> None:
        self.repo = pathlib.Path(tempfile.mkdtemp(dir=SCRATCH_ROOT, prefix="mut-")) / "repo"
        shutil.copytree(self.base, self.repo)
        self.bundle = self.repo / "docs" / "review"

    def tearDown(self) -> None:
        shutil.rmtree(self.repo.parent)

    def _errors(self, **kwargs: Any) -> str:
        errs, _ = validate_refresh.validate(self.repo, **kwargs)
        return "\n".join(errs)

    def _edit(self, rel: str, old: str, new: str) -> None:
        path = self.repo / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def test_copy_is_valid_with_render(self) -> None:
        self.assertEqual(self._errors(render_first=True), "")

    def test_extra_rule_file(self) -> None:
        (self.bundle / "rules" / "misc.md").write_text("# Misc\n\n## R\nWhen x: y.\n", encoding="utf-8")
        self.assertIn("rules/misc.md: not one of the 4 canonical", self._errors())

    def test_rule_without_when_trigger(self) -> None:
        path = self.bundle / "rules" / "unencoded_runtime_hazards.md"
        text = path.read_text(encoding="utf-8")
        idx = text.index("\nWhen ") + 1
        path.write_text(text[:idx] + "Always " + text[idx + len("When "):], encoding="utf-8")
        self.assertIn("must start with 'When <condition>: <check>'", self._errors())

    def test_budget_overflow(self) -> None:
        self.assertIn("over the 5000 byte", self._errors(max_stage1_bytes=5000))

    def test_routing_row_byte_cap(self) -> None:
        self.assertIn("bytes (max 2000)", self._errors(max_row_bytes=2000))

    def test_retention_guard_removed(self) -> None:
        self._edit("AGENTS.md", "never drop, suppress, or downgrade", "may drop")
        self.assertIn("AGENTS.md: Pass-1 Retention Guard phrase missing", self._errors())

    def test_em_dash_in_agents_md(self) -> None:
        with (self.repo / "AGENTS.md").open("a", encoding="utf-8") as f:
            f.write("\nTrailing note \u2014 here.\n")
        self.assertIn("AGENTS.md", self._errors())
        self.assertIn("contains U+2014", self._errors())

    def test_temporal_firewall(self) -> None:
        errs = self._errors(max_pr=3000)
        self.assertIn("docs/review/provenance.json", errs)
        self.assertIn("docs/review_playbook.md", errs)
        self.assertIn("above --max-pr 3000", errs)

    def test_temporal_firewall_scans_agents_md(self) -> None:
        self.assertNotIn("AGENTS.md:", self._errors(max_pr=10**6))
        with (self.repo / "AGENTS.md").open("a", encoding="utf-8") as f:
            f.write("\nSee PR #999999 and https://github.com/llm-d/llm-d-router/issues/999998.\n")
        errs = self._errors(max_pr=999990)
        self.assertRegex(errs, r"AGENTS\.md:\d+: cites PR 999999 above --max-pr 999990")
        self.assertRegex(errs, r"AGENTS\.md:\d+: cites issue 999998 above --max-pr 999990")

    def test_unknown_top_level_watermark_key(self) -> None:
        path = self.bundle / "provenance.json"
        prov = json.loads(path.read_text(encoding="utf-8"))
        prov["mined_through_pr"] = 3600
        path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
        self.assertIn("unknown top-level key 'mined_through_pr'", self._errors())

    def test_stale_rendered_playbook(self) -> None:
        with (self.repo / "docs" / "review_playbook.md").open("a", encoding="utf-8") as f:
            f.write("\nmanual edit\n")
        self.assertIn("--check --check-staleness failed", self._errors())
        self.assertEqual(self._errors(render_first=True), "")

    def test_sync_prdraft_cli(self) -> None:
        dest = self.repo.parent / "prdraft" / "review-playbook.md"
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "validate_refresh.py"), "--repo-root", str(self.repo),
             "--sync-prdraft", "--prdraft-path", str(dest)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(dest.read_bytes(), (self.repo / "docs" / "review_playbook.md").read_bytes())

    def test_sync_prdraft_skipped_on_failure(self) -> None:
        dest = self.repo.parent / "prdraft" / "review-playbook.md"
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "validate_refresh.py"), "--repo-root", str(self.repo),
             "--max-pr", "3000", "--sync-prdraft", "--prdraft-path", str(dest)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
