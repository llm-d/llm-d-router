#!/usr/bin/env python3
"""Validate a refreshed llm-d-router review playbook bundle before opening a PR.

Stdlib-only CLI. Reuses the parsers in <repo-root>/hack/render_playbook.py so
this script checks the same rule-file grammar the repository CI enforces, then
adds the refresh-specific invariants: the four canonical Stage-1 rule files,
rule triggers and citations, the Stage-1 byte budget, the per-row routing cap,
the Pass-1 Retention Guard and High-Signal Filter, zero U+2014 characters, and
the optional --max-pr temporal holdout firewall.
"""

import argparse
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mine_delta import ISSUE_CITATION_RX, PR_CITATION_RX, STAGE1_CATEGORIES  # noqa: E402

CANONICAL_RULE_FILES = frozenset(f"{stem}.md" for stem in STAGE1_CATEGORIES) | {"index.md"}
REQUIRED_BUNDLE_PATHS = (
    "index.md",
    "log.md",
    "provenance.json",
    "rules/index.md",
    "decisions/index.md",
    "open_questions/index.md",
    "playbooks/review_a_pull_request.md",
    "references/index.md",
)
ALLOWED_TOP_KEYS = frozenset({"corpus", "sections", "rules"})
REQUIRED_CORPUS_INT_KEYS = ("pull_requests", "maintainer_pr_items", "issues", "maintainer_issue_items")
OPTIONAL_CORPUS_INT_KEYS = ("mined_through_pr", "mined_through_issue")
ISO_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CITATION_URL_RX = re.compile(r"^https://github\.com/llm-d/llm-d-router/(?:pull|issues)/\d+\b")
TRIGGER_RX = re.compile(r"^When [^:]+:\s+\S")

DEFAULT_MAX_STAGE1_BYTES = 9500
DEFAULT_MAX_ROW_BYTES = 4400
BYTES_PER_TOKEN = 4

RETENTION_GUARD_PHRASES = (
    "Record all verified Pass-1 code findings before opening any Stage-1 rule file",
    "never drop, suppress, or downgrade verified Pass-1 code",
)
RETENTION_GUARD_FILES = ("docs/review/playbooks/review_a_pull_request.md", "AGENTS.md")
HIGH_SIGNAL_FILTER_PHRASE = (
    "Before posting, drop comments that only restyle code, reword comments, rename, "
    "or restructure tests without changing what they verify."
)
HIGH_SIGNAL_FILTER_FILE = "docs/review/playbooks/review_a_pull_request.md"
EM_DASH = "\u2014"
DEFAULT_PRDRAFT_PATH = pathlib.Path.home() / ".config/prdraft/knowledge/llm-d-router/review-playbook.md"


def load_render_module(repo_root: pathlib.Path) -> ModuleType:
    """Imports <repo-root>/hack/render_playbook.py as a module."""
    path = repo_root / "hack" / "render_playbook.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    spec = importlib.util.spec_from_file_location("render_playbook", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check_structure(bundle: pathlib.Path) -> List[str]:
    errs = [f"missing docs/review/{rel}" for rel in REQUIRED_BUNDLE_PATHS if not (bundle / rel).is_file()]
    rules_dir = bundle / "rules"
    if rules_dir.is_dir():
        present = {p.name for p in rules_dir.iterdir()}
        for extra in sorted(present - CANONICAL_RULE_FILES):
            errs.append(f"docs/review/rules/{extra}: not one of the 4 canonical Stage-1 rule files")
        for missing in sorted(CANONICAL_RULE_FILES - present):
            errs.append(f"docs/review/rules/{missing}: canonical Stage-1 rule file is missing")
    return errs


def check_provenance(prov: Any) -> List[str]:
    """Checks the provenance.json schema used by hack/render_playbook.py."""
    if not isinstance(prov, dict):
        return ["provenance.json: top-level value must be an object"]
    errs: List[str] = []
    for key in sorted(set(prov) - ALLOWED_TOP_KEYS):
        errs.append(f"provenance.json: unknown top-level key {key!r} (put watermarks under corpus)")

    corpus = prov.get("corpus")
    if not isinstance(corpus, dict):
        errs.append("provenance.json: corpus must be an object")
    else:
        for key in ("window_start", "window_end"):
            if not ISO_DATE_RX.match(str(corpus.get(key) or "")):
                errs.append(f"provenance.json: corpus.{key} must be YYYY-MM-DD")
        for key in REQUIRED_CORPUS_INT_KEYS:
            val = corpus.get(key)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                errs.append(f"provenance.json: corpus.{key} must be a non-negative integer")
        for key in OPTIONAL_CORPUS_INT_KEYS:
            val = corpus.get(key)
            if key in corpus and (isinstance(val, bool) or not isinstance(val, int) or val < 1):
                errs.append(f"provenance.json: corpus.{key} must be a positive integer")
        if "mined_through_date" in corpus and not ISO_DATE_RX.match(str(corpus["mined_through_date"])):
            errs.append("provenance.json: corpus.mined_through_date must be YYYY-MM-DD")

    sections = prov.get("sections")
    if not isinstance(sections, dict) or set(sections) != set(STAGE1_CATEGORIES):
        found = sorted(sections) if isinstance(sections, dict) else sections
        errs.append(f"provenance.json: sections must be exactly {sorted(STAGE1_CATEGORIES)}, got {found}")

    rules = prov.get("rules")
    if not isinstance(rules, dict):
        errs.append("provenance.json: rules must be an object")
        return errs
    for rkey, rprov in rules.items():
        if rkey.split("#", 1)[0] not in STAGE1_CATEGORIES:
            errs.append(f"provenance.json ({rkey}): rule key must start with a canonical section stem")
        if not isinstance(rprov, dict):
            continue
        for ex in rprov.get("examples") or []:
            url = ex.get("url") if isinstance(ex, dict) else None
            if not isinstance(url, str) or not CITATION_URL_RX.match(url):
                errs.append(f"provenance.json ({rkey}): example url {url!r} is not an llm-d-router PR or issue")
    return errs


def parse_rules(render: ModuleType, bundle: pathlib.Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    parsed: Dict[str, List[Dict[str, Any]]] = {}
    errs: List[str] = []
    for stem in STAGE1_CATEGORIES:
        path = bundle / "rules" / f"{stem}.md"
        if not path.is_file():
            continue
        try:
            _, rules = render.parse_stage1_rule_file(path.read_text(encoding="utf-8"), f"rules/{stem}.md")
        except ValueError as exc:
            errs.append(str(exc))
            continue
        parsed[stem] = rules
    return parsed, errs


def check_rule_triggers_and_citations(
    render: ModuleType, parsed: Dict[str, List[Dict[str, Any]]], prov: Any
) -> List[str]:
    """Every rule opens with 'When <condition>:' and cites at least one PR or issue."""
    rule_prov = prov.get("rules") if isinstance(prov, dict) and isinstance(prov.get("rules"), dict) else {}
    errs: List[str] = []
    for stem, rules in parsed.items():
        for rule in rules:
            title = str(rule["title"])
            body = str(rule["rule"] or "")
            if not TRIGGER_RX.match(body):
                errs.append(f"rules/{stem}.md ({title}): body must start with 'When <condition>: <check>'")
            rkey = f"{stem}#{render.github_heading_anchor(title)}"
            examples = (rule_prov.get(rkey) or {}).get("examples") if isinstance(rule_prov.get(rkey), dict) else None
            urls = [ex.get("url") for ex in examples or [] if isinstance(ex, dict)]
            if not any(isinstance(u, str) and CITATION_URL_RX.match(u) for u in urls):
                errs.append(f"rules/{stem}.md ({title}): no PR or issue citation in provenance.json {rkey}")
    return errs


def stage1_sizes(bundle: pathlib.Path) -> Dict[str, int]:
    return {
        f"{stem}.md": (bundle / "rules" / f"{stem}.md").stat().st_size
        for stem in STAGE1_CATEGORIES
        if (bundle / "rules" / f"{stem}.md").is_file()
    }


def check_budget(sizes: Dict[str, int], max_bytes: int) -> List[str]:
    total = sum(sizes.values())
    if total > max_bytes:
        return [
            f"Stage-1 rule files total {total} bytes (~{total // BYTES_PER_TOKEN} tokens), "
            f"over the {max_bytes} byte (~{max_bytes // BYTES_PER_TOKEN} token) budget"
        ]
    return []


def check_routing_rows(render: ModuleType, bundle: pathlib.Path, sizes: Dict[str, int], max_row_bytes: int) -> List[str]:
    routing = bundle / "playbooks" / "review_a_pull_request.md"
    if not routing.is_file():
        return []
    rows = render.routing_table_rows(routing.read_text(encoding="utf-8"))
    if rows is None:
        return ["playbooks/review_a_pull_request.md: missing diff-path routing table"]
    errs: List[str] = []
    for globs, rule_files, _ in rows:
        label = ", ".join(globs[:2]) + (", ..." if len(globs) > 2 else "")
        if len(rule_files) > render.MAX_ROUTED_RULE_FILES:
            errs.append(f"routing row [{label}]: routes {len(rule_files)} rule files (max {render.MAX_ROUTED_RULE_FILES})")
        unknown = [f for f in rule_files if f not in sizes]
        if unknown:
            errs.append(f"routing row [{label}]: links non-canonical rule files {unknown}")
        row_bytes = sum(sizes.get(f, 0) for f in rule_files)
        if row_bytes > max_row_bytes:
            errs.append(f"routing row [{label}]: loads {row_bytes} bytes (max {max_row_bytes})")
    return errs


def check_retention_guard(repo_root: pathlib.Path) -> List[str]:
    errs: List[str] = []
    for rel in RETENTION_GUARD_FILES:
        path = repo_root / rel
        if not path.is_file():
            errs.append(f"{rel}: missing (Pass-1 Retention Guard cannot be verified)")
            continue
        text = _normalize_ws(path.read_text(encoding="utf-8"))
        for phrase in RETENTION_GUARD_PHRASES:
            if phrase not in text:
                errs.append(f"{rel}: Pass-1 Retention Guard phrase missing: {phrase!r}")
        if rel == HIGH_SIGNAL_FILTER_FILE and HIGH_SIGNAL_FILTER_PHRASE not in text:
            errs.append(f"{rel}: High-Signal Filter phrase missing: {HIGH_SIGNAL_FILTER_PHRASE!r}")
    return errs


def guarded_text_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    bundle = repo_root / "docs" / "review"
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    for rel in ("docs/review_playbook.md", "AGENTS.md"):
        if (repo_root / rel).is_file():
            files.append(repo_root / rel)
    return files


def check_em_dashes(repo_root: pathlib.Path) -> List[str]:
    errs: List[str] = []
    for path in guarded_text_files(repo_root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if EM_DASH in line:
                errs.append(f"{path.relative_to(repo_root).as_posix()}:{lineno}: contains U+2014")
    return errs


def check_temporal_firewall(repo_root: pathlib.Path, max_pr: int) -> List[str]:
    """Rejects llm-d-router PR or issue citations numbered above max_pr.

    GitHub numbers PRs and issues from one counter, so an issue above max_pr
    was opened after the cutoff PR and is also outside the holdout window.
    """
    errs: List[str] = []
    for path in guarded_text_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for kind, rx in (("PR", PR_CITATION_RX), ("issue", ISSUE_CITATION_RX)):
                for num in sorted({int(m.group(1)) for m in rx.finditer(line)}):
                    if num > max_pr:
                        errs.append(f"{rel}:{lineno}: cites {kind} {num} above --max-pr {max_pr}")
    return errs


def run_render(repo_root: pathlib.Path, extra: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "hack/render_playbook.py", *extra],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def validate(
    repo_root: pathlib.Path,
    render_first: bool = False,
    max_pr: int = 0,
    max_stage1_bytes: int = DEFAULT_MAX_STAGE1_BYTES,
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
) -> Tuple[List[str], Dict[str, Any]]:
    """Runs every check and returns (errors, stats). max_pr <= 0 disables the firewall."""
    repo_root = repo_root.resolve()
    bundle = repo_root / "docs" / "review"
    errs: List[str] = []

    if render_first:
        code, out = run_render(repo_root, [])
        if code != 0:
            errs.append(f"hack/render_playbook.py failed (exit {code}): {out}")
    code, out = run_render(repo_root, ["--check", "--check-staleness"])
    if code != 0:
        errs.append(f"hack/render_playbook.py --check --check-staleness failed (exit {code}): {out}")

    render = load_render_module(repo_root)
    errs.extend(check_structure(bundle))
    prov: Any = {}
    prov_path = bundle / "provenance.json"
    if prov_path.is_file():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            errs.append(f"provenance.json: invalid JSON ({exc})")
        else:
            errs.extend(check_provenance(prov))

    parsed, parse_errs = parse_rules(render, bundle)
    errs.extend(parse_errs)
    errs.extend(check_rule_triggers_and_citations(render, parsed, prov))
    sizes = stage1_sizes(bundle)
    errs.extend(check_budget(sizes, max_stage1_bytes))
    errs.extend(check_routing_rows(render, bundle, sizes, max_row_bytes))
    errs.extend(check_retention_guard(repo_root))
    errs.extend(check_em_dashes(repo_root))
    if max_pr > 0:
        errs.extend(check_temporal_firewall(repo_root, max_pr))

    stats = {
        "stage1_rules": sum(len(r) for r in parsed.values()),
        "stage1_bytes": sum(sizes.values()),
        "stage1_tokens_estimate": sum(sizes.values()) // BYTES_PER_TOKEN,
        "max_stage1_bytes": max_stage1_bytes,
    }
    return errs, stats


def sync_prdraft(repo_root: pathlib.Path, dest: pathlib.Path) -> None:
    src = repo_root / "docs" / "review_playbook.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--render", action="store_true", help="Run hack/render_playbook.py before checking.")
    parser.add_argument(
        "--max-pr",
        type=int,
        default=0,
        help="Temporal firewall: fail on any llm-d-router PR or issue citation above this number.",
    )
    parser.add_argument("--max-stage1-bytes", type=int, default=DEFAULT_MAX_STAGE1_BYTES)
    parser.add_argument("--max-row-bytes", type=int, default=DEFAULT_MAX_ROW_BYTES)
    parser.add_argument(
        "--sync-prdraft",
        action="store_true",
        help="After all checks pass, copy docs/review_playbook.md to --prdraft-path.",
    )
    parser.add_argument("--prdraft-path", type=pathlib.Path, default=DEFAULT_PRDRAFT_PATH)
    args = parser.parse_args(argv)

    errs, stats = validate(
        args.repo_root,
        render_first=args.render,
        max_pr=args.max_pr,
        max_stage1_bytes=args.max_stage1_bytes,
        max_row_bytes=args.max_row_bytes,
    )
    print(
        f"Stage-1: {stats['stage1_rules']} rules, {stats['stage1_bytes']} bytes "
        f"(~{stats['stage1_tokens_estimate']} tokens) of {stats['max_stage1_bytes']} byte budget"
    )
    if errs:
        for e in errs:
            sys.stderr.write(f"ERROR: {e}\n")
        sys.stderr.write(f"FAIL: {len(errs)} error(s)\n")
        return 1
    if args.sync_prdraft:
        sync_prdraft(args.repo_root.resolve(), args.prdraft_path)
        print(f"Synced docs/review_playbook.md to {args.prdraft_path}")
    print("OK: review playbook refresh is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
