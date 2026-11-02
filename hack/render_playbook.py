#!/usr/bin/env python3
"""Compile docs/review_playbook.md from Stage-1 docs/review/rules/*.md and docs/review/provenance.json.

Stdlib-only script gating CI via `make verify-review-playbook`
(`python3 hack/render_playbook.py --check`) and opt-in freshness audits via
`python3 hack/render_playbook.py --check --check-staleness`.
"""

import argparse
import datetime
import difflib
import json
import pathlib
import posixpath
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

GENERATED_HEADER = (
    "<!-- Generated from docs/review/ by hack/render_playbook.py. "
    "Do not edit directly; regenerate with: python3 hack/render_playbook.py -->"
)

TWO_STAGE_NOTE = (
    "> **Two-stage workflow:** This file is the compiled human-readable review guide for "
    "contributors and maintainers, combining subsystem rules with rationale, accepted exceptions, "
    "open questions, and representative pull request threads. Coding and review agents must not "
    "load `docs/review_playbook.md` or `docs/review/provenance.json`; agents route changed paths "
    "through [review_a_pull_request.md](review/playbooks/review_a_pull_request.md) and load only "
    "matching rule files under `docs/review/rules/`. See [AGENTS.md](../AGENTS.md) for "
    "repository-wide coding, prose, logging, constant-reuse, and pull request rules."
)

MAX_REPRESENTATIVE_THREADS = 3
STALE_AFTER_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
STALE_AFTER_SHAPE = "YYYY-MM-DDTHH:MM:SSZ"
MAINTENANCE_PLAYBOOK = "playbooks/maintain_review_bundle.md"
ROUTING_TABLE_FILE = "playbooks/review_a_pull_request.md"
ROUTING_TABLE_HEADER = "Touched repository paths"
PROVENANCE_FILE = "provenance.json"
GLOB_CHARS = frozenset("*?[")

ALLOWED_KEYS = {
    "okf_version",
    "type",
    "title",
    "description",
    "tags",
    "status",
    "stale_after",
    "sources",
    "generated",
    "verified",
}


def unquote_yaml(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        return val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        return val[1:-1].replace("''", "'")
    return val


def parse_flow_mapping(flow_str: str) -> Dict[str, str]:
    """Parses a flat YAML flow mapping '{ k1: v1, k2: v2 }' into a dict."""
    inner = flow_str.strip()[1:-1].strip()
    mapping: Dict[str, str] = {}
    if not inner:
        return mapping
    for part in re.findall(r'(?:[^,"]|"(?:\\.|[^"])*")+?(?=,|$)', inner):
        chunk = part.strip()
        if ":" in chunk:
            mk, mv = chunk.split(":", 1)
            mapping[mk.strip()] = unquote_yaml(mv)
    return mapping


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Minimal stdlib YAML frontmatter parser for OKF v0.2 concept files."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("Unclosed YAML frontmatter block")
    fm_block = text[4:end]
    body = text[end + 5 :].lstrip("\n")

    fm: Dict[str, Any] = {}
    lines = fm_block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            items: List[Dict[str, str]] = []
            i += 1
            cur: Dict[str, str] = {}
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                sline = lines[i].strip()
                if not sline:
                    i += 1
                    continue
                if sline.startswith("- "):
                    if cur:
                        items.append(cur)
                        cur = {}
                    kv = sline[2:].strip()
                    if kv.startswith("{") and kv.endswith("}"):
                        items.append(parse_flow_mapping(kv))
                    elif ":" in kv:
                        ik, iv = kv.split(":", 1)
                        cur[ik.strip()] = unquote_yaml(iv)
                elif ":" in sline:
                    ik, iv = sline.split(":", 1)
                    cur[ik.strip()] = unquote_yaml(iv)
                i += 1
            if cur:
                items.append(cur)
            fm[key] = items
            continue
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            fm[key] = [unquote_yaml(x) for x in inner.split(",")] if inner else []
        elif rest.startswith("{") and rest.endswith("}"):
            fm[key] = parse_flow_mapping(rest)
        else:
            fm[key] = unquote_yaml(rest)
        i += 1
    return fm, body


def parse_h1_sections(body: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    cur_heading = ""
    cur_lines: List[str] = []
    for line in body.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            if cur_heading:
                sections[cur_heading] = "\n".join(cur_lines).strip()
            cur_heading = line[2:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_heading:
        sections[cur_heading] = "\n".join(cur_lines).strip()
    return sections


def parse_stage1_rule_file(text: str, rel: str = "") -> Tuple[str, List[Dict[str, Optional[str]]]]:
    """Parses a Stage-1 agent rule file (`# <Section>`, `## <Rule>`, `- Accept:`, `- Open:`)."""
    lines = text.splitlines()
    sec_title = ""
    rules: List[Dict[str, Optional[str]]] = []
    cur_title: Optional[str] = None
    cur_rule_lines: List[str] = []
    cur_accept: Optional[str] = None
    cur_open: Optional[str] = None

    def _flush() -> None:
        if cur_title is not None:
            rules.append(
                {
                    "title": cur_title,
                    "rule": " ".join(s for s in (ln.strip() for ln in cur_rule_lines) if s),
                    "accept": cur_accept,
                    "open": cur_open,
                }
            )

    for idx, line in enumerate(lines, 1):
        if line.startswith("# ") and not line.startswith("## "):
            if sec_title:
                raise ValueError(f"{rel}:{idx}: multiple H1 headings in rule file")
            sec_title = line[2:].strip()
        elif line.startswith("## ") and not line.startswith("### "):
            _flush()
            cur_title = line[3:].strip()
            cur_rule_lines = []
            cur_accept = None
            cur_open = None
        elif line.startswith("### "):
            raise ValueError(f"{rel}:{idx}: Stage-1 rule files must not contain ### subheadings")
        elif line.startswith("- Accept:"):
            if cur_title is None or cur_accept is not None:
                raise ValueError(f"{rel}:{idx}: misplaced or duplicate '- Accept:' line")
            cur_accept = line[len("- Accept:") :].strip()
        elif line.startswith("- Open:"):
            if cur_title is None or cur_open is not None:
                raise ValueError(f"{rel}:{idx}: misplaced or duplicate '- Open:' line")
            cur_open = line[len("- Open:") :].strip()
        elif cur_title is not None:
            if line.strip():
                if cur_accept is not None or cur_open is not None:
                    raise ValueError(
                        f"{rel}:{idx}: rule prose must precede '- Accept:' and '- Open:' lines"
                    )
                cur_rule_lines.append(line)

    _flush()
    if not sec_title:
        raise ValueError(f"{rel}: missing '# <Section Title>' heading")
    return sec_title, rules


def rewrite_links_to_docs(text: str, source_dir_from_docs: str) -> str:
    """Rewrites relative Markdown links authored inside docs/<source_dir_from_docs>/
    to paths relative to docs/ (where docs/review_playbook.md lives).
    """

    def _repl(m: re.Match[str]) -> str:
        label, link = m.group(1), m.group(2)
        if link.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        if "#" in link:
            path_part, frag = link.split("#", 1)
            frag_suffix = "#" + frag
        else:
            path_part, frag_suffix = link, ""
        src_from_repo = posixpath.normpath(posixpath.join("docs", source_dir_from_docs))
        target_from_repo = posixpath.normpath(posixpath.join(src_from_repo, path_part))
        rel_from_docs = posixpath.relpath(target_from_repo, "docs")
        return f"[{label}]({rel_from_docs}{frag_suffix})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


def url_to_short_ref(url: str) -> str:
    m = re.search(r"/(?:pull|issues)/(\d+)", url)
    return f"#{m.group(1)}" if m else url


def restore_inline_citations(text: str, source_map: Dict[str, str]) -> str:
    """Converts OKF inline footnote clusters '.[^id1][^id2]' into
    ' ([#num1](url1), [#num2](url2)).' for human playbook rendering.
    """
    content_lines = [
        line for line in text.splitlines() if not re.match(r"^\[\^[a-zA-Z0-9_.-]+\]:\s+", line)
    ]
    joined = " ".join(l.strip() for l in content_lines if l.strip())

    def _repl(m: re.Match[str]) -> str:
        punct = m.group(1) or ""
        fn_cluster = m.group(2)
        fn_ids = re.findall(r"\[\^([a-zA-Z0-9_.-]+)\]", fn_cluster)
        rendered_links = []
        for fid in fn_ids:
            if fid not in source_map:
                raise KeyError(f"Footnote [^{fid}] missing from concept frontmatter sources")
            url = source_map[fid]
            rendered_links.append(f"[{url_to_short_ref(url)}]({url})")
        return f" ({', '.join(rendered_links)}){punct}"

    return re.sub(r"([.,;]?)((?:\[\^[a-zA-Z0-9_.-]+\])+)", _repl, joined)


def github_heading_anchor(heading: str) -> str:
    """Computes GitHub-flavored Markdown heading anchor slug."""
    cleaned = heading.strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return cleaned.replace(" ", "-")


def parse_rules_index_entries(bundle_dir: pathlib.Path) -> List[Tuple[str, str]]:
    rules_index_path = bundle_dir / "rules" / "index.md"
    if not rules_index_path.is_file():
        raise FileNotFoundError(f"Missing rules index: {rules_index_path}")
    entries: List[Tuple[str, str]] = []
    for line in rules_index_path.read_text(encoding="utf-8").splitlines():
        m_item = re.match(r"^\* \[([^\]]+)\]\(([^)]+\.md)\)", line.strip())
        if m_item:
            entries.append((m_item.group(1).strip(), m_item.group(2).strip()))
    return entries


def decision_link_from_docs(bundle_dir: pathlib.Path, dec_rel: str) -> str:
    dec_path = bundle_dir / dec_rel
    fm, _ = parse_frontmatter(dec_path.read_text(encoding="utf-8"))
    title = str(fm.get("title") or dec_path.stem)
    return f"[{title}](review/{dec_rel})"


def render_playbook(bundle_dir: pathlib.Path, omit_generated_header: bool = False) -> str:
    prov_path = bundle_dir / PROVENANCE_FILE
    if not prov_path.is_file():
        raise FileNotFoundError(f"Missing provenance sidecar: {prov_path}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    sec_prov_map: Dict[str, Any] = prov.get("sections") or {}
    rule_prov_map: Dict[str, Any] = prov.get("rules") or {}

    sec_entries = parse_rules_index_entries(bundle_dir)
    parsed_sections: List[Tuple[str, str, List[Dict[str, Optional[str]]]]] = []
    for sec_title, sec_file in sec_entries:
        sec_path = bundle_dir / "rules" / sec_file
        parsed_title, rules = parse_stage1_rule_file(
            sec_path.read_text(encoding="utf-8"), f"rules/{sec_file}"
        )
        if parsed_title != sec_title:
            raise ValueError(
                f"rules/{sec_file}: H1 {parsed_title!r} does not match rules/index.md {sec_title!r}"
            )
        parsed_sections.append((sec_title, sec_file[:-3], rules))

    out_lines: List[str] = ["# Review Playbook", ""]
    if not omit_generated_header:
        out_lines.extend([GENERATED_HEADER, ""])
    out_lines.extend([TWO_STAGE_NOTE, "", "## Contents", ""])

    for sec_title, _, rules in parsed_sections:
        anchor = github_heading_anchor(sec_title)
        noun = "rule" if len(rules) == 1 else "rules"
        out_lines.append(f"- [{sec_title}](#{anchor}) ({len(rules)} {noun})")
    out_lines.append("")

    for sec_title, sec_stem, rules in parsed_sections:
        sec_meta = sec_prov_map.get(sec_stem) or {}
        sec_desc = str(sec_meta.get("description") or "").strip()
        out_lines.append(f"## {sec_title}")
        out_lines.append("")
        if sec_desc:
            out_lines.append(sec_desc)
            out_lines.append("")

        for r in rules:
            rtitle = str(r["title"] or "").strip()
            ranchor = github_heading_anchor(rtitle)
            rkey = f"{sec_stem}#{ranchor}"
            if rkey not in rule_prov_map:
                raise KeyError(f"Missing provenance entry for {rkey}")
            rprov = rule_prov_map[rkey]

            rule_prose = rewrite_links_to_docs(str(r["rule"] or ""), "review/rules")
            consequence = str(rprov.get("consequence") or "").strip()

            out_lines.append(f"### {rtitle}")
            out_lines.append("")
            out_lines.append(rule_prose)
            out_lines.append("")
            out_lines.append(f"- **Why it matters:** {consequence}")

            if r.get("accept"):
                acc_text = str(r["accept"]).rstrip(".")
                acc_refs: List[str] = []
                for dec_rel in rprov.get("accept_decisions") or []:
                    acc_refs.append(decision_link_from_docs(bundle_dir, dec_rel))
                for ex in rprov.get("accept_examples") or []:
                    url = str(ex["url"])
                    acc_refs.append(f"[{url_to_short_ref(url)}]({url})")
                suffix = f" ({', '.join(acc_refs)})." if acc_refs else "."
                out_lines.append(f"- **Reviewers accept:** {acc_text}{suffix}")

            if rprov.get("decisions"):
                dec_links = [
                    decision_link_from_docs(bundle_dir, d) for d in rprov["decisions"]
                ]
                label = "Settled decision" if len(dec_links) == 1 else "Settled decisions"
                out_lines.append(f"- **{label}:** {', '.join(dec_links)}.")

            oq_meta = rprov.get("open_question")
            if isinstance(oq_meta, dict) and oq_meta.get("file"):
                oq_rel = str(oq_meta["file"])
                oq_path = bundle_dir / oq_rel
                if oq_path.is_file():
                    oq_fm, oq_body = parse_frontmatter(oq_path.read_text(encoding="utf-8"))
                    if (
                        oq_fm.get("type") == "Open Question"
                        and oq_fm.get("status", "stable") == "draft"
                    ):
                        oq_h1 = parse_h1_sections(oq_body)
                        oq_sources = oq_fm.get("sources") or []
                        oq_smap = {
                            str(s["id"]): str(s["resource"])
                            for s in oq_sources
                            if isinstance(s, dict)
                        }
                        oq_restored = restore_inline_citations(
                            oq_h1.get("Question", ""), oq_smap
                        ).rstrip(".")
                        out_lines.append(
                            f"- **Open question:** {oq_restored} ([details](review/{oq_rel}))."
                        )

            obs_prs = int(rprov.get("observed_prs") or 0)
            pr_noun = "PR" if obs_prs == 1 else "PRs"
            ex_links = [
                f"[{url_to_short_ref(str(ex['url']))}]({ex['url']})"
                for ex in (rprov.get("examples") or [])
            ]
            out_lines.append(
                f"- **Examples:** Observed in {obs_prs} {pr_noun}: {', '.join(ex_links)}"
            )
            out_lines.append("")

    return "\n".join(out_lines)


def collect_text_anchors(text: str) -> set[str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    anchors: set[str] = set()
    counts: Dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.+)$", line)
        if m:
            base = github_heading_anchor(m.group(1))
            if base in counts:
                counts[base] += 1
                anchors.add(f"{base}-{counts[base]}")
            else:
                counts[base] = 0
                anchors.add(base)
    return anchors


def collect_file_anchors(md_path: pathlib.Path) -> set[str]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    return collect_text_anchors(text)


def routing_table_scopes(body: str) -> Optional[List[str]]:
    """Returns backticked path globs in the first column of the diff-path routing table,
    or None when no table starts with ROUTING_TABLE_HEADER.
    """
    scopes: List[str] = []
    found = False
    in_table = False
    for line in body.splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        first_cell = line.strip().strip("|").split("|", 1)[0].strip()
        if first_cell == ROUTING_TABLE_HEADER:
            found = True
            in_table = True
            continue
        if not in_table or set(first_cell) <= set("-: "):
            continue
        scopes.extend(re.findall(r"`([^`]+)`", first_cell))
    return scopes if found else None


def path_claim_exists(repo_root: pathlib.Path, claim: str) -> bool:
    if GLOB_CHARS & set(claim):
        return next(repo_root.glob(claim), None) is not None
    return (repo_root / claim).exists()


def parse_stale_after(value: Any) -> Optional[datetime.datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.strptime(value, STALE_AFTER_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=datetime.timezone.utc)


def validate_markdown_links(
    label_rel: str,
    body: str,
    md_dir: pathlib.Path,
    virtual_dir: pathlib.Path,
    bundle_abs: pathlib.Path,
    repo_abs: pathlib.Path,
    self_anchors: set[str],
    get_anchors: Any,
    errs: List[str],
) -> None:
    without_fences = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    for _, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", without_fences):
        if "#" in target:
            tpath_raw, frag = target.split("#", 1)
            tpath = tpath_raw.strip()
            frag = frag.strip()
        else:
            tpath, frag = target.strip(), ""
        if tpath.startswith(("http://", "https://", "mailto:")):
            continue
        if not tpath:
            if frag and frag not in self_anchors:
                errs.append(f"{label_rel}: broken in-file anchor #{frag}")
            continue
        cand_bundle = (md_dir / tpath).resolve()
        if cand_bundle.exists() and cand_bundle.is_relative_to(bundle_abs):
            if frag and cand_bundle.is_file() and cand_bundle.suffix == ".md":
                if frag not in get_anchors(cand_bundle):
                    errs.append(f"{label_rel}: broken anchor in link {target}")
            continue
        cand_repo = (virtual_dir / tpath).resolve()
        if not cand_repo.is_relative_to(repo_abs):
            errs.append(f"{label_rel}: relative link escapes repository root: {target}")
        elif not cand_repo.exists():
            errs.append(f"{label_rel}: broken relative link: {target}")
        elif frag and cand_repo.is_file() and cand_repo.suffix == ".md":
            if frag not in get_anchors(cand_repo):
                errs.append(f"{label_rel}: broken anchor in repo link {target}")


def validate_bundle(
    bundle_dir: pathlib.Path,
    repo_root: pathlib.Path,
    rendered_playbook: Optional[str] = None,
    output_path: Optional[pathlib.Path] = None,
    check_staleness: bool = False,
) -> Tuple[List[str], List[str]]:
    """Validates Stage-1 rules, provenance.json sync, routing globs, and Markdown links."""
    errs: List[str] = []
    warns: List[str] = []
    repo_abs = repo_root.resolve()
    bundle_abs = bundle_dir.resolve()
    anchor_cache: Dict[pathlib.Path, set[str]] = {}
    path_claim_cache: Dict[str, bool] = {}
    now = datetime.datetime.now(datetime.timezone.utc)

    if rendered_playbook is not None and output_path is not None:
        rendered_anchors = collect_text_anchors(rendered_playbook)
        anchor_cache[output_path.resolve()] = rendered_anchors
        anchor_cache[(repo_abs / "docs" / "review_playbook.md").resolve()] = rendered_anchors

    def _get_anchors(p: pathlib.Path) -> set[str]:
        rp = p.resolve()
        if rp not in anchor_cache:
            anchor_cache[rp] = collect_file_anchors(rp)
        return anchor_cache[rp]

    def _claim_exists(claim: str) -> bool:
        if claim not in path_claim_cache:
            path_claim_cache[claim] = path_claim_exists(repo_abs, claim)
        return path_claim_cache[claim]

    # 1. Validate provenance.json structure, ASCII, and stale_after timestamps
    prov_path = bundle_abs / PROVENANCE_FILE
    prov: Dict[str, Any] = {}
    if not prov_path.is_file():
        errs.append(f"missing {PROVENANCE_FILE}")
    else:
        raw_prov = prov_path.read_bytes()
        if any(b > 127 for b in raw_prov):
            errs.append(f"{PROVENANCE_FILE}: contains non-ASCII bytes")
        try:
            prov = json.loads(raw_prov.decode("utf-8"))
        except ValueError as exc:
            errs.append(f"{PROVENANCE_FILE}: invalid JSON ({exc})")

    sec_prov_map: Dict[str, Any] = (prov.get("sections") or {}) if isinstance(prov, dict) else {}
    rule_prov_map: Dict[str, Any] = (prov.get("rules") or {}) if isinstance(prov, dict) else {}

    discovered_sections: set[str] = set()
    discovered_rules: set[str] = set()
    seen_consequences: Dict[str, str] = {}

    # 2. Validate all Markdown files under docs/review/
    for md in sorted(bundle_abs.rglob("*.md")):
        rel = md.relative_to(bundle_abs).as_posix()
        raw = md.read_bytes()
        if any(b > 127 for b in raw):
            errs.append(f"{rel}: contains non-ASCII bytes")
        text = raw.decode("utf-8", errors="replace")
        has_fm = text.startswith("---\n")
        try:
            fm, body = parse_frontmatter(text)
        except ValueError as exc:
            errs.append(f"{rel}: {exc}")
            continue

        if rel == ROUTING_TABLE_FILE:
            if has_fm:
                errs.append(f"{rel}: Stage-1 agent routing file must not have YAML frontmatter")
            routing_scopes = routing_table_scopes(body)
            if routing_scopes is None:
                errs.append(f"{rel}: missing routing table with header {ROUTING_TABLE_HEADER!r}")
            else:
                for scope in routing_scopes:
                    if not _claim_exists(scope):
                        errs.append(
                            f"{rel}: routing table path `{scope}` matches no file or directory in the repository"
                        )
            idx = md.parent / "index.md"
            if not idx.is_file() or f"({md.name})" not in idx.read_text(encoding="utf-8"):
                errs.append(f"{rel}: not listed in {idx.relative_to(bundle_abs).as_posix()}")
        elif md.parent.name == "rules" and md.name != "index.md":
            sec_stem = md.stem
            discovered_sections.add(sec_stem)
            if has_fm:
                errs.append(f"{rel}: Stage-1 rule file must not have YAML frontmatter")
            if re.search(r"^sources:\s*", text, re.MULTILINE):
                errs.append(f"{rel}: Stage-1 rule file must not contain sources:")
            if re.search(r"\[\^[a-zA-Z0-9_.-]+\]", text):
                errs.append(f"{rel}: Stage-1 rule file must not contain [^id] footnotes")
            try:
                sec_title, s_rules = parse_stage1_rule_file(text, rel)
            except ValueError as exc:
                errs.append(str(exc))
                s_rules = []
                sec_title = ""

            if sec_stem not in sec_prov_map:
                errs.append(f"{rel}: missing section '{sec_stem}' in {PROVENANCE_FILE}")
            else:
                smeta = sec_prov_map[sec_stem]
                if smeta.get("title") != sec_title:
                    errs.append(
                        f"{rel}: H1 {sec_title!r} differs from {PROVENANCE_FILE} title {smeta.get('title')!r}"
                    )
                if "stale_after" not in smeta:
                    errs.append(
                        f"{PROVENANCE_FILE} ({sec_stem}): missing stale_after (expected {STALE_AFTER_SHAPE})"
                    )
                else:
                    expiry = parse_stale_after(smeta["stale_after"])
                    if expiry is None:
                        errs.append(
                            f"{PROVENANCE_FILE} ({sec_stem}): malformed stale_after {smeta['stale_after']!r} "
                            f"(expected {STALE_AFTER_SHAPE})"
                        )
                    elif expiry <= now:
                        msg = (
                            f"{PROVENANCE_FILE} ({sec_stem}): stale_after {smeta['stale_after']} has passed; "
                            f"audit rules and advance stale_after per {MAINTENANCE_PLAYBOOK}"
                        )
                        if check_staleness:
                            errs.append(msg)
                        else:
                            warns.append(msg)

            for r in s_rules:
                rtitle = str(r["title"] or "")
                rkey = f"{sec_stem}#{github_heading_anchor(rtitle)}"
                discovered_rules.add(rkey)
                if not r.get("rule"):
                    errs.append(f"{rel} ({rtitle}): empty rule body")
                if rkey not in rule_prov_map:
                    errs.append(f"{rel}: rule '{rkey}' missing from {PROVENANCE_FILE}")
                    continue
                rprov = rule_prov_map[rkey]
                conseq = str(rprov.get("consequence") or "").strip()
                if not conseq:
                    errs.append(f"{PROVENANCE_FILE} ({rkey}): missing consequence")
                elif conseq in seen_consequences:
                    errs.append(
                        f"{PROVENANCE_FILE} ({rkey}): duplicate consequence shared with {seen_consequences[conseq]}"
                    )
                else:
                    seen_consequences[conseq] = rkey

                exs = rprov.get("examples") or []
                if not isinstance(exs, list) or not (1 <= len(exs) <= MAX_REPRESENTATIVE_THREADS):
                    errs.append(
                        f"{PROVENANCE_FILE} ({rkey}): expected 1..{MAX_REPRESENTATIVE_THREADS} examples, got {len(exs)}"
                    )
                if int(rprov.get("observed_prs") or 0) < 1:
                    errs.append(f"{PROVENANCE_FILE} ({rkey}): observed_prs must be >= 1")

                has_acc_prov = bool(rprov.get("accept_examples") or rprov.get("accept_decisions"))
                if bool(r.get("accept")) != has_acc_prov:
                    errs.append(
                        f"{rkey}: '- Accept:' in {rel} out of sync with accept_examples/accept_decisions in {PROVENANCE_FILE}"
                    )

                for dec_field in ("decisions", "accept_decisions"):
                    for dec_rel in rprov.get(dec_field) or []:
                        if not (bundle_abs / dec_rel).is_file():
                            errs.append(f"{PROVENANCE_FILE} ({rkey}): missing {dec_rel}")

                oq_meta = rprov.get("open_question")
                has_draft_oq = False
                if oq_meta is not None:
                    oq_rel = str(oq_meta.get("file") or "")
                    oq_path = bundle_abs / oq_rel
                    if not oq_rel or not oq_path.is_file():
                        errs.append(f"{PROVENANCE_FILE} ({rkey}): missing open_question file {oq_rel!r}")
                    else:
                        oq_fm, _ = parse_frontmatter(oq_path.read_text(encoding="utf-8"))
                        has_draft_oq = oq_fm.get("status", "stable") == "draft"
                if bool(r.get("open")) != has_draft_oq:
                    errs.append(
                        f"{rkey}: '- Open:' in {rel} out of sync with draft open_question in {PROVENANCE_FILE}"
                    )

            idx = md.parent / "index.md"
            if not idx.is_file() or f"({md.name})" not in idx.read_text(encoding="utf-8"):
                errs.append(f"{rel}: not listed in {idx.relative_to(bundle_abs).as_posix()}")
        elif md.name == "log.md" or (md.name == "index.md" and rel != "index.md"):
            if has_fm:
                errs.append(f"{rel}: reserved file must not have YAML frontmatter")
        elif rel == "index.md":
            if fm.get("okf_version") != "0.2":
                errs.append("index.md: expected okf_version: '0.2'")
        else:
            bad_keys = set(fm.keys()) - ALLOWED_KEYS
            if bad_keys or not fm.get("type") or not fm.get("title"):
                errs.append(
                    f"{rel}: invalid frontmatter keys or missing type/title ({sorted(bad_keys)})"
                )
            if fm.get("status", "stable") not in {"draft", "stable", "deprecated"}:
                errs.append(f"{rel}: invalid status {fm.get('status')!r}")
            sids = {str(s.get("id")) for s in (fm.get("sources") or []) if isinstance(s, dict)}
            fn_defs = set(re.findall(r"^\[\^([a-zA-Z0-9_.-]+)\]:\s+", body, re.MULTILINE))
            body_without_defs = "\n".join(
                ln for ln in body.splitlines() if not re.match(r"^\[\^[a-zA-Z0-9_.-]+\]:\s+", ln)
            )
            prose_without_code = re.sub(r"```.*?```", "", body_without_defs, flags=re.DOTALL)
            prose_without_code = re.sub(r"`[^`\n]+`", "", prose_without_code)
            fn_refs = set(re.findall(r"\[\^([a-zA-Z0-9_.-]+)\](?!:)", prose_without_code))
            if fn_refs - sids:
                errs.append(
                    f"{rel}: footnotes missing from frontmatter sources: {sorted(fn_refs - sids)}"
                )
            if sids - fn_refs:
                errs.append(
                    f"{rel}: unused source entry in frontmatter sources: {sorted(sids - fn_refs)}"
                )
            if fn_defs - fn_refs:
                errs.append(
                    f"{rel}: unused footnote definition in body: {sorted(fn_defs - fn_refs)}"
                )
            idx = md.parent / "index.md"
            if not idx.is_file() or f"({md.name})" not in idx.read_text(encoding="utf-8"):
                errs.append(f"{rel}: not listed in {idx.relative_to(bundle_abs).as_posix()}")

        virtual_md = repo_abs / "docs" / "review" / rel
        validate_markdown_links(
            rel,
            body,
            md.parent,
            virtual_md.parent,
            bundle_abs,
            repo_abs,
            _get_anchors(md),
            _get_anchors,
            errs,
        )

    orphan_secs = sorted(set(sec_prov_map.keys()) - discovered_sections)
    if orphan_secs:
        errs.append(f"{PROVENANCE_FILE}: orphan section keys not in rules/*.md: {orphan_secs}")

    orphan_rules = sorted(set(rule_prov_map.keys()) - discovered_rules)
    if orphan_rules:
        errs.append(f"{PROVENANCE_FILE}: orphan rule keys not in rules/*.md: {orphan_rules}")

    # 3. Validate rendered docs/review_playbook.md ASCII and Markdown links
    if rendered_playbook is not None and output_path is not None:
        if any(ord(ch) > 127 for ch in rendered_playbook):
            errs.append(f"{output_path}: rendered playbook contains non-ASCII characters")
        out_abs = output_path.resolve()
        virtual_out_dir = out_abs.parent if out_abs.is_relative_to(repo_abs) else (repo_abs / "docs")
        validate_markdown_links(
            output_path.as_posix(),
            rendered_playbook,
            bundle_abs.parent,
            virtual_out_dir,
            bundle_abs,
            repo_abs,
            _get_anchors(out_abs),
            _get_anchors,
            errs,
        )

    return errs, warns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=pathlib.Path,
        default=pathlib.Path("docs/review"),
        help="Path to the review bundle directory (default: docs/review).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("docs/review_playbook.md"),
        help="Target path for compiled human-readable playbook (default: docs/review_playbook.md).",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root for validating links and routing globs in --check mode (default: .).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate bundle and fail with a unified diff if --output differs from compiled output.",
    )
    parser.add_argument(
        "--check-staleness",
        action="store_true",
        help="Fail when any section stale_after timestamp in provenance.json has passed.",
    )
    parser.add_argument(
        "--omit-header",
        action="store_true",
        help="Omit the generated HTML comment line.",
    )
    args = parser.parse_args()

    try:
        rendered = render_playbook(args.bundle, omit_generated_header=args.omit_header)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"RENDER ERROR: {exc}\n")
        return 1

    if args.check or args.check_staleness:
        v_errs, v_warns = validate_bundle(
            args.bundle,
            args.repo_root,
            rendered_playbook=rendered,
            output_path=args.output,
            check_staleness=args.check_staleness,
        )
        for w in v_warns:
            sys.stderr.write(f"WARNING: {w}\n")
        if v_errs:
            for e in v_errs:
                sys.stderr.write(f"BUNDLE ERROR: {e}\n")
            return 1

    if args.check:
        if not args.output.is_file():
            sys.stderr.write(
                f"ERROR: {args.output} does not exist. Run 'python3 hack/render_playbook.py'.\n"
            )
            return 1
        actual = args.output.read_text(encoding="utf-8")
        if actual != rendered:
            diff = difflib.unified_diff(
                actual.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=str(args.output),
                tofile=f"{args.output} (compiled from {args.bundle})",
            )
            sys.stderr.writelines(diff)
            sys.stderr.write(
                f"\nERROR: {args.output} is out of sync with {args.bundle}.\n"
                "Run 'python3 hack/render_playbook.py' and commit the updated file.\n"
            )
            return 1
        print(f"OK: {args.output} matches compiled output from {args.bundle}.")
        return 0

    if args.check_staleness:
        print(f"OK: all stale_after timestamps in {args.bundle}/{PROVENANCE_FILE} are current.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
