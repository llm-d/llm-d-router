#!/usr/bin/env python3
"""Render docs/review_playbook.md deterministically from the docs/review/ OKF v0.2 bundle.

Stdlib-only script suitable for vendoring into hack/render_playbook.py and
gating CI via `make verify-review-playbook` (`python3 hack/render_playbook.py --check`).
"""

import argparse
import difflib
import pathlib
import posixpath
import re
import sys
from typing import Any, Dict, List, Tuple

GENERATED_HEADER = (
    "<!-- Generated from docs/review/ by hack/render_playbook.py. "
    "Do not edit directly; regenerate with: python3 hack/render_playbook.py -->"
)


def unquote_yaml(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        return val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        return val[1:-1].replace("''", "'")
    return val


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
                    if ":" in kv:
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
            inner = rest[1:-1].strip()
            mapping: Dict[str, str] = {}
            if inner:
                for part in inner.split(","):
                    if ":" in part:
                        mk, mv = part.split(":", 1)
                        mapping[mk.strip()] = unquote_yaml(mv)
            fm[key] = mapping
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


def parse_rule_blocks(body: str) -> List[Tuple[str, Dict[str, str]]]:
    """Parses `## <rule title>` blocks and their `### <subheading>` subsections."""
    rules: List[Tuple[str, Dict[str, str]]] = []
    cur_rule = ""
    cur_sub = ""
    cur_sub_map: Dict[str, str] = {}
    cur_lines: List[str] = []

    def _flush_sub() -> None:
        if cur_sub:
            cur_sub_map[cur_sub] = "\n".join(cur_lines).strip()

    for line in body.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            _flush_sub()
            if cur_rule:
                rules.append((cur_rule, cur_sub_map))
            cur_rule = line[3:].strip()
            cur_sub = ""
            cur_sub_map = {}
            cur_lines = []
        elif line.startswith("### ") and not line.startswith("#### "):
            _flush_sub()
            cur_sub = line[4:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    _flush_sub()
    if cur_rule:
        rules.append((cur_rule, cur_sub_map))
    return rules


def rewrite_links_to_docs(text: str, source_dir_from_docs: str) -> str:
    """Rewrites relative Markdown links authored inside docs/<source_dir_from_docs>/
    back to paths relative to docs/ (where docs/review_playbook.md lives).
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
    """Converts OKF inline footnote clusters '.[^id1][^id2]' back into
    ' ([#num1](url1), [#num2](url2)).' for single-page playbook rendering.
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


def render_playbook(bundle_dir: pathlib.Path, omit_generated_header: bool = False) -> str:
    rules_index_path = bundle_dir / "rules" / "index.md"
    if not rules_index_path.is_file():
        raise FileNotFoundError(f"Missing rules index: {rules_index_path}")

    rules_index_text = rules_index_path.read_text(encoding="utf-8")
    idx_lines = rules_index_text.splitlines()
    intro_lines: List[str] = []
    sec_entries: List[Tuple[str, str]] = []
    for line in idx_lines[1:]:
        s = line.strip()
        if not s:
            continue
        m_item = re.match(r"^\* \[([^\]]+)\]\(([^)]+\.md)\)", s)
        if m_item:
            sec_entries.append((m_item.group(1).strip(), m_item.group(2).strip()))
        elif not sec_entries and not s.startswith("#"):
            intro_lines.append(s)

    intro = rewrite_links_to_docs(" ".join(intro_lines), "review/rules")

    out_lines = ["# Review Playbook", ""]
    if not omit_generated_header:
        out_lines.extend([GENERATED_HEADER, ""])
    out_lines.extend([intro, ""])

    for sec_title, sec_file in sec_entries:
        sec_path = bundle_dir / "rules" / sec_file
        fm, body = parse_frontmatter(sec_path.read_text(encoding="utf-8"))
        sources = fm.get("sources") or []
        source_map = {str(s["id"]): str(s["resource"]) for s in sources if isinstance(s, dict)}
        rules = parse_rule_blocks(body)

        out_lines.append(f"## {sec_title}")
        out_lines.append("")

        for rtitle, subs in rules:
            title = rtitle.strip()
            if not title.endswith("."):
                title += "."

            rule_prose = rewrite_links_to_docs(
                " ".join(l.strip() for l in subs.get("Rule", "").splitlines() if l.strip()),
                "review/rules",
            )

            ev_text = subs.get("Evidence", "")
            m_ex = re.search(
                r"Representative review threads:((?:\[\^[a-zA-Z0-9_.-]+\])+)", ev_text
            )
            if not m_ex:
                raise ValueError(
                    f"{sec_path} ({rtitle}): missing 'Representative review threads:' footnotes"
                )
            ex_ids = re.findall(r"\[\^([a-zA-Z0-9_.-]+)\]", m_ex.group(1))
            ex_links = [
                f"[{url_to_short_ref(source_map[eid])}]({source_map[eid]})" for eid in ex_ids
            ]

            out_lines.append(f"- **{title}**")
            out_lines.append(f"  {rule_prose}")
            out_lines.append(f"  Examples: {', '.join(ex_links)}")

            if "Reviewers accept" in subs:
                ra_rewritten = rewrite_links_to_docs(subs["Reviewers accept"], "review/rules")
                ra_restored = restore_inline_citations(ra_rewritten, source_map)
                out_lines.append(f"  Reviewers accept: {ra_restored}")

            if "Open question" in subs:
                m_oq = re.search(r"\]\(([^)]+\.md)\)", subs["Open question"])
                if m_oq:
                    oq_path = (sec_path.parent / m_oq.group(1)).resolve()
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
                            )
                            out_lines.append(f"  Open question: {oq_restored}")

            out_lines.append("")

    return "\n".join(out_lines)


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


def github_heading_anchor(heading: str) -> str:
    """Computes GitHub-flavored Markdown heading anchor slug."""
    cleaned = heading.strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return cleaned.replace(" ", "-")


def collect_file_anchors(md_path: pathlib.Path) -> set[str]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
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


def validate_bundle(bundle_dir: pathlib.Path, repo_root: pathlib.Path) -> List[str]:
    """Compact repo-aware OKF v0.2 check replacing generic validate_okf.py in CI."""
    errs: List[str] = []
    repo_abs = repo_root.resolve()
    bundle_abs = bundle_dir.resolve()
    anchor_cache: Dict[pathlib.Path, set[str]] = {}

    def _get_anchors(p: pathlib.Path) -> set[str]:
        if p not in anchor_cache:
            anchor_cache[p] = collect_file_anchors(p)
        return anchor_cache[p]

    for md in sorted(bundle_abs.rglob("*.md")):
        rel = md.relative_to(bundle_abs).as_posix()
        raw = md.read_bytes()
        if any(b > 127 for b in raw):
            errs.append(f"{rel}: contains non-ASCII bytes")
        text = raw.decode("utf-8", errors="replace")
        has_fm = text.startswith("---\n")
        fm, body = parse_frontmatter(text)
        if md.name == "log.md" or (md.name == "index.md" and rel != "index.md"):
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
            fn_refs = set(re.findall(r"\[\^([a-zA-Z0-9_.-]+)\](?!:)", body))
            if fn_refs - sids:
                errs.append(
                    f"{rel}: footnotes missing from frontmatter sources: {sorted(fn_refs - sids)}"
                )
            idx = md.parent / "index.md"
            if not idx.is_file() or f"({md.name})" not in idx.read_text(encoding="utf-8"):
                errs.append(f"{rel}: not listed in {idx.relative_to(bundle_abs).as_posix()}")
        virtual_md = repo_abs / "docs" / "review" / rel
        for _, target in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", body):
            if "#" in target:
                tpath_raw, frag = target.split("#", 1)
                tpath = tpath_raw.strip()
                frag = frag.strip()
            else:
                tpath, frag = target.strip(), ""
            if tpath.startswith(("http://", "https://", "mailto:")):
                continue
            if not tpath:
                if frag and frag not in _get_anchors(md):
                    errs.append(f"{rel}: broken in-file anchor #{frag}")
                continue
            cand_bundle = (md.parent / tpath).resolve()
            if cand_bundle.exists() and cand_bundle.is_relative_to(bundle_abs):
                if frag and cand_bundle.is_file() and cand_bundle.suffix == ".md":
                    if frag not in _get_anchors(cand_bundle):
                        errs.append(f"{rel}: broken anchor in link {target}")
                continue
            cand_repo = (virtual_md.parent / tpath).resolve()
            if not cand_repo.is_relative_to(repo_abs):
                errs.append(f"{rel}: relative link escapes repository root: {target}")
            elif not cand_repo.exists():
                errs.append(f"{rel}: broken relative link: {target}")
            elif frag and cand_repo.is_file() and cand_repo.suffix == ".md":
                if frag not in _get_anchors(cand_repo):
                    errs.append(f"{rel}: broken anchor in repo link {target}")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=pathlib.Path,
        default=pathlib.Path("docs/review"),
        help="Path to the OKF v0.2 review bundle directory (default: docs/review).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("docs/review_playbook.md"),
        help="Target path for rendered single-page playbook (default: docs/review_playbook.md).",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root for validating repo-relative links in --check mode (default: .).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate bundle and fail with a unified diff if --output differs from rendered output.",
    )
    parser.add_argument(
        "--omit-header",
        action="store_true",
        help="Omit the generated HTML comment line (used for initial import round-trip tests).",
    )
    args = parser.parse_args()

    if args.check and args.repo_root is not None:
        v_errs = validate_bundle(args.bundle, args.repo_root)
        if v_errs:
            for e in v_errs:
                sys.stderr.write(f"BUNDLE ERROR: {e}\n")
            return 1

    rendered = render_playbook(args.bundle, omit_generated_header=args.omit_header)

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
                tofile=f"{args.output} (rendered from {args.bundle})",
            )
            sys.stderr.writelines(diff)
            sys.stderr.write(
                f"\nERROR: {args.output} is out of sync with {args.bundle}.\n"
                "Run 'python3 hack/render_playbook.py' and commit the updated file.\n"
            )
            return 1
        print(f"OK: {args.output} matches rendered output from {args.bundle}.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
