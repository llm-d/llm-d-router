#!/usr/bin/env python3
"""Mine maintainer PR review threads, reviews, and issues since the last provenance.json watermark.

Stdlib-only CLI that reads docs/review/provenance.json, queries GitHub GraphQL via
`gh api graphql` (or reads a mock fixture for offline testing), filters out bot
and self-talk threads, enforces optional `--until-pr` and `--until-date` holdout
cutoffs, tags each candidate with Stage-1 category heuristics, and writes a
structured JSON report.
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

KNOWN_BOT_LOGINS: Set[str] = {
    "github-actions",
    "github-actions[bot]",
    "codecov",
    "codecov[bot]",
    "dependabot",
    "dependabot[bot]",
    "gemini-code-assist",
    "gemini-code-assist[bot]",
    "k8s-ci-robot",
    "fejta-bot",
    "copilot",
    "copilot-pull-request-reviewer",
    "copilot-pull-request-reviewer[bot]",
    "renovate",
    "renovate[bot]",
    "netlify",
    "netlify[bot]",
    "sonarqubecloud",
    "sonarqubecloud[bot]",
    "cla-bot",
    "google-cla",
    "linux-foundation-easycla",
    "easycla",
}

CATEGORY_PATTERNS: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {
    "external_wire_and_engine_contracts": [
        ("msgpack", re.compile(r"\bmsgpack\b", re.IGNORECASE)),
        ("sse_framing", re.compile(r"(?<!\w)(?:sse(?!\w)|\[done\]|event:|data:)", re.IGNORECASE)),
        ("engine_contract", re.compile(r"\b(?:vllm|sglang|triton|tensorrt)\b", re.IGNORECASE)),
        (
            "wire_tokens",
            re.compile(
                r"\b(?:input_tokens|cache_read_input_tokens|cache_creation_input_tokens|"
                r"continuous_usage_stats|include_usage|stream_options)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "http_wire",
            re.compile(
                r"\b(?:content-length|transfer-encoding|connection:\s*upgrade|"
                r"rfc\s*9110|previous_response_id)\b|"
                r"(?:/v1/responses|/v1/messages|/render)(?!\w)",
                re.IGNORECASE,
            ),
        ),
        (
            "zmq_kvevents",
            re.compile(r"\b(?:zmq|kv[-_]?event|sequence\s+gap|epoch\s+reset|kv_transfer_params)\b", re.IGNORECASE),
        ),
        ("otel_gateway_wire", re.compile(r"\b(?:otel|opentelemetry|traceparent|w3c|diffusion)\b", re.IGNORECASE)),
    ],
    "unencoded_runtime_hazards": [
        (
            "concurrency_lock",
            re.compile(r"\b(?:goroutine|mutex|rwmutex|deadlock|data\s+race|lock\s+held|unjoined)\b", re.IGNORECASE),
        ),
        (
            "buffer_retention",
            re.compile(
                r"\b(?:bytes\.buffer|sync\.pool|backing\s+array|slice\s+alias|use[- ]after[- ]free|buffer\s+lifetime)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "metric_cardinality",
            re.compile(r"\b(?:cardinality|unbounded\s+label|per[- ]pod\s+label|per[- ]request\s+label|histogram)\b", re.IGNORECASE),
        ),
        (
            "streaming_unwrap",
            re.compile(r"\b(?:responsewriter|http\.flusher|unwrap\b|mergeusage|parseresponse)\b", re.IGNORECASE),
        ),
        (
            "ext_proc_skip",
            re.compile(r"\b(?:ext_proc|early[- ]skip|header[- ]only|decode[- ]pod|prefill[- ]only)\b", re.IGNORECASE),
        ),
        (
            "dp_rank_key",
            re.compile(r"\b(?:data[- ]parallel|dp\s+rank|host,\s*port,\s*rank|pod\s+ip\s+alone)\b", re.IGNORECASE),
        ),
    ],
    "governance_and_compatibility_policy": [
        (
            "deprecation_policy",
            re.compile(r"\b(?:deprecat\w*|n\+2|two[- ]release|backward[- ]compat\w*|breaking\s+change)\b", re.IGNORECASE),
        ),
        (
            "plugin_graduation",
            re.compile(r"\b(?:alpha|beta|stable\s+v1|allow-experimental-plugins|graduation|testdata/plugins/stable)\b", re.IGNORECASE),
        ),
        (
            "helm_kustomize_parity",
            re.compile(r"\b(?:config/charts|helm\s+chart|kustomize|values\.yaml|test/e2e/artifacts)\b|\bdeploy/", re.IGNORECASE),
        ),
        (
            "crd_upstream_gate",
            re.compile(r"\b(?:crd\b|gie\b|unmerged\s+upstream|verifyclusterstate|env\s+var)\b", re.IGNORECASE),
        ),
    ],
    "settled_architectural_boundaries": [
        (
            "shedding_boundary",
            re.compile(r"\b(?:saturationdetector|admissionplugin|admitter|load\s+shed\w*)\b", re.IGNORECASE),
        ),
        (
            "plugin_phase_boundary",
            re.compile(r"\b(?:preparerequestdata|endpointcandidates|filter\s+vs\s+scorer|profilehandler)\b", re.IGNORECASE),
        ),
        (
            "pool_homogeneity",
            re.compile(r"\b(?:homogeneous|heterogeneous|inferencepool|models-responder|httproute)\b", re.IGNORECASE),
        ),
        (
            "state_boundary",
            re.compile(r"\b(?:conversation\s+history|stateless\s+router|session\s+affinity|bounded\s+cache)\b", re.IGNORECASE),
        ),
        (
            "coord_sidecar_boundary",
            re.compile(r"\b(?:pkg/coordinator|pkg/sidecar|ssrf\s+allowlist|header\s+strip\w*|spoof\w*)\b", re.IGNORECASE),
        ),
    ],
}

STAGE1_CATEGORIES: Tuple[str, ...] = tuple(CATEGORY_PATTERNS)
DECISIONS_CATEGORY = "decisions"
CODE_LOCAL_REJECT = "code_local_reject"
MIN_CATEGORY_SCORE = 2

DECISION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "settled_direction",
        re.compile(
            r"\b(?:we\s+(?:decided|agreed|settled)|consensus|going\s+forward|"
            r"by\s+design|intentional(?:ly)?|out\s+of\s+scope|not\s+the\s+right\s+layer|"
            r"belongs\s+in|should\s+live\s+in|the\s+direction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "design_process",
        re.compile(
            r"\b(?:design\s+(?:doc|proposal)|proposal|rfc|follow[- ]up\s+issue|"
            r"open\s+question|tradeoff|trade-off)\b",
            re.IGNORECASE,
        ),
    ),
]

CODE_LOCAL_SUSPECT_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("agents_md_logging", re.compile(r"\b(?:logging\.(?:debug|default|verbose|trace)|go-logr|\.v\(\d+\))\b", re.IGNORECASE)),
    ("agents_md_constants", re.compile(r"\b(?:goconst|re-export\s+constant|duplicate\s+constant)\b", re.IGNORECASE)),
    ("agents_md_git", re.compile(r"\b(?:dco|sign-off|signed-off-by|release-note)\b", re.IGNORECASE)),
    ("local_nit", re.compile(r"^(?:nit\b|typo\b|rename\b|style\b|formatting\b)", re.IGNORECASE)),
]

PATH_CATEGORY_HINTS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:pkg/kvevents|pkg/kvcache|pkg/common/usage|pkg/epp/framework/plugins/requesthandling)/"), "external_wire_and_engine_contracts"),
    (re.compile(r"^(?:pkg/epp/handlers|pkg/epp/requestcontrol|pkg/epp/datastore|pkg/epp/datalayer)/"), "unencoded_runtime_hazards"),
    (re.compile(r"^(?:config/charts|config/crd|test/e2e/artifacts|apix|client-go|test/testdata/plugins/stable)/"), "governance_and_compatibility_policy"),
    (re.compile(r"^(?:pkg/coordinator|pkg/sidecar|pkg/epp/flowcontrol|pkg/epp/scheduling)/"), "settled_architectural_boundaries"),
]

PR_DELTA_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 40, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        url
        state
        createdAt
        mergedAt
        updatedAt
        author {
          login
        }
        reviews(first: 50) {
          pageInfo {
            hasNextPage
          }
          nodes {
            databaseId
            state
            body
            createdAt
            submittedAt
            url
            author {
              login
            }
          }
        }
        reviewThreads(first: 50) {
          pageInfo {
            hasNextPage
          }
          nodes {
            isResolved
            path
            line
            comments(first: 30) {
              pageInfo {
                hasNextPage
              }
              nodes {
                databaseId
                body
                createdAt
                updatedAt
                url
                author {
                  login
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

ISSUE_DELTA_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 40, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        url
        state
        createdAt
        updatedAt
        closedAt
        author {
          login
        }
        comments(first: 40) {
          pageInfo {
            hasNextPage
          }
          nodes {
            databaseId
            body
            createdAt
            updatedAt
            url
            author {
              login
            }
          }
        }
      }
    }
  }
}
"""


def is_bot_login(login: Optional[str]) -> bool:
    """Returns True if login belongs to an automated bot or is empty."""
    if not login:
        return True
    norm = login.strip().lower()
    if not norm:
        return True
    if norm in KNOWN_BOT_LOGINS:
        return True
    if norm.endswith("[bot]") or norm.endswith("-bot") or norm.startswith("bot-"):
        return True
    return False


def normalize_iso_cutoff(date_str: Optional[str], is_end: bool = True) -> Optional[str]:
    """Parses YYYY-MM-DD or ISO-8601 and returns a `YYYY-MM-DDTHH:MM:SSZ` UTC string.

    Raises ValueError for unparseable input and for timestamps without a UTC offset.
    """
    if date_str is None:
        return None
    s = date_str.strip()
    if not s:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        day = datetime.date.fromisoformat(s)
        return f"{day.isoformat()}T23:59:59Z" if is_end else f"{day.isoformat()}T00:00:00Z"
    try:
        parsed = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid date {date_str!r}: expected YYYY-MM-DD or ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"invalid date {date_str!r}: ISO-8601 timestamps need a UTC offset or Z")
    return parsed.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date_arg(value: str) -> str:
    """argparse type for --since-date and --until-date."""
    try:
        normalize_iso_cutoff(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def timestamp_in_window(
    ts: Optional[str],
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
) -> bool:
    """Checks whether an ISO-8601 UTC timestamp falls within [since_iso, until_iso].

    A missing timestamp is outside any window with an upper cutoff.
    """
    if not ts:
        return not until_iso
    if since_iso and ts < since_iso:
        return False
    if until_iso and ts > until_iso:
        return False
    return True


def check_nested_truncation(kind: str, number: int, node: Dict[str, Any], fields: Tuple[str, ...]) -> None:
    """Raises RuntimeError when a nested GraphQL connection reports hasNextPage."""
    for field in fields:
        conn = node.get(field) or {}
        if (conn.get("pageInfo") or {}).get("hasNextPage"):
            raise RuntimeError(
                f"{kind} #{number}: nested {field} connection is truncated (hasNextPage=true); "
                "raise the first: limit in the query"
            )
    for thread in (node.get("reviewThreads") or {}).get("nodes") or []:
        if isinstance(thread, dict) and ((thread.get("comments") or {}).get("pageInfo") or {}).get("hasNextPage"):
            raise RuntimeError(
                f"{kind} #{number}: nested reviewThreads.comments connection is truncated "
                f"(hasNextPage=true) on {thread.get('path')!r}; raise the first: limit in the query"
            )


PR_CITATION_RX = re.compile(
    r"(?:github\.com/llm-d/llm-d-router/pull/|\bpr-|\bPR\s+#)(\d+)\b"
)
ISSUE_CITATION_RX = re.compile(
    r"(?:github\.com/llm-d/llm-d-router/issues/|\bissue-|\bIssue\s+#)(\d+)\b"
)


def _scan_text_for_max_ids(text: str) -> Tuple[int, int]:
    """Returns the highest llm-d-router PR and issue numbers cited in text."""
    max_pr = max((int(m.group(1)) for m in PR_CITATION_RX.finditer(text)), default=0)
    max_issue = max((int(m.group(1)) for m in ISSUE_CITATION_RX.finditer(text)), default=0)
    return max_pr, max_issue


def extract_watermarks(
    repo_root: pathlib.Path,
    prov_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Reads provenance.json and derives mined_through_pr, mined_through_issue, and mined_through_date."""
    target_prov = prov_path or (repo_root / "docs" / "review" / "provenance.json")
    if not target_prov.is_file():
        raise FileNotFoundError(f"Missing provenance file: {target_prov}")

    raw_text = target_prov.read_text(encoding="utf-8")
    prov = json.loads(raw_text)
    if not isinstance(prov, dict):
        raise ValueError(f"{target_prov}: top-level JSON must be an object")

    corpus = prov.get("corpus") if isinstance(prov.get("corpus"), dict) else {}

    explicit_pr = prov.get("mined_through_pr") or corpus.get("mined_through_pr")
    explicit_issue = prov.get("mined_through_issue") or corpus.get("mined_through_issue")
    explicit_date = (
        prov.get("mined_through_date")
        or corpus.get("mined_through_date")
        or corpus.get("window_end")
    )

    scanned_pr, scanned_issue = _scan_text_for_max_ids(raw_text)
    bundle_dir = target_prov.parent
    if bundle_dir.is_dir():
        for md_file in sorted(bundle_dir.rglob("*.md")):
            m_pr, m_iss = _scan_text_for_max_ids(md_file.read_text(encoding="utf-8", errors="replace"))
            if m_pr > scanned_pr:
                scanned_pr = m_pr
            if m_iss > scanned_issue:
                scanned_issue = m_iss

    mined_pr = int(explicit_pr) if explicit_pr is not None else scanned_pr
    mined_issue = int(explicit_issue) if explicit_issue is not None else scanned_issue
    mined_date = str(explicit_date) if explicit_date else "1970-01-01"
    try:
        normalize_iso_cutoff(mined_date)
    except ValueError as exc:
        raise ValueError(f"{target_prov}: mined_through_date or window_end: {exc}") from exc

    return {
        "mined_through_pr": mined_pr,
        "mined_through_issue": mined_issue,
        "mined_through_date": mined_date,
    }


def score_candidate_text(text: str, path: str = "") -> Dict[str, Any]:
    """Tags a review thread or issue with one triage category.

    The category is the highest-scoring Stage-1 rule section, else `decisions`
    for settled or debated design direction, else `code_local_reject`. Ties
    resolve in STAGE1_CATEGORIES order, and Stage-1 sections win ties over
    `decisions`.
    """
    matched_signals: List[str] = []
    category_scores: Dict[str, int] = {cat: 0 for cat in STAGE1_CATEGORIES}
    category_scores[DECISIONS_CATEGORY] = 0

    for cat, patterns in CATEGORY_PATTERNS.items():
        for sig_name, rx in patterns:
            if rx.search(text):
                matched_signals.append(f"{cat}:{sig_name}")
                category_scores[cat] += 2
    for sig_name, rx in DECISION_PATTERNS:
        if rx.search(text):
            matched_signals.append(f"{DECISIONS_CATEGORY}:{sig_name}")
            category_scores[DECISIONS_CATEGORY] += 2

    if path:
        for path_rx, cat in PATH_CATEGORY_HINTS:
            if path_rx.search(path):
                category_scores[cat] += 1

    stripped = text.strip()
    suspect_tags = [name for name, rx in CODE_LOCAL_SUSPECT_PATTERNS if rx.search(stripped)]

    best_cat = CODE_LOCAL_REJECT
    best_score = 0
    for cat, sc in category_scores.items():
        if sc >= MIN_CATEGORY_SCORE and sc > best_score:
            best_score = sc
            best_cat = cat

    return {
        "suggested_category": best_cat,
        "category_scores": category_scores,
        "matched_signals": matched_signals,
        "code_local_or_agents_md_suspects": suspect_tags,
        "non_code_candidate": best_cat != CODE_LOCAL_REJECT,
        "priority_score": max(0, best_score - len(suspect_tags)),
    }


def filter_and_extract_pr_candidates(
    pr_nodes: List[Dict[str, Any]],
    since_pr: int,
    until_pr: Optional[int] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filters PR nodes by watermark and extracts human maintainer review threads and top-level reviews.

    stats["safe_pr_watermark"] is the highest PR number that is safe to record as
    mined: the highest merged or closed PR in the window, capped below the
    lowest PR still open (or with unknown state), whose later comments a
    future run must still see.
    """
    since_iso = normalize_iso_cutoff(since_date, is_end=False)
    until_iso = normalize_iso_cutoff(until_date, is_end=True)

    candidates: List[Dict[str, Any]] = []
    prs_in_window = 0
    max_settled_pr = since_pr
    lowest_open_pr: Optional[int] = None
    max_date_seen = since_date or ""

    for pr in pr_nodes:
        number = int(pr.get("number") or 0)
        if number <= since_pr:
            continue
        if until_pr is not None and number > until_pr:
            continue

        pr_merged_at = pr.get("mergedAt")
        pr_created_at = pr.get("createdAt")
        pr_updated_at = pr.get("updatedAt")
        effective_pr_ts = pr_merged_at or pr_updated_at or pr_created_at
        if not timestamp_in_window(pr_created_at, since_iso=None, until_iso=until_iso):
            continue
        check_nested_truncation("PR", number, pr, ("reviews", "reviewThreads"))

        prs_in_window += 1
        if str(pr.get("state") or "").upper() in {"MERGED", "CLOSED"}:
            max_settled_pr = max(max_settled_pr, number)
        elif lowest_open_pr is None or number < lowest_open_pr:
            lowest_open_pr = number
        if effective_pr_ts and effective_pr_ts[:10] > max_date_seen[:10]:
            max_date_seen = effective_pr_ts[:10]

        pr_author = ((pr.get("author") or {}).get("login") or "").strip()
        pr_author_norm = pr_author.lower()
        pr_title = pr.get("title") or ""
        pr_url = pr.get("url") or f"https://github.com/llm-d/llm-d-router/pull/{number}"

        # 1. Process inline review threads
        threads = ((pr.get("reviewThreads") or {}).get("nodes")) or []
        for thread in threads:
            if not isinstance(thread, dict):
                continue
            path = str(thread.get("path") or "")
            line = thread.get("line")
            raw_comments = ((thread.get("comments") or {}).get("nodes")) or []
            human_comments: List[Dict[str, Any]] = []
            non_author_reviewers: List[str] = []

            for c in raw_comments:
                if not isinstance(c, dict):
                    continue
                c_author = ((c.get("author") or {}).get("login") or "").strip()
                if is_bot_login(c_author):
                    continue
                c_ts = c.get("createdAt") or c.get("updatedAt")
                if not timestamp_in_window(c_ts, since_iso=None, until_iso=until_iso):
                    continue
                body = str(c.get("body") or "").strip()
                if not body:
                    continue
                human_comments.append(
                    {
                        "id": c.get("databaseId"),
                        "author": c_author,
                        "body": body,
                        "created_at": c_ts,
                        "url": c.get("url") or pr_url,
                    }
                )
                if c_author.lower() != pr_author_norm:
                    non_author_reviewers.append(c_author)

            # Skip self-talk threads where only the PR author commented
            if not human_comments or not non_author_reviewers:
                continue

            lead = next(
                (hc for hc in human_comments if hc["author"].lower() != pr_author_norm),
                human_comments[0],
            )
            combined_text = "\n\n".join(f"{hc['author']}: {hc['body']}" for hc in human_comments)
            scoring = score_candidate_text(combined_text, path=path)
            db_id = lead.get("id")
            prov_id = f"pr-{number}-r{db_id}" if db_id else f"pr-{number}"

            candidates.append(
                {
                    "kind": "review_thread",
                    "provenance_id": prov_id,
                    "pr_number": number,
                    "pr_title": pr_title,
                    "pr_author": pr_author,
                    "reviewer": f"human:{lead['author']}",
                    "path": path,
                    "line": line,
                    "is_resolved": bool(thread.get("isResolved")),
                    "url": lead["url"],
                    "last_modified": human_comments[-1].get("created_at") or effective_pr_ts,
                    "lead_body": lead["body"],
                    "thread_comments": human_comments,
                    "heuristics": scoring,
                }
            )

        # 2. Process top-level review bodies
        reviews = ((pr.get("reviews") or {}).get("nodes")) or []
        for rev in reviews:
            if not isinstance(rev, dict):
                continue
            r_author = ((rev.get("author") or {}).get("login") or "").strip()
            if is_bot_login(r_author) or r_author.lower() == pr_author_norm:
                continue
            r_ts = rev.get("submittedAt") or rev.get("createdAt")
            if not timestamp_in_window(r_ts, since_iso=None, until_iso=until_iso):
                continue
            r_body = str(rev.get("body") or "").strip()
            # Skip empty or pure "/lgtm" one-word bodies
            if not r_body or r_body.lower() in {"/lgtm", "lgtm", "/approve"}:
                continue
            scoring = score_candidate_text(r_body, path="")
            db_id = rev.get("databaseId")
            prov_id = f"pr-{number}-rev-{db_id}" if db_id else f"pr-{number}-rev"

            candidates.append(
                {
                    "kind": "pull_request_review",
                    "provenance_id": prov_id,
                    "pr_number": number,
                    "pr_title": pr_title,
                    "pr_author": pr_author,
                    "reviewer": f"human:{r_author}",
                    "state": rev.get("state"),
                    "path": "",
                    "line": None,
                    "url": rev.get("url") or pr_url,
                    "last_modified": r_ts or effective_pr_ts,
                    "lead_body": r_body,
                    "heuristics": scoring,
                }
            )

    candidates.sort(
        key=lambda item: (
            -int(item["heuristics"]["priority_score"]),
            int(item["pr_number"]),
            str(item["provenance_id"]),
        )
    )

    safe_pr = max_settled_pr if lowest_open_pr is None else min(max_settled_pr, lowest_open_pr - 1)
    stats = {
        "prs_scanned": len(pr_nodes),
        "prs_in_window": prs_in_window,
        "candidate_threads_count": len(candidates),
        "safe_pr_watermark": max(since_pr, safe_pr),
        "lowest_open_pr": lowest_open_pr,
        "max_date_seen": max_date_seen or (until_date or since_date or ""),
    }
    return candidates, stats


def filter_and_extract_issue_candidates(
    issue_nodes: List[Dict[str, Any]],
    since_issue: int,
    until_issue: Optional[int] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filters issue nodes and extracts human maintainer discussions."""
    since_iso = normalize_iso_cutoff(since_date, is_end=False)
    until_iso = normalize_iso_cutoff(until_date, is_end=True)

    issue_candidates: List[Dict[str, Any]] = []
    max_issue_seen = since_issue

    for iss in issue_nodes:
        number = int(iss.get("number") or 0)
        if until_issue is not None and number > until_issue:
            continue

        updated_at = iss.get("updatedAt") or iss.get("createdAt")
        if number <= since_issue and (not since_iso or (updated_at and updated_at <= since_iso)):
            continue
        check_nested_truncation("Issue", number, iss, ("comments",))

        iss_author = ((iss.get("author") or {}).get("login") or "").strip()
        if is_bot_login(iss_author):
            iss_author = ""
        iss_author_norm = iss_author.lower()

        raw_comments = ((iss.get("comments") or {}).get("nodes")) or []
        human_comments: List[Dict[str, Any]] = []
        for c in raw_comments:
            if not isinstance(c, dict):
                continue
            c_author = ((c.get("author") or {}).get("login") or "").strip()
            if is_bot_login(c_author):
                continue
            c_ts = c.get("createdAt") or c.get("updatedAt")
            if not timestamp_in_window(c_ts, since_iso=since_iso, until_iso=until_iso):
                continue
            c_body = str(c.get("body") or "").strip()
            if not c_body:
                continue
            human_comments.append(
                {
                    "id": c.get("databaseId"),
                    "author": c_author,
                    "body": c_body,
                    "created_at": c_ts,
                    "url": c.get("url") or iss.get("url"),
                }
            )

        if not any(hc["author"].lower() != iss_author_norm for hc in human_comments):
            continue

        if number > max_issue_seen:
            max_issue_seen = number

        title = str(iss.get("title") or "")
        combined = title + "\n" + "\n".join(hc["body"] for hc in human_comments)
        scoring = score_candidate_text(combined, path="")
        lead_comment = human_comments[-1] if human_comments else None
        prov_id = (
            f"issue-{number}-ic-{lead_comment['id']}"
            if lead_comment and lead_comment.get("id")
            else f"issue-{number}"
        )

        issue_candidates.append(
            {
                "kind": "issue",
                "provenance_id": prov_id,
                "issue_number": number,
                "title": title,
                "state": iss.get("state"),
                "author": f"human:{lead_comment['author']}" if lead_comment else f"human:{iss_author}",
                "url": (lead_comment["url"] if lead_comment else iss.get("url"))
                or f"https://github.com/llm-d/llm-d-router/issues/{number}",
                "last_modified": (lead_comment["created_at"] if lead_comment else updated_at) or "",
                "comments": human_comments,
                "heuristics": scoring,
            }
        )

    issue_candidates.sort(
        key=lambda item: (
            -int(item["heuristics"]["priority_score"]),
            int(item["issue_number"]),
        )
    )
    return issue_candidates, max_issue_seen


def _run_gh_graphql(query: str, owner: str, name: str, cursor: Optional[str]) -> Dict[str, Any]:
    cmd = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-F",
        f"owner={owner}",
        "-F",
        f"name={name}",
    ]
    if cursor:
        cmd.extend(["-F", f"cursor={cursor}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api graphql failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def fetch_github_nodes(
    repo: str,
    since_pr: int,
    since_issue: int,
    since_date: Optional[str],
    max_pages: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Queries GitHub GraphQL with pagination for PRs > since_pr and updated issues.

    Raises RuntimeError when GraphQL reports errors or when max_pages runs out
    before the watermark boundary is reached.
    """
    if "/" not in repo:
        raise ValueError(f"Expected --repo in owner/name format, got {repo!r}")
    owner, name = repo.split("/", 1)
    since_iso = normalize_iso_cutoff(since_date, is_end=False)

    def _pr_boundary_reached(batch: List[Dict[str, Any]]) -> bool:
        return min(int(n.get("number") or 0) for n in batch) <= since_pr

    def _issue_boundary_reached(batch: List[Dict[str, Any]]) -> bool:
        if since_iso:
            return min(str(n.get("updatedAt") or "") for n in batch) < since_iso
        return min(int(n.get("number") or 0) for n in batch) <= since_issue

    pr_nodes = _paginate(PR_DELTA_QUERY, "pullRequests", owner, name, max_pages, _pr_boundary_reached)
    issue_nodes = _paginate(ISSUE_DELTA_QUERY, "issues", owner, name, max_pages, _issue_boundary_reached)
    return pr_nodes, issue_nodes


def _paginate(
    query: str,
    field: str,
    owner: str,
    name: str,
    max_pages: int,
    boundary_reached: Any,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        payload = _run_gh_graphql(query, owner, name, cursor)
        if payload.get("errors"):
            raise RuntimeError(f"gh api graphql returned errors for {field}: {payload['errors']}")
        conn = ((payload.get("data") or {}).get("repository") or {}).get(field)
        if not isinstance(conn, dict):
            raise RuntimeError(f"gh api graphql response missing repository.{field}")
        batch = [n for n in (conn.get("nodes") or []) if isinstance(n, dict)]
        nodes.extend(batch)
        page_info = conn.get("pageInfo") or {}
        if not batch or boundary_reached(batch) or not page_info.get("hasNextPage"):
            return nodes
        cursor = page_info.get("endCursor")
    raise RuntimeError(
        f"{field}: reached --max-pages={max_pages} before the watermark boundary; "
        "rerun with a larger --max-pages"
    )


def build_delta_report(
    repo_root: pathlib.Path,
    pr_nodes: List[Dict[str, Any]],
    issue_nodes: List[Dict[str, Any]],
    since_pr: Optional[int] = None,
    until_pr: Optional[int] = None,
    since_issue: Optional[int] = None,
    until_issue: Optional[int] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    prov_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Builds the structured delta candidate report from PR and Issue nodes."""
    wm = extract_watermarks(repo_root, prov_path=prov_path)
    eff_since_pr = since_pr if since_pr is not None else int(wm["mined_through_pr"])
    eff_since_issue = since_issue if since_issue is not None else int(wm["mined_through_issue"])
    eff_since_date = since_date if since_date is not None else str(wm["mined_through_date"])
    if until_issue is None:
        until_issue = until_pr
    cand_threads, pr_stats = filter_and_extract_pr_candidates(
        pr_nodes,
        since_pr=eff_since_pr,
        until_pr=until_pr,
        since_date=eff_since_date,
        until_date=until_date,
    )
    cand_issues, max_issue_seen = filter_and_extract_issue_candidates(
        issue_nodes,
        since_issue=eff_since_issue,
        until_issue=until_issue,
        since_date=eff_since_date,
        until_date=until_date,
    )

    cat_counts: Dict[str, int] = {}
    for item in cand_threads + cand_issues:
        cat = item["heuristics"]["suggested_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    after_pr = max(eff_since_pr, int(pr_stats["safe_pr_watermark"]))
    if until_pr is not None:
        lowest_open = pr_stats["lowest_open_pr"]
        after_pr = until_pr if lowest_open is None else max(eff_since_pr, min(until_pr, lowest_open - 1))
    after_issue = until_issue if until_issue is not None else max(eff_since_issue, max_issue_seen)
    after_date = (
        until_date[:10]
        if until_date
        else (
            str(pr_stats["max_date_seen"])[:10]
            if pr_stats.get("max_date_seen")
            else datetime.date.today().isoformat()
        )
    )

    return {
        "summary": {
            "prs_scanned": pr_stats["prs_scanned"],
            "prs_in_window": pr_stats["prs_in_window"],
            "candidate_threads": len(cand_threads),
            "non_code_flagged_threads": sum(
                1 for c in cand_threads if c["heuristics"]["non_code_candidate"]
            ),
            "candidate_issues": len(cand_issues),
            "category_breakdown": cat_counts,
            "temporal_firewall": {
                "since_pr": eff_since_pr,
                "until_pr": until_pr,
                "since_date": eff_since_date,
                "until_date": until_date,
            },
        },
        "watermark_before": {
            "mined_through_pr": eff_since_pr,
            "mined_through_issue": eff_since_issue,
            "mined_through_date": eff_since_date,
        },
        "watermark_after": {
            "mined_through_pr": after_pr,
            "mined_through_issue": after_issue,
            "mined_through_date": after_date,
        },
        "candidate_threads": cand_threads,
        "candidate_issues": cand_issues,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Path to the llm-d-router repository or worktree root.",
    )
    parser.add_argument(
        "--provenance",
        type=pathlib.Path,
        default=None,
        help="Optional explicit path to provenance.json.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="llm-d/llm-d-router",
        help="GitHub owner/repo to query (default: llm-d/llm-d-router).",
    )
    parser.add_argument(
        "--since-pr",
        type=int,
        default=None,
        help="Override lower watermark PR number (exclusive).",
    )
    parser.add_argument(
        "--until-pr",
        type=int,
        default=None,
        help="Optional upper cutoff PR number (inclusive) for temporal holdout firewall.",
    )
    parser.add_argument(
        "--since-issue",
        type=int,
        default=None,
        help="Override lower watermark Issue number (exclusive).",
    )
    parser.add_argument(
        "--until-issue",
        type=int,
        default=None,
        help="Optional upper cutoff Issue number (inclusive).",
    )
    parser.add_argument(
        "--since-date",
        type=parse_date_arg,
        default=None,
        help="Override lower watermark date (YYYY-MM-DD or ISO-8601).",
    )
    parser.add_argument(
        "--until-date",
        type=parse_date_arg,
        default=None,
        help="Optional upper cutoff date (YYYY-MM-DD or ISO-8601) for temporal holdout firewall.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum GraphQL pages to fetch per resource type (default: 10).",
    )
    parser.add_argument(
        "--mock-graphql-file",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file containing {'pullRequests': [...], 'issues': [...]} for offline testing.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output path for the delta candidates JSON report.",
    )
    args = parser.parse_args(argv)
    if args.until_pr is not None and args.until_date is None:
        sys.stderr.write(
            "WARNING: --until-pr without --until-date keeps review comments posted after the "
            "cutoff on PRs <= --until-pr; pass --until-date for a strict holdout firewall.\n"
        )

    wm = extract_watermarks(args.repo_root, prov_path=args.provenance)
    eff_since_pr = args.since_pr if args.since_pr is not None else int(wm["mined_through_pr"])
    eff_since_issue = (
        args.since_issue if args.since_issue is not None else int(wm["mined_through_issue"])
    )
    eff_since_date = (
        args.since_date if args.since_date is not None else str(wm["mined_through_date"])
    )

    if args.mock_graphql_file is not None:
        raw_mock = json.loads(args.mock_graphql_file.read_text(encoding="utf-8"))
        pr_nodes = raw_mock.get("pullRequests") or []
        issue_nodes = raw_mock.get("issues") or []
    else:
        pr_nodes, issue_nodes = fetch_github_nodes(
            repo=args.repo,
            since_pr=eff_since_pr,
            since_issue=eff_since_issue,
            since_date=eff_since_date,
            max_pages=args.max_pages,
        )

    report = build_delta_report(
        repo_root=args.repo_root,
        pr_nodes=pr_nodes,
        issue_nodes=issue_nodes,
        since_pr=eff_since_pr,
        until_pr=args.until_pr,
        since_issue=eff_since_issue,
        until_issue=args.until_issue,
        since_date=eff_since_date,
        until_date=args.until_date,
        prov_path=args.provenance,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"OK: mined {report['summary']['prs_in_window']} PRs and "
        f"{report['summary']['candidate_threads']} maintainer review threads "
        f"into {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
