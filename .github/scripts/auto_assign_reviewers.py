#!/usr/bin/env python3
"""
Auto Pick Reviewers and Assignee for PRs.

Rules:
1. Auto pick 2 reviewers (excluding PR author).
2. Auto assign 1 owner as the assignee.
3. Ideally if at least one of the reviewers is an owner, pick that as the assignee.
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys

# Default maintainers (from LEADS.md / @llm-d/router-maintainers)
DEFAULT_MAINTAINERS = ["ahg-g", "elevran", "vMaroon", "liu-cong"]


def parse_codeowners(codeowners_path):
    """
    Parses CODEOWNERS file into a list of tuples: (pattern, owners_list)
    """
    rules = []
    if not os.path.exists(codeowners_path):
        return rules

    with open(codeowners_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pattern = parts[0]
            owners = [o.lstrip("@") for o in parts[1:]]
            rules.append((pattern, owners))
    return rules


def parse_leads(leads_path):
    """
    Parses LEADS.md for maintainers if available.
    """
    maintainers = list(DEFAULT_MAINTAINERS)
    if not os.path.exists(leads_path):
        return maintainers

    with open(leads_path, "r", encoding="utf-8") as f:
        in_maintainers = False
        for line in f:
            line = line.strip()
            if line.startswith("## Maintainers"):
                in_maintainers = True
                continue
            elif line.startswith("## ") and in_maintainers:
                break
            if in_maintainers and line.startswith("- @"):
                user = line.replace("- @", "").strip()
                if user and user not in maintainers and not user.startswith("llm-d/"):
                    maintainers.append(user)
    return maintainers


def match_file_against_pattern(filepath, pattern):
    """
    Checks if a relative file path matches a CODEOWNERS pattern.
    """
    filepath = filepath.lstrip("/")
    pattern_clean = pattern.lstrip("/")

    if pattern == "*":
        return True

    if pattern.startswith("/"):
        if pattern.endswith("/"):
            prefix = pattern_clean
            return filepath.startswith(prefix)
        else:
            return filepath == pattern_clean or fnmatch.fnmatch(filepath, pattern_clean)
    else:
        if pattern.endswith("/"):
            return filepath.startswith(pattern_clean) or f"/{pattern_clean}" in filepath
        else:
            return fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(os.path.basename(filepath), pattern)


def get_owners_for_files(changed_files, codeowners_rules, default_maintainers):
    """
    For a list of changed files, returns:
    - specific_file_owners: set of individual users listed as owners for touched paths
    - all_known_owners: set of all known individual owners
    """
    specific_file_owners = set()
    all_known_owners = set(default_maintainers)

    for _, owners in codeowners_rules:
        for owner in owners:
            if "/" not in owner:
                all_known_owners.add(owner)

    for filepath in changed_files:
        matched_owners = []
        for pattern, owners in codeowners_rules:
            if match_file_against_pattern(filepath, pattern):
                matched_owners = owners  # last matching pattern wins

        for owner in matched_owners:
            if "/" not in owner:
                specific_file_owners.add(owner)

    return specific_file_owners, all_known_owners


def select_reviewers_and_assignee(
    pr_author,
    changed_files,
    codeowners_rules,
    maintainers,
    pr_number=0
):
    """
    Implements the core logic:
    1. Auto pick 2 reviewers (excluding PR author).
    2. Auto assign 1 owner as the assignee (excluding PR author).
    3. Ideally if at least one of the reviewers is an owner, pick that as the assignee.
    """
    specific_file_owners, all_known_owners = get_owners_for_files(
        changed_files, codeowners_rules, maintainers
    )

    # Candidate pools excluding PR author
    cand_file_owners = [o for o in sorted(specific_file_owners) if o.lower() != pr_author.lower()]
    cand_maintainers = [m for m in sorted(maintainers) if m.lower() != pr_author.lower()]
    cand_all_owners = [o for o in sorted(all_known_owners) if o.lower() != pr_author.lower()]

    # Deterministic sorting using PR number as seed offset
    def deterministic_sort(items):
        return sorted(items, key=lambda x: (hash(f"{pr_number}:{x}") & 0x7fffffff))

    sorted_file_owners = deterministic_sort(cand_file_owners)
    sorted_maintainers = deterministic_sort(cand_maintainers)
    sorted_all_owners = deterministic_sort(cand_all_owners)

    # 1. Pick 2 reviewers
    selected_reviewers = []

    # Priority 1: specific file owners
    for user in sorted_file_owners:
        if user not in selected_reviewers:
            selected_reviewers.append(user)
        if len(selected_reviewers) == 2:
            break

    # Priority 2: general maintainers
    if len(selected_reviewers) < 2:
        for user in sorted_maintainers:
            if user not in selected_reviewers:
                selected_reviewers.append(user)
            if len(selected_reviewers) == 2:
                break

    # Priority 3: all known owners
    if len(selected_reviewers) < 2:
        for user in sorted_all_owners:
            if user not in selected_reviewers:
                selected_reviewers.append(user)
            if len(selected_reviewers) == 2:
                break

    # 2 & 3. Select 1 assignee (an owner)
    selected_assignee = None

    # Check if any reviewer is a specific file owner
    reviewer_file_owners = [r for r in selected_reviewers if r in cand_file_owners]
    if reviewer_file_owners:
        selected_assignee = deterministic_sort(reviewer_file_owners)[0]
    else:
        # Check if any reviewer is a maintainer or owner
        reviewer_owners = [r for r in selected_reviewers if r in cand_maintainers or r in cand_all_owners]
        if reviewer_owners:
            selected_assignee = deterministic_sort(reviewer_owners)[0]
        elif cand_file_owners:
            selected_assignee = sorted_file_owners[0]
        elif cand_maintainers:
            selected_assignee = sorted_maintainers[0]
        elif cand_all_owners:
            selected_assignee = sorted_all_owners[0]

    return selected_reviewers, selected_assignee


def run_gh_command(args):
    """Executes a gh CLI command and returns stdout."""
    res = subprocess.run(["gh"] + args, capture_output=True, text=True, check=True)
    return res.stdout


def main():
    parser = argparse.ArgumentParser(description="Auto assign PR reviewers and assignee.")
    parser.add_argument("--pr-number", required=True, type=int, help="PR number")
    parser.add_argument("--pr-author", required=True, type=str, help="PR author login")
    parser.add_argument("--repo", type=str, default="", help="GitHub repository (OWNER/REPO)")
    parser.add_argument("--codeowners-path", type=str, default="CODEOWNERS", help="Path to CODEOWNERS")
    parser.add_argument("--leads-path", type=str, default="LEADS.md", help="Path to LEADS.md")
    parser.add_argument("--apply", action="store_true", help="Apply changes via gh CLI")

    args = parser.parse_args()

    repo_flag = ["--repo", args.repo] if args.repo else []

    try:
        # Get changed files in PR
        files_json = run_gh_command(["pr", "view", str(args.pr_number)] + repo_flag + ["--json", "files"])
        files_data = json.loads(files_json)
        changed_files = [f["path"] for f in files_data.get("files", [])]

        # Parse CODEOWNERS and LEADS.md
        codeowners_rules = parse_codeowners(args.codeowners_path)
        maintainers = parse_leads(args.leads_path)

        # Pick reviewers and assignee
        selected_reviewers, selected_assignee = select_reviewers_and_assignee(
            pr_author=args.pr_author,
            changed_files=changed_files,
            codeowners_rules=codeowners_rules,
            maintainers=maintainers,
            pr_number=args.pr_number
        )

        print(f"PR #{args.pr_number} Author: {args.pr_author}")
        print(f"Changed files count: {len(changed_files)}")
        print(f"Selected Reviewers: {selected_reviewers}")
        print(f"Selected Assignee: {selected_assignee}")

        if not args.apply:
            print("Dry-run mode. Use --apply to execute gh pr edit.")
            return

        # Fetch current requested reviewers & assignees to remove extras if needed
        pr_json = run_gh_command(["pr", "view", str(args.pr_number)] + repo_flag + ["--json", "reviewRequests,assignees"])
        pr_data = json.loads(pr_json)

        current_reviewers = [r.get("login") for r in pr_data.get("reviewRequests", []) if "login" in r]
        current_teams = [r.get("slug") or r.get("name") for r in pr_data.get("reviewRequests", []) if "slug" in r or "name" in r]
        current_assignees = [a.get("login") for a in pr_data.get("assignees", []) if "login" in a]

        # Remove reviewers that are not in selected_reviewers, as well as team review requests
        reviewers_to_remove = [r for r in current_reviewers if r not in selected_reviewers]
        teams_to_remove = [t for t in current_teams]

        # Apply reviewer changes
        if selected_reviewers:
            reviewers_str = ",".join(selected_reviewers)
            print(f"Adding reviewers: {reviewers_str}")
            run_gh_command(["pr", "edit", str(args.pr_number)] + repo_flag + ["--add-reviewer", reviewers_str])

        for r in reviewers_to_remove:
            print(f"Removing excess reviewer: {r}")
            run_gh_command(["pr", "edit", str(args.pr_number)] + repo_flag + ["--remove-reviewer", r])

        for t in teams_to_remove:
            print(f"Removing team reviewer request: {t}")
            run_gh_command(["pr", "edit", str(args.pr_number)] + repo_flag + ["--remove-reviewer", t])

        # Apply assignee changes
        if selected_assignee and selected_assignee not in current_assignees:
            print(f"Assigning owner: {selected_assignee}")
            run_gh_command(["pr", "edit", str(args.pr_number)] + repo_flag + ["--add-assignee", selected_assignee])

    except Exception as e:
        print(f"::warning::Auto-assign failed: {e}. Falling back to default GitHub CODEOWNERS behavior.")
        sys.exit(0)


if __name__ == "__main__":
    main()

