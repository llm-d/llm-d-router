#!/usr/bin/env python3
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from auto_assign_reviewers import (
    parse_codeowners,
    match_file_against_pattern,
    get_owners_for_files,
    select_reviewers_and_assignee,
)

class TestAutoAssignReviewers(unittest.TestCase):
    def setUp(self):
        self.codeowners_rules = [
            ("*", ["llm-d/router-maintainers"]),
            ("/pkg/epp/flowcontrol/", ["LukeAVanDrie", "shmuelk", "llm-d/router-maintainers"]),
            ("/pkg/epp/framework/plugins/scheduling/scorer/prefix/", ["liu-cong", "vMaroon", "llm-d/router-maintainers"]),
            ("/cmd/coordinator/", ["shmuelk", "roytman", "llm-d/router-maintainers"]),
        ]
        self.maintainers = ["ahg-g", "elevran", "vMaroon", "liu-cong"]

    def test_match_file_against_pattern(self):
        self.assertTrue(match_file_against_pattern("/pkg/epp/flowcontrol/main.go", "/pkg/epp/flowcontrol/"))
        self.assertTrue(match_file_against_pattern("pkg/epp/flowcontrol/sub/main.go", "/pkg/epp/flowcontrol/"))
        self.assertFalse(match_file_against_pattern("/pkg/epp/config/main.go", "/pkg/epp/flowcontrol/"))
        self.assertTrue(match_file_against_pattern("/README.md", "*"))

    def test_get_owners_for_files(self):
        changed_files = ["/pkg/epp/flowcontrol/main.go"]
        specific, all_owners = get_owners_for_files(changed_files, self.codeowners_rules, self.maintainers)
        self.assertEqual(specific, {"LukeAVanDrie", "shmuelk"})
        self.assertIn("ahg-g", all_owners)
        self.assertIn("LukeAVanDrie", all_owners)

    def test_select_2_reviewers_and_assignee_external_author(self):
        # External author touching flowcontrol
        changed_files = ["/pkg/epp/flowcontrol/main.go"]
        reviewers, assignee = select_reviewers_and_assignee(
            pr_author="external-contributor",
            changed_files=changed_files,
            codeowners_rules=self.codeowners_rules,
            maintainers=self.maintainers,
            pr_number=101
        )
        self.assertEqual(len(reviewers), 2)
        self.assertIn("LukeAVanDrie", reviewers)
        self.assertIn("shmuelk", reviewers)
        # Requirement 3: assignee should be one of the reviewers who is an owner
        self.assertIn(assignee, reviewers)
        self.assertIn(assignee, ["LukeAVanDrie", "shmuelk"])

    def test_select_2_reviewers_and_assignee_author_is_file_owner(self):
        # Author is shmuelk (file owner for flowcontrol)
        changed_files = ["/pkg/epp/flowcontrol/main.go"]
        reviewers, assignee = select_reviewers_and_assignee(
            pr_author="shmuelk",
            changed_files=changed_files,
            codeowners_rules=self.codeowners_rules,
            maintainers=self.maintainers,
            pr_number=102
        )
        self.assertEqual(len(reviewers), 2)
        self.assertNotIn("shmuelk", reviewers)
        self.assertIn("LukeAVanDrie", reviewers)  # remaining file owner
        # Requirement 3: since LukeAVanDrie is both reviewer and file owner, assignee must be LukeAVanDrie
        self.assertEqual(assignee, "LukeAVanDrie")

    def test_select_2_reviewers_general_files(self):
        # General file touching root README.md
        changed_files = ["/README.md"]
        reviewers, assignee = select_reviewers_and_assignee(
            pr_author="external-user",
            changed_files=changed_files,
            codeowners_rules=self.codeowners_rules,
            maintainers=self.maintainers,
            pr_number=103
        )
        self.assertEqual(len(reviewers), 2)
        self.assertNotIn("external-user", reviewers)
        # Both reviewers are maintainers (owners)
        self.assertIn(assignee, reviewers)
        self.assertIn(assignee, self.maintainers)

    def test_select_reviewers_maintainer_author(self):
        # Author is liu-cong (maintainer)
        changed_files = ["/README.md"]
        reviewers, assignee = select_reviewers_and_assignee(
            pr_author="liu-cong",
            changed_files=changed_files,
            codeowners_rules=self.codeowners_rules,
            maintainers=self.maintainers,
            pr_number=104
        )
        self.assertEqual(len(reviewers), 2)
        self.assertNotIn("liu-cong", reviewers)
        self.assertNotIn("liu-cong", [assignee])
        self.assertIn(assignee, reviewers)

if __name__ == "__main__":
    unittest.main()
