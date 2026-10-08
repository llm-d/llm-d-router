# Issue Triage

Every open issue carries either `needs-triage` or exactly one `triage/*` label. `.github/workflows/issue-triage.yaml` adds `needs-triage` to a new or reopened issue that has no `triage/*` label, and removes it when a `triage/*` label is applied.

## Triage labels

| Label | Meaning |
|---|---|
| `triage/accepted` | Ready to be worked on |
| `triage/needs-information` | The reporter must add details before work can start |
| `triage/not-reproducible` | Could not be reproduced as described |
| `triage/duplicate` | Covered by another open issue |

`triage/*` labels are mutually exclusive. Applying one removes any other.

`triage/accepted` requires at least one `kind/*` label and one `area/*` label on the issue. Without them, the workflow rejects `/triage accepted`, removes a `triage/accepted` label applied through the UI, and posts a comment that says what is needed. `/kind` and `/area` lines in the same comment as `/triage accepted` count.

## Comment commands

Each command goes on its own line and starts at the first column. Commands inside code blocks and HTML comments are ignored. A command accepts several values separated by spaces, for example `/area epp scheduling`.

| Command | Effect | Who can use it |
|---|---|---|
| `/triage <name>` | Replace the `triage/*` label with `triage/<name>` | Triage role or higher |
| `/kind <name>`, `/area <name>` | Add `kind/<name>` or `area/<name>` (see [area_taxonomy.md](area_taxonomy.md)) | Triage role or higher |
| `/remove-triage <name>`, `/remove-kind <name>`, `/remove-area <name>` | Remove the label | Triage role or higher |
| `/assign`, `/unassign` | Assign or unassign the commenter | Anyone |
| `/assign @user`, `/unassign @user` | Assign or unassign the named user | Triage role or higher |

GitHub only allows assigning users who have access to the repository or who have commented on the issue. A request to assign any other user is rejected.

The workflow reacts to the comment with a thumbs up when every command was applied. It reacts with a confused face when a command was rejected, for example because the label does not exist or the commenter lacks permission. The workflow run log names the rejected command.

The commands apply to issues only. Comments on pull requests are ignored. Pull requests take `/kind` and `/area` from the PR body.

### Interaction with the issue body

`issue-kind-label.yaml` sets `kind/*` and `area/*` labels from `/kind` and `/area` lines in the issue body. It runs only when the issue is opened, so that editing the body later does not overwrite labels a triager set. After an issue is opened, change its labels with comment commands or through the UI.

## Assigned issues

An assigned issue is expected to get a pull request. `.github/workflows/issue-assignment-check.yaml` runs daily and checks every open issue that has an assignee and no pull request. An issue has a pull request when an open pull request, draft or not, is linked to it, or when an open pull request by one of the assignees mentions it. Closed pull requests do not count.

For an issue without one:

1. If no assignee has commented for 30 days, it posts one comment asking the assignees for a status update.
2. If no assignee comments in the 14 days after that, it removes the assignees, adds `help wanted`, and posts a comment explaining why.

Only comments from assignees count as activity. Comments from other users and from bots do not. An assignee who has not commented is counted from the date of the assignment. A comment from an assignee after the reminder restarts the 30 days. The workflow does not close issues.

The workflow reads the last 100 comments, the last 100 cross-references and the first 20 linked pull requests of an issue. An issue with more than that is skipped and named in the run log, so it is never unassigned on a partial view.

This check is independent of `stale.yaml`, which marks issues stale after 90 days of inactivity from anyone.

