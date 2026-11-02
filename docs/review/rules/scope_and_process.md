# Scope and process

## Settle design tradeoffs on public issues and scope auto-close keywords
Discuss architectural changes and behavior-changing requirements on public GitHub issues rather than private trackers before iterating on pull requests. Use auto-closing keywords (`Fixes #`, `Closes #`) only when a pull request resolves the entire issue, and use `Part of #` on multi-slice work.
- Accept: Narrow wire-formatting adjustments motivated by an external conformance verifier when the issue and pull request state that compatibility scope.
- Open: Whether accepted design proposals belong in an in-repo `proposals/` directory or in GitHub issues and linked design documents.

## Use `NONE` release notes for internal changes and keep PR metadata synchronized
Set the `release-note` block to `NONE` for internal refactors, tests, CI edits, and fixes to unshipped features on `main`, and write operator-focused notes when defaults, flags, metrics, headers, or upgrade steps change. Update the pull request title, description, and release note whenever review iterations rename symbols or narrow scope.
