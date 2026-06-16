---
name: New Release
about: Propose a new release
title: Release vX.Y.Z
labels: kind/release
assignees: ''

---

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Release Process](#release-process)
- [Announce the Release](#announce-the-release)
- [Final Steps](#final-steps)

## Introduction

This document defines the process for releasing llm-d-router.

## Prerequisites

1. Permissions to push to the llm-d-router repository.

1. Membership in the `@llm-d/router-release-managers` team. Tag protection on
   `refs/tags/v*` restricts who can push release tags, which is what triggers
   the release build.

1. Set the required environment variables based on the expected release number:

   ```shell
   export MAJOR=0
   export MINOR=1
   export PATCH=0
   export REMOTE=origin
   ```

1. If creating a release candidate, set the release candidate number.

   ```shell
   export RC=1
   ```
1. If needed, clone the llm-d-router [repo].

   ```shell
   git clone -o ${REMOTE} git@github.com:llm-d/llm-d-router.git
   ```

## Release Process

### Create or Checkout branch 

1. If you already have the repo cloned, ensure it's up-to-date and your local branch is clean.

1. Release Branch Handling:
   - For a Release Candidate:
     Create a new release branch from the `main` branch. The branch should be named `release-${MAJOR}.${MINOR}`, for example, `release-0.1`:

     ```shell
     git checkout -b release-${MAJOR}.${MINOR}
     ```

   - For a Major, Minor or Patch Release:
     A release branch should already exist. In this case, check out the existing branch:

     ```shell
     git checkout release-${MAJOR}.${MINOR} ${REMOTE}/release-${MAJOR}.${MINOR}
     ```

1. Push your release branch to the llm-d-router remote.

    ```shell
    git push ${REMOTE} release-${MAJOR}.${MINOR}
    ```

### Tag commit and trigger image build

1. Tag the head of your release branch with the sem-ver release version.

   For a release candidate:

    ```shell
    git tag -s -a v${MAJOR}.${MINOR}.${PATCH}-rc.${RC} -m "llm-d-router v${MAJOR}.${MINOR}.${PATCH}-rc.${RC} Release Candidate"
    ```

   For a major, minor or patch release:

    ```shell
    git tag -s -a v${MAJOR}.${MINOR}.${PATCH} -m "llm-d-router v${MAJOR}.${MINOR}.${PATCH} Release"
    ```

1. Push the tag to the llm-d-router repo.

   For a release candidate:

    ```shell
    git push ${REMOTE} v${MAJOR}.${MINOR}.${PATCH}-rc.${RC}
    ```

   For a major, minor or patch release:

    ```shell
    git push ${REMOTE} v${MAJOR}.${MINOR}.${PATCH}
    ```

1. Pushing the tag triggers CI action to build and publish the EPP image (`ghcr.io/llm-d/llm-d-router-endpoint-picker`) and sidecar image (`ghcr.io/llm-d/llm-d-router-disagg-sidecar`) to the [ghcr registry].
1. Verify the [CI release workflow] completed successfully before proceeding.
1. Test the steps in the tagged quickstart guide after the PR merges.

### Create the release!

1. Create a [new release]:
    1. Choose the tag that you created for the release.
    1. Use the tag as the release title, e.g. `v0.1.0`.
    1. Click "Generate release notes" and preview the release body.
    1. Ensure the release body includes: highlights, breaking changes (if any), known issues, and upgrade steps.
    1. If this is a release candidate, select the "This is a pre-release" checkbox.
1. If you find any bugs in this process, create an [issue].

## Announce the Release

Use the following steps to announce the release.

1. Send an announcement email to `llm-d-contributors@googlegroups.com` with the subject:

   ```shell
   [ANNOUNCE] llm-d-router v${MAJOR}.${MINOR}.${PATCH} is released
   ```

1. Add a link to the final release in this issue.

1. Close this issue.

[repo]: https://github.com/llm-d/llm-d-router
[ghcr registry]: https://github.com/orgs/llm-d/packages?repo_name=llm-d-router
[new release]: https://github.com/llm-d/llm-d-router/releases/new
[issue]: https://github.com/llm-d/llm-d-router/issues/new/choose
[CI release workflow]: https://github.com/llm-d/llm-d-router/actions/workflows/ci-release.yaml
