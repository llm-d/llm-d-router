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

1. Choose whether you are releasing a release candidate or an official release, and set the environment variables accordingly:

   - For a **Release Candidate** (e.g. `v0.9.0-rc.1`):
     ```shell
     export VERSION=v0.9.0-rc.1
     export BRANCH_VERSION=0.9
     export REMOTE=origin
     ```

   - For an **Official Release** (e.g. `v0.9.0`):
     ```shell
     export VERSION=v0.9.0
     export BRANCH_VERSION=0.9
     export REMOTE=origin
     ```

1. (Optional) If the latency predictor release version does **not** align with the router version, also set the expected tag (refer to the [latency predictor releases] to find the latest valid release tag):

   ```shell
   export LATENCY_PREDICTOR_TAG=v0.8.0-rc.1
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
     Create a new release branch from the `main` branch. The branch should be named `release-${BRANCH_VERSION}`, for example, `release-0.9`:

     ```shell
     git checkout -b release-${BRANCH_VERSION}
     ```

   - For a Major, Minor or Patch Release:
     A release branch should already exist. In this case, check out the existing branch:

     ```shell
     git checkout release-${BRANCH_VERSION} ${REMOTE}/release-${BRANCH_VERSION}
     ```

1. By default, `LATENCY_PREDICTOR_TAG` in the `Makefile` automatically resolves to the router release version (`${BUILD_REF}`). If the latency predictor tag does **not** align with the router version, update the default value of `LATENCY_PREDICTOR_TAG` in the `Makefile` to match your exported `${LATENCY_PREDICTOR_TAG}`.
   Commit the change (if modified):

    ```shell
    # Update LATENCY_PREDICTOR_TAG ?= vX.Y.Z in Makefile
    git commit -a -s -m "release: set LATENCY_PREDICTOR_TAG to ${LATENCY_PREDICTOR_TAG}"
    ```

1. Push your release branch to the llm-d-router remote.

    ```shell
    git push ${REMOTE} release-${BRANCH_VERSION}
    ```

### Tag commit and trigger image build

1. Tag the head of your release branch with the version:

     ```shell
     git tag -s -a ${VERSION} -m "llm-d-router ${VERSION} Release"
     ```

1. Push the tag to the llm-d-router repo:

     ```shell
     git push ${REMOTE} ${VERSION}
     ```

1. Pushing the tag triggers CI action to build and publish the EPP image (`ghcr.io/llm-d/llm-d-router-endpoint-picker`) and sidecar image (`ghcr.io/llm-d/llm-d-router-disagg-sidecar`) to the [ghcr registry].
1. Verify the [CI release workflow] completed successfully before proceeding.
1. Test the steps in the tagged quickstart guide after the PR merges.

### Create the release!

1. Create a [new release]:
    1. Choose the tag that you created for the release.
    1. Use the tag as the release title, e.g. `v0.1.0`.
    1. Click "Generate release notes" to auto-populate the list of PRs and contributors.
    1. Summarize the release notes using an LLM of your choice (e.g., Gemini, Copilot, ChatGPT). Provide the newly compiled release notes block from `RELEASE-NOTES.md` (or the unreleased fragments in `release-notes.d/unreleased/`) with the following prompt:

       ```text
       Please summarize these release notes into three clear sections:
       1. Highlights (key features, performance wins, bug fixes)
       2. Upgrade Steps & Deprecations (configuration changes, deprecated flags/metrics)
       3. Known Issues (if any, otherwise omit)
       ```

       Review the generated content, edit it if necessary to ensure accuracy, and then copy and prepend this summary at the very top of the release description box on GitHub.
    1. If this is a release candidate, select the "This is a pre-release" checkbox.
1. If you find any bugs in this process, create an [issue].

## Announce the Release

Use the following steps to announce the release.

1. Generate the announcement email content by running the following block in your terminal (make sure `${VERSION}` is set in your current shell):

   ```shell
   cat <<EOF
   Subject: [ANNOUNCE] llm-d-router ${VERSION} is released

   Hi all,

   We are pleased to announce the release of llm-d-router ${VERSION}!

   ### Container Images
   * Endpoint Picker: ghcr.io/llm-d/llm-d-router-endpoint-picker:${VERSION}
   * Disaggregated Sidecar: ghcr.io/llm-d/llm-d-router-disagg-sidecar:${VERSION}

   ### Helm Charts (OCI)
   * Standalone Chart: oci://ghcr.io/llm-d/charts/llm-d-router-standalone (version ${VERSION})
   * Gateway Chart: oci://ghcr.io/llm-d/charts/llm-d-router-gateway (version ${VERSION})

   ### Release Notes
   For more details, please see the GitHub release notes: https://github.com/llm-d/llm-d-router/releases/tag/${VERSION}
   EOF
   ```

1. Copy the generated subject and body, and send an email to `llm-d-contributors@googlegroups.com`.

1. Add a link to the final release in this issue.

1. Close this issue.

[repo]: https://github.com/llm-d/llm-d-router
[ghcr registry]: https://github.com/orgs/llm-d/packages?repo_name=llm-d-router
[new release]: https://github.com/llm-d/llm-d-router/releases/new
[issue]: https://github.com/llm-d/llm-d-router/issues/new/choose
[CI release workflow]: https://github.com/llm-d/llm-d-router/actions/workflows/ci-release.yaml
[latency predictor releases]: https://github.com/orgs/llm-d/packages?repo_name=llm-d-latency-predictor
