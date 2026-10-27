# Verifying Published Artifacts

Container images published by this repository carry a signed SLSA provenance
attestation and an SPDX software bill of materials (SBOM). Verification
establishes that an image was built by this repository's build workflow and
the SBOM enumerates what the image contains.

## Coverage

| Artifact | Provenance attestation | SBOM |
|---|---|---|
| `ghcr.io/llm-d/llm-d-router-endpoint-picker` | yes | yes |
| `ghcr.io/llm-d/llm-d-router-disagg-sidecar` | yes | yes |
| `ghcr.io/llm-d/llm-d-router-coordinator` | yes | yes |
| Helm charts under `ghcr.io/llm-d/charts/` | no | no |

Development images built from `main` use the same names with a `-dev` suffix
(`llm-d-router-endpoint-picker-dev`) and are covered identically.

Release images tagged `v0.10.0` and earlier carry neither an attestation nor
an SBOM. The commands below report a missing attestation for them.

## Prerequisites

Verification needs either [`gh`](https://cli.github.com) or
[`cosign`](https://github.com/sigstore/cosign). Reading the SBOM needs
`docker buildx` and `jq` for processing the output.

`gh` must be 2.51.0 or later for `--signer-workflow` and 2.64.0 or later for
`--bundle-from-oci`. It requires authentication (`gh auth login`) because it
queries the GitHub API for the attestation.

`cosign` must be 2.4.1 or later which reads Sigstore bundles. It needs no
authentication.

## Resolve the tag to a digest first

Verify by digest, not by tag. A tag such as `main` or `latest` is mutable and
the build workflow repoints it a few seconds before the attestation for the new
digest is published. A tag based verification run in that window fails against a
correctly signed image.

```console
$ IMAGE=ghcr.io/llm-d/llm-d-router-endpoint-picker-dev
$ DIGEST=$(docker buildx imagetools inspect "$IMAGE:main" --format '{{.Manifest.Digest}}')
$ echo "$DIGEST"
sha256:0dc1fdbd6de4dc6f0cb542561f3ecdde2bea0a66407adabea20a1948e608d265
```

Use `$IMAGE@$DIGEST` for every command below.

## Verify provenance

### The signing identity is the reusable workflow

Images are built by `.github/workflows/ci-build-images.yaml` which runs as a
reusable workflow called by `ci-dev.yaml` and `ci-release.yaml`. The OIDC
certificate identity is the reusable workflow's path, not the caller's.
Verification commands must name it. A command that names `ci-release.yaml`
fails against a correctly signed release image.

### With `gh`

```console
$ gh attestation verify "oci://$IMAGE@$DIGEST" \
    --repo llm-d/llm-d-router \
    --signer-workflow llm-d/llm-d-router/.github/workflows/ci-build-images.yaml
Loaded digest sha256:0dc1fdbd6de4dc6f0cb542561f3ecdde2bea0a66407adabea20a1948e608d265 for oci://ghcr.io/llm-d/llm-d-router-endpoint-picker-dev@sha256:0dc1fdbd6de4dc6f0cb542561f3ecdde2bea0a66407adabea20a1948e608d265
Loaded 1 attestation from GitHub API

The following policy criteria will be enforced:
- Predicate type must match:................ https://slsa.dev/provenance/v1
- Source Repository Owner URI must match:... https://github.com/llm-d
- Source Repository URI must match:......... https://github.com/llm-d/llm-d-router
- Subject Alternative Name must match regex: ^https://github.com/llm-d/llm-d-router/.github/workflows/ci-build-images.yaml
- OIDC Issuer must match:................... https://token.actions.githubusercontent.com

Verification succeeded!
```

The attestation bundle is stored both in the GitHub API and in the registry as
an OCI referrer. Adding `--bundle-from-oci` reads it from the registry, which
avoids a dependency on GitHub API availability:

```console
$ gh attestation verify "oci://$IMAGE@$DIGEST" --bundle-from-oci \
    --repo llm-d/llm-d-router \
    --signer-workflow llm-d/llm-d-router/.github/workflows/ci-build-images.yaml
```

`gh attestation verify` writes its report to standard error and exits 0 on
success. When output is redirected it prints nothing, so scripts should test
the exit status.

### With `cosign`

```console
$ cosign verify-attestation \
    --type slsaprovenance1 \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp '^https://github\.com/llm-d/llm-d-router/\.github/workflows/ci-build-images\.yaml@.*$' \
    "$IMAGE@$DIGEST"

Verification for ghcr.io/llm-d/llm-d-router-endpoint-picker-dev@sha256:0dc1fdbd6de4dc6f0cb542561f3ecdde2bea0a66407adabea20a1948e608d265 --
The following checks were performed on each of these signatures:
  - The cosign claims were validated
  - Existence of the claims in the transparency log was verified offline
  - The code-signing certificate was verified using trusted certificate authority certificates
```

The verified in-toto statement is written to standard output. Its `predicate`
holds the SLSA provenance, including the workflow run that produced the image:

```console
$ cosign verify-attestation ... "$IMAGE@$DIGEST" 2>/dev/null \
    | jq -r '.payload | @base64d | fromjson | .predicate.runDetails.builder.id'
https://github.com/llm-d/llm-d-router/.github/workflows/ci-build-images.yaml@refs/heads/main
```

### Confirm the check can fail

A verification command that silently matches nothing is indistinguishable from
one that passes. To confirm a command discriminates, rerun it with a signer
identity that should not match and check that it fails:

```console
$ gh attestation verify "oci://$IMAGE@$DIGEST" \
    --repo llm-d/llm-d-router \
    --signer-workflow llm-d/llm-d-router/.github/workflows/ci-release.yaml
Sigstore verification failed

Error: verifying with issuer "sigstore.dev"
```

## What the signature covers

The attestation subject is the digest of the image index and the index
references the per-architecture images alongside the buildkit attestation
manifests that hold the provenance and the SBOM:

```console
$ docker buildx imagetools inspect "$IMAGE@$DIGEST" --raw \
    | jq -r '.manifests[] | "\(.platform.os)/\(.platform.architecture)  \(.annotations["vnd.docker.reference.type"] // "image")"'
linux/amd64  image
linux/arm64  image
unknown/unknown  attestation-manifest
unknown/unknown  attestation-manifest
```

Because the index digest commits to every entry, verifying it covers both
architectures and the SBOM without a separate SBOM signature.

## Read the SBOM

The SBOM is attached in-registry as an SPDX document per architecture.

```console
$ docker buildx imagetools inspect "$IMAGE@$DIGEST" --format '{{ json .SBOM }}' | jq -r 'keys[]'
linux/amd64
linux/arm64
```

Count the packages in one architecture:

```console
$ docker buildx imagetools inspect "$IMAGE@$DIGEST" --format '{{ json .SBOM }}' \
    | jq '.["linux/amd64"].SPDX.packages | length'
116
```

The document covers both OS packages and Go modules. To check whether an image
contains a package affected by an advisory and at which version:

```console
$ docker buildx imagetools inspect "$IMAGE@$DIGEST" --format '{{ json .SBOM }}' \
    | jq -r '.["linux/amd64"].SPDX.packages[]
             | select(.name | test("golang.org/x/net"))
             | "\(.name) \(.versionInfo)"'
golang.org/x/net v0.58.0
```

Write the SPDX document to a file for a scanner that consumes SPDX:

```console
$ docker buildx imagetools inspect "$IMAGE@$DIGEST" --format '{{ json .SBOM }}' \
    | jq '.["linux/amd64"].SPDX' > epp-sbom.spdx.json
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no attestations found` against a tag which succeeds against a digest | The tag moved to an image whose attestation is not yet published. Resolve the tag to a digest and retry. |
| `no attestations found` for a release at `v0.10.0` or earlier | Those images carry no attestation. See Coverage. |
| `Sigstore verification failed` (`gh`) or `failed to verify certificate identity` (`cosign`) | The signer identity names a calling workflow. Use `ci-build-images.yaml`. |
| `gh` prints nothing and exits 0 | Output is redirected. The report goes to standard error. Test the exit status. |
| `.SBOM` returns `{}` | The artifact carries no SBOM. See Coverage. |

## Related

- [SECURITY.md](../SECURITY.md) — vulnerability reporting
- [`.github/workflows/ci-build-images.yaml`](../.github/workflows/ci-build-images.yaml) — the build, signing, and SBOM workflow
