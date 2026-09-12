#!/usr/bin/env bash
# collect-sboms.sh <tag> <image>...
#
# Extracts the SPDX documents buildx attached to each pushed image index and
# writes them as release ready files. There is one per image per platform and a
# sbom-digests.txt recording the image digest each file describes.
#
# Usage:
#   REGISTRY=ghcr.io/llm-d OUTDIR=sbom ./scripts/collect-sboms.sh v0.11.0 \
#     llm-d-router-endpoint-picker llm-d-router-disagg-sidecar llm-d-router-coordinator
set -euo pipefail

registry="${REGISTRY:-ghcr.io/llm-d}"
outdir="${OUTDIR:-sbom}"

tag="${1:?tag required}"
shift

mkdir -p "$outdir"
: > "$outdir/sbom-digests.txt"

for image in "$@"; do
  ref="$registry/$image:$tag"
  # The index listing drives the loop so the platforms collected are the ones
  # the image was built for.
  digests=$(docker buildx imagetools inspect "$ref" --format \
    '{{ range .Manifest.Manifests }}{{ if .Platform }}{{ .Platform.OS }}/{{ .Platform.Architecture }} {{ .Digest }}{{ println }}{{ end }}{{ end }}')
  [ -n "$digests" ] || { echo "$ref has no platform manifests" >&2; exit 1; }

  while read -r platform digest; do
    # buildx lists its attestation manifests as unknown/unknown.
    if [ "$platform" = "unknown/unknown" ]; then
      continue
    fi

    file="$image-$tag-${platform//\//-}.spdx.json"
    # Validate before moving so a failed extraction never lands at the final path.
    docker buildx imagetools inspect "$ref" \
      --format "{{ json (index .SBOM \"$platform\").SPDX }}" > "$outdir/.$file"
    jq -e '.spdxVersion' "$outdir/.$file" > /dev/null
    mv "$outdir/.$file" "$outdir/$file"

    printf '%s  %s@%s\n' "$file" "$ref" "$digest" >> "$outdir/sbom-digests.txt"
  done <<< "$digests"
done
