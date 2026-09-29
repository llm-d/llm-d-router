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
  # The index digest is captured alongside the platform listing so the SBOM
  # lookups below can pin to it instead of the mutable tag. Otherwise a
  # second push to the same tag between the two inspections could pair one
  # image's platform digest with another image's SBOM. buildx lists its
  # attestation manifests as unknown/unknown, so they are left out here and
  # the check below also catches an index holding only attestations.
  listing=$(docker buildx imagetools inspect "$ref" --format \
    '{{ .Manifest.Digest }}{{ println }}{{ range .Manifest.Manifests }}{{ if .Platform }}{{ if ne .Platform.OS "unknown" }}{{ .Platform.OS }}/{{ .Platform.Architecture }} {{ .Digest }}{{ println }}{{ end }}{{ end }}{{ end }}')
  index_digest=$(head -n1 <<< "$listing")
  digests=$(tail -n +2 <<< "$listing")
  [ -n "$digests" ] || { echo "$ref has no platform manifests" >&2; exit 1; }

  pinned="$registry/$image@$index_digest"

  while read -r platform digest; do
    file="$image-$tag-${platform//\//-}.spdx.json"
    # Validate before moving so a failed extraction never lands at the final path.
    docker buildx imagetools inspect "$pinned" \
      --format "{{ json (index .SBOM \"$platform\").SPDX }}" > "$outdir/.$file"
    jq -e '.spdxVersion' "$outdir/.$file" > /dev/null
    mv "$outdir/.$file" "$outdir/$file"

    printf '%s  %s@%s\n' "$file" "$ref" "$digest" >> "$outdir/sbom-digests.txt"
  done <<< "$digests"
done
