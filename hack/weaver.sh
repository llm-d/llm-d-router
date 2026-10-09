#!/usr/bin/env bash

# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=0.27.0
case "$(uname -s)/$(uname -m)" in
    Linux/x86_64)
        platform=x86_64-unknown-linux-gnu
        checksum=7e497eb48215a5240a0bf77747cf431596c468f1039d481807b1f1e096cb5ac2 ;;
    Linux/aarch64|Linux/arm64)
        platform=aarch64-unknown-linux-gnu
        checksum=5bb1ebc3c5c06ee0ceac8c65243805fe819b1484cb2ca1c22d9941f53c989962 ;;
    Darwin/x86_64)
        platform=x86_64-apple-darwin
        checksum=b5a01ddd360bd27676732fc44922c1fe9e5f494db24c4d353e66e9b97533b70e ;;
    Darwin/arm64)
        platform=aarch64-apple-darwin
        checksum=11c9ce7364b21c9d73a02dc3e448346a2b1db4a06ae6d045cb383eecbfa700b2 ;;
    *) echo "Unsupported Weaver platform: $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

binary="$root/bin/weaver-$version-$platform"
if [[ ! -x "$binary" ]]; then
    mkdir -p "$root/bin"
    tmp=$(mktemp -d "$root/bin/.weaver.XXXXXX")
    trap 'rm -rf "$tmp"' EXIT
    archive="weaver-$platform.tar.xz"
    curl --fail --location --silent --show-error --retry 3 --connect-timeout 15 --max-time 180 \
        "https://github.com/open-telemetry/weaver/releases/download/v$version/$archive" \
        --output "$tmp/$archive"
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$tmp/$archive")
    else
        actual=$(shasum -a 256 "$tmp/$archive")
    fi
    [[ "${actual%% *}" == "$checksum" ]] || { echo 'Weaver checksum mismatch' >&2; exit 1; }
    tar -xJf "$tmp/$archive" -C "$tmp"
    mv "$tmp/weaver-$platform/weaver" "$binary"
    rm -rf "$tmp"
    trap - EXIT
fi

[[ "$("$binary" --version)" == "weaver $version" ]] || { echo 'Unexpected Weaver version' >&2; exit 1; }
exec "$binary" "$@"
